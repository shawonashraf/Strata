// src/deepseekv4/dsv4ref.cpp - the deepseek4 CPU reference (include/strata/dsv4/dsv4ref.hpp carries the
// contract).  Transcribed piece by piece from deepseek-ai/DeepSeek-V4-Flash-0731's inference/model.py and
// inference/kernel.py; the line numbers cited below are that file's (fetched Oct 2026).
//
// What deliberately differs from the official code, and why (all stated for stage 2's acceptance):
//   * no QAT activation simulation - act_quant on kv's nope dims (model.py 512), Hadamard rotation plus
//     fp4_act_quant in the indexer (420-422) and the indexer compressor (374-376): those emulate the
//     FP8/FP4 deployment's rounding; a GGUF repack feeds a plain fp32 matmul, the q36ref convention.
//   * fp32 activations everywhere instead of bf16, and fp64 accumulation inside matvec (one ordering, one
//     source of truth; the dumps are compared with tolerance).
//   * one token at a time for the prompt as well, so only model.py's start_pos > 0 branches matter - its
//     start_pos == 0 branches with seqlen == 1 do the same thing (walked through at pos 0, pos 1..3 and
//     pos 127/128: window row = position while pos < 128, ring after; no pool entry before pos 3).
//   * DSpark (model.py 743-961: DSparkBlock/DSparkAttention/Markov and confidence heads, and the three
//     extra compress_ratios rows) is out of scope, as in stage 1: those tensors ship in a separate file.
#include "strata/dsv4/dsv4ref.hpp"

#include "strata/artifact/dequant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata {
namespace dsv4 {

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kInf = std::numeric_limits<double>::infinity();
double softplus(double x) { return x > 20.0 ? x : std::log1p(std::exp(x)); }
double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
}   // namespace

// ---- rope (precompute_freqs_cis + apply_rotary_emb, model.py 206-250) ----------------------------------

void Ref::Rope::build(int64_t upto, double base_theta, double factor, int64_t original, double beta_fast,
                      double beta_slow) {
    if (upto <= built) return;
    const int half = rd / 2;
    std::vector<double> freqs(static_cast<size_t>(half));
    for (int i = 0; i < half; ++i) freqs[i] = 1.0 / std::pow(base_theta, (2.0 * i) / rd);
    if (original > 0) {   // YaRN: a smooth ramp between the beta_fast and beta_slow correction ranges
        auto corr_dim = [&](double rot) {
            return rd * std::log(original / (rot * 2.0 * kPi)) / (2.0 * std::log(base_theta));
        };
        const int64_t low = std::max<int64_t>(0, (int64_t) std::floor(corr_dim(beta_fast)));
        const int64_t high = std::min<int64_t>(rd - 1, (int64_t) std::ceil(corr_dim(beta_slow)));
        const double span = (low == high) ? 0.001 : (double) (high - low);   // model.py 220-221
        for (int i = 0; i < half; ++i) {
            const double smooth = 1.0 - std::min(1.0, std::max(0.0, ((double) i - (double) low) / span));
            freqs[i] = freqs[i] / factor * (1.0 - smooth) + freqs[i] * smooth;
        }
    }
    const size_t old = (size_t) built * (size_t) half;
    cos.resize((size_t) upto * (size_t) half);
    sin.resize((size_t) upto * (size_t) half);
    for (int64_t p = built; p < upto; ++p)
        for (int i = 0; i < half; ++i) {
            cos[old + (size_t) (p - built) * (size_t) half + i] = (float) std::cos(freqs[i] * (double) p);
            sin[old + (size_t) (p - built) * (size_t) half + i] = (float) std::sin(freqs[i] * (double) p);
        }
    built = upto;
}

void Ref::Rope::apply(float* x, int64_t pos, int64_t head_dim, bool inverse) const {
    // apply_rotary_emb rotates x[..., head_dim-rd : head_dim] in adjacent pairs ((x0,x1),(x2,x3),...);
    // inverse takes the conjugate (model.py 238-250).
    const int half = rd / 2;
    const float* c = cos.data() + (size_t) pos * (size_t) half;
    const float* s = sin.data() + (size_t) pos * (size_t) half;
    float* p = x + head_dim - rd;
    for (int i = 0; i < half; ++i) {
        const float x0 = p[2 * i], x1 = p[2 * i + 1];
        const float cs = c[i], sn = inverse ? -s[i] : s[i];
        p[2 * i] = x0 * cs - x1 * sn;
        p[2 * i + 1] = x0 * sn + x1 * cs;
    }
}

void Ref::rope_for(const Layer& L, float* x, int64_t pos, int64_t head_dim, bool inverse) const {
    // sliding layers run plain rope_theta with YaRN off, compressed layers YaRN over the compress base
    // (model.py 481-487)
    (L.kind == core::Dsv4LayerKind::Sliding ? rope_plain : rope_comp).apply(x, pos, head_dim, inverse);
}

// ---- tensors -------------------------------------------------------------------------------------------

Ref::Weight& Ref::wref(const std::string& name) {
    auto it = weights_.find(name);
    if (it == weights_.end()) throw std::runtime_error("dsv4ref: internal, tensor not resolved: " + name);
    return it->second;
}

void Ref::dequant_row(const Weight& t, int64_t expert, int64_t row, float* out) const {
    const uint8_t* p = t.base + (uint64_t) expert * t.expert_bytes + (uint64_t) row * t.row_bytes;
    const int64_t n_in = (int64_t) t.info->shape[0];
    switch (t.info->type) {
        case 0: dequantize_f32(p, out, (int) n_in); break;
        case 1: dequantize_f16(p, out, (int) n_in); break;
        case 30: dequantize_bf16(p, out, (int) n_in); break;
        case 8:
            for (int64_t b = 0; b < n_in / 32; ++b) dequantize_q8_0(p + (size_t) b * 34, out + b * 32);
            break;
        case 39:
            for (int64_t b = 0; b < n_in / 32; ++b) dequantize_mxfp4(p + (size_t) b * 17, out + b * 32);
            break;
        case 42:
            for (int64_t b = 0; b < n_in / 64; ++b) dequantize_q2_0(p + (size_t) b * 18, out + b * 64);
            break;
        default:
            throw std::runtime_error("dsv4ref: tensor " + t.info->name + " type " +
                                     std::to_string(t.info->type) + " has no reference dequantizer");
    }
}

void Ref::matvec(const Weight& t, float* y, int64_t n_out, const float* x, int64_t n_in, int64_t expert,
                 int64_t row0) const {
    for (int64_t r = 0; r < n_out; ++r) {
        dequant_row(t, expert, row0 + r, scratch_.data());
        const float* w = scratch_.data();
        double acc = 0.0;
        for (int64_t k = 0; k < n_in; ++k) acc += (double) w[k] * x[k];
        y[r] = (float) acc;
    }
}

void Ref::rmsnorm(const float* x, const float* wt, int64_t n, float eps, float* y) {
    double ss = 0.0;
    for (int64_t i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const double rs = 1.0 / std::sqrt(ss / (double) n + (double) eps);   // RMSNorm, model.py 189-204
    for (int64_t i = 0; i < n; ++i) y[i] = (float) ((double) x[i] * rs * (double) wt[i]);
}

// ---- mHC (Block.hc_pre/hc_post model.py 680-693; hc_split_sinkhorn kernel.py 372-440) -------------------

void Ref::hc_pre(const Layer& L, bool ffn, const float* streams, float* y, float* post, float* comb) {
    const Weight& fn = ffn ? L.hc_ffn_fn : L.hc_attn_fn;
    const Weight& base = ffn ? L.hc_ffn_base : L.hc_attn_base;
    const Weight& scale = ffn ? L.hc_ffn_scale : L.hc_attn_scale;
    const int hc = (int) g_.hc;
    double ss = 0.0;                                       // rsqrt over the flattened streams (684)
    for (int64_t i = 0; i < hcdim_; ++i) ss += (double) streams[i] * streams[i];
    const double rsqrt = 1.0 / std::sqrt(ss / (double) hcdim_ + norm_eps);
    std::vector<float> mixes((size_t) mix_);
    matvec(fn, mixes.data(), mix_, streams, hcdim_);
    for (auto& m : mixes) m = (float) ((double) m * rsqrt);   // mixes carry the rsqrt (model.py 685)
    std::vector<float> fbase((size_t) mix_), fsc(3);
    dequant_row(base, 0, 0, fbase.data());
    dequant_row(scale, 0, 0, fsc.data());
    std::vector<float> pre((size_t) hc);
    for (int j = 0; j < hc; ++j)                          // pre = sigmoid(mix*s0 + base) + eps
        pre[j] = (float) (sigmoid((double) mixes[j] * fsc[0] + fbase[j]) + hc_eps);
    for (int j = 0; j < hc; ++j)                          // post = 2 * sigmoid(mix*s1 + base)
        post[j] = (float) (2.0 * sigmoid((double) mixes[hc + j] * fsc[1] + fbase[hc + j]));
    // comb: row softmax + eps, a column normalize, then sinkhorn_iters - 1 alternating rounds
    std::vector<double> cm((size_t) hc * hc);
    for (int j = 0; j < hc; ++j)
        for (int k = 0; k < hc; ++k) {
            const int t = 2 * hc + j * hc + k;
            cm[j * hc + k] = (double) mixes[t] * fsc[2] + fbase[t];
        }
    for (int j = 0; j < hc; ++j) {
        double mx = -kInf;
        for (int k = 0; k < hc; ++k) mx = std::max(mx, cm[j * hc + k]);
        double sum = 0.0;
        for (int k = 0; k < hc; ++k) cm[j * hc + k] = std::exp(cm[j * hc + k] - mx), sum += cm[j * hc + k];
        for (int k = 0; k < hc; ++k) cm[j * hc + k] = cm[j * hc + k] / sum + hc_eps;
    }
    for (int k = 0; k < hc; ++k) {
        double col = 0.0;
        for (int j = 0; j < hc; ++j) col += cm[j * hc + k];
        for (int j = 0; j < hc; ++j) cm[j * hc + k] /= col + hc_eps;
    }
    for (int it = 1; it < (int) g_.hc_sinkhorn; ++it) {
        for (int j = 0; j < hc; ++j) {
            double row = 0.0;
            for (int k = 0; k < hc; ++k) row += cm[j * hc + k];
            for (int k = 0; k < hc; ++k) cm[j * hc + k] /= row + hc_eps;
        }
        for (int k = 0; k < hc; ++k) {
            double col = 0.0;
            for (int j = 0; j < hc; ++j) col += cm[j * hc + k];
            for (int j = 0; j < hc; ++j) cm[j * hc + k] /= col + hc_eps;
        }
    }
    for (size_t i = 0; i < cm.size(); ++i) comb[i] = (float) cm[i];
    for (int64_t d = 0; d < g_.n_embd; ++d) {             // y = sum_j pre_j * x_j (model.py 687)
        double acc = 0.0;
        for (int j = 0; j < hc; ++j) acc += (double) pre[j] * streams[(size_t) j * g_.n_embd + d];
        y[d] = (float) acc;
    }
}

void Ref::hc_post(float* streams, const float* y, const float* residual, const float* post, const float* comb) {
    const int hc = (int) g_.hc;                           // y_j = post_j*attn + sum_k comb_jk*res_k (692)
    for (int j = 0; j < hc; ++j)
        for (int64_t d = 0; d < g_.n_embd; ++d) {
            double acc = (double) post[j] * y[d];
            for (int k = 0; k < hc; ++k) acc += (double) comb[j * hc + k] * residual[(size_t) k * g_.n_embd + d];
            streams[(size_t) j * g_.n_embd + d] = (float) acc;
        }
}

// ---- pool compressors (Compressor.forward's decode branch, model.py 349-383) ---------------------------

void Ref::compressor_step(Compressor& C, const float* x, int64_t pos, std::vector<float>& pool, int64_t entry) {
    const int64_t d = C.head_dim, cw = C.coff * d, ratio = C.ratio;
    std::vector<float> kv((size_t) cw), score((size_t) cw);
    matvec(C.wkv, kv.data(), cw, x, g_.n_embd);
    matvec(C.wgate, score.data(), cw, x, g_.n_embd);
    dequant_row(C.ape, 0, pos % ratio, scratch_.data());      // one APE row per slot of the block
    for (int64_t c = 0; c < cw; ++c) score[c] += scratch_[(size_t) c];
    float* ks = C.kv_state.data();
    float* ss = C.score_state.data();
    // every token writes its state row (model.py 349-354); the block pools only when it is complete
    const int64_t row = (C.coff == 2) ? ratio + pos % ratio : pos % ratio;
    std::memcpy(ks + (size_t) row * cw, kv.data(), (size_t) cw * 4);
    std::memcpy(ss + (size_t) row * cw, score.data(), (size_t) cw * 4);
    if ((pos + 1) % ratio != 0) return;
    // the block is complete: gate every state row per column and pool (model.py 355-365); with overlap the
    // row set is [old-window cols 0..d) + [current cols d..2d), and the current window becomes the old one
    std::vector<float> out((size_t) d, 0.0f);
    for (int64_t c = 0; c < d; ++c) {
        double mx = -kInf, sum = 0.0;
        if (C.coff == 2) {
            for (int64_t r = 0; r < ratio; ++r) mx = std::max(mx, (double) ss[(size_t) r * cw + c]);
            for (int64_t r = ratio; r < 2 * ratio; ++r) mx = std::max(mx, (double) ss[(size_t) r * cw + d + c]);
            for (int64_t r = 0; r < ratio; ++r)
                out[c] += (float) (std::exp((double) ss[(size_t) r * cw + c] - mx) * ks[(size_t) r * cw + c]);
            for (int64_t r = ratio; r < 2 * ratio; ++r)
                out[c] += (float) (std::exp((double) ss[(size_t) r * cw + d + c] - mx) *
                                   ks[(size_t) r * cw + d + c]);
            for (int64_t r = 0; r < ratio; ++r) sum += std::exp((double) ss[(size_t) r * cw + c] - mx);
            for (int64_t r = ratio; r < 2 * ratio; ++r) sum += std::exp((double) ss[(size_t) r * cw + d + c] - mx);
        } else {
            for (int64_t r = 0; r < ratio; ++r) mx = std::max(mx, (double) ss[(size_t) r * cw + c]);
            for (int64_t r = 0; r < ratio; ++r) {
                out[c] += (float) (std::exp((double) ss[(size_t) r * cw + c] - mx) * ks[(size_t) r * cw + c]);
                sum += std::exp((double) ss[(size_t) r * cw + c] - mx);
            }
        }
        out[c] = (float) ((double) out[c] / sum);
    }
    if (C.coff == 2) {   // shift: the rows just used as "current" become the "overlap" window
        std::vector<float> nk(ks, ks + (size_t) 2 * ratio * cw), ns(ss, ss + (size_t) 2 * ratio * cw);
        for (int64_t r = 0; r < ratio; ++r) {
            std::memcpy(ks + (size_t) r * cw, nk.data() + (size_t) (ratio + r) * cw, (size_t) cw * 4);
            std::memcpy(ss + (size_t) r * cw, ns.data() + (size_t) (ratio + r) * cw, (size_t) cw * 4);
        }
    }
    pool.resize((size_t) (entry + 1) * (size_t) d);
    float* dst = pool.data() + (size_t) entry * (size_t) d;
    std::vector<float> wn((size_t) d);
    dequant_row(C.norm, 0, 0, wn.data());                 // RMSNorm then rope (model.py 368-373)
    rmsnorm(out.data(), wn.data(), d, (float) norm_eps, dst);
    rope_comp.apply(dst, entry * ratio, d, false);        // entry e sits at token position e*ratio
}

// ---- attention (Attention.forward's decode branch, model.py 490-548; sparse_attn kernel.py 277-370) -----

void Ref::attention(Layer& L, const float* x, int64_t pos, float* out) {
    const int64_t NH = g_.n_head, HD = g_.head_dim, QL = g_.q_lora_rank;
    // qr = q_norm(wq_a(x)); q = wq_b(qr), each head scaled by its own rsqrt, rope on the last rd dims (501-505)
    matvec(L.q_a, qr.data(), QL, x, g_.n_embd);
    std::vector<float> qn((size_t) QL);
    dequant_row(L.q_a_norm, 0, 0, qn.data());
    rmsnorm(qr.data(), qn.data(), QL, (float) norm_eps, qr.data());
    matvec(L.q_b, q.data(), NH * HD, qr.data(), QL);
    for (int64_t h = 0; h < NH; ++h) {
        float* qh = q.data() + (size_t) h * (size_t) HD;
        double sq = 0.0;
        for (int64_t i = 0; i < HD; ++i) sq += (double) qh[i] * qh[i];
        const double rs = 1.0 / std::sqrt(sq / (double) HD + norm_eps);
        for (int64_t i = 0; i < HD; ++i) qh[i] = (float) ((double) qh[i] * rs);
        rope_for(L, qh, pos, HD);
    }
    // the shared K=V latent: normed, roped, written into the window ring (508-510, 535)
    matvec(L.kv, kv512.data(), HD, x, g_.n_embd);
    std::vector<float> kn((size_t) HD);
    dequant_row(L.kv_norm, 0, 0, kn.data());
    rmsnorm(kv512.data(), kn.data(), HD, (float) norm_eps, kv512.data());
    rope_for(L, kv512.data(), pos, HD);
    std::memcpy(L.ring.data() + (size_t) (pos % win_) * (size_t) HD, kv512.data(), (size_t) HD * 4);
    // candidates: the window rows (get_window_topk_idxs's decode branch, model.py 261-271)
    std::vector<const float*> cand;
    const int64_t nwin = std::min(win_, pos + 1);
    cand.reserve((size_t) nwin + (size_t) g_.idx_top_k + 8);
    for (int64_t r = 0; r < nwin; ++r) cand.push_back(L.ring.data() + (size_t) r * (size_t) HD);
    if (L.kind != core::Dsv4LayerKind::Sliding) {
        const int64_t entries = (pos + 1) / L.ratio;
        compressor_step(L.comp, x, pos, L.pool, entries - 1);      // may append entry entries-1
        if (L.kind == core::Dsv4LayerKind::Csa && L.idx_q_b.info) {
            // the Lightning indexer: its own pool, scores over it, top-k (Indexer.forward, model.py 408-439)
            compressor_step(L.idx_comp, x, pos, L.ikv, entries - 1);
            matvec(L.idx_q_b, iq.data(), g_.idx_q_heads * g_.idx_key_dim, qr.data(), QL);
            for (int64_t h = 0; h < g_.idx_q_heads; ++h)
                rope_comp.apply(iq.data() + (size_t) h * (size_t) g_.idx_key_dim, pos, g_.idx_key_dim, false);
            std::vector<float> iw((size_t) g_.idx_q_heads);
            matvec(L.idx_proj, iw.data(), g_.idx_q_heads, x, g_.n_embd);
            const double wsc = 1.0 / (std::sqrt((double) g_.idx_key_dim) * std::sqrt((double) g_.idx_q_heads));
            for (auto& v : iw) v = (float) ((double) v * wsc);
            std::vector<double> sc((size_t) entries);
            for (int64_t t = 0; t < entries; ++t) {
                const float* kvt = L.ikv.data() + (size_t) t * (size_t) g_.idx_key_dim;
                double acc = 0.0;
                for (int64_t h = 0; h < g_.idx_q_heads; ++h) {
                    double dot = 0.0;
                    const float* qh = iq.data() + (size_t) h * (size_t) g_.idx_key_dim;
                    for (int64_t i = 0; i < g_.idx_key_dim; ++i) dot += (double) qh[i] * kvt[i];
                    if (dot > 0.0) acc += dot * iw[h];
                }
                sc[t] = acc;
            }
            const int64_t topk = std::min(g_.idx_top_k, entries);
            std::vector<int64_t> ord((size_t) entries);
            for (int64_t t = 0; t < entries; ++t) ord[(size_t) t] = t;
            std::partial_sort(ord.begin(), ord.begin() + topk, ord.end(),
                              [&](int64_t a, int64_t b) { return sc[a] > sc[b]; });
            for (int64_t k = 0; k < topk; ++k)
                cand.push_back(L.pool.data() + (size_t) ord[(size_t) k] * (size_t) HD);
        } else {
            for (int64_t t = 0; t < entries; ++t) cand.push_back(L.pool.data() + (size_t) t * (size_t) HD);
        }
    }
    // sparse_attn: scores q.kv * HD^-0.5; the per-head sink is an extra denominator term exp(sink - max)
    // with no numerator (kernel.py 325-345)
    std::vector<float> sink((size_t) NH);
    dequant_row(L.sinks, 0, 0, sink.data());
    const double scale = 1.0 / std::sqrt((double) HD);
    for (int64_t h = 0; h < NH; ++h) {
        const float* qh = q.data() + (size_t) h * (size_t) HD;
        std::vector<double> s(cand.size());
        double mx = -kInf;
        for (size_t j = 0; j < cand.size(); ++j) {
            double dot = 0.0;
            for (int64_t i = 0; i < HD; ++i) dot += (double) qh[i] * cand[j][i];
            s[j] = dot * scale;
            if (s[j] > mx) mx = s[j];
        }
        std::vector<double> p(cand.size());
        double sum = std::exp((double) sink[h] - mx);
        for (size_t j = 0; j < cand.size(); ++j) { p[j] = std::exp(s[j] - mx); sum += p[j]; }
        for (int64_t i = 0; i < HD; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < cand.size(); ++j) acc += p[j] * cand[j][i];
            o_attn[(size_t) h * (size_t) HD + i] = (float) (acc / sum);
        }
    }
    // inverse rope on o (model.py 539), then the grouped low-rank output (541-547): wo_a's rows run
    // group-major (weight.view(n_groups, o_lora, -1)), one group of n_head/n_groups heads at a time
    for (int64_t h = 0; h < NH; ++h) rope_for(L, o_attn.data() + (size_t) h * (size_t) HD, pos, HD, true);
    const int64_t gs = (NH * HD) / g_.o_groups;
    for (int64_t gr = 0; gr < g_.o_groups; ++gr)
        matvec(L.o_a, olo.data() + (size_t) gr * (size_t) g_.o_lora_rank, g_.o_lora_rank,
               o_attn.data() + (size_t) gr * (size_t) gs, gs, 0, gr * g_.o_lora_rank);
    matvec(L.o_b, out, g_.n_embd, olo.data(), g_.o_groups * g_.o_lora_rank);
}

// ---- MoE (Gate/Expert/MoE, model.py 551-649) ------------------------------------------------------------

void Ref::moe(const Layer& L, const float* x, int token, float* out) {
    const int64_t NE = g_.n_expert, K = g_.n_expert_used, FF = g_.n_ff;
    matvec(L.gate_inp, gate_scores.data(), NE, x, g_.n_embd);
    for (auto& s : gate_scores) s = (float) std::sqrt(softplus(s));    // sqrtsoftplus (model.py 576)
    gate_w.assign((size_t) K, 0.0f);
    gate_ids.assign((size_t) K, 0);
    if (L.tid2eid.info) {   // hash layers: the frozen per-token table (582: tid2eid[input_ids]); GGUF
                            // [K, vocab] stores it exactly as torch's [vocab, K] lays it out
        const int32_t* tt = (const int32_t*) (L.tid2eid.base + (uint64_t) token * (uint64_t) K * 4);
        // real files hold ids in [0, NE); keep the reference safe if a fixture's pattern bytes do not
        for (int64_t k = 0; k < K; ++k)
            gate_ids[k] = (tt[k] >= 0 && (uint64_t) tt[k] < (uint64_t) NE) ? (int64_t) tt[k] : (int64_t) (0u);
    } else {                // the bias shifts the selection only, never the weights (579-584)
        std::vector<float> biased((size_t) NE);
        dequant_row(L.exp_probs_b, 0, 0, scratch_.data());
        for (int64_t e = 0; e < NE; ++e) biased[e] = gate_scores[e] + scratch_[(size_t) e];
        std::vector<int64_t> ord((size_t) NE);
        for (int64_t e = 0; e < NE; ++e) ord[(size_t) e] = e;
        std::partial_sort(ord.begin(), ord.begin() + K, ord.end(),
                          [&](int64_t a, int64_t b) { return biased[a] > biased[b]; });
        for (int64_t k = 0; k < K; ++k) gate_ids[k] = ord[(size_t) k];
    }
    double wsum = 0.0;
    for (int64_t k = 0; k < K; ++k) wsum += gate_scores[gate_ids[k]];   // weights are the ORIGINAL scores (585)
    for (int64_t k = 0; k < K; ++k)
        gate_w[k] = (float) (gate_scores[gate_ids[k]] / wsum * route_scale_);
    std::fill_n(out, g_.n_embd, 0.0f);
    std::vector<float> up(FF), gate(FF), dn((size_t) g_.n_embd);
    const float lim = (float) clamp_exp[cur_layer_];
    const bool cl = lim > 0.0f;
    for (int64_t k = 0; k < K; ++k) {   // gate = w1, up = w3, down = w2; clamp only over a positive limit (603-610)
        const int64_t e = gate_ids[k];
        matvec(L.gate_exps, gate.data(), FF, x, g_.n_embd, e);
        matvec(L.up_exps, up.data(), FF, x, g_.n_embd, e);
        if (cl) {
            for (int64_t i = 0; i < FF; ++i) gate[i] = std::min(gate[i], lim);
            for (int64_t i = 0; i < FF; ++i) up[i] = std::min(std::max(up[i], -lim), lim);
        }
        for (int64_t i = 0; i < FF; ++i)
            act[i] = (float) ((double) gate[i] / (1.0 + std::exp(-(double) gate[i])) * up[i] * gate_w[k]);
        matvec(L.down_exps, dn.data(), g_.n_embd, act.data(), FF, e);
        for (int64_t d = 0; d < g_.n_embd; ++d) out[d] = (float) ((double) out[d] + dn[d]);
    }
    const float lim2 = (float) clamp_shexp[cur_layer_];    // one shared expert, no routing weight (648)
    matvec(L.sh_gate, gate.data(), FF, x, g_.n_embd);
    matvec(L.sh_up, up.data(), FF, x, g_.n_embd);
    if (lim2 > 0.0f) {
        for (int64_t i = 0; i < FF; ++i) gate[i] = std::min(gate[i], lim2);
        for (int64_t i = 0; i < FF; ++i) up[i] = std::min(std::max(up[i], -lim2), lim2);
    }
    for (int64_t i = 0; i < FF; ++i)
        act[i] = (float) ((double) gate[i] / (1.0 + std::exp(-(double) gate[i])) * up[i]);
    matvec(L.sh_down, dn.data(), g_.n_embd, act.data(), FF);
    for (int64_t d = 0; d < g_.n_embd; ++d) out[d] = (float) ((double) out[d] + dn[d]);
}

// ---- load -----------------------------------------------------------------------------------------------

Ref::Ref(const std::string& any_shard, int64_t ctx) : m_(GgufModel::open(any_shard)), ctx_(ctx) {
    const GgufFile& meta = m_.shard(0);          // a split GGUF carries its metadata in shard 1 (reader 541)
    std::string err;
    err = check_architecture(meta);
    if (!err.empty()) throw std::runtime_error("dsv4ref: " + err);
    if (!core::dsv4_geometry_from_gguf(meta, g_, err)) throw std::runtime_error("dsv4ref: " + err);
    if (!core::check_gguf_layout_dsv4(m_, g_, err)) throw std::runtime_error("dsv4ref: " + err);
    auto num = [&](const char* key, double dflt) {
        const MetaValue* v = meta.get(key);
        if (!v) return dflt;
        return (v->type == MetaType::F32 || v->type == MetaType::F64) ? v->f : (double) v->u;
    };
    norm_eps = num("deepseek4.attention.layer_norm_rms_epsilon", 1e-6);
    hc_eps = num("deepseek4.hyper_connection.epsilon", 1e-6);
    route_scale_ = num("deepseek4.expert_weights_scale", 1.0);
    win_ = g_.sliding_window;
    mix_ = g_.hc_mix();
    hcdim_ = g_.hc_dim();
    const MetaValue* ce = meta.get("deepseek4.swiglu_clamp_exp");
    const MetaValue* cs = meta.get("deepseek4.swiglu_clamp_shexp");
    auto arrf = [](const MetaValue* v, size_t i) {   // published files ship these as F32 arrays; take any
        const MetaValue& it = v->items[i];           // numeric item (test fixtures write u32 arrays)
        return (it.type == MetaType::F32 || it.type == MetaType::F64) ? it.f : (double) it.u;
    };
    for (int64_t l = 0; l < g_.n_layers; ++l) {   // the reader samples at most 64 array items (reader 372)
        clamp_exp.push_back(ce && (size_t) l < ce->items.size() ? arrf(ce, (size_t) l) : 0.0);
        clamp_shexp.push_back(cs && (size_t) l < cs->items.size() ? arrf(cs, (size_t) l) : 0.0);
    }
    layers_.resize((size_t) g_.n_layers);
    auto resolve = [&](const std::string& name, Weight& outw) {
        size_t s = 0;
        const TensorInfo* t = m_.find(name, &s);
        if (!t) throw std::runtime_error("dsv4ref: missing tensor " + name);
        outw.info = t;
        outw.shard = s;
        outw.base = m_.shard(s).tensor_data(*t);
        outw.row_bytes = tensor_payload_bytes(*t) / (t->shape.size() >= 2 ? t->shape[1] : 1);
        outw.expert_bytes = t->shape.size() == 3 ? tensor_payload_bytes(*t) / t->shape[2] : 0;
        weights_[name] = outw;
    };
    auto comp_setup = [&](Compressor& C, int64_t hd, int64_t ratio, int64_t coff) {
        C.head_dim = hd;
        C.ratio = ratio;
        C.coff = coff;
        C.kv_state.assign((size_t) coff * (size_t) ratio * (size_t) coff * (size_t) hd, 0.0f);
        C.score_state.assign((size_t) coff * (size_t) ratio * (size_t) coff * (size_t) hd,
                             (float) -std::numeric_limits<float>::infinity());
    };
    for (int64_t l = 0; l < g_.n_layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        Layer& L = layers_[l];
        resolve(p + "attn_norm.weight", L.attn_norm);
        resolve(p + "attn_q_a.weight", L.q_a);
        resolve(p + "attn_q_a_norm.weight", L.q_a_norm);
        resolve(p + "attn_q_b.weight", L.q_b);
        resolve(p + "attn_kv.weight", L.kv);
        resolve(p + "attn_kv_a_norm.weight", L.kv_norm);
        resolve(p + "attn_output_a.weight", L.o_a);
        resolve(p + "attn_output_b.weight", L.o_b);
        resolve(p + "attn_sinks.weight", L.sinks);
        resolve(p + "hc_attn_base.weight", L.hc_attn_base);
        resolve(p + "hc_attn_fn.weight", L.hc_attn_fn);
        resolve(p + "hc_attn_scale.weight", L.hc_attn_scale);
        resolve(p + "hc_ffn_base.weight", L.hc_ffn_base);
        resolve(p + "hc_ffn_fn.weight", L.hc_ffn_fn);
        resolve(p + "hc_ffn_scale.weight", L.hc_ffn_scale);
        resolve(p + "ffn_norm.weight", L.ffn_norm);
        resolve(p + "ffn_gate_inp.weight", L.gate_inp);
        if (l < g_.n_hash) resolve(p + "ffn_gate_tid2eid.weight", L.tid2eid);
        else resolve(p + "exp_probs_b.bias", L.exp_probs_b);
        resolve(p + "ffn_gate_shexp.weight", L.sh_gate);
        resolve(p + "ffn_up_shexp.weight", L.sh_up);
        resolve(p + "ffn_down_shexp.weight", L.sh_down);
        resolve(p + "ffn_gate_exps.weight", L.gate_exps);
        resolve(p + "ffn_up_exps.weight", L.up_exps);
        resolve(p + "ffn_down_exps.weight", L.down_exps);
        L.kind = core::dsv4_layer_kind(g_, l);
        L.ratio = g_.compress[(size_t) l];
        if (L.kind == core::Dsv4LayerKind::Csa) {
            resolve(p + "attn_compressor_kv.weight", L.comp.wkv);
            resolve(p + "attn_compressor_gate.weight", L.comp.wgate);
            resolve(p + "attn_compressor_norm.weight", L.comp.norm);
            resolve(p + "attn_compressor_ape.weight", L.comp.ape);
            comp_setup(L.comp, g_.head_dim, g_.csa_block, 2);
            resolve(p + "indexer.attn_q_b.weight", L.idx_q_b);
            resolve(p + "indexer.proj.weight", L.idx_proj);
            resolve(p + "indexer_compressor_norm.weight", L.idx_norm);
            resolve(p + "indexer_compressor_kv.weight", L.idx_comp.wkv);
            resolve(p + "indexer_compressor_gate.weight", L.idx_comp.wgate);
            L.idx_comp.norm = L.idx_norm;
            resolve(p + "indexer_compressor_ape.weight", L.idx_comp.ape);
            comp_setup(L.idx_comp, g_.idx_key_dim, g_.csa_block, 2);
            L.ikv.reserve((size_t) (ctx / g_.csa_block + 1) * (size_t) g_.idx_key_dim);
        } else if (L.kind == core::Dsv4LayerKind::Hca) {
            resolve(p + "attn_compressor_kv.weight", L.comp.wkv);
            resolve(p + "attn_compressor_gate.weight", L.comp.wgate);
            resolve(p + "attn_compressor_norm.weight", L.comp.norm);
            resolve(p + "attn_compressor_ape.weight", L.comp.ape);
            comp_setup(L.comp, g_.head_dim, g_.hca_block, 1);
        }
        L.ring.assign((size_t) win_ * (size_t) g_.head_dim, 0.0f);
        if (L.kind != core::Dsv4LayerKind::Sliding)
            L.pool.reserve((size_t) (ctx / L.ratio + 1) * (size_t) g_.head_dim);
    }
    resolve("token_embd.weight", token_embd);
    resolve("output.weight", output);
    resolve("output_norm.weight", output_norm);
    resolve("output_hc_base.weight", hc_head_base);
    resolve("output_hc_fn.weight", hc_head_fn);
    resolve("output_hc_scale.weight", hc_head_scale);

    // the two rope tables (model.py 481-487): compressed layers YaRN over compress_rope_freq_base,
    // sliding layers plain rope.freq_base with YaRN off
    const int rd = (int) g_.n_rot;
    rope_plain.rd = rope_comp.rd = rd;
    rope_plain.build(ctx, num("deepseek4.rope.freq_base", 10000.0), 1.0, 0, 0.0, 0.0);
    rope_comp.build(ctx, num("deepseek4.attention.compress_rope_freq_base", 10000.0),
                    num("deepseek4.rope.scaling.factor", 1.0),
                    (int64_t) num("deepseek4.rope.scaling.original_context_length", 0.0),
                    num("deepseek4.rope.scaling.yarn_beta_fast", 32.0),
                    num("deepseek4.rope.scaling.yarn_beta_slow", 1.0));

    scratch_.resize((size_t) hcdim_ + 64);
    logits_.resize((size_t) g_.n_vocab);
    qr.resize((size_t) g_.q_lora_rank);
    q.resize((size_t) g_.n_head * g_.head_dim);
    iq.resize((size_t) g_.idx_q_heads * g_.idx_key_dim);
    kv512.resize((size_t) g_.head_dim);
    o_attn.resize((size_t) g_.n_head * g_.head_dim);
    olo.resize((size_t) g_.o_groups * g_.o_lora_rank);
    yd.resize((size_t) g_.n_embd);
    xin.resize((size_t) g_.n_embd);
    hbuf.resize((size_t) g_.n_embd);
    streams.resize((size_t) hcdim_);
    streams2.resize((size_t) hcdim_);
    residual.resize((size_t) hcdim_);
    post_.resize((size_t) g_.hc);
    comb_.resize((size_t) g_.hc * g_.hc);
    gate_scores.resize((size_t) g_.n_expert);
    act.resize((size_t) g_.n_ff);
}

// ---- step -----------------------------------------------------------------------------------------------

int Ref::step(int token) {
    if (pos_ >= ctx_) throw std::runtime_error("dsv4ref: context exhausted at " + std::to_string(pos_));
    // embedding, copied into every hc stream (forward_embed's unsqueeze(2).repeat, model.py 857)
    dequant_row(token_embd, 0, token, streams.data());
    for (int64_t s = 1; s < g_.hc; ++s)
        std::memcpy(streams.data() + (size_t) s * (size_t) g_.n_embd, streams.data(), (size_t) g_.n_embd * 4);
    for (size_t li = 0; li < layers_.size(); ++li) {
        cur_layer_ = (int64_t) li;
        Layer& L = layers_[li];
        residual = streams;
        hc_pre(L, false, streams.data(), xin.data(), post_.data(), comb_.data());
        dequant_row(L.attn_norm, 0, 0, scratch_.data());
        rmsnorm(xin.data(), scratch_.data(), g_.n_embd, (float) norm_eps, hbuf.data());
        attention(L, hbuf.data(), pos_, yd.data());
        hc_post(streams.data(), yd.data(), residual.data(), post_.data(), comb_.data());

        residual = streams;
        hc_pre(L, true, streams.data(), xin.data(), post_.data(), comb_.data());
        dequant_row(L.ffn_norm, 0, 0, scratch_.data());
        rmsnorm(xin.data(), scratch_.data(), g_.n_embd, (float) norm_eps, hbuf.data());
        moe(L, hbuf.data(), token, yd.data());
        hc_post(streams2.data(), yd.data(), residual.data(), post_.data(), comb_.data());
        streams.swap(streams2);
        if (dump_layers_) {   // parity dumps: the summed stream after each layer (tools/dsv4/parity.py)
            for (int64_t d = 0; d < g_.n_embd; ++d) {
                double acc = 0.0;
                for (int64_t s = 0; s < g_.hc; ++s) acc += streams[(size_t) s * (size_t) g_.n_embd + d];
                const float f = (float) acc;
                std::fwrite(&f, 4, 1, dump_layers_);
            }
        }
    }
    // the HyperHead: pre only, the single scale (model.py 709-716 used by forward_head, 862)
    double ss = 0.0;
    for (int64_t i = 0; i < hcdim_; ++i) ss += (double) streams[i] * streams[i];
    const double rsqrt = 1.0 / std::sqrt(ss / (double) hcdim_ + norm_eps);
    std::vector<float> mixes((size_t) g_.hc);
    matvec(hc_head_fn, mixes.data(), g_.hc, streams.data(), hcdim_);
    std::vector<float> hbase((size_t) g_.hc), hsc(1);
    dequant_row(hc_head_base, 0, 0, hbase.data());
    dequant_row(hc_head_scale, 0, 0, hsc.data());
    for (int64_t d = 0; d < g_.n_embd; ++d) {
        double acc = 0.0;
        for (int64_t j = 0; j < g_.hc; ++j) {
            const double pre = sigmoid((double) mixes[j] * rsqrt * hsc[0] + hbase[j]) + hc_eps;
            acc += pre * streams[(size_t) j * (size_t) g_.n_embd + d];
        }
        yd[d] = (float) acc;
    }
    // output_norm, then the full fp32 head (ParallelHead.forward, model.py 731-740)
    std::vector<float> onorm((size_t) g_.n_embd);
    dequant_row(output_norm, 0, 0, onorm.data());
    rmsnorm(yd.data(), onorm.data(), g_.n_embd, (float) norm_eps, hbuf.data());
    matvec(output, logits_.data(), g_.n_vocab, hbuf.data(), g_.n_embd);
    if (dump_logits_) std::fwrite(logits_.data(), 4, logits_.size(), dump_logits_);
    const int best = (int) (std::max_element(logits_.begin(), logits_.end()) - logits_.begin());
    ++pos_;
    return best;
}

// ---- CLI -------------------------------------------------------------------------------------------------

int ref_main(int argc, char** argv) {
    std::string model, tokens;
    int64_t ctx = 4096, extra = 8;
    const char *dump_layers = nullptr, *dump_logits = nullptr;
    try {
        for (int i = 1; i < argc; ++i) {
            auto need = [&](const char* flag) {
                if (i + 1 >= argc) throw std::runtime_error(std::string("dsv4ref: ") + flag + " needs a value");
                return argv[++i];
            };
            if (!std::strcmp(argv[i], "--model")) model = need("--model");
            else if (!std::strcmp(argv[i], "--tokens")) tokens = need("--tokens");
            else if (!std::strcmp(argv[i], "--ctx")) ctx = std::atoll(need("--ctx"));
            else if (!std::strcmp(argv[i], "--n")) extra = std::atoll(need("--n"));
            else if (!std::strcmp(argv[i], "--dump-layers")) dump_layers = need("--dump-layers");
            else if (!std::strcmp(argv[i], "--dump-logits")) dump_logits = need("--dump-logits");
            else throw std::runtime_error(std::string("dsv4ref: unknown flag ") + argv[i]);
        }
        if (model.empty() || tokens.empty()) {
            std::fprintf(stderr,
                         "usage: strata-dsv4ref --model SHARD.gguf --tokens 1,2,3 [--ctx N] [--n N]"
                         " [--dump-layers F] [--dump-logits F]\n");
            return 2;
        }
        Ref r(model, ctx);
        FILE* fl = dump_layers ? std::fopen(dump_layers, "wb") : nullptr;
        FILE* fg = dump_logits ? std::fopen(dump_logits, "wb") : nullptr;
        r.set_dump_layers(fl);
        r.set_dump_logits(fg);
        std::vector<int> toks;
        for (size_t p = 0; p < tokens.size();) {
            const size_t c = tokens.find(',', p);
            toks.push_back(std::atoi(tokens.substr(p, c == std::string::npos ? c - p : c - p).c_str()));
            if (c == std::string::npos) break;
            p = c + 1;
        }
        int next = 0;
        for (size_t i = 0; i < toks.size(); ++i) next = r.step(toks[i]);
        std::printf("fed %zu tokens; greedy next: %d\n", toks.size(), next);
        for (int64_t k = 0; k < extra; ++k) {
            next = r.step(next);
            std::printf("step %lld: %d\n", (long long) (toks.size() + k), next);
        }
        if (fl) std::fclose(fl);
        if (fg) std::fclose(fg);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "dsv4ref: %s\n", e.what());
        return 1;
    }
}

}   // namespace dsv4
}   // namespace strata
