// include/strata/dsv4/dsv4ref.hpp - docs/DEEPSEEK_V4_PLAN.md stage 2: the reference forward pass for
// DeepSeek-V4-Flash (GGUF architecture deepseek4), CPU, exactness first.
//
// Transcribed from the official inference code (deepseek-ai/DeepSeek-V4-Flash-0731's inference/model.py and
// inference/kernel.py, fetched Oct 2026) and from the GGUF convention q36ref.cu set for Qwen3.6: one token
// at a time (the prompt too), fp32 activations, every weight read in its GGUF form (MXFP4 experts,
// Q8_0/BF16 dense, F32 mHC/norms, I32 tid2eid) straight out of the mmap.  No GPU, no expert cache, no
// speculative path: the DSpark layers ship as a separate file and are out of scope, as in stage 1.
//
// Unlike the official code, no QAT activation simulation (act_quant / fp4_act_quant / Hadamard rotation):
// those emulate the FP8/FP4 deployment's rounding; a GGUF repack's dequantized weights feed a plain fp32
// matmul.  Parity against llama.cpp's build is with tolerance (KL, per-layer cosine), not bit-exactness -
// and until a >=160 GB machine runs it, parity is stated as not run (docs/DEEPSEEK_V4_PLAN.md stage 2).
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace strata {
namespace dsv4 {

/// One loaded deepseek4 model plus its running state; feed tokens with step(), it returns the greedy id.
class Ref {
public:
    /// Opens any shard of the model (siblings found), checks architecture, geometry and layout first.
    /// Throws std::runtime_error with the reason.  `ctx` bounds the rope tables and the pool reserves.
    Ref(const std::string& any_shard, int64_t ctx);

    const core::Dsv4Geometry& geom() const { return g_; }
    int64_t pos() const { return pos_; }

    /// Feeds one token at position pos_, advances pos_, fills logits(), and returns argmax(logits()).
    int step(int token);

    const std::vector<float>& logits() const { return logits_; }

    /// Optional parity dumps (stage 2's tools/dsv4/parity.py hook): raw fp32, step-major - after every
    /// layer the stream sum (n_embd floats), and after every step the full logits.
    void set_dump_layers(FILE* f) { dump_layers_ = f; }
    void set_dump_logits(FILE* f) { dump_logits_ = f; }

private:
    struct Weight {                     // one resolved tensor: where it lives and how to walk its rows
        const TensorInfo* info = nullptr;
        size_t shard = 0;
        const uint8_t* base = nullptr;  // tensor payload start (mmap)
        uint64_t row_bytes = 0;         // one output row's bytes (GGUF dim 0 is the input, contiguous)
        uint64_t expert_bytes = 0;      // one expert's slice of a 3-D blade (0 for anything else)
    };

    struct Rope {                       // cos/sin per position over rd/2 pairs (YaRN or plain, model.py 206)
        int rd = 0;
        std::vector<float> cos, sin;    // [pos][rd/2]
        int64_t built = 0;
        void build(int64_t upto, double base_theta, double factor, int64_t original, double beta_fast,
                   double beta_slow);
        void apply(float* x, int64_t pos, int64_t head_dim, bool inverse) const;   // rotates x's last rd dims
    };

    struct Compressor {                 // Compressor in model.py 285-383 (decode branch)
        Weight wkv, wgate, norm, ape;
        int64_t head_dim = 0, ratio = 0, coff = 1;    // coff = 1 + overlap (overlap iff ratio == 4)
        std::vector<float> kv_state, score_state;     // coff*ratio rows x coff*head_dim (scores start at -inf)
    };

    struct Layer {
        Weight attn_norm, q_a, q_a_norm, q_b, kv, kv_norm, o_a, o_b, sinks;
        Weight hc_attn_base, hc_attn_fn, hc_attn_scale, hc_ffn_base, hc_ffn_fn, hc_ffn_scale;
        Weight ffn_norm, gate_inp, tid2eid, exp_probs_b, sh_gate, sh_up, sh_down;
        Weight gate_exps, up_exps, down_exps;
        Compressor comp;                                // the main pool (unused on sliding layers)
        Weight idx_q_b, idx_proj, idx_norm;             // the Lightning indexer (CSA layers only)
        Compressor idx_comp;
        core::Dsv4LayerKind kind = core::Dsv4LayerKind::Sliding;
        int64_t ratio = 0;                              // the file's compress_ratios[layer] (0 = sliding)
        std::vector<float> ring;                        // window_size x head_dim (model.py's kv rows 0..win-1)
        std::vector<float> pool;                        // entries x head_dim (kv rows win..)
        std::vector<float> ikv;                         // indexer entries x idx_key_dim (its own cache)
    };

    // tensor + row helpers
    Weight& wref(const std::string& name);
    void dequant_row(const Weight& t, int64_t expert, int64_t row, float* out) const;
    void matvec(const Weight& t, float* y, int64_t n_out, const float* x, int64_t n_in, int64_t expert = 0,
                int64_t row0 = 0) const;
    static void rmsnorm(const float* x, const float* wt, int64_t n, float eps, float* y);   // y = rms(x)*w
    void rope_for(const Layer& L, float* x, int64_t pos, int64_t head_dim, bool inverse = false) const;

    // per-token pieces (single token, batch 1; model.py's start_pos > 0 branches)
    void hc_pre(const Layer& L, bool ffn, const float* streams, float* y, float* post, float* comb);
    void hc_post(float* streams, const float* y, const float* residual, const float* post, const float* comb);
    void attention(Layer& L, const float* x, int64_t pos, float* out);
    void compressor_step(Compressor& C, const float* x, int64_t pos, std::vector<float>& pool, int64_t entry);
    void moe(const Layer& L, const float* x, int token, float* out);

    core::Dsv4Geometry g_;
    GgufModel m_;
    std::map<std::string, Weight> weights_;
    std::vector<Layer> layers_;
    Weight token_embd, output, output_norm, hc_head_base, hc_head_fn, hc_head_scale;
    Rope rope_plain, rope_comp;
    double norm_eps = 1e-6, hc_eps = 1e-6, route_scale_ = 1.0;
    std::vector<double> clamp_exp, clamp_shexp;      // per-layer swiglu limits (arrays in the file)
    int64_t ctx_ = 0, pos_ = 0;
    int64_t win_ = 0, mix_ = 0, hcdim_ = 0;          // window, (2+hc)*hc, hc*dim (row scratch bound)
    int64_t cur_layer_ = 0;                          // for the per-layer swiglu limits
    std::vector<float> logits_;
    mutable std::vector<float> scratch_;               // one row's worth of dequantized values (matvec temp)
    std::vector<float> qr, q, iq, kv512, o_attn, olo, yd, xin, hbuf;
    std::vector<float> streams, streams2, residual, post_, comb_;
    std::vector<float> gate_scores, gate_w, act;     // MoE scores / weights / FF intermediate
    std::vector<int64_t> gate_ids;
    FILE* dump_layers_ = nullptr;
    FILE* dump_logits_ = nullptr;
};

/// The `strata-dsv4ref` entry point (docs/DEEPSEEK_V4_PLAN.md stage 2).  Returns 0 on a clean run.
int ref_main(int argc, char** argv);

}   // namespace dsv4
}   // namespace strata
