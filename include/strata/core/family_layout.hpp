// include/strata/core/family_layout.hpp - per model family: the geometry read from a GGUF's metadata, and the
// tensor layout table (every tensor of a layer, by name and GGUF shape) checked against a model file's directory.
//
// Header-only and GPU-free: the engine, the tools and the unit tests (tests/core/model_family_test.cpp) share it.
//
//   * geometry_from_gguf: fills core::ModelGeometry from "<arch>.*" keys.  For qwen4exp the keys a file does not
//     carry keep Flash-Next's values (the struct defaults, which every published qwen4exp file matches); for
//     qwen35moe every key is required (check_architecture_qwen35moe).
//   * compiled_geometry_mismatch: the first field where a geometry differs from the shapes this engine's kernels
//     are built for (the expert count and top-k are run-time values and are not compared).
//   * layer_tensors / global_tensors: the layout tables.  Names are GGUF names (blk.N.<suffix>), shapes are in
//     GGUF order (dim 0 contiguous), as tools/gguf_reader.py prints them.
//   * check_gguf_layout: every table entry present with its shape, inside its file, of a type with a known block
//     size; and, for a family whose table is complete (qwen35moe), no blk.* tensor the table does not name.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/model_family.hpp"
#include "strata/core/geometry.hpp"

#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace strata::core {

/// Fills `g` (and `top_k`, the experts per token) from the metadata shard `meta`.  False with `err` when the
/// architecture guard refuses the file or a key has an impossible value.
inline bool geometry_from_gguf(const strata::GgufFile& meta, ModelGeometry& g, int64_t& top_k, std::string& err) {
    err = strata::check_architecture(meta);
    if (!err.empty()) return false;
    const strata::ModelFamily family = strata::model_family(meta);
    const std::string p = std::string(strata::model_family_arch(family)) + ".";
    g = ModelGeometry{};
    g.family = family;
    if (family == strata::ModelFamily::DeepseekV4) {   // its shape is not this struct's (see below in this header)
        err = meta.path() + ": deepseek4 has its own geometry (core::Dsv4Geometry, dsv4_geometry_from_gguf)";
        return false;
    }
    // a key the file carries replaces the default; `required` keys were checked present by the guard
    auto num = [&](const char* key, int64_t& dst) {
        if (const strata::MetaValue* v = meta.get(p + key))
            if (v->is_num()) dst = (int64_t) v->u;
    };
    int64_t block_count = g.n_layers, key_length = g.head_dim, value_length = -1;
    num("block_count", block_count);
    num("embedding_length", g.n_embd);
    num("attention.head_count", g.n_head);
    num("attention.head_count_kv", g.n_head_kv);
    num("attention.key_length", key_length);
    num("attention.value_length", value_length);
    num("expert_count", g.n_expert);
    top_k = 10;
    num("expert_used_count", top_k);
    g.n_expert_used = top_k;
    num("expert_feed_forward_length", g.n_ff);
    num("expert_shared_feed_forward_length", g.n_ff_shexp);
    num("ssm.conv_kernel", g.ssm_d_conv);
    num("ssm.state_size", g.ssm_state_size);
    num("ssm.group_count", g.ssm_k_heads);
    num("ssm.time_step_rank", g.ssm_v_heads);
    num("ssm.inner_size", g.ssm_value_dim);
    num("full_attention_interval", g.qsa_interval);
    g.head_dim = key_length;
    if (family == strata::ModelFamily::Qwen35Moe) {
        num("nextn_predict_layers", g.n_mtp_layers);
        g.hc = g.hc_lr = 0;                 // a plain pre-norm residual
        g.idx_q_heads = g.idx_key_dim = 0;  // dense attention, no sparse indexer
    } else {
        num("hyper_connection.count", g.hc);
        num("hyper_connection.low_rank", g.hc_lr);
        num("attention.indexer.head_count", g.idx_q_heads);
        num("attention.indexer.key_length", g.idx_key_dim);
    }
    g.n_layers = block_count - g.n_mtp_layers;
    g.ssm_conv_channels = 2 * g.ssm_state_size * g.ssm_k_heads + g.ssm_value_dim;
    if (const strata::MetaValue* v = meta.get("tokenizer.ggml.tokens"))
        if (v->type == strata::MetaType::ARRAY && v->count > 0) g.n_vocab = (int64_t) v->count;

    char buf[256];
    auto bad = [&](const char* what) {
        std::snprintf(buf, sizeof buf, "%s: %s", meta.path().c_str(), what);
        err = buf;
        return false;
    };
    if (g.n_layers <= 0 || g.n_embd <= 0 || g.n_head <= 0 || g.n_head_kv <= 0 || g.head_dim <= 0 || g.n_expert <= 0 ||
        top_k <= 0 || top_k > g.n_expert || g.n_ff <= 0 || g.ssm_state_size <= 0 || g.ssm_k_heads <= 0 ||
        g.ssm_v_heads <= 0 || g.ssm_d_conv <= 0 || g.qsa_interval < 2)
        return bad("a geometry key is zero or out of range");
    if (value_length >= 0 && value_length != key_length) return bad("attention.key_length != attention.value_length");
    if (g.ssm_value_dim != g.ssm_state_size * g.ssm_v_heads) return bad("ssm.inner_size != ssm.state_size * ssm.time_step_rank");
    if (g.ssm_v_heads % g.ssm_k_heads) return bad("ssm.time_step_rank is not a multiple of ssm.group_count");
    if (g.n_head % g.n_head_kv) return bad("attention.head_count is not a multiple of attention.head_count_kv");
    if (g.n_layers % g.qsa_interval) return bad("the layer count is not a multiple of full_attention_interval");
    return true;
}

/// The first geometry field that differs from the shapes this engine's kernels are built for (ModelGeometry's
/// defaults, Flash-Next's), as "<what> is <got>, this engine's kernels are built for <want>"; empty when none.
/// n_expert and top-k are run-time sizes (the Coder ships 256 experts) and n_vocab is the embedding's business.
inline std::string compiled_geometry_mismatch(const ModelGeometry& g) {
    const ModelGeometry c;
    if (g.family != c.family)
        return std::string("the model is ") + strata::model_family_title(g.family) + ", this engine's kernels are built "
               "for " + strata::model_family_title(c.family);
    struct F {
        const char* what;
        int64_t got, want;
    };
    const F fields[] = {
        {"embedding_length", g.n_embd, c.n_embd},
        {"the layer count", g.n_layers, c.n_layers},
        {"full_attention_interval", g.qsa_interval, c.qsa_interval},
        {"ssm.state_size", g.ssm_state_size, c.ssm_state_size},
        {"ssm.group_count", g.ssm_k_heads, c.ssm_k_heads},
        {"ssm.time_step_rank", g.ssm_v_heads, c.ssm_v_heads},
        {"ssm.conv_kernel", g.ssm_d_conv, c.ssm_d_conv},
        {"the GDN conv channels", g.ssm_conv_channels, c.ssm_conv_channels},
        {"ssm.inner_size", g.ssm_value_dim, c.ssm_value_dim},
        {"attention.head_count", g.n_head, c.n_head},
        {"attention.head_count_kv", g.n_head_kv, c.n_head_kv},
        {"attention.key_length", g.head_dim, c.head_dim},
        {"attention.indexer.head_count", g.idx_q_heads, c.idx_q_heads},
        {"attention.indexer.key_length", g.idx_key_dim, c.idx_key_dim},
        {"hyper_connection.count", g.hc, c.hc},
        {"hyper_connection.low_rank", g.hc_lr, c.hc_lr},
        {"expert_feed_forward_length", g.n_ff, c.n_ff},
        {"expert_shared_feed_forward_length", g.n_ff_shexp, c.n_ff_shexp},
        {"the in-file MTP layers", g.n_mtp_layers, c.n_mtp_layers},
    };
    for (const F& f : fields)
        if (f.got != f.want) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "%s is %lld, this engine's kernels are built for %lld", f.what,
                          (long long) f.got, (long long) f.want);
            return buf;
        }
    return {};
}

enum class LayerKind { Gdn, Attention, Mtp };

inline LayerKind layer_kind(const ModelGeometry& g, int64_t layer) {
    if (layer >= g.n_layers) return LayerKind::Mtp;
    return is_qsa_layer(g, layer) ? LayerKind::Attention : LayerKind::Gdn;
}

/// One tensor of the layout table: its name (a blk.N. suffix for a layer's, the full name for a global one) and
/// its GGUF shape.
struct TensorWant {
    std::string name;
    std::vector<int64_t> shape;
};

/// Every tensor of layer `layer` (kind from `layer_kind`), by family.
inline std::vector<TensorWant> layer_tensors(const ModelGeometry& g, int64_t layer) {
    const LayerKind kind = layer_kind(g, layer);
    const int64_t N = g.n_embd, NE = g.n_expert, FF = g.n_ff, FS = g.n_ff_shexp;
    std::vector<TensorWant> t;
    auto add = [&](const char* s, std::vector<int64_t> shape) { t.push_back({s, std::move(shape)}); };
    // MoE, every layer (the MTP block's too)
    add("ffn_gate_inp.weight", {N, NE});
    add("ffn_gate_inp_shexp.weight", {N});
    add("ffn_gate_shexp.weight", {N, FS});
    add("ffn_up_shexp.weight", {N, FS});
    add("ffn_down_shexp.weight", {FS, N});
    add("ffn_gate_exps.weight", {N, FF, NE});
    add("ffn_up_exps.weight", {N, FF, NE});
    add("ffn_down_exps.weight", {FF, N, NE});
    if (g.has_hyper_connections()) {
        // gated residual (Flash-Next): `gr_read` transposes, so the file's orientation is [hc_dim, hc_lr] / [hc_lr, hc_dim]
        const int64_t H = g.hc_dim();
        add("hc_attn_down.weight", {H, g.hc_lr});
        add("hc_attn_up.weight", {g.hc_lr, H});
        add("hc_attn_inject.weight", {H, g.hc});
        add("hc_attn_norm.weight", {H});
        add("hc_ffn_down.weight", {H, g.hc_lr});
        add("hc_ffn_up.weight", {g.hc_lr, H});
        add("hc_ffn_inject.weight", {H, g.hc});
        add("hc_ffn_norm.weight", {H});
    } else {
        // plain pre-norm residual (qwen35moe): x += mixer(rms(x)); x += moe(rms(x))
        add("attn_norm.weight", {N});
        add("post_attention_norm.weight", {N});
    }
    if (kind == LayerKind::Gdn) {
        add("attn_qkv.weight", {N, g.ssm_conv_channels});
        add("attn_gate.weight", {N, g.ssm_value_dim});
        add("ssm_out.weight", {g.ssm_value_dim, N});
        add("ssm_conv1d.weight", {g.ssm_d_conv, g.ssm_conv_channels});
        add("ssm_alpha.weight", {N, g.ssm_v_heads});
        add("ssm_beta.weight", {N, g.ssm_v_heads});
        add("ssm_a", {g.ssm_v_heads});
        add("ssm_dt.bias", {g.ssm_v_heads});
        add("ssm_norm.weight", {g.ssm_state_size});
    } else {
        // attention (query + output gate in attn_q), also the MTP block's
        add("attn_q.weight", {N, 2 * g.n_head * g.head_dim});
        add("attn_k.weight", {N, g.n_head_kv * g.head_dim});
        add("attn_v.weight", {N, g.n_head_kv * g.head_dim});
        add("attn_output.weight", {g.n_head * g.head_dim, N});
        add("attn_q_norm.weight", {g.head_dim});
        add("attn_k_norm.weight", {g.head_dim});
        if (g.has_indexer() && kind == LayerKind::Attention) {
            // `q_proj` is the QUERY count, `k_proj` the KEY width (round 194: they are different numbers)
            add("indexer.q_proj.weight", {N, g.idx_q_heads * g.idx_key_dim});
            add("indexer.k_proj.weight", {N, g.idx_key_dim});
            add("indexer.q_norm.weight", {g.idx_key_dim});
            add("indexer.k_norm.weight", {g.idx_key_dim});
        }
    }
    if (kind == LayerKind::Mtp) {
        // the GGUF-native MTP block: eh_proj over concat(enorm(embedding), hnorm(hidden))
        add("nextn.eh_proj.weight", {2 * N, N});
        add("nextn.enorm.weight", {N});
        add("nextn.hnorm.weight", {N});
        add("nextn.shared_head_norm.weight", {N});
    }
    return t;
}

/// The tensors outside the layers, by family.  (Flash-Next's output_norm, PLE and n-gram tables are not listed:
/// the published files place them differently, and their loaders check them.)
inline std::vector<TensorWant> global_tensors(const ModelGeometry& g) {
    std::vector<TensorWant> t;
    t.push_back({"token_embd.weight", {g.n_embd, g.n_vocab}});
    t.push_back({"output.weight", {g.n_embd, g.n_vocab}});
    if (g.has_hyper_connections()) {
        t.push_back({"output_hc_down.weight", {g.hc_dim(), g.hc_lr}});
        t.push_back({"output_hc_up.weight", {g.hc_lr, g.hc_dim()}});
        t.push_back({"output_hc_norm.weight", {g.hc_dim()}});
    } else {
        t.push_back({"output_norm.weight", {g.n_embd}});
    }
    return t;
}

/// Whether the family's table names every blk.* tensor its files carry (then an unnamed one is an error).
inline bool layout_table_complete(const ModelGeometry& g) { return g.family == strata::ModelFamily::Qwen35Moe; }

/// Checks the model `m` (all shards) against the layout table of `g`.  `checked` counts the tensors found.
inline bool check_gguf_layout(const strata::GgufModel& m, const ModelGeometry& g, std::string& err,
                              size_t* checked = nullptr) {
    std::set<std::string> named;
    size_t n = 0;
    auto check = [&](const std::string& name, const std::vector<int64_t>& shape) {
        named.insert(name);
        size_t s = 0;
        const strata::TensorInfo* t = m.find(name, &s);
        if (!t) {
            err = "missing " + name;
            return false;
        }
        bool same = t->shape.size() == shape.size();
        for (size_t i = 0; same && i < shape.size(); ++i) same = t->shape[i] == (uint64_t) shape[i];
        if (!same) {
            std::string got, want;
            for (uint64_t d : t->shape) got += (got.empty() ? "" : ", ") + std::to_string(d);
            for (int64_t d : shape) want += (want.empty() ? "" : ", ") + std::to_string(d);
            err = name + " is [" + got + "], the " + strata::model_family_title(g.family) + " layout wants [" + want + "]";
            return false;
        }
        if (!m.in_bounds(*t, s)) {
            err = name + " (" + t->type_name() + ") has no known size inside " + m.shard(s).path();
            return false;
        }
        ++n;
        return true;
    };
    for (const TensorWant& w : global_tensors(g))
        if (!check(w.name, w.shape)) return false;
    for (int64_t l = 0; l < g.n_layers + g.n_mtp_layers; ++l)
        for (const TensorWant& w : layer_tensors(g, l))
            if (!check("blk." + std::to_string(l) + "." + w.name, w.shape)) return false;
    if (layout_table_complete(g)) {
        for (size_t s = 0; s < m.size(); ++s)
            for (const strata::TensorInfo& t : m.shard(s).tensors())
                if (!named.count(t.name)) {
                    err = t.name + " is not in the " + std::string(strata::model_family_title(g.family)) + " layout";
                    return false;
                }
    }
    if (checked) *checked = n;
    return true;
}

// ---- DeepSeek-V4 (`deepseek4`; docs/DEEPSEEK_V4_PLAN.md stage 1).  ModelGeometry above is the kernel contract of
// the Flash-Next path; V4 shares almost none of it with Flash-Next: one shared K=V latent (MQA), a 128-token sliding
// window plus 4x / 128x compressed pools read through a learned indexer, low-rank Q and grouped O projections,
// Sinkhorn-mixed hyper-connections instead of residuals, and a frozen token-id to expert-id table for the first
// layers.  So V4 gets its own geometry struct, schedule and layout tables; nothing above changes.
//
// Names and shapes are those of the GA files (unsloth/DeepSeek-V4-Flash-0731-GGUF, checked tensor by tensor
// against its header: 43 layers x (23 + 1 routing) + 21 CSA layers x 10 + 20 HCA layers x 4 + 6 global = 1,328).

enum class Dsv4LayerKind {
    Sliding,  ///< compress_ratios 0: only the 128-token window (the bootstrap layers)
    Csa,      ///< compress_ratios 4: the overlapping 4x pool plus the Lightning indexer (and its own pool)
    Hca,      ///< compress_ratios 128: the whole-context 128x pool, no indexer
};

struct Dsv4Geometry {
    int64_t n_layers = 0, n_embd = 0, n_vocab = 0;
    int64_t n_head = 0, head_dim = 0;     ///< head_count_kv is 1 and key_length == value_length (guard-checked)
    int64_t q_lora_rank = 0;              ///< attention.q_lora_rank: attn_q_a / attn_q_b (and the indexer's q_b)
    int64_t o_groups = 0, o_lora_rank = 0; ///< attention.output_group_count / output_lora_rank
    int64_t n_rot = 0;                    ///< rope.dimension_count: the rotated slice - RoPE turns the last 64 dims
                                          ///  of every 512-wide head (nope 448 + rope 64, the V3-style split)
    int64_t sliding_window = 0;           ///< attention.sliding_window (128)
    int64_t n_expert = 0, n_expert_used = 0, n_ff = 0;   ///< one shared expert at width n_ff (guard-checked)
    int64_t idx_q_heads = 0, idx_key_dim = 0, idx_top_k = 0;   ///< attention.indexer.* (64 x 128, top 512)
    int64_t hc = 0, hc_sinkhorn = 0;      ///< hyper_connection.count / .sinkhorn_iterations (4, 20)
    int64_t n_hash = 0;                   ///< hash_layer_count: the layers with ffn_gate_tid2eid and no exp_probs_b
    int64_t csa_block = 0, hca_block = 0; ///< the two non-zero pooling factors of the schedule (4 and 128)
    std::vector<int64_t> compress;        ///< the file's per-layer schedule (0 | csa_block | hca_block); GA may
                                          ///  carry extra all-sliding rows past the trunk for its DSpark layers

    constexpr int64_t hc_dim() const { return hc * n_embd; }
    /// columns of hc_*_base / hc_*_fn: the (pre, post, comb) mix of hc streams and the two mixers (4 -> 24)
    constexpr int64_t hc_mix() const { return hc * (hc + 2); }
    /// the per-expert byte stride inside one blade is constant: blades are [n_embd, n_ff, n_expert] in GGUF order
    int64_t n_csa_layers() const {
        int64_t n = 0;
        for (int64_t r : compress) n += (r == csa_block);
        return n;
    }
    int64_t n_hca_layers() const {
        int64_t n = 0;
        for (int64_t r : compress) n += (r == hca_block);
        return n;
    }
};

/// Reads the geometry of the deepseek4 metadata shard `meta` (its own struct; ModelGeometry is Flash-Next's).
/// The guard (artifact/check_architecture_deepseek4) has already required every key below.
inline bool dsv4_geometry_from_gguf(const strata::GgufFile& meta, Dsv4Geometry& g, std::string& err) {
    err = strata::check_architecture(meta);
    if (!err.empty()) return false;
    if (strata::model_family(meta) != strata::ModelFamily::DeepseekV4) {
        err = meta.path() + ": this is not a deepseek4 file";
        return false;
    }
    g = Dsv4Geometry{};
    const std::string p = "deepseek4.";
    auto num = [&](const char* key, int64_t& dst) {
        if (const strata::MetaValue* v = meta.get(p + key))
            if (v->is_num()) dst = (int64_t) v->u;
    };
    num("block_count", g.n_layers);
    num("embedding_length", g.n_embd);
    num("attention.head_count", g.n_head);
    num("attention.key_length", g.head_dim);
    num("attention.q_lora_rank", g.q_lora_rank);
    num("attention.output_group_count", g.o_groups);
    num("attention.output_lora_rank", g.o_lora_rank);
    num("attention.sliding_window", g.sliding_window);
    num("rope.dimension_count", g.n_rot);
    num("expert_count", g.n_expert);
    num("expert_used_count", g.n_expert_used);
    num("expert_feed_forward_length", g.n_ff);
    num("attention.indexer.head_count", g.idx_q_heads);
    num("attention.indexer.key_length", g.idx_key_dim);
    num("attention.indexer.top_k", g.idx_top_k);
    num("hyper_connection.count", g.hc);
    num("hyper_connection.sinkhorn_iterations", g.hc_sinkhorn);
    num("hash_layer_count", g.n_hash);
    if (const strata::MetaValue* v = meta.get(p + "attention.compress_ratios"))
        for (uint64_t l = 0; l < v->count; ++l) g.compress.push_back((int64_t) v->items[(size_t) l].u);
    if (const strata::MetaValue* v = meta.get("tokenizer.ggml.tokens"))
        if (v->type == strata::MetaType::ARRAY && v->count > 0) g.n_vocab = (int64_t) v->count;
    // the two pooling factors, smallest = CSA / largest = HCA (the guard proved at most two non-zero values)
    {
        int64_t a = 0, b = 0;
        for (int64_t r : g.compress)
            if (r && r != a && r != b) {
                if (!a) a = r;
                else if (!b) { if (r < a) { b = a; a = r; } else b = r; }
            }
        g.csa_block = a;
        g.hca_block = b;
    }
    if ((int64_t) g.compress.size() < g.n_layers) {
        err = meta.path() + ": deepseek4.attention.compress_ratios has fewer entries than the trunk's layers";
        return false;
    }
    return true;
}

inline Dsv4LayerKind dsv4_layer_kind(const Dsv4Geometry& g, int64_t layer) {
    const int64_t r = g.compress[(size_t) layer];
    return r == 0 ? Dsv4LayerKind::Sliding : (r == g.csa_block ? Dsv4LayerKind::Csa : Dsv4LayerKind::Hca);
}

/// Every tensor of layer `layer` of a deepseek4 file (the GA names; the routing block decides tid2eid vs
/// exp_probs_b, the schedule decides the compressor / indexer tensors).
inline std::vector<TensorWant> dsv4_layer_tensors(const Dsv4Geometry& g, int64_t layer) {
    const int64_t N = g.n_embd, NE = g.n_expert, FF = g.n_ff, HD = g.hc_dim(), MIX = g.hc_mix(), QL = g.q_lora_rank;
    std::vector<TensorWant> t;
    auto add = [&](const char* s, std::vector<int64_t> shape) { t.push_back({s, std::move(shape)}); };
    // attention: low-rank Q, one shared K=V latent, per-head sinks, grouped low-rank O
    add("attn_norm.weight", {N});
    add("attn_q_a.weight", {N, QL});
    add("attn_q_a_norm.weight", {QL});
    add("attn_q_b.weight", {QL, g.n_head * g.head_dim});
    add("attn_kv.weight", {N, g.head_dim});
    add("attn_kv_a_norm.weight", {g.head_dim});
    add("attn_output_a.weight", {N, g.o_groups * g.o_lora_rank});
    add("attn_output_b.weight", {g.o_groups * g.o_lora_rank, N});
    add("attn_sinks.weight", {g.n_head});
    // mHC (manifold-constrained hyper-connections): base + fn per mixer, three learned scales (the Sinkhorn
    // comb is computed from these at run time; hyper_connection.sinkhorn_iterations rounds)
    add("hc_attn_base.weight", {MIX});
    add("hc_attn_fn.weight", {HD, MIX});
    add("hc_attn_scale.weight", {3});
    add("hc_ffn_base.weight", {MIX});
    add("hc_ffn_fn.weight", {HD, MIX});
    add("hc_ffn_scale.weight", {3});
    // MoE: scores from ffn_gate_inp always; which experts come from the frozen table (hash layers, I32
    // [n_expert_used, n_vocab]) or from the top-k of the biased scores (exp_probs_b, every other layer)
    add("ffn_norm.weight", {N});
    add("ffn_gate_inp.weight", {N, NE});
    if (layer < g.n_hash) add("ffn_gate_tid2eid.weight", {g.n_expert_used, g.n_vocab});
    else add("exp_probs_b.bias", {NE});
    add("ffn_gate_shexp.weight", {N, FF});
    add("ffn_up_shexp.weight", {N, FF});
    add("ffn_down_shexp.weight", {FF, N});
    add("ffn_gate_exps.weight", {N, FF, NE});
    add("ffn_up_exps.weight", {N, FF, NE});
    add("ffn_down_exps.weight", {FF, N, NE});
    const Dsv4LayerKind kind = dsv4_layer_kind(g, layer);
    if (kind == Dsv4LayerKind::Sliding) return t;
    // the pool compressor: CSA compresses overlapping blocks (2 entries per block: width 2x head_dim) at rate
    // csa_block; HCA one entry per block (width head_dim) at hca_block; APE = a learned positional embedding
    // per rate slots of the pool entry
    const int64_t cw = (kind == Dsv4LayerKind::Csa) ? 2 * g.head_dim : g.head_dim;
    add("attn_compressor_kv.weight", {N, cw});
    add("attn_compressor_gate.weight", {N, cw});
    add("attn_compressor_norm.weight", {g.head_dim});
    add("attn_compressor_ape.weight", {cw, (kind == Dsv4LayerKind::Csa) ? g.csa_block : g.hca_block});
    if (kind == Dsv4LayerKind::Csa) {
        // the Lightning indexer: QL low-rank Q over idx_q_heads x idx_key_dim, one score weight per head from
        // one shared projection (idx_q_heads wide - it equals rope.dimension_count in every published file, but
        // it IS the head count: inference/kernel.py's weights_proj(dim, n_heads)), and its own CSA-shaped pool
        add("indexer.attn_q_b.weight", {QL, g.idx_q_heads * g.idx_key_dim});
        add("indexer.proj.weight", {N, g.idx_q_heads});
        add("indexer_compressor_kv.weight", {N, 2 * g.idx_key_dim});
        add("indexer_compressor_gate.weight", {N, 2 * g.idx_key_dim});
        add("indexer_compressor_norm.weight", {g.idx_key_dim});
        add("indexer_compressor_ape.weight", {2 * g.idx_key_dim, g.csa_block});
    }
    return t;
}

/// The tensors outside the layers (the HyperHead collapses the hc streams before output_norm).
inline std::vector<TensorWant> dsv4_global_tensors(const Dsv4Geometry& g) {
    std::vector<TensorWant> t;
    t.push_back({"token_embd.weight", {g.n_embd, g.n_vocab}});
    t.push_back({"output.weight", {g.n_embd, g.n_vocab}});
    t.push_back({"output_norm.weight", {g.n_embd}});
    t.push_back({"output_hc_base.weight", {g.hc}});
    t.push_back({"output_hc_fn.weight", {g.hc_dim(), g.hc}});
    t.push_back({"output_hc_scale.weight", {1}});
    return t;
}

/// Checks the model `m` (all shards) against the deepseek4 layout table of `g` - and, the table being complete,
/// refuses any blk.* tensor it does not name.  `checked` counts the tensors found.
inline bool check_gguf_layout_dsv4(const strata::GgufModel& m, const Dsv4Geometry& g, std::string& err,
                                   size_t* checked = nullptr) {
    std::set<std::string> named;
    size_t n = 0;
    auto check = [&](const std::string& name, const std::vector<int64_t>& shape) {
        named.insert(name);
        size_t s = 0;
        const strata::TensorInfo* t = m.find(name, &s);
        if (!t) {
            err = "missing " + name;
            return false;
        }
        bool same = t->shape.size() == shape.size();
        for (size_t i = 0; same && i < shape.size(); ++i) same = t->shape[i] == (uint64_t) shape[i];
        if (!same) {
            std::string got, want;
            for (uint64_t d : t->shape) got += (got.empty() ? "" : ", ") + std::to_string(d);
            for (int64_t d : shape) want += (want.empty() ? "" : ", ") + std::to_string(d);
            err = name + " is [" + got + "], the deepseek4 layout wants [" + want + "]";
            return false;
        }
        if (!m.in_bounds(*t, s)) {
            err = name + " (" + t->type_name() + ") has no known size inside " + m.shard(s).path();
            return false;
        }
        ++n;
        return true;
    };
    for (const TensorWant& w : dsv4_global_tensors(g))
        if (!check(w.name, w.shape)) return false;
    for (int64_t l = 0; l < g.n_layers; ++l)
        for (const TensorWant& w : dsv4_layer_tensors(g, l))
            if (!check("blk." + std::to_string(l) + "." + w.name, w.shape)) return false;
    for (size_t s = 0; s < m.size(); ++s)
        for (const strata::TensorInfo& t : m.shard(s).tensors())
            if (!named.count(t.name)) {
                err = t.name + " is not in the " + std::string(strata::model_family_title(strata::ModelFamily::DeepseekV4)) +
                      " layout";
                return false;
            }
    if (checked) *checked = n;
    return true;
}

}  // namespace strata::core
