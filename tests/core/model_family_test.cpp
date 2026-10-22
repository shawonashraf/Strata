// tests/core/model_family_test.cpp - the model families (NEW-MODELS-PLAN.md T0) on synthetic GGUFs: no model, no GPU.
//
//   * the architecture guard accepts qwen4exp, qwen35moe and deepseek4 and refuses anything else; the engine's
//     guard refuses qwen35moe with "not ... yet" (no engine path for Qwen3.6 before T1) and deepseek4 with the
//     llama-server pointer (docs/DEEPSEEK_V4_PLAN.md: recognised, geometry read, no engine path yet);
//   * geometry_from_gguf reads Qwen3.6-35B-A3B's real metadata values (hidden 2048, 40 + 1 MTP layers, 32 GDN V
//     heads, conv 8192, no hyper-connections, no indexer) and Flash-Next's as exactly the compiled defaults;
//     dsv4_geometry_from_gguf reads DeepSeek-V4-Flash's GA values (43 layers, 21 CSA + 20 HCA + 2 sliding, 3
//     hash-routed first, 256 experts top-6 MXFP4, mHC 4 with 20 Sinkhorn rounds);
//   * the layout tables, on tiny files of either family: a complete file passes, a wrong shape, a missing tensor
//     and (qwen35moe) a tensor the table does not name are refused, naming the tensor.
//
// `model_family_test --real SHARD`: the same view of a real model (headers only): family, geometry, the kernels'
// verdict and the layout check over every shard.
#include "gguf_fixture.hpp"

#include "strata/core/family_layout.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-90s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-model-family-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

using fixture::Kv;
using fixture::Tensor;
const uint32_t F32 = 0;

struct Dims {
    uint64_t block_count, embd, heads, heads_kv, key_len, experts, used, ff, ff_sh, conv, state, groups, v_heads,
        inner, interval, nextn, hc, hc_lr, idx_heads, idx_key;
};
// Qwen3.6-35B-A3B (unsloth/Qwen3.6-35B-A3B-MTP-GGUF, every quant) and Qwen3.8-Flash-Next, as their files say
const Dims kQwen36 = {41, 2048, 16, 2, 256, 256, 8, 512, 512, 4, 128, 16, 32, 4096, 4, 1, 0, 0, 0, 0};
const Dims kFlashNext = {48, 2560, 24, 2, 256, 512, 10, 640, 640, 4, 128, 16, 48, 6144, 4, 0, 4, 320, 4, 128};
// tiny models of each family for the layout checks (every tensor F32, a few KB)
const Dims kTiny36 = {5, 64, 2, 1, 16, 4, 2, 32, 32, 4, 8, 2, 4, 32, 4, 1, 0, 0, 0, 0};
const Dims kTinyFlash = {8, 64, 2, 1, 16, 4, 2, 32, 32, 4, 8, 2, 4, 32, 4, 0, 2, 8, 2, 8};

// DeepSeek-V4 (deepseek4): the real GA geometry from unsloth/DeepSeek-V4-Flash-0731-GGUF's header (checked key
// by key against it), and a tiny stand-in with the same layer pattern: two sliding bootstraps, then CSA / HCA
// alternating; hash layers 0..n_hash-1; schedule = layer 0-1 rate 0, even rate csa, odd rate hca.  GA's
// schedule has mtp=3 extra sliding rows past the 43-layer trunk (the DSpark layers, a separate file).
struct V4Dims {
    uint64_t block_count, embd, heads, key_len, q_lora, o_groups, o_lora, rot, window, experts, used, ff,
        idx_heads, idx_key, idx_topk, hc, sink, hash, csa, hca, mtp;
};
const V4Dims kV4Real = {43, 4096, 64, 512, 1024, 8, 1024, 64, 128, 256, 6, 2048, 64, 128, 512, 4, 20, 3, 4, 128, 3};
const V4Dims kTinyV4 = {6, 64, 2, 16, 8, 2, 16, 8, 4, 4, 2, 32, 2, 8, 3, 4, 2, 3, 4, 8, 0};

std::vector<uint64_t> v4_schedule(const V4Dims& d) {
    std::vector<uint64_t> r;
    for (uint64_t l = 0; l < d.block_count + d.mtp; ++l)
        r.push_back(l < d.block_count ? (l < 2 ? 0 : (l % 2 ? d.hca : d.csa)) : 0);
    return r;
}

std::vector<Kv> metadata_v4(const V4Dims& d, uint64_t vocab) {
    const std::string p = "deepseek4.";
    std::vector<Kv> kv = {fixture::str("general.architecture", "deepseek4"),
                          fixture::u32(p + "block_count", d.block_count),
                          fixture::u32(p + "embedding_length", d.embd),
                          fixture::u32(p + "attention.head_count", d.heads),
                          fixture::u32(p + "attention.head_count_kv", 1),
                          fixture::u32(p + "attention.key_length", d.key_len),
                          fixture::u32(p + "attention.value_length", d.key_len),
                          fixture::u32(p + "attention.q_lora_rank", d.q_lora),
                          fixture::u32(p + "attention.output_group_count", d.o_groups),
                          fixture::u32(p + "attention.output_lora_rank", d.o_lora),
                          fixture::u32(p + "attention.sliding_window", d.window),
                          fixture::u32(p + "rope.dimension_count", d.rot),
                          fixture::u32(p + "expert_count", d.experts),
                          fixture::u32(p + "expert_used_count", d.used),
                          fixture::u32(p + "expert_feed_forward_length", d.ff),
                          fixture::u32(p + "expert_shared_count", 1),
                          fixture::u32(p + "attention.indexer.head_count", d.idx_heads),
                          fixture::u32(p + "attention.indexer.key_length", d.idx_key),
                          fixture::u32(p + "attention.indexer.top_k", d.idx_topk),
                          fixture::u32(p + "hyper_connection.count", d.hc),
                          fixture::u32(p + "hyper_connection.sinkhorn_iterations", d.sink),
                          fixture::u32(p + "hash_layer_count", d.hash),
                          fixture::u32s(p + "attention.compress_ratios", v4_schedule(d))};
    std::vector<std::string> tokens;
    for (uint64_t i = 0; i < vocab; ++i) tokens.push_back("t" + std::to_string(i));
    kv.push_back(fixture::str_array("tokenizer.ggml.tokens", tokens));
    return kv;
}

std::vector<Kv> metadata(const char* arch, const Dims& d, uint64_t vocab) {
    const std::string p = std::string(arch) + ".";
    std::vector<Kv> kv = {fixture::str("general.architecture", arch),
                          fixture::u32(p + "block_count", d.block_count),
                          fixture::u32(p + "embedding_length", d.embd),
                          fixture::u32(p + "attention.head_count", d.heads),
                          fixture::u32(p + "attention.head_count_kv", d.heads_kv),
                          fixture::u32(p + "attention.key_length", d.key_len),
                          fixture::u32(p + "attention.value_length", d.key_len),
                          fixture::u32(p + "expert_count", d.experts),
                          fixture::u32(p + "expert_used_count", d.used),
                          fixture::u32(p + "expert_feed_forward_length", d.ff),
                          fixture::u32(p + "expert_shared_feed_forward_length", d.ff_sh),
                          fixture::u32(p + "ssm.conv_kernel", d.conv),
                          fixture::u32(p + "ssm.state_size", d.state),
                          fixture::u32(p + "ssm.group_count", d.groups),
                          fixture::u32(p + "ssm.time_step_rank", d.v_heads),
                          fixture::u32(p + "ssm.inner_size", d.inner),
                          fixture::u32(p + "full_attention_interval", d.interval)};
    if (std::string(arch) == "qwen35moe") {
        kv.push_back(fixture::u32(p + "nextn_predict_layers", d.nextn));
    } else {
        kv.push_back(fixture::u32(p + "hyper_connection.count", d.hc));
        kv.push_back(fixture::u32(p + "hyper_connection.low_rank", d.hc_lr));
        kv.push_back(fixture::u32(p + "attention.indexer.head_count", d.idx_heads));
        kv.push_back(fixture::u32(p + "attention.indexer.key_length", d.idx_key));
    }
    std::vector<std::string> tokens;
    for (uint64_t i = 0; i < vocab; ++i) tokens.push_back("t" + std::to_string(i));
    kv.push_back(fixture::str_array("tokenizer.ggml.tokens", tokens));
    return kv;
}

/// The tensors the layout table of `g` names, as F32 fixtures.
std::vector<Tensor> tensors_of(const strata::core::ModelGeometry& g) {
    std::vector<Tensor> ts;
    uint8_t seed = 1;
    auto add = [&](const std::string& name, const std::vector<int64_t>& shape) {
        std::vector<uint64_t> s(shape.begin(), shape.end());
        ts.push_back({name, s, F32, seed++});
    };
    for (const auto& w : strata::core::global_tensors(g)) add(w.name, w.shape);
    for (int64_t l = 0; l < g.n_layers + g.n_mtp_layers; ++l)
        for (const auto& w : strata::core::layer_tensors(g, l)) add("blk." + std::to_string(l) + "." + w.name, w.shape);
    return ts;
}

bool geometry_of(const fs::path& p, strata::core::ModelGeometry& g, int64_t& k, std::string& err) {
    const strata::GgufFile f(p.string());
    return strata::core::geometry_from_gguf(f, g, k, err);
}

/// The tensors a deepseek4 layout table names, as F32 fixtures.
std::vector<Tensor> tensors_of_v4(const strata::core::Dsv4Geometry& g) {
    std::vector<Tensor> ts;
    uint8_t seed = 1;
    auto add = [&](const std::string& name, const std::vector<int64_t>& shape) {
        std::vector<uint64_t> s(shape.begin(), shape.end());
        ts.push_back({name, s, F32, seed++});
    };
    for (const auto& w : strata::core::dsv4_global_tensors(g)) add(w.name, w.shape);
    for (int64_t l = 0; l < g.n_layers; ++l)
        for (const auto& w : strata::core::dsv4_layer_tensors(g, l)) add("blk." + std::to_string(l) + "." + w.name, w.shape);
    return ts;
}

std::string layout_error_v4(const fs::path& dir, const V4Dims& d, uint64_t vocab,
                            void (*mutate)(std::vector<Tensor>&)) {
    const fs::path meta = dir / "v4-meta.gguf";
    const std::vector<Kv> kv = metadata_v4(d, vocab);
    fixture::write(meta, kv, {});
    strata::core::Dsv4Geometry g;
    std::string err;
    if (!strata::core::dsv4_geometry_from_gguf(strata::GgufFile(meta.string()), g, err)) return "geometry: " + err;
    std::vector<Tensor> ts = tensors_of_v4(g);
    if (mutate) mutate(ts);
    const fs::path model = dir / "deepseek4-tiny.gguf";
    fixture::write(model, kv, ts);
    const strata::GgufModel m({model.string()});
    size_t n = 0;
    if (!strata::core::check_gguf_layout_dsv4(m, g, err, &n)) return err;
    return {};
}

std::string layout_error(const fs::path& dir, const char* arch, const Dims& d, uint64_t vocab,
                         void (*mutate)(std::vector<Tensor>&)) {
    const fs::path meta = dir / "meta.gguf";
    fixture::write(meta, metadata(arch, d, vocab), {});
    strata::core::ModelGeometry g;
    int64_t k = 0;
    std::string err;
    if (std::string(arch) == "qwen35moe") {
        if (!geometry_of(meta, g, k, err)) return "geometry: " + err;
    } else {
        // the qwen4exp guard pins Flash-Next's own block count and widths, so the tiny model's geometry is set here
        g.n_layers = (int64_t) d.block_count;
        g.n_embd = (int64_t) d.embd;
        g.n_head = (int64_t) d.heads;
        g.n_head_kv = (int64_t) d.heads_kv;
        g.head_dim = (int64_t) d.key_len;
        g.n_expert = (int64_t) d.experts;
        g.n_ff = (int64_t) d.ff;
        g.n_ff_shexp = (int64_t) d.ff_sh;
        g.ssm_d_conv = (int64_t) d.conv;
        g.ssm_state_size = (int64_t) d.state;
        g.ssm_k_heads = (int64_t) d.groups;
        g.ssm_v_heads = (int64_t) d.v_heads;
        g.ssm_value_dim = (int64_t) d.inner;
        g.ssm_conv_channels = 2 * g.ssm_state_size * g.ssm_k_heads + g.ssm_value_dim;
        g.hc = (int64_t) d.hc;
        g.hc_lr = (int64_t) d.hc_lr;
        g.idx_q_heads = (int64_t) d.idx_heads;
        g.idx_key_dim = (int64_t) d.idx_key;
        g.n_vocab = (int64_t) vocab;
    }
    std::vector<Tensor> ts = tensors_of(g);
    if (mutate) mutate(ts);
    const fs::path model = dir / (std::string(arch) + "-tiny.gguf");
    fixture::write(model, metadata(arch, d, vocab), ts);
    const strata::GgufModel m({model.string()});
    size_t n = 0;
    if (!strata::core::check_gguf_layout(m, g, err, &n)) return err;
    return {};
}

int real_model(const std::string& any) {
    try {
        const strata::GgufModel m = strata::GgufModel::open(any);
        const std::string arch = strata::check_architecture(m.meta());
        std::printf("  %zu shard(s); architecture guard: %s; engine guard: %s\n", m.size(),
                    arch.empty() ? "ok" : arch.c_str(),
                    strata::check_engine_architecture(m.meta()).empty() ? "runs"
                                                                         : strata::check_engine_architecture(m.meta()).c_str());
        if (!arch.empty()) return 1;
        if (strata::model_family(m.meta()) == strata::ModelFamily::DeepseekV4) {
            strata::core::Dsv4Geometry g;
            std::string err;
            if (!strata::core::dsv4_geometry_from_gguf(m.meta(), g, err)) {
                std::printf("  geometry: %s\n", err.c_str());
                return 1;
            }
            std::printf("  family %s: layers %lld (%lld sliding + %lld CSA + %lld HCA, %lld hash-routed first),"
                        " hidden %lld, attention %lld x %lld shared K=V window %lld, Q rank %lld, O %lld x %lld,"
                        " indexer %lld x %lld top %lld, MoE %lld experts top-%lld FF %lld, hc %lld sinkhorn %lld,"
                        " vocab %lld\n",
                        strata::model_family_title(strata::ModelFamily::DeepseekV4), (long long) g.n_layers,
                        (long long) (g.n_layers - g.n_csa_layers() - g.n_hca_layers()), (long long) g.n_csa_layers(),
                        (long long) g.n_hca_layers(), (long long) g.n_hash, (long long) g.n_embd,
                        (long long) g.n_head, (long long) g.head_dim, (long long) g.sliding_window,
                        (long long) g.q_lora_rank, (long long) g.o_groups, (long long) g.o_lora_rank,
                        (long long) g.idx_q_heads, (long long) g.idx_key_dim, (long long) g.idx_top_k,
                        (long long) g.n_expert, (long long) g.n_expert_used, (long long) g.n_ff, (long long) g.hc,
                        (long long) g.hc_sinkhorn, (long long) g.n_vocab);
            size_t n = 0;
            const bool ok = strata::core::check_gguf_layout_dsv4(m, g, err, &n);
            std::printf("  layout: %s (%zu tensors checked)\n", ok ? "ok" : err.c_str(), n);
            return ok ? 0 : 1;
        }
        strata::core::ModelGeometry g;
        int64_t k = 0;
        std::string err;
        if (!strata::core::geometry_from_gguf(m.meta(), g, k, err)) {
            std::printf("  geometry: %s\n", err.c_str());
            return 1;
        }
        std::printf("  family %s: layers %lld (+%lld MTP), hidden %lld, attention %lld/%lld x %lld every %lld, GDN K %lld"
                    " V %lld x %lld conv %lld (%lld), MoE %lld experts top-%lld FF %lld shared %lld, hc %lld, indexer"
                    " %lld, vocab %lld\n",
                    strata::model_family_title(g.family), (long long) g.n_layers, (long long) g.n_mtp_layers,
                    (long long) g.n_embd, (long long) g.n_head, (long long) g.n_head_kv, (long long) g.head_dim,
                    (long long) g.qsa_interval, (long long) g.ssm_k_heads, (long long) g.ssm_v_heads,
                    (long long) g.ssm_state_size, (long long) g.ssm_conv_channels, (long long) g.ssm_d_conv,
                    (long long) g.n_expert, (long long) k, (long long) g.n_ff, (long long) g.n_ff_shexp,
                    (long long) g.hc, (long long) g.idx_q_heads, (long long) g.n_vocab);
        const std::string mm = strata::core::compiled_geometry_mismatch(g);
        std::printf("  compiled kernels: %s\n", mm.empty() ? "match" : mm.c_str());
        size_t n = 0;
        const bool ok = strata::core::check_gguf_layout(m, g, err, &n);
        std::printf("  layout: %s (%zu tensors checked)\n", ok ? "ok" : err.c_str(), n);
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("  refused: %s\n", e.what());
        return 1;
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--real") return real_model(argv[2]);
    std::printf("model_family_test\n");
    TempDir d;
    {
        // the guards
        const fs::path q36 = d.path / "q36.gguf", fn = d.path / "fn.gguf", other = d.path / "other.gguf";
        fixture::write(q36, metadata("qwen35moe", kQwen36, 8), {});
        fixture::write(fn, metadata("qwen4exp", kFlashNext, 8), {});
        fixture::write(other, {fixture::str("general.architecture", "llama")}, {});
        const strata::GgufFile a(q36.string()), b(fn.string()), c(other.string());
        check(strata::model_family(a) == strata::ModelFamily::Qwen35Moe &&
                  strata::model_family(b) == strata::ModelFamily::Qwen4Exp &&
                  strata::model_family(c) == strata::ModelFamily::Unknown,
              "model_family: qwen35moe, qwen4exp, unknown");
        check(strata::check_architecture(a).empty(), "the architecture guard accepts qwen35moe");
        check(strata::check_architecture(b).empty(), "the architecture guard accepts qwen4exp");
        check(strata::check_architecture(c).find("'llama'") != std::string::npos, "and refuses llama, naming it");
        const std::string e = strata::check_engine_architecture(a);
        check(e.find("Qwen3.6-35B-A3B") != std::string::npos && e.find("does not run it yet") != std::string::npos,
              "the engine guard refuses Qwen3.6: recognised, not supported yet");
        check(strata::check_engine_architecture(b).empty(), "the engine guard runs qwen4exp");
        std::vector<Kv> kv = metadata("qwen35moe", kQwen36, 8);
        for (auto it = kv.begin(); it != kv.end(); ++it)
            if (it->key == "qwen35moe.ssm.time_step_rank") { kv.erase(it); break; }
        const fs::path miss = d.path / "miss.gguf";
        fixture::write(miss, kv, {});
        check(strata::check_architecture(strata::GgufFile(miss.string())).find("ssm.time_step_rank") != std::string::npos,
              "a qwen35moe file without ssm.time_step_rank is refused, naming the key");
    }
    {
        // DeepSeek-V4 (deepseek4): recognised and read, yet no engine runs it (stage 1)
        const fs::path v4 = d.path / "v4.gguf";
        fixture::write(v4, metadata_v4(kV4Real, 8), {});
        const strata::GgufFile f(v4.string());
        check(strata::model_family(f) == strata::ModelFamily::DeepseekV4, "model_family: deepseek4");
        check(strata::check_architecture(f).empty(),
              "the architecture guard accepts deepseek4 (the GA keys and the per-layer schedule)");
        const std::string e = strata::check_engine_architecture(f);
        check(e.find("DeepSeek-V4") != std::string::npos && e.find("deepseek4") != std::string::npos &&
                  e.find("llama-server") != std::string::npos,
              "the engine guard refuses deepseek4 by name and points at the llama-server route (" + e + ")");
        // a missing key is refused, naming it
        std::vector<Kv> kv = metadata_v4(kV4Real, 8);
        for (auto it = kv.begin(); it != kv.end(); ++it)
            if (it->key == "deepseek4.attention.indexer.top_k") { kv.erase(it); break; }
        const fs::path miss = d.path / "v4miss.gguf";
        fixture::write(miss, kv, {});
        check(strata::check_architecture(strata::GgufFile(miss.string())).find("indexer.top_k") != std::string::npos,
              "a deepseek4 file without attention.indexer.top_k is refused, naming the key");
        // the invariants: one shared K=V latent, at most two pooling factors in the schedule
        std::vector<Kv> kv2 = metadata_v4(kV4Real, 8);
        for (auto& k : kv2)
            if (k.key == "deepseek4.attention.head_count_kv") k.u = 8;
        const fs::path mqa = d.path / "v4mqa.gguf";
        fixture::write(mqa, kv2, {});
        check(strata::check_architecture(strata::GgufFile(mqa.string())).find("head_count_kv") != std::string::npos,
              "a deepseek4 file with head_count_kv 8 is refused (V4 shares one K=V latent)");
        V4Dims three = kV4Real;
        std::vector<Kv> kv3 = metadata_v4(three, 8);
        for (auto& k : kv3)
            if (k.key == "deepseek4.attention.compress_ratios") k.arr[5] = 4096;   // a third pooling factor
        const fs::path sched = d.path / "v4sched.gguf";
        fixture::write(sched, kv3, {});
        check(strata::check_architecture(strata::GgufFile(sched.string())).find("third pooling factor") !=
                  std::string::npos,
              "a schedule with three pooling factors is refused, naming the layer");
        // GA's schedule has 46 rows for 43 trunk layers: the extra ones (DSpark speculative layers, shipped
        // separately) must be sliding; a compressed row there is refused
        std::vector<Kv> kv4 = metadata_v4(kV4Real, 8);
        for (auto& k : kv4)
            if (k.key == "deepseek4.attention.compress_ratios") k.arr[45] = 4;
        const fs::path tail = d.path / "v4tail.gguf";
        fixture::write(tail, kv4, {});
        check(strata::check_architecture(strata::GgufFile(tail.string())).find("must be sliding") != std::string::npos,
              "a compressed row past the trunk is refused, naming the row");
    }
    {
        // the geometry of the real models, from their metadata values
        const fs::path q36 = d.path / "q36g.gguf", fn = d.path / "fng.gguf";
        fixture::write(q36, metadata("qwen35moe", kQwen36, 248320), {});
        fixture::write(fn, metadata("qwen4exp", kFlashNext, 248320), {});
        strata::core::ModelGeometry g;
        int64_t k = 0;
        std::string err;
        const bool ok = geometry_of(q36, g, k, err);
        check(ok, "Qwen3.6 geometry reads" + (ok ? std::string() : ": " + err));
        check(g.family == strata::ModelFamily::Qwen35Moe && g.n_layers == 40 && g.n_mtp_layers == 1 && g.n_embd == 2048,
              "Qwen3.6: 40 trunk layers + 1 MTP block, hidden 2048");
        check(g.n_qsa_layers() == 10 && g.n_gdn_layers() == 30 && strata::core::layer_kind(g, 3) ==
                  strata::core::LayerKind::Attention && strata::core::layer_kind(g, 40) == strata::core::LayerKind::Mtp,
              "Qwen3.6: 10 attention / 30 GDN layers, layer 40 is the MTP block");
        check(g.ssm_k_heads == 16 && g.ssm_v_heads == 32 && g.ssm_conv_channels == 8192 && g.ssm_value_dim == 4096,
              "Qwen3.6 GDN: 16 K groups, 32 V heads, conv 8192, inner 4096");
        check(g.n_head == 16 && g.n_head_kv == 2 && g.head_dim == 256 && !g.has_indexer() && !g.has_hyper_connections(),
              "Qwen3.6 attention: 16 Q / 2 KV x 256, no indexer, no hyper-connections");
        check(g.n_expert == 256 && k == 8 && g.n_ff == 512 && g.n_ff_shexp == 512 && g.n_vocab == 248320,
              "Qwen3.6 MoE: 256 experts top-8, FF 512, shared 512; vocab 248320");
        check(!strata::core::compiled_geometry_mismatch(g).empty(), "Qwen3.6 does not match the compiled kernels");
        strata::core::ModelGeometry f;
        const bool okf = geometry_of(fn, f, k, err);
        const strata::core::ModelGeometry c;
        check(okf && strata::core::compiled_geometry_mismatch(f).empty() && k == 10 && f.n_expert == c.n_expert &&
                  f.ssm_conv_channels == c.ssm_conv_channels && f.n_vocab == c.n_vocab && f.hc_dim() == c.hc_dim(),
              "Flash-Next's metadata gives exactly the compiled geometry");
        Dims coder = kFlashNext;
        coder.experts = 256;
        const fs::path cp = d.path / "coder.gguf";
        fixture::write(cp, metadata("qwen4exp", coder, 8), {});
        check(geometry_of(cp, f, k, err) && f.n_expert == 256 && strata::core::compiled_geometry_mismatch(f).empty(),
              "the Coder's 256 experts are a run-time size, not a kernel mismatch");
        Dims wrong = kQwen36;
        wrong.inner = 4000;
        const fs::path wp = d.path / "wrong.gguf";
        fixture::write(wp, metadata("qwen35moe", wrong, 8), {});
        check(!geometry_of(wp, f, k, err) && err.find("ssm.inner_size") != std::string::npos,
              "an inconsistent ssm.inner_size is refused");
    }
    {
        // the layout tables
        check(layout_error(d.path, "qwen35moe", kTiny36, 16, nullptr).empty(),
              "a complete qwen35moe file passes its layout (incl. the nextn MTP block)");
        const std::string ef = layout_error(d.path, "qwen4exp", kTinyFlash, 16, nullptr);
        check(ef.empty(), "a complete qwen4exp file passes its layout (hyper-connections, indexer)" +
                              (ef.empty() ? std::string() : ": " + ef));
        std::string e = layout_error(d.path, "qwen35moe", kTiny36, 16, [](std::vector<Tensor>& ts) {
            for (auto& t : ts)
                if (t.name == "blk.1.ssm_conv1d.weight") t.shape[1] += 1;
        });
        check(e.find("blk.1.ssm_conv1d.weight") != std::string::npos && e.find("wants") != std::string::npos,
              "a wrong shape is refused, naming the tensor and both shapes");
        e = layout_error(d.path, "qwen35moe", kTiny36, 16, [](std::vector<Tensor>& ts) {
            for (auto it = ts.begin(); it != ts.end(); ++it)
                if (it->name == "blk.4.nextn.eh_proj.weight") { ts.erase(it); break; }
        });
        check(e == "missing blk.4.nextn.eh_proj.weight", "a missing MTP tensor is refused");
        e = layout_error(d.path, "qwen35moe", kTiny36, 16, [](std::vector<Tensor>& ts) {
            ts.push_back({"blk.0.hc_attn_norm.weight", {64}, F32, 99});
        });
        check(e.find("blk.0.hc_attn_norm.weight") != std::string::npos && e.find("not in the") != std::string::npos,
              "a tensor the qwen35moe table does not name is refused");
        e = layout_error(d.path, "qwen4exp", kTinyFlash, 16, [](std::vector<Tensor>& ts) {
            for (auto it = ts.begin(); it != ts.end(); ++it)
                if (it->name == "blk.3.indexer.k_proj.weight") { ts.erase(it); break; }
        });
        check(e == "missing blk.3.indexer.k_proj.weight", "Flash-Next: a missing indexer tensor is refused (" + e + ")");
    }
    {
        // DeepSeek-V4's geometry: the GA file's own values, and its tensor count (1,328, key by key against
        // unsloth/DeepSeek-V4-Flash-0731-GGUF's header)
        const fs::path v4 = d.path / "v4g.gguf";
        fixture::write(v4, metadata_v4(kV4Real, 129280), {});
        strata::core::Dsv4Geometry g;
        std::string err;
        const bool ok = strata::core::dsv4_geometry_from_gguf(strata::GgufFile(v4.string()), g, err);
        check(ok, "deepseek4 geometry reads" + (ok ? std::string() : ": " + err));
        check(g.n_layers == 43 && g.n_embd == 4096 && g.n_vocab == 129280 && g.n_hash == 3,
              "deepseek4: 43 layers, hidden 4096, vocab 129280, 3 hash-routed first");
        check(g.n_head == 64 && g.head_dim == 512 && g.sliding_window == 128 && g.q_lora_rank == 1024 &&
                  g.o_groups == 8 && g.o_lora_rank == 1024 && g.n_rot == 64,
              "deepseek4 attention: 64 x 512 shared K=V, window 128, Q rank 1024, O 8 x 1024, rope slice 64");
        check(g.n_expert == 256 && g.n_expert_used == 6 && g.n_ff == 2048,
              "deepseek4 MoE: 256 experts top-6, FF 2048, one shared");
        check(g.idx_q_heads == 64 && g.idx_key_dim == 128 && g.idx_top_k == 512,
              "deepseek4 indexer: 64 x 128, top 512");
        check(g.hc == 4 && g.hc_sinkhorn == 20 && g.hc_dim() == 16384 && g.hc_mix() == 24,
              "deepseek4 mHC: 4 streams (dim 16384), 24 mix columns, 20 Sinkhorn rounds");
        check(g.csa_block == 4 && g.hca_block == 128 && g.n_csa_layers() == 21 && g.n_hca_layers() == 20,
              "deepseek4 schedule: 2 sliding + 21 CSA (4x) + 20 HCA (128x)");
        check(strata::core::dsv4_layer_kind(g, 0) == strata::core::Dsv4LayerKind::Sliding &&
                  strata::core::dsv4_layer_kind(g, 2) == strata::core::Dsv4LayerKind::Csa &&
                  strata::core::dsv4_layer_kind(g, 3) == strata::core::Dsv4LayerKind::Hca &&
                  strata::core::dsv4_layer_kind(g, 42) == strata::core::Dsv4LayerKind::Csa,
              "deepseek4 layer kinds: 0 sliding, 2 CSA, 3 HCA, 42 CSA");
        size_t tensors = strata::core::dsv4_global_tensors(g).size();
        for (int64_t l = 0; l < g.n_layers; ++l) tensors += strata::core::dsv4_layer_tensors(g, l).size();
        check(tensors == 1328, "the deepseek4 table names 1328 tensors, as the GA file has (" +
                                   std::to_string(tensors) + ")");
        strata::core::ModelGeometry mg;
        int64_t mk = 0;
        check(!strata::core::geometry_from_gguf(strata::GgufFile(v4.string()), mg, mk, err) &&
                  err.find("Dsv4Geometry") != std::string::npos,
              "the Flash-Next geometry struct refuses a deepseek4 file, pointing at its own");
    }
    {
        // the deepseek4 layout table on a tiny file of the same pattern
        check(layout_error_v4(d.path, kTinyV4, 17, nullptr).empty(),
              "a complete deepseek4 file passes its layout (schedule, mHC, hash-routed prefix)");
        std::string e = layout_error_v4(d.path, kTinyV4, 17, [](std::vector<Tensor>& ts) {
            for (auto& t : ts)
                if (t.name == "blk.2.indexer_compressor_ape.weight") t.shape[1] += 1;
        });
        check(e.find("blk.2.indexer_compressor_ape.weight") != std::string::npos && e.find("wants") != std::string::npos,
              "a wrong CSA shape is refused, naming the tensor and both shapes (" + e + ")");
        e = layout_error_v4(d.path, kTinyV4, 17, [](std::vector<Tensor>& ts) {
            for (auto it = ts.begin(); it != ts.end(); ++it)
                if (it->name == "blk.0.ffn_gate_tid2eid.weight") { ts.erase(it); break; }
        });
        check(e == "missing blk.0.ffn_gate_tid2eid.weight", "a missing hash-layer table is refused (" + e + ")");
        e = layout_error_v4(d.path, kTinyV4, 17, [](std::vector<Tensor>& ts) {
            ts.push_back({"blk.0.exp_probs_b.bias", {4}, F32, 99});   // a hash layer has no learned bias
        });
        check(e.find("blk.0.exp_probs_b.bias") != std::string::npos && e.find("not in the") != std::string::npos,
              "an exp_probs_b on a hash layer is refused (" + e + ")");
        e = layout_error_v4(d.path, kTinyV4, 17, [](std::vector<Tensor>& ts) {
            for (auto it = ts.begin(); it != ts.end(); ++it)
                if (it->name == "blk.3.attn_compressor_kv.weight") { ts.erase(it); break; }
        });
        check(e == "missing blk.3.attn_compressor_kv.weight", "a missing HCA compressor is refused (" + e + ")");
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
