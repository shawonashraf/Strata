// tests/core/dsv4ref_test.cpp - the deepseek4 CPU reference on a tiny whole model (docs/DEEPSEEK_V4_PLAN.md
// stage 2): no GPU, no download.  The file carries the GA tensor names and quantization types (MXFP4
// experts, Q8_0 dense, BF16 router, F32 norms/mHC, I32 tid2eid); the fixture's pattern bytes would decode to
// NaN and Inf floats, so after writing we refill every payload with plain finite values per type (1.0, 0.5,
// d=2.0 Q8_0 blocks, all-1.0 MXFP4 blocks, in-range tid2eid ids).  The values stay simple - what this test
// proves is that the whole forward pass runs: every piece (schedule, mHC sinkhorn, CSA overlap pooling,
// indexer top-k, HCA pooling, hash routing, clamped Swiglu, HyperHead, rope tables) is reached, nothing
// throws, activations stay finite, and the greedy path is deterministic.  Numerical parity against
// llama.cpp is stage 2's other half: not runnable here (needs a >=160 GB machine) and stated as not run.
#include "strata/dsv4/dsv4ref.hpp"

#include "gguf_fixture.hpp"

#include "strata/core/family_layout.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-96s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

using Kv = fixture::Kv;
using Tensor = fixture::Tensor;

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-dsv4ref-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

// a six-layer tiny deepseek4: two sliding bootstraps (hash-routed MoE), CSA at 2 and 4, HCA at 3 and 5
constexpr int64_t kN = 64, kVocab = 17, kHeads = 2, kHD = 16, kQL = 32, kOG = 2, kOL = 16, kRot = 8,
                  kWin = 4, kNE = 4, kUsed = 2, kFF = 32, kIH = 2, kIKD = 12, kTopK = 3, kHC = 4,
                  kCsa = 4, kHca = 8, kHash = 2;

const uint32_t F32 = 0, BF16 = 30, Q8_0 = 8, MXFP4 = 39, I32 = 26;

std::vector<Kv> metadata() {
    const std::string p = "deepseek4.";
    std::vector<Kv> k = {fixture::str("general.architecture", "deepseek4"),
                         fixture::u32(p + "block_count", 6),
                         fixture::u32(p + "embedding_length", kN),
                         fixture::u32(p + "expert_feed_forward_length", kFF),
                         fixture::u32(p + "attention.head_count", kHeads),
                         fixture::u32(p + "attention.head_count_kv", 1),
                         fixture::u32(p + "attention.key_length", kHD),
                         fixture::u32(p + "attention.value_length", kHD),
                         fixture::u32(p + "attention.q_lora_rank", kQL),
                         fixture::u32(p + "attention.output_group_count", kOG),
                         fixture::u32(p + "attention.output_lora_rank", kOL),
                         fixture::u32(p + "attention.sliding_window", kWin),
                         fixture::u32(p + "rope.dimension_count", kRot),
                         fixture::u32(p + "expert_count", kNE),
                         fixture::u32(p + "expert_used_count", kUsed),
                         fixture::u32(p + "expert_shared_count", 1),
                         fixture::u32(p + "attention.indexer.head_count", kIH),
                         fixture::u32(p + "attention.indexer.key_length", kIKD),
                         fixture::u32(p + "attention.indexer.top_k", kTopK),
                         fixture::u32(p + "hyper_connection.count", kHC),
                         fixture::u32(p + "hyper_connection.sinkhorn_iterations", 2),
                         fixture::u32(p + "hash_layer_count", kHash),
                         fixture::u32s(p + "attention.compress_ratios", {0, 0, kCsa, kHca, kCsa, kHca})};
    // the numeric keys the reference reads (all optional for the guard, required for a faithful run)
    k.push_back(fixture::f32(p + "rope.freq_base", 10000.0f));
    k.push_back(fixture::f32(p + "attention.compress_rope_freq_base", 160000.0f));
    k.push_back(fixture::f32(p + "attention.layer_norm_rms_epsilon", 1e-6f));
    k.push_back(fixture::u32(p + "rope.scaling.factor", 16));
    k.push_back(fixture::u32(p + "rope.scaling.original_context_length", 32));
    k.push_back(fixture::f32(p + "rope.scaling.yarn_beta_fast", 32.0f));
    k.push_back(fixture::f32(p + "rope.scaling.yarn_beta_slow", 1.0f));
    k.push_back(fixture::f32(p + "hyper_connection.epsilon", 1e-6f));
    k.push_back(fixture::f32(p + "expert_weights_scale", 1.5f));
    k.push_back(fixture::u32s(p + "swiglu_clamp_exp", {10, 10, 10, 10, 10, 10}));
    k.push_back(fixture::u32s(p + "swiglu_clamp_shexp", {10, 10, 10, 10, 10, 10}));
    std::vector<std::string> tokens;
    for (int64_t i = 0; i < kVocab; ++i) tokens.push_back("t" + std::to_string(i));
    k.push_back(fixture::str_array("tokenizer.ggml.tokens", tokens));
    return k;
}

// every tensor the tables name, in the GA file's quantization types
std::vector<Tensor> tensors_of(const strata::core::Dsv4Geometry& g) {
    std::vector<Tensor> ts;
    uint8_t seed = 1;
    auto type_of = [](const std::string& leaf) {
        if (leaf.find("_exps.weight") != std::string::npos) return MXFP4;   // routed blades
        if (leaf.find("tid2eid") != std::string::npos) return I32;          // the frozen hash table
        if (leaf.find("gate_inp") != std::string::npos) return BF16;        // the router
        if (leaf.find("_norm.") != std::string::npos || leaf.find("sinks") != std::string::npos ||
            leaf.find("ape.weight") != std::string::npos)
            return F32;   // norms, sinks and the APE tables ship F32 (too short for a quant block too)
        if (leaf == "token_embd.weight" || leaf == "output.weight" || leaf.rfind("attn_", 0) == 0 ||
            leaf.rfind("indexer.", 0) == 0 || leaf.find("shexp") != std::string::npos)
            return Q8_0;                                                    // dense projections
        return F32;                                                         // mHC, biases, APE
    };
    auto add = [&](const std::string& name, const std::vector<int64_t>& shape) {
        const size_t leaf = name.compare(0, 4, "blk.") == 0 ? name.find('.', 4) + 1 : 0;
        std::vector<uint64_t> s(shape.begin(), shape.end());
        ts.push_back({name, s, type_of(name.substr(leaf)), seed++});
    };
    for (const auto& w : strata::core::dsv4_global_tensors(g)) add(w.name, w.shape);
    for (int64_t l = 0; l < g.n_layers; ++l)
        for (const auto& w : strata::core::dsv4_layer_tensors(g, l))
            add("blk." + std::to_string(l) + "." + w.name, w.shape);
    return ts;
}

// Replace every payload with plain finite values by type: F32 1.0f, BF16 0.5, Q8_0 d=2.0 x q=16, MXFP4
// all 1.0 (e2m1 code 2, e8m0 2^0), I32 the ids 0..n_expert-1 repeated (valid tid2eid routing).
void refill(const fs::path& path, const fixture::Written& w, const std::vector<fixture::Tensor>& ts) {
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    for (size_t i = 0; i < ts.size(); ++i) {
        strata::TensorInfo info;
        info.shape = ts[i].shape;
        info.type = ts[i].type;
        const uint64_t n = strata::tensor_payload_bytes(info);
        std::vector<uint8_t> b;
        switch (ts[i].type) {
            case F32:
                while (b.size() < n) { b.push_back(0x00); b.push_back(0x00); b.push_back(0x80); b.push_back(0x3F); }
                break;
            case BF16:
                while (b.size() < n) { b.push_back(0x00); b.push_back(0x3F); }
                break;
            case Q8_0:
                for (uint64_t blk = 0; blk * 34 < n; ++blk) {
                    b.push_back(0x00); b.push_back(0x40);   // fp16 2.0
                    for (int j = 0; j < 32; ++j) b.push_back(0x10);
                }
                break;
            case MXFP4:
                for (uint64_t blk = 0; blk * 17 < n; ++blk) {
                    for (int j = 0; j < 16; ++j) b.push_back(0x22);   // both nibbles = 1.0
                    b.push_back(0x7F);                                // scale 2^0
                }
                break;
            case I32: {
                int64_t e = 0;
                while (b.size() < n) {
                    const int32_t v = (int32_t) (e++ % kNE);
                    b.push_back((uint8_t) v); b.push_back(0); b.push_back(0); b.push_back(0);
                }
                break;
            }
            default: b.assign((size_t) n, 0);
        }
        b.resize((size_t) n);
        f.seekp((std::streamoff) (w.data_start + w.offsets[i]));
        f.write((const char*) b.data(), (std::streamsize) b.size());
    }
    if (!f) throw std::runtime_error("refill: cannot rewrite " + path.string());
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v)
        if (!std::isfinite(x)) return false;
    return true;
}
}   // namespace

int main() {
    TempDir d;
    const fs::path p = d.path / "tiny.gguf";
    fixture::write(p, metadata(), {});   // metadata only, to read the geometry the tables need
    strata::core::Dsv4Geometry g;
    std::string err;
    if (!strata::core::dsv4_geometry_from_gguf(strata::GgufFile(p.string()), g, err)) {
        std::printf("geometry from the tiny metadata failed: %s\nFAILED (1)\n", err.c_str());
        return 1;
    }
    const auto ts = tensors_of(g);
    const fixture::Written w = fixture::write(p, metadata(), ts);
    refill(p, w, ts);
    try {
        strata::dsv4::Ref r(p.string(), 24);
        check(r.geom().n_layers == 6 && r.pos() == 0,
              "the ref loads the file (guard, geometry, " + std::to_string(ts.size()) +
                  "-tensor layout) and starts at pos 0");
        std::vector<int> greedy;
        std::vector<float> logits_prev;
        bool finite = true, in_vocab = true, moved = false;
        int tok = 3;
        for (int i = 0; i < 12; ++i) {   // past the CSA block boundary (3), the window (4) and HCA's (7)
            const int next = r.step(tok);
            for (float x : r.logits())
                if (!std::isfinite(x)) finite = false;
            if (next < 0 || next >= (int) kVocab) in_vocab = false;
            if (!logits_prev.empty())
                for (size_t j = 0; j < r.logits().size() && !moved; ++j)
                    if (r.logits()[j] != logits_prev[j]) moved = true;
            logits_prev = r.logits();
            greedy.push_back(next);
            tok = next;
        }
        check(finite, "12 single-token steps (window ring, CSA pooling + indexer, HCA pooling, hash layers) "
                      "keep every activation finite");
        check(in_vocab, "the greedy ids stay inside the 17-token vocab");
        check(moved, "the logits change from step to step (the state, not just the weights, drives them)");
        strata::dsv4::Ref r2(p.string(), 24);
        std::vector<int> greedy2;
        tok = 3;
        for (int i = 0; i < 12; ++i) tok = r2.step(tok), greedy2.push_back(tok);
        check(greedy == greedy2, "the same model gives the same greedy ids twice (deterministic)");
        bool threw = false;
        try {
            for (int i = 0; i < 20; ++i) r2.step(1);   // ctx 24, 12 used
        } catch (const std::exception&) { threw = true; }
        check(threw, "running past --ctx refuses instead of reading past the rope tables");
    } catch (const std::exception& e) {
        check(false, std::string("the reference ran end to end (threw: ") + e.what() + ")");
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
