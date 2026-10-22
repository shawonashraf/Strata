# DeepSeek-V4-Flash as a third model family (`deepseek4`) — the plan

**Status: proposal for scope agreement. Nothing in the engine has been changed.**
Everything about Strata's code below comes from reading this repository at v0.2.2 (engine 0.1.47); where it
describes behavior that was not executed, treat it as **unconfirmed static analysis**. Everything about the
DeepSeek file comes from the Hugging Face model card and a community-posted llama.cpp load log
([llama.cpp#22319](https://github.com/ggml-org/llama.cpp/issues/22319)); it is a third-party report, not our
measurement. Sizes we computed ourselves are marked *estimated* and need a real file header to confirm.

## The user problem

Strata runs Qwen3.8-Flash-Next (and Qwen3.6-35B-A3B) on a normal PC by keeping experts out of VRAM: a pinned RAM
arena, an SSD file tier, and a VRAM slot tier ranked by a routing profile. People with 16 GB cards, ~96 GB RAM and
an NVMe drive ask for the same treatment for DeepSeek-V4-Flash (284B total, 13B active, MIT). Today `strata`
refuses its files. The workaround is llama-server as a Custom model API; it serves the model but has no expert
tier, and per community reports its V4 MoE path is currently slow (see "Outside reference" below).

## What the code does today

- `include/strata/artifact/model_family.hpp` knows `qwen4exp` and `qwen35moe` only; a `deepseek4` GGUF resolves to
  `Unknown` and the reader's guard refuses it (`gguf_reader.hpp`, `check_architecture`).
- The main engine path compiles the shape in (`core/geometry.hpp`: `n_embd=2560`, 48 layers, 512 experts top-10,
  expert FF 640) and `compiled_geometry_mismatch` refuses anything else. The kernels `static_assert` the same.
- The expert offload assumes one fixed blob: `H=2560`, `FF=640`, `BLOB=1,382,400` bytes
  (`core/expert_source.hpp`), 48 × 512 = 24,576 ranked pairs in `profile.bin`, a top-10 router kernel.
- The KV cache (`core/layer.hpp`, `QsaState`) is a paged 2-KV-head × 256-dim pool with INT8/Q4_0 quant paths. V4's
  attention (below) has no representation in it.
- MXFP4 (GGUF type 39) is in the reader's type table (`gguf_reader.hpp`) but appears in **no kernel**: QUANTS.md
  lists no coverage. If it runs at all today it is through ggml-cpu's native expert dots, which is untested for
  this family.
- The precedent to follow is Qwen3.6-35B-A3B: its own engine path (`src/qwen35/q36_engine.cu`, chosen at the top
  of `main` in `src/program/generate.cpp`, "THE FAMILY SWITCH"), its reference file (`src/qwen35/q36ref.cu`,
  exactness first, checked against llama.cpp with `tools/qwen35/parity.py` and `kl.py`), and its packaging entry
  (`setup.py`, `OWN_PATH_FAMILIES["qwen36"]`).

## What the file looks like (verified from GA headers by HTTP range reads, Oct 2026)

Read with `tools/dsv4_fetch_headers.py` (range reads only, ~5 MB) from `unsloth/DeepSeek-V4-Flash-0731-GGUF`
UD-Q4_K_XL (5 shards, 1,328 tensors, 144.44 GiB of declared payload) and the older `nsparks` preview file:

- `general.architecture = deepseek4`; 43 layers, embedding 4096, 64 heads, **1 KV head** (shared K=V latent, 512
  wide, `attn_kv.weight`), `q_lora_rank=1024` (`attn_q_a` + `attn_q_b [1024, 64×512]`), grouped output projection
  (`output_group_count=8`, `output_lora_rank=1024`: `attn_output_a [4096, 8192]` + `attn_output_b`), per-head
  attention sinks (`attn_sinks.weight [64]`), partial RoPE `rope.dimension_count=64`, `sliding_window=128`,
  YaRN 16× over 65,536.
- The per-layer attention schedule **is in the file**: `attention.compress_ratios` has **46 rows for the 43-layer
  trunk** — `[0, 0, 4, 128, 4, 128, …, 4]` plus three trailing zeros for the DSpark speculative layers, which
  ship as a separate `dspark-*.gguf`. So: 2 sliding bootstrap layers, then CSA (4×) and HCA (128×) alternating —
  21 CSA, 20 HCA. CSA layers carry `attn_compressor_*` (kv/gate `[4096, 2×512]`, norm 512, ape `[1024, 4]`) plus
  the Lightning indexer (`indexer.attn_q_b [1024, 64×128]`, `indexer.proj [4096, 64]` — one shared 64-wide key
  per token) and its own pool (`indexer_compressor_*`, ape `[256, 4]`); HCA layers carry the same compressor at
  `[4096, 512]`, ape `[512, 128]`, no indexer.
- MoE: 256 routed experts + 1 shared, **top-6** (`expert_used_count=6`), expert FF 2048,
  `expert_weights_scale=1.5`, `hyper_connection.*` (count 4, sinkhorn_iterations 20, epsilon), per-layer
  `swiglu_clamp_exp`/`swiglu_clamp_shexp` arrays, `hash_layer_count=3`. The three hash layers carry
  `ffn_gate_tid2eid.weight` I32 `[6, 129280]` (the frozen token-id → expert-id table, 6 = top-k) and **no**
  `exp_probs_b.bias`; the other 40 layers have the bias and no table.
- mHC tensors per layer: `hc_attn_base [24]`, `hc_attn_fn [16384, 24]`, `hc_attn_scale [3]` (and the ffn twins);
  24 = hc_mult × (hc_mult + 2), 16384 = hc_mult × 4096. Globals: `token_embd`, `output`, `output_norm` and the
  HyperHead triple `output_hc_{base,fn,scale}`. Table total: 23 base + 1 routing per layer, +10 CSA, +4 HCA, +6
  global = **1,328 — exactly what the GA file carries** (checked by `model_family_test --real` on the fetched
  headers: guard ok, geometry ok, layout ok).
- Tensor **types** vary by repack: UD-Q4_K_XL stores the experts as MXFP4 (bit-exact, blades
  `ffn_{gate,up,down}_exps.weight` `[4096, 2048, 256]` / `[2048, 4096, 256]` — per-expert stride 4,456,448 B per
  blade, constant) and the rest as Q8_0/BF16. **No published file uses `f8_e4m3_b128`**: Unsloth dequantizes the
  official FP8 tensors (147 BF16 + 365 FP8 in the source) without rounding loss, and `nsparks`'s
  "FP4-FP8-native.gguf" turned out to be a preview with those 365 tensors re-quantized to Q2_0 and the older
  preview tensor names (`attn_kv_latent`, `attn_compress_*`). Stage 2 needs no new GGML block type for published
  files; if an official native repack lands later, its reader work is a small addition.

Sizes computed from the file directory itself (arithmetic on declared shapes and block formats, not measured):

| Quantity | Flash-Next today | V4-Flash (from the GA directory) |
|---|---:|---:|
| one routed expert, native MXFP4 | 1,382,400 B | 13,369,344 B (3 blades × 8,388,608 elements × 17/32 B) |
| expert pairs to rank | 48 × 512 = 24,576 | 43 × 256 = 11,008 |
| expert bytes read per decode token | ≈ 663 MB (10 × 48) | ≈ 3.4 GB routed (6 × 13.37 MB × 43) + the shared expert and dense weights |
| KV/cache bytes per context token | 576 B × 13 QSA layers (Q4_0) | order 160 B × ≈ 40 layers (window ring + CSA + HCA + indexer pools), TBC at Stage 2 |

The per-token read is ~5× Flash-Next's. The KV per token is a wash despite the 1M context, because V4 keeps only
a 128-token window plus compressed pools.

## Proposed design

One new family, one new engine path, exactly the shape `src/qwen35/` took. Five stages; each is its own PR per
CONTRIBUTING.md. Everything is opt-in; the Flash-Next default path stays byte-identical.

**Stage 1 — know the file, refuse it cleanly (no GPU needed). DONE on branch `dsv4/stage1-recognize`.**
`DeepseekV4` is in `model_family.hpp` (arch string `deepseek4`); the engine refuses it with: recognised, no
engine path yet, use the Custom model API (llama-server) meanwhile. `check_architecture_deepseek4`
(`gguf_reader.hpp`) proves the file declares the published shape (keys, invariants, the per-layer schedule);
`core::Dsv4Geometry` + the layout tables live in `family_layout.hpp` beside the Qwen tables. `ModelGeometry`
(Flash-Next's kernel contract) is untouched. How it was checked: `tests/core/model_family_test.cpp` (62 checks,
synthetic tiny V4 files) and `model_family_test --real` over the fetched GA UD-Q4_K_XL headers (sparse
replicas made by `tools/dsv4_fetch_headers.py`): architecture guard ok, geometry reads exactly (43 = 2 sliding +
21 CSA + 20 HCA, 3 hash-routed), layout ok with 1,328/1,328 tensors named. Not run: any GPU path (there is
none yet), the 5-GPU llama.cpp comparison from the outside reports. Files: `model_family.hpp`,
`gguf_reader.hpp`, `family_layout.hpp`, `tests/core/model_family_test.cpp`, `strata_control/models.py` (a named
suggestion instead of `unsupported_or_missing_architecture`), `tools/gguf_reader.py` (I32 block geometry),
`tools/dsv4_fetch_headers.py` (new). `setup.py` unchanged: the family is recognized, not offered.

**Stage 2 — the reference path, exactness first (`src/deepseekv4/dsv4ref.cu`).**
Copy of `q36ref.cu`: one GPU, whole model resident (a ≥160 GB RAM machine or a large unified-memory Mac), one
token at a time, fp32 activations, every weight read in its GGUF form (MXFP4, Q8_0/BF16 as published; a native
FP8 repack would add its block type later). `--kernels
plain|strata`. Checked against llama.cpp the way q36 was: per-layer hidden states, greedy ids, teacher-forced KL
(`tools/dsv4/parity.py`, `kl.py`, a llama-side dump). This stage forces all the numerics down at once —
sqrt-softplus router scores with the correction bias and the 1.5 scale, the `tid2eid` hash layers, clamped
SwiGLU, mHC with Sinkhorn, sliding-window K=V attention plus the CSA/HCA pools and the indexer, the grouped
output projection, the per-head attention sink, interleaved partial RoPE — with plain kernels, so every piece has
a reference before anything is fast.

**Stage 3 — the tiers (this is why the family is in Strata at all).**
V4's blades are already per-expert contiguous, so the GGUF can be read in place: a fixed byte stride per expert
per blade. Reuse `FileExpertSource` (SSD tier: mmap or unbuffered direct reads at fixed offsets), the pinned
arena (`ArenaExpertSource`), the VRAM slot tier (`expert_cache.cpp`), and `profile.bin` rebuilt over the 11,008
pairs (`tools/dsv4/make_profile.py` from `collect_profile_traces.py`). No pack step: unlike Flash-Next's
`tools/pack_layer.py`, the GGUF *is* the arena-sized file here. Files: `expert_source.*` (a second blob
geometry), `expert_cache.cpp`, `session.cpp`, `setup.py` (`OWN_PATH_FAMILIES["deepseek"]` with pinned file
sizes/SHA-256s, `ram_gb`, `kv_bytes_per_token` from Stage 2's measured cache footprint).

**Stage 4 — the speed kernels (CUDA, then HIP).**
MXFP4 decode GEMV and grouped-expert kernels on V4's shape (H 4096, FF 2048) — the same family of kernels as
`s2_gemv`/`native_moe` on Q2_0 and the MMQ IQ path; plus a CPU pool kernel for the hybrid mode. Router top-6
(sqrt-softplus + bias). mHC stream mixing. The attention: the sliding-window K=V ring plus gathered CSA/HCA pool
reads is close in shape to what `qsa_select`/`qsa_prompt_attn` already do for QSA (pooled keys, top-k gather),
with the CSA indexer (top-512 over the pool) in place of QSA's (top-2048 over cells). Per CONTRIBUTING, HIP must
build and validate before review; SYCL keeps refusing the family (as it does `qwen35moe`) and says so.

**Stage 5 — the server and the Terminal.**
V4's chat template (official GGUF ships a ChatML-style `<｜...｜>` template with `enable_thinking` /
`reasoning_effort`), its own tool-call format (TBC from the template — not Qwen's XML), EOS/BOS ids from the
file, `--context` mapped onto the YaRN settings. Then the setup menu entry, `strata launch` (family inference by
shard names and header), the VRAM-fit tables, and docs (MODELS.md, a DEEPSEEK_V4.md in today's plain style).

## Memory fit on the reference PC (RTX 5080 16 GB + 93 GB RAM + NVMe)

*Estimated arithmetic, not a measurement.* The native file (145 GiB) exceeds RAM: Stage 2 does not run here at
all; it runs on a 160 GB+ machine. Stage 3 is the path for this PC: roughly 75–80 GB of experts pinned in the RAM
arena, the rest served from the NVMe tier, a few GB of hot experts in VRAM slots. With about half the expert
reads from RAM, the ~4 GB/token of expert traffic projects to low single-digit tokens/s on this class of machine.
That projection is an estimate; no claim is made until it is measured here against the same-day llama-server run.

## Outside reference (third-party, not measured by us)

Community reports in llama.cpp#22319 (April 2026, WIP branch): 15.2 t/s full-precision native GGUF on dual
RTX 3090 with layer split; ~4 t/s fully VRAM-resident on three GPUs (171 GB); ~6 t/s Q2_K on RTX 5090 + 96 GB
DDR5. Reports also say the FP4 tensor-core path and the MoE placement flags were not effective for V4 at that
revision. Unsloth's UD-IQ3_XXS is 103 GB with 110 GB RAM recommended. These numbers are quoted to size the
problem, not as evidence for Strata.

## Out of scope (proposed)

DeepSeek-V4-Pro (1.6T); V4.1-Flash and its Engram modules; the Vision-Exp variant; DSpark speculative decoding
(the MTP path in `core/mtp.hpp` is Flash-Next-specific); the SYCL port; any change to defaults. V4.1's Engram
(196B of hashed N-gram lookups) is the best future fit for the SSD tier, but a later proposal.

## Open questions for maintainers

1. Is a third family in scope at all, or does Strata stay Qwen-only with llama-server as the custom route?
2. If in scope: is the `src/qwen35/` shape (own engine path, own reference file, `OWN_PATH_FAMILIES` packaging)
   the agreed pattern, and is `strata-dsv4` the right binary name?
3. MXFP4 compute: native Strata kernels, or reuse ggml-cpu's native expert dots for CPU/hybrid and require the
   GPU kernels before the family ships?
4. Which published GGUF to pin first — Unsloth's UD-Q4_K_XL (155.1 GB: experts bit-exact MXFP4, the official
   FP8 tensors kept with no rounding loss as Q8_0), UD-Q8_K_XL (161.9 GB, fully lossless), or UD-IQ3_XXS
   (103 GB, re-quantized, the only one that fits 96 GB RAM)? There is no official native FP8 repack to pin
   (none published uses the FP8 block type as of the Oct 2026 header reads). The first Stage-2 PR needs one.
5. ~~The GA header may lack the schedule keys.~~ Resolved by the header reads: `attention.compress_ratios`
   carries the whole schedule (46 rows: 43 trunk + 3 sliding DSpark rows), and the GA layout table is fixed by
   the file itself (1,328 tensors, above).
