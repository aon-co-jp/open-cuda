# open-cuda

> ## 🎯🕒 aruaru-db × RPoem SET policy + "current-state-only" caveat
> (2026-08-29, cross-repo review, pinned to the very top per user
> instruction)
>
> **Canonical source: aruaru-db/CLAUDE.md's opening "most important"
> note.** aruaru-db only delivers "no REST API needed, compatible with
> WunderGraph Cosmo's paid Enterprise tier" when paired (SET) with
> RPoem — and warns against building REST replacements aimlessly.
>
> **This repository is out of scope for now**: re-checked and
> confirmed to have no REST API / API-key mentions anywhere in its
> `CLAUDE.md` — it is a GPU compute library with no HTTP server
> surface, so this SET policy currently has no applicable target here.
> See aruaru-db/CLAUDE.md's 2026-08-29 HANDOFF for the investigation.
>
> **⚠️ But this "out of scope" is a provisional call for right now, not
> a permanent conclusion (user's own caveat, 2026-08-29)**: as
> open-directx's DirectX-compatibility implementation and development
> progress, the expectation is that **more scenarios will emerge where
> issuing commands directly at the OS level, or having the app run via
> hardware accelerators (open-directx/open-cpu), becomes the bigger
> win**. If that happens, ties to open-cuda, aruaru-llm, aruaru-db,
> open-web-server, and RPoem will genuinely deepen, and this SET
> policy's scope (REST-free, Cosmo-Enterprise-compatible) should be
> re-evaluated at that point. Re-ask "is this still out of scope?"
> each time development advances.

*English*: [README-English.md](README-English.md) ·
*Other languages*: [Deutsch](README-German.md) · [Italiano](README-Italian.md) ·
[Français](README-French.md) · [Русский](README-Russian.md) ·
[Українська](README-Ukrainian.md) · [עברית](README-Hebrew.md) ·
[فارسی](README-Persian.md) · [العربية](README-Arabic.md)

> **Recent update (2026-09-03)**: Per user instruction to target
> 32GB-VRAM-class NVIDIA/AMD/Intel GPUs going forward and support
> F16/F32/F64/F128, added `F16`/`F64`/`F128` variants to
> `opencuda-core::KernelArg`/`ResolvedArg` and `hgemm`/`dgemm`/`qgemm`
> CPU reference GEMMs in `opencuda-blas`. **F128 is a from-scratch
> software double-double type** (`DoubleDouble`) — no NVIDIA/AMD/Intel
> GPU has native FP128 hardware, so this exists for type-system
> consistency and numerical-accuracy use cases only. This dev machine
> only has a GT 730 (2GB, Kepler), so no 32GB-VRAM-class multi-vendor
> hardware verification was possible. See [CLAUDE.md](CLAUDE.md) /
> [OmniGPU-Design.md](OmniGPU-Design.md) §13.
>
> **Recent update (2026-08-20)**: Implemented FlexQ-style INT6
> quantization (`quantize_int6`/`dequantize_int6`/`QuantizedInt6Tensor`
> in `opencuda-blas`, packing 4 6-bit values into 3 bytes). PuzzleMoE
> was declined after checking its prerequisites — it requires an
> existing MoE architecture that this repo's dense GPT-2/BERT models
> don't have. Also corrected an earlier inaccurate note claiming
> `open-directx` and `open-cuda` had an "in-design link" — in reality
> there are just two unrelated same-named things: the in-repo
> `opencuda-directx` crate and the independent `aon-co-jp/open-directx`
> repository. Separately, investigated and declined adding a
> self-update mechanism (`open-english`'s `self_update.rs` pattern) —
> this repo has no resident service (10 library crates + 12 throwaway
> example binaries), and confirmed real Microsoft DirectX/NVIDIA CUDA
> are themselves runtime-library architectures rather than resident
> background services, so the current design is not a defect. See the
> 2026-08-19/2026-08-20 HANDOFF entries in [CLAUDE.md](CLAUDE.md).

> **Recent update (2026-08-10)**: Added `generate_with_repetition_penalty`
> (CTRL-style repetition penalty — penalty>1.0 weakens the logits of tokens
> already seen) to `open-cuda-llm::GptModel`. The existing `generate()` is
> now a thin wrapper calling it with `penalty=1.0` (byte-identical
> behavior, no regression to existing tests/callers). This directly
> addresses a reported `aruaru-llm` bug where the base (non-fine-tuned)
> GPT-2's greedy decoding loops the same string forever (e.g.
> "Student: Hello"). Added a new test on real GPT-2 124M weights,
> `repetition_penalty_reduces_degenerate_loop_on_real_gpt2_weights`,
> confirming the loop actually reproduces without the penalty and actually
> stops (producing grammatically natural text) at `penalty=1.3`.
> `aruaru-llm`'s `/v1/generate` now calls this new API with `penalty=1.3`
> by default (override via `ARUARU_LLM_REPETITION_PENALTY`). See the
> 2026-08-10 HANDOFF entry in [CLAUDE.md](CLAUDE.md).

> **Recent update (2026-08-08)**: Re-verified the 2026-08-06 "MLA
> implemented" note on real hardware
> (`cargo test -p opencuda-blas mla -- --nocapture` → `1 passed; 0
> failed`, exercised the GT730's real Vulkan path). Investigated FP8
> mixed-precision and DeepSeekMoE but **declined to implement either**:
> FP8 has no real hardware support on this machine's only GPU (GT730,
> Kepler, CC 3.5 — no FP8 Tensor Cores, would be software emulation
> only); DeepSeekMoE has no genuine integration point since
> `open-cuda-llm`'s `DecoderLayer` has only a single dense FFN (no
> expert/router structure) and no real MoE checkpoint exists. See the
> 2026-08-08 HANDOFF entry in [CLAUDE.md](CLAUDE.md) for details.

> **Recent update (2026-08-07)**: Wired the fused flash-attention kernel
> (`flash_attention_with_spirv`) into `open-cuda-llm`'s `DecoderLayer`,
> via `GptModel::set_flash_attention_spirv()` — falls back to the
> existing 3-dispatch path when unset (fully backward compatible).
> Verified on real NVIDIA GT 730 hardware: Vulkan-generated token
> sequences are byte-identical to the CPU path. See [CLAUDE.md](CLAUDE.md)
> HANDOFF for details.

> **Implemented 2026-08-06**: low-rank KV-cache compression inspired by
> DeepSeek-V3's Multi-Head Latent Attention (MLA) —
> `opencuda-blas::mla_compress_kv`/`mla_decompress_kv`. Researched the
> technical report ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437))
> and implementation blogs in Japanese and English, then built the
> low-rank projection mechanism (the design behind DeepSeek's reported
> 93.3% KV-cache reduction) on top of the existing, real-hardware-verified
> `sgemm` backend. Verified on real hardware (GT730) that the CPU and
> Vulkan paths match numerically. Does not carry trained weights (this
> demonstrates the mechanism, not the trained compression quality — see
> the HANDOFF section of `CLAUDE.md` for the honest disclosure). Applying
> Toshiba SBM / DeepSeek techniques to the other 7 repos is still under
> consideration.

> **Updated 2026-07-25**: The dev-policy file (`CLAUDE.md`) heading was
> renamed from "Development Policy & Dev Environment Rules" to "Design
> Philosophy & Development Policy & Dev Environment Rules", to more
> clearly separate the project's design philosophy (what we value),
> development policy (how we work), and dev environment rules (concrete
> operational conventions). See `CLAUDE.md` for details.

**Development started: 2026-06-26** (this repo's GitHub creation date)

"The second CUDA" — a GPU abstraction/compute foundation (the `OmniGPU`
design) aiming for Windows/macOS/Linux compatibility and
Intel/AMD/NVIDIA compatibility. Paired ("SET") with `aruaru-llm`, which
is the implementation consumer of the GPU/CPU execution pipeline.

## What this is

- **`opencuda-core`**: The `GpuDevice` trait shared by all backends
  (CUDA Runtime API equivalent: `alloc`/`memcpy`/`launch_kernel`).
- **`opencuda-cpu`**: The CPU backend (data parallelism via `rayon`).
- **`opencuda-vulkan`**: The Vulkan Compute backend (cross-platform,
  native on Windows/Linux/Android, macOS/iOS via MoltenVK). GEMM /
  Attention / INT4·INT8 quantization verified on real Vulkan execution.
- **`opencuda-directx`** (added 2026-07-23): A DirectX 12 Compute
  backend (Windows-only, an opt-in backend that coexists with Vulkan).
  GPU dispatch of `vector_add`/`matmul`/`ChaCha20` verified on real
  hardware (NVIDIA GT 730) — output matches CPU reference
  implementations (e.g. the RustCrypto `chacha20` crate) exactly in
  tests. Real vendor name/VRAM capacity retrieval via DXGI adapter
  enumeration is also implemented.
- **`opencuda-blas`**: NumPy equivalent (GEMM/Attention/quantization).
- **`open-cuda-bert`**: Forward pass for BERT-family encoders (supports
  multilingual-e5-small).
- **`open-cuda-llm`**: vLLM equivalent (greedy decoding with KV cache).
  Implements `GptModel::load`, which loads GPT-2 (Hugging Face
  `openai-community/gpt2`) `safetensors` (2026-07-25, same design as
  `open-cuda-bert::BertModel::load`). Verified on real hardware:
  downloading and loading real GPT-2 124M weights produces clearly more
  fluent greedy-decoded English than random initialization (meaningless
  output) — e.g. "The quick brown fox" → "es are a great way to get a
  little bit of a". See the `CLAUDE.md` HANDOFF for details.
- **`open-cuda-whisper`** (added 2026-07-31): Whisper equivalent
  (speech recognition, #6 on the market-research roadmap).
  Log-mel-spectrogram extraction + encoder (same Multi-Head Attention
  design as `open-cuda-bert`) + KV-cached decoder (same design as
  `open-cuda-llm`) + cross-attention. **Currently a random-init MVP
  only** (no trained Whisper weight loader yet — see `CLAUDE.md`
  HANDOFF for details).

## Why we have both DirectX and Vulkan (2026-07-23 technical decision)

Initially, this project was thought to be "under development as a
DirectX plugin." After backing that up with Japanese/English web
research, it turned out DXVK/vkd3d-proton (the technology Valve's Proton
actually uses) both convert in the direction "DirectX (Windows-only
API) → Vulkan (cross-platform API)" — no examples of the reverse
direction were found. **For the goal of cross-platform support, the
existing Vulkan Compute approach is technically the more direct path.**
Based on that, we adopted the policy of "keep Vulkan, and additionally
add DirectX for Windows, coexisting" and implemented `opencuda-directx`.

## Honest disclosure

- **Cross-platform support is a work in progress**: Vulkan Compute is
  designed for native support on Windows/Linux/Android and MoltenVK-based
  support on macOS/iOS, but real-hardware verification has only been
  done on this machine (Windows, NVIDIA GT 730).
- **`opencuda-directx`'s kernel dispatch is only partially Phase 2**:
  `vector_add`, `matmul`, and `ChaCha20` (encryption only — does not
  include Poly1305 authentication tag computation) are implemented.
  Vendor detection via DXGI adapter enumeration (`GpuVendor::Nvidia`
  etc.) is also implemented, but detailed info like
  `compute_capability` remains a placeholder since it can't be
  obtained from DXGI.
- **The real-world benefit of GPU compression/encryption is
  unverified**: for a payload as small as one tunnel frame, there's a
  technical concern that host↔device transfer overhead may cancel out
  the GPU's compute advantage (real benchmarking remains a future
  task).

## Relationships within this ecosystem

- [aruaru-llm](https://github.com/aon-co-jp/aruaru-llm) — a reference
  implementation of this repo (consumer of the GPU/CPU execution
  pipeline).
- [RS-LinkFusion](https://github.com/aon-co-jp/RS-LinkFusion) — evaluating
  use of GPU compression/encryption acceleration (the ChaCha20 kernel in
  `opencuda-directx`).
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — the
  canonical source for dev-policy rules.

## Build & test

```bash
cargo build --workspace
cargo test --workspace

# DirectX 12 real-hardware tests (Windows-only, real-dx12 feature)
cargo test -p opencuda-directx --features real-dx12
```

### Try it on your own GPU (added 2026-07-27)

If you just want to run one thing first to check it works, each
sub-crate under `examples/` (a workspace member) can be run with
`cargo run -p <name>`. `vulkan_info` in particular is a minimal example
that just enumerates and prints the real Vulkan physical devices (GPU
vendor name, VRAM capacity) on your machine, making it the best first
command to check whether a GPU is detected in your environment:

```bash
cargo run -p vulkan_info
```

Other examples (`matmul`, `matmul_vulkan_real`, `vector_add`,
`vector_add_vulkan`, `vector_add_vulkan_real`, `vector_add_omniir`) can
likewise be run with `cargo run -p <name>`. See `OmniGPU-Design.md`
§8.5 for the per-vendor (Intel/AMD/NVIDIA etc.) support-status matrix.

## License

Apache-2.0


---

## Update 2026-08-23 — CPU feature detection unified into `open-cpu`, two dispatch bugs fixed

`opencuda-blas`'s `simd.rs` used to call `is_x86_feature_detected!`
itself. Detection is now delegated to the shared
[`open-cpu`](https://github.com/aon-co-jp/open-cpu) crate (path
dependency). The `CpuFeatures` struct keeps its existing fields, so callers
are unchanged.

**Two real bugs were found and fixed during this work. Both are the same
mistake: branching on a single feature flag where the code actually
requires a combination of features.**

1. **The AVX-512 path was reachable without opt-in.** `dot_f32` and `axpy`
   entered the 512-bit path on `if f.avx512f` alone, meaning
   **hardware-unverified code would run automatically on any AVX-512
   machine**. A new `avx512_f32_path()` now requires AVX-512F+BW+VL *and*
   `OPEN_CPU_ENABLE_AVX512=1`.
2. **The int8 VNNI branch did not match its own `target_feature`
   declaration.** `dot_i8_avx512vnni` is declared
   `#[target_feature(enable = "avx512vnni,avx512bw,avx512f")]` but the
   caller only checked `f.avx512vnni`. Both it and `dot_i8_avxvnni`
   (`avxvnni,avx2`) now use full combination checks.

Also added: an `avx512bw` field, plus `isa_profile()`, `has_avx2_fma()`,
`has_vnni_path()` and `cpu_runtime_line()` for logging and APIs.

**Verification**: `cargo test -p opencuda-blas --release` — **34 tests
pass**, including scalar-equivalence tests. On the development machine
(Ryzen 9 3950X, Zen 2) the profile is `avx2+fma3`. **AVX-512 and VNNI
paths remain unverified on real hardware.**

Weight repacking (llama.cpp-style online repack), GFNI and AMX were
surveyed and judged excessive for this repository's scale; the full survey
with source links is recorded in `open-cpu/CLAUDE.md` (2026-08-23 HANDOFF).
