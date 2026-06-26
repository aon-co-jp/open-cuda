# OpenCUDA

**A clean-room, cross-vendor GPU runtime written in Rust.**
*Write your kernel once. Run it on any GPU — or none at all.*

> ⚠️ **v0.1 — design + prototype.** This is an early release showing the design and a working prototype.
> At this point, the only backend that actually runs is the **CPU backend** (no GPU required). GPU backends
> have been designed, but implementation is still ahead. We do not make exaggerated promises. Instead,
> we clearly separate "what works today" from "where the project is heading."

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-v0.1%20prototype-orange.svg)]()

---

## What is this?

OpenCUDA is a Rust runtime for running a CUDA-inspired programming model **without being locked to a single vendor**.
It aims to let the same kernel code run on NVIDIA, AMD, or Intel GPUs — and on the CPU when no GPU is available.

LLM inference is the first main battlefield. The reason is simple: most inference time is concentrated in a **small number of kernels**, mainly GEMM and Attention. By optimizing those kernels across vendors, OpenCUDA can reach "practically useful compatibility" by the shortest path.

---

## Two-layer scope — reality and ambition

This project deliberately separates the achievable near-term reality from the long-term horizon.
This is not to hide overreach, but to be **trustworthy OSS**.

### 🟢 Reality — the range we can reliably reach (Phase 1–3)

> **A Rust runtime that runs LLM inference with the same code on Windows / Linux / macOS / Android, across NVIDIA / AMD / Intel GPUs.**

That alone is already a large goal. llama.cpp has achieved something close, but the seat is still open for a **clean Rust implementation that is source-compatible with a CUDA API subset**.
If we focus on this target, even a small team — in principle, even one person — can reach Phase 1–3.

| Platform | Positioning |
|---|---|
| Linux / Windows / macOS | First-class support (low-level GPU access is flexible) |
| Android | Planned support (JIT restrictions must be handled carefully) |

The primary use case is **LLM inference**. Compatibility is described honestly as **"compatibility with the CUDA API subset required for LLM inference"**, not "full CUDA compatibility."

### 🔭 Ambition — the final horizon (long-term vision)

Even if it costs some short-term credibility, the destination should be written honestly. The author is looking beyond the ordinary extension of current tools — not as a mad scientist, but as a staircase that can be climbed step by step.

Ultimately, OpenCUDA aims to become a universal runtime that can run **any vendor's GPU on any compute device with one codebase**: desktop platforms (Windows / macOS / Linux), mobile platforms (Android / iOS), and someday — if formal agreements are reached with platform holders — even more closed environments.

However, we also state honestly that this ambition has **walls that cannot be crossed immediately**.

- **iOS**: GPU APIs other than Metal are effectively unavailable, and JIT is prohibited by App Store rules. Support would have to be limited to AOT-compiled kernels.
- **Game consoles / some smart TVs**: These are closed platforms, and low-level GPU access requires developer agreements with each company. This is an authorization issue, not merely a technical issue. They are out of scope for now.
- **CAD and general graphics workloads**: Their requirements are the opposite of LLM workloads centered on GEMM/Attention. Supporting them would be close to reimplementing an entire graphics driver stack. They are outside the current scope.

The ambition remains. But **the README promises only the 🟢 scope at the top**.

---

## Run it now (CPU backend, no GPU required)

```bash
# Vector addition C = A + B (1 million elements)
cargo run --release -p vector_add

# Matrix multiplication C = A·B
cargo run --release -p matmul
```

Expected output (`vector_add`):

```
device: OpenCUDA CPU Device (rayon, 32 threads)
OK: all 1000000 elements equal 1000000
c[0]=1000000, c[999999]=1000000
```

Even without a single GPU, kernels run on a 16-core / 32-thread CPU using `rayon` multi-threading.
This is the core of Phase 1: **validate the design on real hardware before buying more hardware**.

---

## Architecture

```
        ┌──────────────────────────────────────────────┐
        │  User kernels (Native / SPIR-V / PTX / OmniIR)│
        └──────────────────────────────────────────────┘
                              │
        ┌──────────────────────────────────────────────┐
        │  opencuda-core   ← backbone (GpuDevice trait) │
        │   ・two-layer memory (DevicePtr + DeviceBuffer)│
        │   ・KernelSource enum                          │
        │   ・DeviceRegistry                             │
        └──────────────────────────────────────────────┘
                              │
   ┌──────────┬──────────┬──────────┬──────────┬──────────┐
   │ cpu ✅   │ vulkan ⏳ │ nvidia ⏳ │ amd ⏳   │ intel ⏳ │
   │ (rayon)  │ (SPIR-V) │ (CUDA)   │ (ROCm)   │ (oneAPI) │
   └──────────┴──────────┴──────────┴──────────┴──────────┘
                              │
        ┌──────────────────────────────────────────────┐
        │  opencuda-blas      GEMM / Attention / Quant  │  ⏳ Phase 3
        │  opencuda-multidev  Pipeline parallel / VRAM  │  ⏳ Phase 3
        └──────────────────────────────────────────────┘
```

✅ = works in v0.1　⏳ = designed, implementation planned

### Fixed design decisions (Phase 1)

1. **Memory has two layers**. Internally, raw pointers use `DevicePtr` (embedding `device_id` to prevent mix-ups in multi-GPU environments at the type level). The native API uses RAII-based `DeviceBuffer` (holding `Arc<dyn GpuDevice>` and freeing automatically on Drop).
2. **Errors are based on `anyhow` + a minimal `GpuError` enum**. Everyday code stays easy to write, while the enum captures representative failures that the CUDA compatibility layer may need to convert into integer error codes.
3. **Kernels are stored in multiple formats through the `KernelSource` enum**. v0.1 implements `Native` (CPU). `SpirV` / `Ptx` / `OmniIr` are planned. Because this is an enum, formats can be added later without breaking the design.
4. **The CPU backend is the first execution target**. The design can be validated without a GPU.

---

## Workspace layout

```
opencuda/
├── crates/
│   ├── opencuda-core/       backbone (traits, memory, kernel representation) ✅
│   ├── opencuda-cpu/        CPU backend (rayon)                              ✅
│   ├── opencuda-blas/       AI kernels (GEMM / Attention / quantization)     ⏳
│   └── opencuda-multidev/   multi-GPU partitioning and pipeline parallelism  ⏳
└── examples/
    ├── vector_add/          C = A + B                                        ✅
    └── matmul/              C = A·B                                          ✅
```

---

## Roadmap

- **Phase 1 (foundation)** — run on CPU first, then reach all GPUs through Vulkan
  - [x] `opencuda-core`: traits, two-layer memory, `KernelSource`, `DeviceRegistry`
  - [x] `opencuda-cpu`: rayon multi-threaded execution
  - [x] `examples/vector_add`, `examples/matmul` (running on CPU)
  - [ ] `opencuda-vulkan`: run SPIR-V kernels through Vulkan Compute
  - [ ] Confirm that the same binary runs on NVIDIA / AMD / Intel GPUs
- **Phase 2 (CUDA compatibility)**
  - [ ] OmniIR (common intermediate representation) and SPIR-V output
  - [ ] CUDA C++ subset parser → OmniIR
  - [ ] Hook layer for major CUDA APIs (malloc/memcpy/free/launch)
  - [ ] NVIDIA / AMD backends
- **Phase 3 (AI optimization)**
  - [ ] `opencuda-blas`: GEMM / Flash Attention / quantization
  - [ ] `opencuda-multidev`: pipeline parallelism and unified VRAM handling
  - [ ] Run LLM inference across two GPUs
- **Phase 4 (extension)** — Intel oneAPI, nvcc-compatible driver, PyTorch backend

---

## Honest estimate

Achieving "true full CUDA compatibility" in every direction is work at the scale of a GPU vendor (5–10 years for one person). The goal of each phase in this roadmap is **not "full compatibility," but "a practical subset required for LLM inference that works well enough to be useful."**
If we confuse the two and promise "full compatibility," we betray users and lose the most important asset of OSS: trust. Therefore, we say what can be done, and we say what cannot be done.

---

## Why the name "OpenCUDA"?

The name directly expresses the intent: an open implementation inspired by CUDA.
`CUDA` is a trademark of NVIDIA. If trademark concerns arise, the project is prepared to change its name to **iLumi** (derived from "light").

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

Like the aruaru ecosystem (`aruaru-DB`, `aruaru-llm`), OpenCUDA will be released under Apache-2.0 and positioned as its GPU compute foundation.

---

## Contributing

v0.1 is at the design and prototype stage. Feedback on the design, improvements to the CPU backend, and work on the Vulkan backend (the next step in Phase 1) are welcome.

# open-cuda
