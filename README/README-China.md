# OpenCUDA

[British English](README.en-GB.md) | [English](README.en.md) | [Deutsch](README.de.md) | [Italiano](README.it.md) | [Français](README.fr.md) | [Русский](README.ru.md) | [Українська](README.uk.md) | [العربية](README.ar.md) | [فارسی](README.fa.md) | [简体中文](README.zh-CN.md) | [한국어](README.ko.md) | [繁體中文 / 台灣](README.zh-TW.md)

**用 Rust 编写的 clean-room、跨厂商 GPU 运行时。**
*内核只写一次，即可在任何 GPU 上运行——也可以在没有 GPU 的情况下运行。*

> ⚠️ **v0.1 — 设计 + 原型。** 这是一个早期版本，用于展示设计和可运行原型。目前真正可运行的只有 **CPU 后端**；不需要 GPU。GPU 后端已经完成设计，但实现仍在进行中。我们不做夸大承诺，而是清楚区分今天能运行的内容和项目未来目标。

## OpenCUDA 是什么？

OpenCUDA 是一个 Rust 运行时，用来执行受 CUDA 启发的编程模型，**但不绑定到单一厂商**。目标是让同一份 kernel 代码运行在 NVIDIA、AMD、Intel GPU 上；如果没有 GPU，也可以运行在 CPU 上。

第一个重点场景是 LLM 推理，因为大部分推理时间集中在少数 kernel 上，主要是 GEMM 和 Attention。只要跨厂商优化这些 kernel，OpenCUDA 就能以最短路径达到实用级兼容性。

## 双层范围 — 现实与野心

项目有意区分短期可达成的现实目标和长期愿景，这是为了成为可信赖的 OSS。

### 🟢 现实 — Phase 1–3

> **一个 Rust 运行时，可在 Windows / Linux / macOS / Android 上，用同一份代码在 NVIDIA / AMD / Intel GPU 上运行 LLM 推理。**

这已经是很大的目标。llama.cpp 做到了接近的事情，但一个 clean 的 Rust 实现、并与 CUDA API 子集保持源码兼容，仍然是空位。

| 平台 | 定位 |
|---|---|
| Linux / Windows / macOS | 一等支持；底层 GPU 访问较自由 |
| Android | 计划支持；需要注意 JIT 限制 |

兼容性会诚实描述为 **LLM 推理所需的 CUDA API 子集兼容**，而不是完整 CUDA 兼容。

### 🔭 野心 — 长期愿景

长期来看，OpenCUDA 目标是成为通用运行时，用一套代码在尽可能多的计算设备上运行各厂商 GPU。iOS、游戏机和部分智能电视存在技术与许可限制；CAD 和通用图形工作负载不在当前范围内。


[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-v0.1%20prototype-orange.svg)]()

---

## Run it now

```bash
# Vector addition: C = A + B, 1,000,000 elements
cargo run --release -p vector_add

# Matrix multiplication: C = A·B
cargo run --release -p matmul
```

Expected output for `vector_add`:

```text
device: OpenCUDA CPU Device (rayon, 32 threads)
OK: all 1000000 elements equal 1000000
c[0]=1000000, c[999999]=1000000
```

---

## Architecture

```text
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

✅ = working in v0.1.  ⏳ = designed, planned for implementation.

---

## Workspace layout

```text
opencuda/
├── crates/
│   ├── opencuda-core/       core traits, memory, kernel representation  ✅
│   ├── opencuda-cpu/        CPU backend using rayon                       ✅
│   ├── opencuda-blas/       AI kernels: GEMM / Attention / Quantisation   ⏳
│   └── opencuda-multidev/   multi-GPU partitioning and pipeline parallel  ⏳
└── examples/
    ├── vector_add/          C = A + B                                     ✅
    └── matmul/              C = A·B                                       ✅
```

## Roadmap

- **Phase 1 — Foundation**: CPU first, then Vulkan for all GPUs.
  - [x] `opencuda-core`: traits, two-layer memory, `KernelSource`, `DeviceRegistry`
  - [x] `opencuda-cpu`: rayon multithreaded execution
  - [x] `examples/vector_add`, `examples/matmul` running on CPU
  - [ ] `opencuda-vulkan`: run SPIR-V kernels through Vulkan Compute
  - [ ] Confirm one binary running on NVIDIA / AMD / Intel GPUs
- **Phase 2 — CUDA compatibility**
  - [ ] OmniIR common intermediate representation and SPIR-V output
  - [ ] CUDA C++ subset parser → OmniIR
  - [ ] Hook layer for major CUDA APIs: malloc, memcpy, free, launch
  - [ ] NVIDIA and AMD backends
- **Phase 3 — AI optimisation**
  - [ ] `opencuda-blas`: GEMM / Flash Attention / quantisation
  - [ ] `opencuda-multidev`: pipeline parallelism and unified VRAM strategy
  - [ ] Run LLM inference across two GPUs
- **Phase 4 — Expansion**: Intel oneAPI, nvcc-compatible driver, PyTorch backend

---

## Honest estimate

True full CUDA compatibility across every direction is GPU-vendor-scale work — roughly five to ten years for one person. This roadmap therefore targets **a practical subset required for LLM inference**, not universal full compatibility. Calling it full compatibility too early would betray users and damage the trust that OSS needs most.

---

## Why the name "OpenCUDA"?

The name directly expresses the intent: an open implementation inspired by CUDA. `CUDA` is a trademark of NVIDIA. If trademark concerns arise, the project is prepared to rename itself to **iLumi**, derived from “light”.

---

## 许可证

Apache License 2.0. See [LICENSE](LICENSE).

OpenCUDA is intended to be released under Apache-2.0 as the GPU compute foundation of the aruaru ecosystem.

---

## 贡献

v0.1 is a design and prototype stage. Feedback on the design, CPU backend improvements, and work on the Vulkan backend are welcome.
