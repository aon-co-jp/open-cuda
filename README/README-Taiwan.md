# OpenCUDA

[British English](README.en-GB.md) | [English](README.en.md) | [Deutsch](README.de.md) | [Italiano](README.it.md) | [Français](README.fr.md) | [Русский](README.ru.md) | [Українська](README.uk.md) | [العربية](README.ar.md) | [فارسی](README.fa.md) | [简体中文](README.zh-CN.md) | [한국어](README.ko.md) | [繁體中文 / 台灣](README.zh-TW.md)

**以 Rust 撰寫、clean-room、跨廠商的 GPU Runtime。**
*Kernel 只寫一次，就能在任何 GPU 上執行——也能在沒有 GPU 的情況下執行。*

> ⚠️ **v0.1 — 設計 + 原型。** 這是早期版本，用來展示設計與可運作的原型。目前真正能執行的只有 **CPU Backend**；不需要 GPU。GPU Backend 已完成設計，但實作仍在後續階段。我們不做誇大的承諾，而是清楚區分今天能運作的內容與專案未來的方向。

## OpenCUDA 是什麼？

OpenCUDA 是一個 Rust Runtime，用於執行受 CUDA 啟發的程式設計模型，**但不被單一廠商綁住**。目標是讓同一份 kernel 程式碼可以在 NVIDIA、AMD、Intel GPU 上執行；如果沒有 GPU，也能在 CPU 上執行。

第一個主要戰場是 LLM 推論，因為推論時間大多集中在少數 kernel，例如 GEMM 與 Attention。只要跨廠商最佳化這些 kernel，OpenCUDA 就能用最短路徑達到實用層級的相容性。

## 雙層範圍 — 現實與野心

專案刻意區分短期可達成的現實，以及長期想抵達的地平線。這是為了成為可信任的 OSS。

### 🟢 現實 — Phase 1–3

> **一個 Rust Runtime，可在 Windows / Linux / macOS / Android 上，用同一份程式碼在 NVIDIA / AMD / Intel GPU 上執行 LLM 推論。**

這已經是很大的目標。llama.cpp 做到了接近的事情，但以 Rust 寫成、設計 clean、並與 CUDA API 子集合保持原始碼相容的實作，仍然是空位。

| 平台 | 定位 |
|---|---|
| Linux / Windows / macOS | 第一級支援；底層 GPU 存取較自由 |
| Android | 預計支援；需注意 JIT 限制 |

相容性會誠實描述為 **LLM 推論所需的 CUDA API 子集合相容**，不是完整 CUDA 相容。

### 🔭 野心 — 長期願景

長期而言，OpenCUDA 目標是成為通用 Runtime，用同一套 codebase 在盡可能多的運算裝置上執行各廠商 GPU。iOS、遊戲主機與部分智慧電視存在技術與授權限制；CAD 與一般圖形工作負載不在目前範圍內。


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

## 授權

Apache License 2.0. See [LICENSE](LICENSE).

OpenCUDA is intended to be released under Apache-2.0 as the GPU compute foundation of the aruaru ecosystem.

---

## 貢獻

v0.1 is a design and prototype stage. Feedback on the design, CPU backend improvements, and work on the Vulkan backend are welcome.
