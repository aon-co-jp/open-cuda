# OpenCUDA

[British English](README.en-GB.md) | [English](README.en.md) | [Deutsch](README.de.md) | [Italiano](README.it.md) | [Français](README.fr.md) | [Русский](README.ru.md) | [Українська](README.uk.md) | [العربية](README.ar.md) | [فارسی](README.fa.md) | [简体中文](README.zh-CN.md) | [한국어](README.ko.md) | [繁體中文 / 台灣](README.zh-TW.md)

**Eine Clean-Room-GPU-Runtime für mehrere Hersteller, geschrieben in Rust.**
*Schreibe deinen Kernel einmal. Führe ihn auf jeder GPU aus — oder ganz ohne GPU.*

> ⚠️ **v0.1 — Design + Prototyp.** Diese frühe Version zeigt das Design und einen funktionierenden Prototyp. Derzeit läuft tatsächlich nur das **CPU-Backend**; eine GPU ist nicht erforderlich. Die GPU-Backends sind entworfen, ihre Implementierung steht jedoch noch aus. Wir machen keine übertriebenen Versprechen und trennen klar, was heute funktioniert, von dem Ziel des Projekts.

## Was ist OpenCUDA?

OpenCUDA ist eine Rust-Runtime für ein von CUDA inspiriertes Programmiermodell **ohne Bindung an einen einzelnen Hersteller**. Derselbe Kernel-Code soll auf NVIDIA-, AMD- und Intel-GPUs laufen — und auch auf der CPU, wenn keine GPU vorhanden ist.

LLM-Inferenz ist das erste Hauptziel, weil der größte Teil der Rechenzeit auf wenige Kernel wie GEMM und Attention fällt. Wenn diese Kernel für verschiedene Hersteller optimiert werden, ist praktische Kompatibilität auf dem kürzesten Weg erreichbar.

## Umfang in zwei Ebenen — Realität und Ambition

Das Projekt trennt bewusst die kurzfristig erreichbare Realität vom langfristigen Horizont. Das dient der Glaubwürdigkeit als OSS.

### 🟢 Realität — Phase 1–3

> **Eine Rust-Runtime, die LLM-Inferenz mit demselben Code unter Windows / Linux / macOS / Android und auf NVIDIA-, AMD- sowie Intel-GPUs ausführt.**

Das ist bereits ein großes Ziel. llama.cpp kommt dem nahe, aber eine saubere Rust-Implementierung mit Quellkompatibilität zu einem CUDA-API-Subset ist noch ein offener Platz.

| Plattform | Einordnung |
|---|---|
| Linux / Windows / macOS | Erstklassige Unterstützung; Low-Level-GPU-Zugriff ist flexibel |
| Android | Geplante Unterstützung; JIT-Beschränkungen beachten |

Die Kompatibilität wird ehrlich als **CUDA-API-Subset für LLM-Inferenz** beschrieben, nicht als vollständige CUDA-Kompatibilität.

### 🔭 Ambition — langfristige Vision

Langfristig strebt OpenCUDA eine universelle Runtime an, die GPUs aller Hersteller auf möglichst vielen Rechengeräten mit einer Codebasis ausführen kann. iOS, Spielkonsolen und einige Smart-TVs haben jedoch rechtliche und technische Grenzen. CAD und allgemeine Grafiklasten liegen ausserhalb des aktuellen Fokus.


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

## Lizenz

Apache License 2.0. See [LICENSE](LICENSE).

OpenCUDA is intended to be released under Apache-2.0 as the GPU compute foundation of the aruaru ecosystem.

---

## Mitwirken

v0.1 is a design and prototype stage. Feedback on the design, CPU backend improvements, and work on the Vulkan backend are welcome.
