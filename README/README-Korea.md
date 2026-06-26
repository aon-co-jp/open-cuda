# OpenCUDA

[British English](README.en-GB.md) | [English](README.en.md) | [Deutsch](README.de.md) | [Italiano](README.it.md) | [Français](README.fr.md) | [Русский](README.ru.md) | [Українська](README.uk.md) | [العربية](README.ar.md) | [فارسی](README.fa.md) | [简体中文](README.zh-CN.md) | [한국어](README.ko.md) | [繁體中文 / 台灣](README.zh-TW.md)

**Rust로 작성된 클린룸 방식의 크로스 벤더 GPU 런타임.**
*커널을 한 번 작성하고, 어떤 GPU에서든 실행하세요 — GPU가 없어도 실행할 수 있습니다.*

> ⚠️ **v0.1 — 설계 + 프로토타입.** 이 초기 릴리스는 설계와 동작하는 프로토타입을 보여줍니다. 현재 실제로 동작하는 것은 **CPU 백엔드**뿐이며 GPU가 필요 없습니다. GPU 백엔드는 설계가 완료되었지만 구현은 앞으로 진행됩니다. 과장된 약속을 하지 않고, 오늘 동작하는 것과 프로젝트가 향하는 목표를 명확히 구분합니다.

## OpenCUDA란?

OpenCUDA는 CUDA에서 영감을 받은 프로그래밍 모델을 **특정 벤더에 묶이지 않고** 실행하기 위한 Rust 런타임입니다. 같은 커널 코드를 NVIDIA, AMD, Intel GPU에서 실행하고, GPU가 없을 때는 CPU에서도 실행하는 것을 목표로 합니다.

첫 번째 주요 목표는 LLM 추론입니다. 추론 시간의 대부분이 GEMM과 Attention 같은 소수의 커널에 집중되기 때문입니다. 이 커널들을 벤더별로 최적화하면 실용적인 호환성에 가장 짧은 경로로 도달할 수 있습니다.

## 두 계층의 범위 — 현실과 야망

이 프로젝트는 단기적으로 달성 가능한 현실과 장기적인 지평을 의도적으로 분리합니다. 이는 신뢰할 수 있는 OSS가 되기 위한 선택입니다.

### 🟢 현실 — Phase 1–3

> **Windows / Linux / macOS / Android에서 동일한 코드로 NVIDIA / AMD / Intel GPU의 LLM 추론을 실행하는 Rust 런타임.**

이것만으로도 큰 목표입니다. llama.cpp가 비슷한 일을 해냈지만, CUDA API 서브셋과 소스 호환되는 clean한 Rust 구현은 아직 빈자리입니다.

| 플랫폼 | 위치づけ |
|---|---|
| Linux / Windows / macOS | 1급 지원; 저수준 GPU 접근이 비교적 자유로움 |
| Android | 지원 예정; JIT 제약에 주의 필요 |

호환성은 전체 CUDA 호환이 아니라 **LLM 추론에 필요한 CUDA API 서브셋 호환**으로 솔직하게 설명합니다.

### 🔭 야망 — 장기 비전

장기적으로 OpenCUDA는 하나의 코드베이스로 여러 계산 장치에서 모든 벤더의 GPU를 실행하는 범용 런타임을 지향합니다. iOS, 게임 콘솔, 일부 스마트 TV에는 기술 및 계약상의 제한이 있으며, CAD와 일반 그래픽 워크로드는 현재 범위 밖입니다.


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

## 라이선스

Apache License 2.0. See [LICENSE](LICENSE).

OpenCUDA is intended to be released under Apache-2.0 as the GPU compute foundation of the aruaru ecosystem.

---

## 기여

v0.1 is a design and prototype stage. Feedback on the design, CPU backend improvements, and work on the Vulkan backend are welcome.
