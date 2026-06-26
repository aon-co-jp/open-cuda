# OpenCUDA

[British English](README.en-GB.md) | [English](README.en.md) | [Deutsch](README.de.md) | [Italiano](README.it.md) | [Français](README.fr.md) | [Русский](README.ru.md) | [Українська](README.uk.md) | [العربية](README.ar.md) | [فارسی](README.fa.md) | [简体中文](README.zh-CN.md) | [한국어](README.ko.md) | [繁體中文 / 台灣](README.zh-TW.md)

**Clean-room GPU runtime для разных производителей, написанный на Rust.**
*Напишите kernel один раз. Запускайте его на любой GPU — или вообще без GPU.*

> ⚠️ **v0.1 — дизайн + прототип.** Это ранний релиз, показывающий архитектуру и рабочий прототип. Сейчас реально работает только **CPU backend**; GPU не требуется. GPU backend'ы уже спроектированы, но их реализация ещё впереди. Мы не даём завышенных обещаний и ясно разделяем то, что работает сегодня, и то, к чему проект стремится.

## Что такое OpenCUDA?

OpenCUDA — это runtime на Rust для модели программирования, вдохновлённой CUDA, **без привязки к одному поставщику**. Одна и та же kernel-программа должна запускаться на GPU NVIDIA, AMD и Intel — а также на CPU, если GPU отсутствует.

Первый основной сценарий — LLM inference, потому что большая часть времени вычислений сосредоточена в небольшом числе kernels, прежде всего GEMM и Attention. Оптимизируя их для разных производителей, OpenCUDA может быстрее всего прийти к практически полезной совместимости.

## Две области — реальность и амбиция

Проект сознательно разделяет достижимую краткосрочную цель и долгосрочный горизонт. Это нужно для доверия к OSS.

### 🟢 Реальность — Phase 1–3

> **Rust runtime, который запускает LLM inference одним кодом на Windows / Linux / macOS / Android и на GPU NVIDIA / AMD / Intel.**

Это уже большая цель. llama.cpp достиг близкого результата, но чистая реализация на Rust с исходной совместимостью с подмножеством CUDA API всё ещё остаётся свободной нишей.

| Платформа | Позиционирование |
|---|---|
| Linux / Windows / macOS | Поддержка первого класса; низкоуровневый доступ к GPU гибок |
| Android | Планируемая поддержка; ограничения JIT нужно учитывать |

Совместимость честно описывается как **подмножество CUDA API, нужное для LLM inference**, а не как полная совместимость с CUDA.

### 🔭 Амбиция — долгосрочное видение

В долгосрочной перспективе OpenCUDA стремится стать универсальным runtime для GPU всех производителей на разных вычислительных устройствах с одной кодовой базой. iOS, игровые консоли и некоторые smart TV имеют технические и юридические ограничения; CAD и общая графика сейчас вне области проекта.


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

## Лицензия

Apache License 2.0. See [LICENSE](LICENSE).

OpenCUDA is intended to be released under Apache-2.0 as the GPU compute foundation of the aruaru ecosystem.

---

## Участие

v0.1 is a design and prototype stage. Feedback on the design, CPU backend improvements, and work on the Vulkan backend are welcome.
