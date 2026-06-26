# OpenCUDA

[British English](README.en-GB.md) | [English](README.en.md) | [Deutsch](README.de.md) | [Italiano](README.it.md) | [Français](README.fr.md) | [Русский](README.ru.md) | [Українська](README.uk.md) | [العربية](README.ar.md) | [فارسی](README.fa.md) | [简体中文](README.zh-CN.md) | [한국어](README.ko.md) | [繁體中文 / 台灣](README.zh-TW.md)

**Clean-room GPU runtime для різних виробників, написаний на Rust.**
*Напишіть kernel один раз. Запускайте його на будь-якій GPU — або взагалі без GPU.*

> ⚠️ **v0.1 — дизайн + прототип.** Це ранній реліз, який показує архітектуру і робочий прототип. Наразі реально працює лише **CPU backend**; GPU не потрібна. GPU backend'и вже спроєктовані, але їх реалізація ще попереду. Ми не даємо перебільшених обіцянок і чітко розділяємо те, що працює сьогодні, і те, куди рухається проєкт.

## Що таке OpenCUDA?

OpenCUDA — це runtime на Rust для моделі програмування, натхненної CUDA, **без прив'язки до одного виробника**. Один і той самий kernel-код має запускатися на GPU NVIDIA, AMD та Intel — а також на CPU, якщо GPU немає.

Перший основний сценарій — LLM inference, бо більшість часу обчислень зосереджена у невеликій кількості kernels, насамперед GEMM та Attention. Оптимізуючи їх для різних виробників, OpenCUDA може найкоротшим шляхом досягти практично корисної сумісності.

## Два рівні — реальність і амбіція

Проєкт свідомо розділяє досяжну короткострокову мету та довгостроковий горизонт. Це потрібно для довіри до OSS.

### 🟢 Реальність — Phase 1–3

> **Rust runtime, який запускає LLM inference одним кодом на Windows / Linux / macOS / Android та на GPU NVIDIA / AMD / Intel.**

Це вже велика мета. llama.cpp досяг близького результату, але чиста реалізація на Rust із сумісністю на рівні вихідного коду з підмножиною CUDA API ще залишається відкритою нішею.

| Платформа | Позиціонування |
|---|---|
| Linux / Windows / macOS | Підтримка першого класу; низькорівневий доступ до GPU гнучкий |
| Android | Запланована підтримка; обмеження JIT слід враховувати |

Сумісність чесно описується як **підмножина CUDA API, потрібна для LLM inference**, а не як повна сумісність із CUDA.

### 🔭 Амбіція — довгострокове бачення

У довгостроковій перспективі OpenCUDA прагне стати універсальним runtime для GPU всіх виробників на різних обчислювальних пристроях з однією кодовою базою. iOS, ігрові консолі та деякі smart TV мають технічні й дозвільні обмеження; CAD і загальна графіка наразі поза областю проєкту.


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

## Ліцензія

Apache License 2.0. See [LICENSE](LICENSE).

OpenCUDA is intended to be released under Apache-2.0 as the GPU compute foundation of the aruaru ecosystem.

---

## Участь

v0.1 is a design and prototype stage. Feedback on the design, CPU backend improvements, and work on the Vulkan backend are welcome.
