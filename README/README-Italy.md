# OpenCUDA

[British English](README.en-GB.md) | [English](README.en.md) | [Deutsch](README.de.md) | [Italiano](README.it.md) | [Français](README.fr.md) | [Русский](README.ru.md) | [Українська](README.uk.md) | [العربية](README.ar.md) | [فارسی](README.fa.md) | [简体中文](README.zh-CN.md) | [한국어](README.ko.md) | [繁體中文 / 台灣](README.zh-TW.md)

**Un runtime GPU clean-room e multi-vendor scritto in Rust.**
*Scrivi il kernel una volta. Eseguilo su qualsiasi GPU — oppure senza GPU.*

> ⚠️ **v0.1 — design + prototipo.** Questa è una versione iniziale che mostra il design e un prototipo funzionante. Al momento funziona davvero solo il **backend CPU**; non serve alcuna GPU. I backend GPU sono stati progettati, ma l'implementazione deve ancora arrivare. Non facciamo promesse esagerate: separiamo ciò che funziona oggi da ciò che il progetto vuole diventare.

## Che cos'è OpenCUDA?

OpenCUDA è un runtime in Rust per eseguire un modello di programmazione ispirato a CUDA **senza dipendere da un singolo vendor**. L'obiettivo è eseguire lo stesso codice kernel su GPU NVIDIA, AMD e Intel — e anche sulla CPU se non è disponibile una GPU.

Il primo obiettivo è l'inferenza LLM, perché la maggior parte del tempo di calcolo è concentrata in pochi kernel, soprattutto GEMM e Attention. Ottimizzando questi kernel per più vendor, OpenCUDA può raggiungere una compatibilità pratica nel modo più diretto.

## Ambito a due livelli — realtà e ambizione

Il progetto separa intenzionalmente la realtà raggiungibile a breve termine dall'orizzonte di lungo periodo, per essere un OSS affidabile.

### 🟢 Realtà — Phase 1–3

> **Un runtime Rust che esegue inferenza LLM con lo stesso codice su Windows / Linux / macOS / Android e su GPU NVIDIA / AMD / Intel.**

È già un obiettivo grande. llama.cpp ha ottenuto qualcosa di simile, ma c'è ancora spazio per un'implementazione Rust pulita e sorgente-compatibile con un sottoinsieme delle API CUDA.

| Piattaforma | Posizionamento |
|---|---|
| Linux / Windows / macOS | Supporto di prima classe; accesso GPU low-level flessibile |
| Android | Supporto pianificato; attenzione ai vincoli JIT |

La compatibilità viene descritta onestamente come **sottoinsieme API CUDA necessario per l'inferenza LLM**, non come compatibilità CUDA completa.

### 🔭 Ambizione — visione a lungo termine

A lungo termine OpenCUDA mira a una runtime universale capace di eseguire GPU di ogni vendor su molti dispositivi con una sola base di codice. iOS, console e alcune smart TV hanno limiti tecnici e di permesso; CAD e grafica generale sono fuori dallo scope attuale.


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

## Licenza

Apache License 2.0. See [LICENSE](LICENSE).

OpenCUDA is intended to be released under Apache-2.0 as the GPU compute foundation of the aruaru ecosystem.

---

## Contribuire

v0.1 is a design and prototype stage. Feedback on the design, CPU backend improvements, and work on the Vulkan backend are welcome.
