# OpenCUDA

[British English](README.en-GB.md) | [English](README.en.md) | [Deutsch](README.de.md) | [Italiano](README.it.md) | [Français](README.fr.md) | [Русский](README.ru.md) | [Українська](README.uk.md) | [العربية](README.ar.md) | [فارسی](README.fa.md) | [简体中文](README.zh-CN.md) | [한국어](README.ko.md) | [繁體中文 / 台灣](README.zh-TW.md)

**Un runtime GPU clean-room et multi-fournisseur écrit en Rust.**
*Écrivez votre kernel une fois. Exécutez-le sur n'importe quel GPU — ou sans GPU.*

> ⚠️ **v0.1 — conception + prototype.** Cette version initiale présente l'architecture et un prototype fonctionnel. Pour l'instant, seul le **backend CPU** fonctionne réellement ; aucune GPU n'est nécessaire. Les backends GPU sont conçus, mais leur implémentation reste à faire. Nous évitons les promesses excessives et séparons clairement ce qui fonctionne aujourd'hui de l'objectif du projet.

## Qu'est-ce qu'OpenCUDA ?

OpenCUDA est un runtime écrit en Rust pour exécuter un modèle de programmation inspiré de CUDA **sans dépendre d'un seul fournisseur**. Le même code kernel doit pouvoir s'exécuter sur des GPU NVIDIA, AMD et Intel — ainsi que sur CPU lorsqu'aucune GPU n'est disponible.

La première cible est l'inférence LLM, car la plupart du temps de calcul se concentre sur quelques kernels, principalement GEMM et Attention. En optimisant ces kernels pour plusieurs fournisseurs, OpenCUDA peut atteindre une compatibilité pratique par le chemin le plus court.

## Portée à deux niveaux — réalité et ambition

Le projet sépare volontairement la réalité atteignable à court terme de l'horizon à long terme afin de rester un OSS crédible.

### 🟢 Réalité — Phase 1–3

> **Un runtime Rust qui exécute l'inférence LLM avec le même code sur Windows / Linux / macOS / Android et sur GPU NVIDIA / AMD / Intel.**

C'est déjà un objectif majeur. llama.cpp réalise quelque chose de proche, mais une implémentation Rust propre et compatible au niveau source avec un sous-ensemble de l'API CUDA reste une place ouverte.

| Plateforme | Positionnement |
|---|---|
| Linux / Windows / macOS | Support de premier rang ; accès GPU bas niveau flexible |
| Android | Support prévu ; contraintes JIT à traiter avec prudence |

La compatibilité est décrite honnêtement comme **sous-ensemble de l'API CUDA nécessaire à l'inférence LLM**, et non comme compatibilité CUDA complète.

### 🔭 Ambition — vision à long terme

À long terme, OpenCUDA vise un runtime universel capable d'exécuter les GPU de tous les fournisseurs sur de nombreux appareils avec une seule base de code. iOS, les consoles et certaines smart TV imposent des limites techniques et contractuelles ; la CAO et les charges graphiques générales restent hors périmètre actuel.


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

## Licence

Apache License 2.0. See [LICENSE](LICENSE).

OpenCUDA is intended to be released under Apache-2.0 as the GPU compute foundation of the aruaru ecosystem.

---

## Contribuer

v0.1 is a design and prototype stage. Feedback on the design, CPU backend improvements, and work on the Vulkan backend are welcome.
