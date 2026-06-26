# OpenCUDA

[British English](README.en-GB.md) | [English](README.en.md) | [Deutsch](README.de.md) | [Italiano](README.it.md) | [Français](README.fr.md) | [Русский](README.ru.md) | [Українська](README.uk.md) | [العربية](README.ar.md) | [فارسی](README.fa.md) | [简体中文](README.zh-CN.md) | [한국어](README.ko.md) | [繁體中文 / 台灣](README.zh-TW.md)

<div dir="rtl">

**بيئة تشغيل GPU نظيفة التصميم ومتعددة المورّدين مكتوبة بلغة Rust.**
*اكتب النواة مرة واحدة. شغّلها على أي GPU — أو حتى من دون GPU.*

> ⚠️ **v0.1 — تصميم + نموذج أولي.** هذا إصدار مبكر يوضح التصميم ونموذجاً أولياً يعمل. حالياً يعمل فعلياً **Backend الخاص بالـ CPU** فقط؛ لا حاجة إلى GPU. تم تصميم Backends الخاصة بالـ GPU، لكن التنفيذ لم يكتمل بعد. لا نقدم وعوداً مبالغاً فيها، بل نفصل بوضوح بين ما يعمل اليوم وما نهدف إلى بنائه.

## ما هو OpenCUDA؟

OpenCUDA هي Runtime مكتوبة بـ Rust لنموذج برمجة مستوحى من CUDA **من دون الارتباط بمورّد واحد**. الهدف هو تشغيل كود النواة نفسه على GPU من NVIDIA أو AMD أو Intel — وكذلك على CPU عندما لا تتوفر GPU.

الهدف الأول هو LLM inference، لأن معظم وقت الحساب يتركز في عدد قليل من Kernels مثل GEMM و Attention. بتحسين هذه Kernels عبر المورّدين يمكن الوصول إلى توافق عملي بأقصر طريق.

## نطاق ذو طبقتين — الواقع والطموح

يفصل المشروع عمداً بين الواقع الممكن على المدى القريب والرؤية طويلة المدى، وذلك للحفاظ على الثقة كمشروع OSS.

### 🟢 الواقع — Phase 1–3

> **Runtime مكتوبة بـ Rust لتشغيل LLM inference بالكود نفسه على Windows / Linux / macOS / Android وعلى GPU من NVIDIA / AMD / Intel.**

هذا هدف كبير بالفعل. حقق llama.cpp شيئاً قريباً، لكن لا يزال هناك مجال لتطبيق نظيف بـ Rust متوافق على مستوى المصدر مع جزء فرعي من CUDA API.

| المنصة | الوضع |
|---|---|
| Linux / Windows / macOS | دعم من الدرجة الأولى؛ الوصول منخفض المستوى إلى GPU مرن |
| Android | دعم مخطط له؛ يجب التعامل بحذر مع قيود JIT |

يتم وصف التوافق بصدق على أنه **جزء فرعي من CUDA API اللازم لـ LLM inference** وليس توافقاً كاملاً مع CUDA.

### 🔭 الطموح — الرؤية طويلة المدى

على المدى الطويل، يهدف OpenCUDA إلى Runtime عالمية لتشغيل GPU من جميع المورّدين على أجهزة حوسبة متعددة بقاعدة كود واحدة. تفرض iOS وأجهزة الألعاب وبعض أجهزة التلفاز الذكية حدوداً تقنية وتعاقدية؛ أما CAD والرسوميات العامة فهي خارج النطاق الحالي.

</div>


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

## الترخيص

Apache License 2.0. See [LICENSE](LICENSE).

OpenCUDA is intended to be released under Apache-2.0 as the GPU compute foundation of the aruaru ecosystem.

---

## المساهمة

v0.1 is a design and prototype stage. Feedback on the design, CPU backend improvements, and work on the Vulkan backend are welcome.
