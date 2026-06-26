# OpenCUDA

[British English](README.en-GB.md) | [English](README.en.md) | [Deutsch](README.de.md) | [Italiano](README.it.md) | [Français](README.fr.md) | [Русский](README.ru.md) | [Українська](README.uk.md) | [العربية](README.ar.md) | [فارسی](README.fa.md) | [简体中文](README.zh-CN.md) | [한국어](README.ko.md) | [繁體中文 / 台灣](README.zh-TW.md)

<div dir="rtl">

**یک Runtime پاک‌طراحی‌شده و چندسازنده‌ای برای GPU، نوشته‌شده با Rust.**
*Kernel را یک‌بار بنویسید. آن را روی هر GPU — یا حتی بدون GPU — اجرا کنید.*

> ⚠️ **v0.1 — طراحی + نمونهٔ اولیه.** این نسخهٔ اولیه، معماری و یک نمونهٔ عملی را نشان می‌دهد. در حال حاضر فقط **Backend مربوط به CPU** واقعاً اجرا می‌شود؛ GPU لازم نیست. Backendهای GPU طراحی شده‌اند، اما پیاده‌سازی آن‌ها هنوز در آینده است. ما وعدهٔ اغراق‌آمیز نمی‌دهیم و روشن جدا می‌کنیم که امروز چه چیزی کار می‌کند و پروژه به کجا می‌رود.

## OpenCUDA چیست؟

OpenCUDA یک Runtime نوشته‌شده با Rust برای اجرای مدل برنامه‌نویسی الهام‌گرفته از CUDA است، **بدون وابستگی به یک سازندهٔ خاص**. هدف این است که همان کد Kernel روی GPUهای NVIDIA، AMD و Intel — و در نبود GPU روی CPU — اجرا شود.

هدف اول، LLM inference است، زیرا بیشتر زمان محاسبه در چند Kernel مانند GEMM و Attention متمرکز است. با بهینه‌سازی این Kernelها برای سازندگان مختلف، OpenCUDA می‌تواند سریع‌تر به سازگاری عملی برسد.

## دامنهٔ دو لایه — واقعیت و بلندپروازی

پروژه آگاهانه هدف کوتاه‌مدت قابل دستیابی را از افق بلندمدت جدا می‌کند تا به‌عنوان OSS قابل اعتماد باقی بماند.

### 🟢 واقعیت — Phase 1–3

> **یک Runtime مبتنی بر Rust که LLM inference را با یک کد روی Windows / Linux / macOS / Android و GPUهای NVIDIA / AMD / Intel اجرا می‌کند.**

همین هدف نیز بزرگ است. llama.cpp به چیزی نزدیک رسیده، اما هنوز جای یک پیاده‌سازی تمیز Rust با سازگاری سطح منبع با زیرمجموعه‌ای از CUDA API خالی است.

| پلتفرم | جایگاه |
|---|---|
| Linux / Windows / macOS | پشتیبانی درجه‌یک؛ دسترسی سطح پایین به GPU انعطاف‌پذیر است |
| Android | پشتیبانی برنامه‌ریزی‌شده؛ محدودیت‌های JIT باید با دقت مدیریت شوند |

سازگاری به‌صورت صادقانه **زیرمجموعهٔ CUDA API لازم برای LLM inference** توصیف می‌شود، نه سازگاری کامل با CUDA.

### 🔭 بلندپروازی — چشم‌انداز بلندمدت

در بلندمدت، OpenCUDA به Runtime جهانی برای اجرای GPU همهٔ سازندگان روی دستگاه‌های محاسباتی گوناگون با یک codebase واحد می‌اندیشد. iOS، کنسول‌ها و برخی Smart TVها محدودیت‌های فنی و قراردادی دارند؛ CAD و گرافیک عمومی فعلاً خارج از دامنه هستند.

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

## مجوز

Apache License 2.0. See [LICENSE](LICENSE).

OpenCUDA is intended to be released under Apache-2.0 as the GPU compute foundation of the aruaru ecosystem.

---

## مشارکت

v0.1 is a design and prototype stage. Feedback on the design, CPU backend improvements, and work on the Vulkan backend are welcome.
