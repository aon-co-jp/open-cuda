# OpenCUDA v0.3.0-dev 追記

この開発版では、v0.3.0 の CPU / OmniIR / VulkanMock 経路を維持したまま、optional feature `real-vulkan` として `ash` ベースの最小実Vulkan Computeバックエンドを追加しました。

通常の開発確認はGPUなしで可能です。

```powershell
cargo check --workspace
cargo run --release -p vector_add
cargo run --release -p vector_add_omniir
cargo run --release -p vector_add_vulkan
```

実Vulkanで `vector_add` を走らせる場合は、Vulkan SDK などで `glslc` を使える状態にしてから、次を実行します。

```powershell
.\tools\compile-vulkan-shaders.ps1
cargo run --release --manifest-path examples\vector_add_vulkan_real\Cargo.toml
```

この実Vulkanサンプルは環境依存なので、`cargo check --workspace` の通常経路からは分離しています。

---

# OpenCUDA

**A clean-room, cross-vendor GPU runtime written in Rust.**
*Write your kernel once. Run it on any GPU — or none at all.*

> ⚠️ **v0.3 — CPU + Mock + 最小OmniIR prototype.** これは設計と動くプロトタイプを示す初期リリースです。
> 現時点で実際に動くのは **CPUバックエンド**（GPU不要）と、GPUなしでSPIR-V/OmniIR経路を検証する **Mockバックエンド** です。
> 本物のGPUバックエンドは実装途中です。誇大な約束はしません。代わりに,
> 「今どこまで動くか」と「どこを目指すか」を明確に分けて書きます。

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-v0.1%20prototype-orange.svg)]()

---

## これは何か

OpenCUDA は、CUDA に着想を得たプログラミングモデルを、**特定ベンダーに縛られずに**
実行するための Rust 製ランタイムです。NVIDIA / AMD / Intel のどの GPU でも、
そして GPU が無ければ CPU でも、同じカーネルコードが動くことを目指します。

LLM 推論を最初の主戦場に据えています。理由は単純で、推論の計算時間は
GEMM と Attention という**少数のカーネル**にほぼ集中するため、そこを各ベンダーで
最適化すれば「実用的に十分な互換性」に最短で到達できるからです。

---

## 二層のスコープ ― 現実と野心

このプロジェクトは、達成可能な現実と、長期に目指す地平を、意図的に分けて宣言します。
これは背伸びを隠すためではなく、**信用できる OSS であるため**です。

### 🟢 現実 — 確実に到達する範囲（Phase 1〜3）

> **Windows / Linux / macOS / Android で、NVIDIA / AMD / Intel のどの GPU でも、
> LLM 推論が同じコードで動く Rust 製ランタイム。**

これだけでも十分に大きな目標です。llama.cpp が近いことを実現していますが、
**Rust 製で、CUDA API にソース互換で、設計が clean な実装**という席はまだ空いています。
ここに的を絞れば、少人数 — 原理的には 1 人 — でも Phase 1〜3 に到達できます。

| プラットフォーム | 位置づけ |
|---|---|
| Linux / Windows / macOS | 第一級サポート（低レベルGPUアクセスが自由） |
| Android | 対応予定（JIT制約に注意） |

ユースケースは **LLM 推論を主軸**に据えます。互換性は「CUDA 全互換」ではなく
**「LLM 推論に必要な CUDA API サブセットの互換」** と正直に表現します。

### 🔭 野心 — 最終的に目指す地平（長期ビジョン）

短期的に信用を一部失ってでも、目指す先は正直に書いておきます。作者は、
普通の延長線ではない地平を見ています ── ただしマッドサイエンティストではなく、
一歩ずつ到達可能な階段として。

最終的には、**あらゆる計算デバイス上で、あらゆるベンダーの GPU を、
1 つのコードで** 動かせる普遍ランタイムを目指します。デスクトップ（Windows /
macOS / Linux）、モバイル（Android / iOS）、そしていつか — 各プラットフォーム
ホルダーとの正規の合意が得られれば — より閉じた環境まで。

ただし、その野心には**今すぐには越えられない壁**があることも同じく正直に書きます。

- **iOS**: Metal 以外の GPU API が事実上使えず、App Store 規約で JIT が禁止。
  対応するなら全カーネル AOT コンパイル限定になる。
- **ゲーム機 / 一部スマートTV**: クローズドプラットフォームで、低レベル GPU
  アクセスには各社との開発者契約が必要。技術ではなく許諾の問題。現時点では対象外。
- **CAD 等のグラフィックス全般**: GEMM/Attention 中心の LLM とは要求が正反対で、
  グラフィックスドライバスタック全体の再実装に近い。現スコープからは外す。

野心は捨てません。が、**README のトップで約束するのは 🟢 の範囲だけ**です。

---

## 今すぐ動かせる（CPU バックエンド、GPU 不要）

```bash
# ベクトル加算 C = A + B（100万要素）
cargo run --release -p vector_add

# GPUなしでSPIR-V経路を検証
cargo run --release -p vector_add_vulkan

# v0.3: 同じOmniIRをCPU NativeとVulkanMockで検証
cargo run --release -p vector_add_omniir

# 行列積 C = A·B
cargo run --release -p matmul
```

期待される出力（vector_add）:

```
device: OpenCUDA CPU Device (rayon, 32 threads)
OK: all 1000000 elements equal 1000000
c[0]=1000000, c[999999]=1000000
```

GPU を 1 枚も持っていなくても、16 コア 32 スレッドの CPU 上で、`rayon` による
マルチスレッド実行でカーネルが走ります。これが Phase 1 の核心 ──
**ハードウェアを買う前に、設計の正しさを実機で検証できる**。


---

## v0.1.1 で追加: GPUなしの Vulkan/SPIR-V 経路テスト

実機GPUや Vulkan SDK が無くても開発を進めるため、`opencuda-vulkan` に
**VulkanMockDevice** を追加しました。これは実Vulkanではありません。
目的は、Vulkan系バックエンドが `Native` カーネルを拒否し、`SpirV` カーネルだけを
受け付ける契約を、GPUなしで検証することです。

```bash
# CPU Native 経路
cargo run --release -p vector_add

# GPUなしの Vulkan/SPIR-V 経路シミュレーション
cargo run --release -p vector_add_vulkan

# CPU Native の小型 matmul 検証
cargo run --release -p matmul
```

`vector_add_vulkan` は最初に `Native` カーネルが Vulkan 系バックエンドで拒否されることを確認し、
次に `KernelSource::SpirV` 経路で `vector_add` をシミュレーション実行します。
これにより、OmniIR → SPIR-V コンパイラや実Vulkanバックエンドが未完成でも、
メモリ、引数、起動設定、SPIR-V分岐のテストを進められます。

本物の Vulkan 実装では、`examples/vector_add_vulkan/shaders/vector_add.comp` を
SPIR-Vへコンパイルし、`VkShaderModule` → Compute Pipeline → Dispatch に流します。

---

## アーキテクチャ

```
        ┌──────────────────────────────────────────────┐
        │  User kernels (Native / SPIR-V / PTX / OmniIR)│
        └──────────────────────────────────────────────┘
                              │
        ┌──────────────────────────────────────────────┐
        │  opencuda-core   ← 背骨（GpuDevice trait）     │
        │   ・二層メモリ（DevicePtr + DeviceBuffer）     │
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

✅ = v0.1 で動作　⏳ = 設計済み・実装予定

### 設計の確定事項（Phase 1）

1. **メモリは二層**。内部は生ポインタ `DevicePtr`（`device_id` を埋め込み、
   マルチGPU混在時の取り違えを型で防ぐ）、ネイティブ API は RAII の
   `DeviceBuffer`（`Arc<dyn GpuDevice>` を保持し Drop で自動解放）。
2. **エラーは `anyhow` 基調 + 最小 `GpuError` enum**。普段は気軽に書き、
   CUDA 互換層が整数コードに変換する必要のある代表的失敗だけ enum で持つ。
3. **カーネルは `KernelSource` enum** で多形式を保持。v0.1 は `Native`(CPU) を
   実装、`SpirV`/`Ptx`/`OmniIr` は今後。enum なので後から足しても壊れない。
4. **CPU バックエンドが最初の実行ターゲット**。GPU 無しで設計を検証する。

---

## ワークスペース構成

```
opencuda/
├── crates/
│   ├── opencuda-core/       背骨（trait・メモリ・カーネル表現）  ✅
│   ├── opencuda-cpu/        CPUバックエンド（rayon）            ✅
│   ├── opencuda-blas/       AIカーネル（GEMM/Attention/量子化）  ⏳
│   └── opencuda-multidev/   マルチGPU分割・パイプライン並列      ⏳
└── examples/
    ├── vector_add/          C = A + B                          ✅
    └── matmul/              C = A·B                            ✅
```

---

## ロードマップ

- **Phase 1（基盤）** — CPU で動かし、次に Vulkan で全GPUへ
  - [x] `opencuda-core`: trait・二層メモリ・`KernelSource`・`DeviceRegistry`
  - [x] `opencuda-cpu`: rayon マルチスレッド実行
  - [x] `examples/vector_add`, `examples/matmul`（CPUで動作）
  - [ ] `opencuda-vulkan`: SPIR-V カーネルを Vulkan Compute で実行
  - [ ] 同一バイナリが NVIDIA / AMD / Intel GPU で動くことを確認
- **Phase 2（CUDA互換）**
  - [ ] OmniIR（共通中間表現）と SPIR-V 出力
  - [ ] CUDA C++ サブセットのパーサ → OmniIR
  - [ ] 主要 CUDA API（malloc/memcpy/free/launch）のフック層
  - [ ] NVIDIA / AMD バックエンド
- **Phase 3（AI最適化）**
  - [ ] `opencuda-blas`: GEMM / Flash Attention / 量子化
  - [ ] `opencuda-multidev`: パイプライン並列・統合VRAM
  - [ ] LLM 推論を 2 枚の GPU にまたがって実行
- **Phase 4（拡張）** — Intel oneAPI、nvcc 互換ドライバ、PyTorch backend

---

## 正直な見積もり

「真の CUDA 完全互換」を全方位で実現するのは、GPU ベンダークラスの仕事量です
（1 人なら 5〜10 年規模）。本ロードマップが各 Phase のゴールに据えるのは
**「完全互換」ではなく「LLM 推論に必要なサブセットが実用的に動く」**こと。
ここを混同して "full compatibility" と約束すると、ユーザーを裏切ることになり、
OSS として最も失ってはいけない信用を失います。だからこそ、できることを
できると言い、できないことをできないと言います。

---

## なぜ "OpenCUDA" という名前か

「CUDA に着想を得た、開かれた実装」という意図そのままの名前です。
なお `CUDA` は NVIDIA の商標です。商標上の懸念が生じた場合、本プロジェクトは
名称を **iLumi**（"光" に由来）へ変更する用意があります。

---

## ライセンス

Apache License 2.0. See [LICENSE](LICENSE).

aruaru エコシステム（aruaru-DB, aruaru-llm）と同じく Apache-2.0 で公開し、
その GPU 計算基盤として位置づけます。

---

## 貢献

v0.1 は設計とプロトタイプの段階です。設計への指摘、CPU バックエンドの改善、
そして Vulkan バックエンド（Phase 1 の次の一手）への着手を歓迎します。
# open-cuda


---

## v0.3 で追加したもの

OpenCUDA v0.3 は「本物Vulkan実装」の直前に必要な、GPUなしで壊れにくい設計を固める版です。

- `opencuda-ir`: 最小 OmniIR クレートを追加。まず `vector_add_f32` だけをIR化。
- `CompiledKernel::omniir(...)`: core に OmniIR カーネル生成APIを追加。
- `vector_add_omniir`: 同じ `IrModule::vector_add_f32()` を CPU Native と VulkanMock の両方で検証。
- `opencuda-vulkan`: `KernelSource::OmniIr` を受け取り、v0.3 の fixture SPIR-V 経路へ下げるシミュレーションを追加。

### 正直な制限

- v0.3 の `opencuda-vulkan` はまだ実Vulkanではありません。
- OmniIR はまだ `vector_add_f32` のみです。
- `compile_omniir_to_spirv_fixture` は本物のSPIR-V生成器ではなく、パイプライン契約を固定するためのfixture生成器です。

次の v0.3 で、`ash` による Vulkan Instance / Device / Queue / Buffer / ShaderModule / ComputePipeline / Dispatch の最小実装へ進めます。
