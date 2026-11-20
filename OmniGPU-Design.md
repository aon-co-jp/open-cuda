# OmniGPU 設計書

**Universal GPU Runtime — Write Once, Run Anywhere**

CUDA互換 × Rust製 × NVIDIA / AMD / Intel 全対応

- ライセンス: Apache-2.0
- 言語: Rust (edition 2021)
- 最初のターゲットバックエンド: **Vulkan Compute**（全GPU共通・最優先）
- 検証環境: RTX 4090 24GB + RX 7900 XTX 24GB（48GB統合）

---

## 0. このプロジェクトの立ち位置

「AMD GPUでもNVIDIA GPUでも同じコードが動く、CUDA完全互換レイヤー」を作ることは技術的に可能だが、実質「第二のNVIDIA CUDA」を作る規模になる。本設計書は、それを**段階的に到達可能なマイルストーンへ分解**したもの。

到達点の定義を3段階に分けて考える。

| 互換レベル | 内容 | 難易度 | 本プロジェクトの方針 |
|---|---|---|---|
| ソースレベル互換 | CUDA/HIP/SYCLソースを共通IRに変換して各GPUで実行 | 中〜高 | 中核として実装 |
| バイナリレベル互換 | 既存のCUDAバイナリ（libcuda呼び出し）をフックしてAMD/Intelで実行 | 非常に高 | ZLUDA方式で段階導入 |
| 新規抽象化レイヤー | OmniGPUネイティブAPIを設計し全ベンダーを統一 | 中 | 最も現実的、最優先 |

Vulkan Compute を最初のバックエンドに選ぶ理由は、**1つの実装でNVIDIA / AMD / Intel / Apple(MoltenVK) すべてが動く**ため。ここで「全GPUで動く」骨格を最短で立てる。

---

## 1. アーキテクチャ全体図

```
┌─────────────────────────────────────────────────────────┐
│                   USER CODE LAYER                        │
│  CUDA C/C++  │  HIP  │  SYCL  │  OmniGPU Native API      │
└──────────────┴───────┴────────┴─────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              OMNIGPU FRONTEND LAYER                       │
│   CUDA Parser │ HIP Parser │ SYCL Parser                 │
│                    ↓                                      │
│        OmniIR（独自中間表現 / LLVM IR + SPIR-V拡張）      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              OMNIGPU RUNTIME CORE (Rust)                  │
│   Memory Mgr │ Scheduler │ Kernel Optimizer              │
│   (Malloc等)   (Stream/Event/Graph)  (Fusion/Tiling)     │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                  BACKEND LAYER                           │
│  NVIDIA(CUDA/PTX) │ AMD(ROCm/HIP) │ Intel(oneAPI/SYCL)   │
│  ───────── Vulkan Compute (全GPU共通フォールバック) ──── │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              MULTI-GPU ORCHESTRATION                     │
│   RTX 4090（主推論）+ RX 7900 XTX（補助）= 統合48GB      │
└─────────────────────────────────────────────────────────┘
```

Vulkan Compute は他のベンダー専用バックエンドの「下敷き」であり、専用バックエンド（CUDA/ROCm/oneAPI）が使えない環境では必ずここに落ちる。これにより「どこでも動く」が保証される。

---

## 2. Cargoワークスペース構成

```
omnigpu/
├── Cargo.toml                     # [workspace]
├── crates/
│   ├── omnigpu-core/              # Runtime Core: Device抽象, Memory, Scheduler
│   ├── omnigpu-ir/                # OmniIR 中間表現とパス
│   ├── omnigpu-frontend/          # CUDA / HIP / SYCL パーサー
│   ├── omnigpu-backend-cpu/       # ★最初の実行ターゲット: rayon マルチスレッド
│   ├── omnigpu-backend-vulkan/    # ★GPU最優先: Vulkan Compute バックエンド
│   ├── omnigpu-backend-nvidia/    # CUDA / PTX バックエンド
│   ├── omnigpu-backend-amd/       # ROCm / HIP バックエンド
│   ├── omnigpu-backend-intel/     # oneAPI / SYCL バックエンド
│   ├── omnigpu-compat/            # バイナリ互換層 (ZLUDA方式 APIフック)
│   ├── omnigpu-multidev/          # マルチGPU管理 / Pipeline並列 / 統合VRAM
│   └── omnigpu-blas/              # AIカーネル: GEMM / FlashAttention / Quantize
├── tools/
│   ├── omni-cc/                   # nvcc互換コンパイラドライバ
│   └── omni-profiler/             # nvprof互換プロファイラ
└── examples/
    ├── vector_add/                # 最小サンプル（まずCPU、次にVulkanで動作確認）
    ├── matmul/                    # 行列乗算
    └── llm_inference/             # aruaru-llm 統合 (Qwen3-14B)
```

### feature フラグ設計

```toml
[features]
default = ["cpu"]            # まずCPUで実際に動かす（GPU不要で検証可能）
cpu     = ["dep:rayon"]      # 16コア32スレッドCPUでカーネル実行
vulkan  = ["dep:ash", "dep:wgpu"]
nvidia  = ["dep:cudarc"]
amd     = []                  # hip-sys 等
intel   = []                  # oneAPI level-zero
all-backends = ["cpu", "vulkan", "nvidia", "amd", "intel"]
```

各バックエンドは feature で切り離す。まず `cpu` バックエンドで設計の正しさを実機検証し（GPU不要）、次に `vulkan` でドライバさえあればどのGPUでも動く状態を作る、という二段構えを最初のゴールにする。

---

## 3. コア抽象の定義（最も重要な契約）

すべてのバックエンドが実装する `GpuDevice` trait が、このプロジェクトの背骨になる。Phase 1 の設計詰めで以下4点を確定した。

**確定1: メモリ所有モデルは二層（内部生ポインタ + RAIIラッパー）**
**確定2: `DeviceBuffer` は `Arc<dyn GpuDevice>` を保持し Drop で自動解放**
**確定3: エラーは `anyhow::Result` 基調 + CUDA変換用の最小 `GpuError` enum**
**確定4: カーネル表現は `KernelSource` enum で多形式を保持（Phase 1 は Native と SpirV のみ実装）**

### 3.1 メモリの二層構造（確定1・2）

内部は生ポインタ `DevicePtr`、ネイティブAPIは RAII の `DeviceBuffer` で包む。`omnigpu-compat`（CUDA互換層）は生ポインタを直接扱い、ネイティブAPIユーザーは安全な `DeviceBuffer` を使う。一粒で互換層と安全APIの両方をまかなう。

```rust
/// 内部・生ポインタ層（compat層とバックエンドが使う）
/// device_id を埋め込むことで「どのGPU上のメモリか」を型で持つ。
/// RTX 4090 と RX 7900 XTX の混在時、ポインタだけ見れば所属GPUが分かり、
/// 取り違えをマルチGPUバグになる前に防げる。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DevicePtr {
    pub addr:      u64,
    pub device_id: u32,
}

/// ネイティブAPI層（ユーザーが使う、Dropで自動解放）
pub struct DeviceBuffer {
    ptr:    DevicePtr,
    len:    usize,
    device: Arc<dyn GpuDevice>,   // 解放先デバイスを保持（確定2）
}

impl DeviceBuffer {
    pub fn as_ptr(&self) -> DevicePtr { self.ptr }
    pub fn len(&self) -> usize { self.len }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // 解放失敗は握りつぶす（Dropはパニック不可）。
        // 重大なら別途ログ。
        let _ = self.device.free(self.ptr);
    }
}
```

`Arc` を持つぶんバッファはやや重いが、安全側に倒す。速度が要る高速経路は compat 層が `DevicePtr` を直接触れるので、そちらで稼ぐ。

### 3.2 エラー型（確定3）

ふだんは `anyhow` で気軽に書く。ただし compat 層は `cudaMalloc` 等が**整数のCUDAエラーコード**を返す義務があるため、変換が必要な代表的失敗だけ小さな enum に切り出す。`thiserror` を1個足すだけで、`anyhow` の軽さはほぼ保てる。

```rust
pub type Result<T> = anyhow::Result<T>;

/// CUDAコードへ変換が必要な代表的失敗のみ。
/// compat層で err.downcast_ref::<GpuError>() で拾い、cudaError_t に対応させる。
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("out of memory: requested {0} bytes")]
    OutOfMemory(usize),     // → cudaErrorMemoryAllocation (2)
    #[error("invalid device pointer")]
    InvalidPtr,             // → cudaErrorInvalidValue (1)
    #[error("no device found")]
    NoDevice,               // → cudaErrorNoDevice (100)
    #[error("kernel launch failed")]
    LaunchFailed,           // → cudaErrorInvalidDeviceFunction (8)
}
```

### 3.3 カーネル表現（確定4）

カーネルは「事前コンパイル済みバイナリ」と「実行時JIT」の両方をありうる。薄い enum で多形式を保持し、バックエンドが対応する形式だけ受け付ける。**Phase 1 では `Native`（CPU）と `SpirV`（Vulkan）だけ実装**し、`Ptx` と `OmniIr` のJITは Phase 2 に回す。enum なので後から足しても既存コードは壊れない。

```rust
pub enum KernelSource {
    SpirV(Vec<u8>),          // 事前コンパイル済み（Vulkan / Intel）
    Ptx(String),             // CUDA互換層から（NVIDIA）        ※Phase 2
    OmniIr(OmniModule),      // JIT変換用（全バックエンド共通）  ※Phase 2
    Native(NativeKernelFn),  // CPUバックエンド用 Rust 関数ポインタ
}

/// CPUバックエンドのカーネル本体。grid/block座標を受け取り1スレッド分を計算。
pub type NativeKernelFn =
    Arc<dyn Fn(ThreadCtx, &[DevicePtr]) + Send + Sync>;

/// カーネル起動時にバックエンドへ渡す実行単位の位置情報（CUDA threadIdx等に相当）
#[derive(Clone, Copy)]
pub struct ThreadCtx {
    pub block_idx:  (u32, u32, u32),
    pub thread_idx: (u32, u32, u32),
    pub block_dim:  (u32, u32, u32),
    pub grid_dim:   (u32, u32, u32),
}

pub struct CompiledKernel {
    pub name:   String,
    pub source: KernelSource,
    pub entry:  String,   // エントリ関数名
}
```

### 3.4 GpuDevice trait（背骨）

```rust
pub trait GpuDevice: Send + Sync {
    fn info(&self) -> &DeviceInfo;

    // メモリ管理（CUDA Runtime API互換のセマンティクス）
    fn alloc(&self, bytes: usize) -> Result<DevicePtr>;
    fn free(&self, ptr: DevicePtr) -> Result<()>;
    fn memcpy_h2d(&self, dst: DevicePtr, src: &[u8]) -> Result<()>;
    fn memcpy_d2h(&self, dst: &mut [u8], src: DevicePtr) -> Result<()>;
    fn memcpy_d2d(&self, dst: DevicePtr, src: DevicePtr, bytes: usize) -> Result<()>;

    // カーネル実行
    fn launch_kernel(&self, kernel: &CompiledKernel, cfg: &LaunchConfig) -> Result<()>;

    // 同期
    fn synchronize(&self) -> Result<()>;
}

/// DeviceBuffer を作るのはネイティブ層のヘルパー（Arc<dyn GpuDevice> を渡す）
pub fn alloc_buffer(device: &Arc<dyn GpuDevice>, len: usize) -> Result<DeviceBuffer> {
    let ptr = device.alloc(len)?;
    Ok(DeviceBuffer { ptr, len, device: Arc::clone(device) })
}

pub enum GpuVendor {
    Nvidia { compute_capability: (u32, u32) },
    Amd    { gfx_version: String },
    Intel  { architecture: String },
    Cpu,                                       // CPUバックエンド（確定4で追加）
    Unknown,
}

pub struct LaunchConfig {
    pub grid:  (u32, u32, u32),   // gridDim
    pub block: (u32, u32, u32),   // blockDim
    pub smem:  u32,               // shared memory bytes
}
```

この trait が固まっていれば、バックエンドは後から好きな順で足せる。**まず CPU 版を完成させて設計の正しさをGPUなしで検証**し、同じ trait で Vulkan → CUDA/ROCm/oneAPI を順次実装していく。

---

## 4. OmniIR（共通中間表現）

CUDA/HIP/SYCL のカーネルを一旦この中間表現に落とし、各バックエンドのコード（SPIR-V / PTX / AMDGPU-IR）へ下げる。

設計指針:

- ベースは LLVM IR の考え方を踏襲しつつ、GPU特有の概念（threadIdx, __syncthreads, shared memory, warp shuffle）を一級市民として持つ。
- 最初の出力ターゲットは **SPIR-V**（Vulkan Compute用）。これが動けば Intel oneAPI も SPIR-V を食えるので流用できる。

主要命令カテゴリ:

| カテゴリ | 命令例 | CUDA対応 |
|---|---|---|
| スレッド位置 | ThreadId / BlockId / BlockDim / GridDim | threadIdx, blockIdx 等 |
| メモリ | Load / Store（Global/Shared/Local/Constant） | グローバル/共有メモリ |
| 同期 | Barrier / WarpBarrier | __syncthreads() / warp sync |
| アトミック | AtomicAdd / AtomicCas | atomicAdd 等 |
| 数値 | FAdd / FMul / FMa（F16/BF16/F32/F64/INT8） | 浮動小数演算 |
| Warp | WarpShuffleDown / WarpVote | __shfl_down_sync 等 |
| 制御 | Branch / Jump / Return | 分岐 |

---

## 5. バイナリ互換層（ZLUDA方式）

既存のCUDAアプリ（`libcuda.so` / `cuda.dll` を呼ぶバイナリ）を**再コンパイルせず**にAMD/Intel GPUで動かすための層。

仕組み:

1. `libcuda` のシンボル（`cudaMalloc`, `cudaMemcpy`, `cudaLaunchKernel` 等）を OmniGPU が `#[no_mangle]` でエクスポートして差し替える。
2. 呼び出しを OmniGPU Runtime にリダイレクト。
3. カーネル（PTX）は JIT で OmniIR に変換し、Vulkan/ROCm/oneAPI で実行。

これは難易度が最も高く、CUDA APIの膨大な表面積を埋める作業になる。**Phase 2以降**に回し、まずは主要な十数個のAPIだけ実装して「簡単なCUDAバイナリが動く」ことを示すのが現実的なマイルストーン。

---

## 6. マルチGPU統合（RTX 4090 + RX 7900 XTX）

48GB統合VRAMの活用方針。aruaru-llm の Qwen3-14B 展開を具体例にする。

戦略: **Pipeline Parallelism（パイプライン並列）**

- モデルのTransformerレイヤーをVRAM容量比でデバイスに分割。
- 例: 全40層を VRAM比に応じて RTX 4090 と RX 7900 XTX に配分（24GB:24GB なので約20層ずつ）。
- 前段GPUの出力(activation)を後段GPUへ転送しながら順伝播。

注意点:

- NVIDIA↔AMD 間は NVLink/Infinity Fabric の直結が使えないため、転送は PCIe 経由（ホストメモリ経由のステージング）になる。ここがボトルネックになりやすいので、転送量の少ないパイプライン分割が向いている（テンソル並列より有利）。
- KV Cache は後段GPU側に置くと転送が減る。

---

## 7. AI最適化カーネル（omnigpu-blas）

各ベンダーの最速ライブラリを自動選択し、無ければVulkanで自前実装にフォールバックする。

| 演算 | NVIDIA | AMD | Intel | フォールバック |
|---|---|---|---|---|
| GEMM | cuBLAS | rocBLAS | oneMKL | Vulkan自前 |
| 畳み込み/Attention | cuDNN | MIOpen | oneDNN | Vulkan自前 |
| 集団通信 | NCCL | RCCL | oneCCL | PCIeステージング |

自前実装の優先順位は GEMM → Flash Attention → 量子化(INT4/INT8) の順。これが揃えばLLM推論の主要部分が動く。

---

## 8. 開発ロードマップ

各Phaseはローカル環境（RTX 4090 + RX 7900 XTX）での `cargo build` / 実機テストを前提とする。

### Phase 1（基盤・約3ヶ月）— まずCPUで動かし、次にVulkanで全GPUへ

- [ ] `omnigpu-core`: `GpuDevice` trait, `DevicePtr`/`DeviceBuffer`, `GpuError`, `KernelSource` 確定（4つの確定事項を実装）
- [ ] `omnigpu-backend-cpu`: rayon で `Native` カーネルをマルチスレッド実行（GPU不要・現行マシンで検証可能）
- [ ] `examples/vector_add`: **まずCPUバックエンドで正しい結果が出る**ことを確認
- [ ] `omnigpu-backend-vulkan`: ash/wgpu で `SpirV` カーネルを Vulkan Compute 実行
- [ ] vector_add / matmul が CPU と Vulkan の両方で同一結果を返す
- [ ] （将来）RTX 4090 と RX 7900 XTX の両方で同一バイナリが動くことを確認
- **当面の完了条件: GPUを買う前に、CPUバックエンドで設計の正しさを実証する**

### Phase 2（CUDA互換・約3ヶ月）

- [ ] `omnigpu-ir`: OmniIR 基本命令セット + SPIR-V出力
- [ ] `omnigpu-frontend/cuda-parser`: CUDA C++ サブセット → OmniIR
- [ ] `omnigpu-compat`: 主要CUDA API（malloc/memcpy/free/launch）フック
- [ ] `omnigpu-backend-nvidia` / `omnigpu-backend-amd` 実装
- **完了条件: 簡単なCUDAソースが無改造でAMD GPU上で動く**

### Phase 3（AI最適化・約3ヶ月）

- [ ] `omnigpu-blas`: GEMM / Flash Attention / 量子化
- [ ] `omnigpu-multidev`: Pipeline並列, 統合VRAM管理
- [ ] aruaru-llm 統合: Qwen3-14B を48GBに展開して推論
- **完了条件: 2枚のGPUにまたがってLLM推論が回る**

### Phase 4（Intel + エコシステム・約3ヶ月）

- [ ] `omnigpu-backend-intel`: oneAPI/SYCL
- [ ] `tools/omni-cc`: nvcc互換ドライバ
- [ ] PyTorch backend 対応の検討
- [ ] Apache-2.0 で公開、aruaruエコシステムのGPU基盤として位置づけ

---

## 8.5. ベンダー対応状況マトリクス（2026-07-25追記、正直な現状）

「INTEL＋AMD＋nVIDIA互換」という目標に対する、実際の仕組みと検証状況を
誤解の無いよう明記する。

| 層 | 実体 | 検証状況 |
|---|---|---|
| **Vulkan Compute統合（実働・主要な統合機構）** | `opencuda-vulkan`が`ash`経由でVulkan 1.x Computeを呼ぶ。ディスパッチ経路（`real.rs`の`launch_kernel`/`dispatch_matmul`相当）に**ベンダー分岐は一切無い**（`vendor_from_id`はデバイス情報の"報告"用途のみで、実行経路はSPIR-Vカーネルを渡すだけの単一コードパス）。Vulkan Computeに対応するGPUなら理論上NVIDIA/AMD/Intel/Qualcomm Adreno/ARM Mali/Imagination PowerVRのどれでも同じコードで動く。 | **実機検証はNVIDIA GeForce GT 730のみ**（このマシンの唯一のGPU、`vulkaninfo --summary`で確認、vendorID`0x10de`）。このマシンに統合Intel GPU等の第二のGPUは存在しない（`vulkaninfo`の`Devices`列挙がGPU0〈NVIDIA〉の1台のみ）ため、AMD/Intel/モバイルGPUでの実機Vulkan列挙・実行は**未検証**。 |
| **`GpuVendor`列挙（報告用の情報層）** | `opencuda-core::device::GpuVendor`に`Nvidia`/`Amd`/`Intel`/`Cpu`/`Unknown`に加え、2026-07-25に`Qualcomm`/`Arm`/`ImaginationPowerVr`を追加。`opencuda-vulkan::real::vendor_from_id`がPCI/VulkanベンダーID（`0x10DE`=NVIDIA、`0x1002`/`0x1022`=AMD、`0x8086`=Intel、`0x5143`=Qualcomm、`0x13B5`=ARM、`0x1010`=Imagination Technologies〈旧称VideoLogic〉、いずれもpci-ids.ucw.cz/Web検索で裏取り済み）から変換する。 | ID→列挙の変換ロジック自体は`cargo test -p opencuda-core`ではテスト対象外の純関数（既存の実機テストはNVIDIAの実IDでしか通らない）。追加した3ベンダーのマッチ分岐は型チェック・ビルド成功のみ確認、実機Qualcomm/ARM/Imagination GPUでの検証手段はこのマシンには無い。 |
| **ベンダー専用最適化ライブラリ経路（cuBLAS/rocBLAS/oneMKL）** | `opencuda-blas::select_gemm_path`が`GemmPath::CuBlas`/`RocBlas`/`OneMkl`を返しうるが、いずれも実装はスタブ（`sgemm`に渡すと未実装エラー）。実機でVulkan経由アクセスの場合は`GpuDevice::supports_spirv()`により自動的に動く`VulkanGeneric`へフォールバックする設計（2026-07-22実装済み）。 | **検証不能、既知の限界のまま**——このマシンにはCUDA Toolkitのみ存在し（`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`確認済み）、AMD ROCm・Intel oneAPI/oneMKLは未インストール。ROCm/oneMKLを実際にコンパイル・検証する手段がこのマシンには無いため、これらのスタブを「動く」と主張することはしない。 |

**結論（誇張を避けるための明記）**: 「Intel/AMD/nVIDIA互換の統合」という
目標に対する**実際に機能している統合機構はVulkan Computeであり、これは
ベンダー中立設計としてすでに実装済み**。ベンダー専用ライブラリ層は
将来の追加最適化であって、統合そのものの前提条件ではない。今回追加した
`GpuVendor`のQualcomm/ARM/Imagination対応は、Android/モバイルGPUという
より広いベンダー分類をVulkan統合の枠組みへ正しく載せるための土台整備
であり、実機検証済みの新機能ではない。

---

## 9. 現実的なリスクと正直な見積もり

- **規模**: フル機能（真のCUDA完全互換）は1人で5〜10年、10人チームで2〜4年。GPUベンダークラスの仕事量。本ロードマップは「完全互換」ではなく「実用的に動くサブセット」を各Phaseのゴールにしている。
- **CUDA APIの表面積**: 数千の関数・型がある。全部は埋めない。LLM推論に必要な経路から埋める。
- **NVIDIA↔AMD混在の転送**: PCIe経由でしか繋がらず帯域が制約。分割戦略で吸収する。
- **ドライバ依存**: cuBLAS等のクローズドライブラリはバイナリ互換層から直接は呼べない。ソース互換経路では各ベンダーライブラリを正規に呼ぶ。

最短で価値が出るのは **Phase 1 の Vulkan バックエンド**。ここだけで「1つのコードがNVIDIA/AMD両方で動く」というプロジェクトの核心が実証できる。

---

## 10. 命名・メタ情報

```toml
[workspace.package]
name    = "omnigpu"
version = "0.1.0"
edition = "2021"
license = "Apache-2.0"
authors = ["PHI"]
# tagline: Universal GPU Runtime — Write Once, Run Anywhere
```

名前候補: **OmniGPU** / Universal CUDA / OpenCUDA / Aruaru CUDA / CrossFire CUDA

---

## 11. クロスベンダー × クロスOS 移植性設計(2026-09-03、一次資料調査に基づく)

ユーザー指示「NVIDIA・AMD・Intel のどの GPU でも互換性を保ち、Windows・
macOS・Linux・Unix でも機能するよう、世界中の言語で Google/GitHub を調査
して最新理論・設計思想を実装へ活かす」への対応。**この節は §8.5 の
ベンダーマトリクスを OS 軸へ拡張し、2025 年時点の一次資料で裏を取った
設計判断をまとめる。**

### 11.1 一次資料調査(2025〜2026、英語)

- **「Rust running on every GPU」**([rust-gpu.github.io, 2025-07-25](https://rust-gpu.github.io/blog/2025/07/25/rust-on-every-gpu/))
  ——**⚠️ 2025-10-31 に `EmbarkStudios/rust-gpu` はアーカイブ(読み取り専用)化**。
  デモの結論(単一 Rust ソース → SPIR-V/naga で全ベンダー・全 OS)は
  概念として有効だが、**メンテされている実装経路は CubeCL と wgpu+naga**
  へ移った(2026-09 更新、§11.6)。
  **単一の Rust コードベース**を、`rustc_codegen_nvvm`(→ PTX を CPU
  バイナリへ埋め込み、実行時に CUDA ドライバへ)と `rustc_codegen_spirv`
  (→ SPIR-V)でビルド時に GPU バイナリ化し、**CUDA(NVIDIA)/ SPIR-V
  (Vulkan = AMD・Intel・NVIDIA・Android)/ Metal(Apple)/ DirectX 12
  (Windows)/ WebGPU(ブラウザ)/ CPU フォールバック**を賄うデモ。
  実行時は埋め込んだ SPIR-V を **`naga`** に渡してプラットフォームの
  シェーディング言語へ翻訳。**「カーネルは素の Rust なので GPU 無しの
  CI でロジックをテストできる」**——open-cuda が `KernelSource::Native`
  (Rust クロージャ)+ CPU バックエンドでやっているのと同じ思想。
- **CubeCL**([tracel-ai/cubecl](https://github.com/tracel-ai/cubecl)、
  Burn ML フレームワークの計算バックエンド):`#[cube]` 注釈した Rust
  関数を JIT で **CUDA / HIP(ROCm)/ Metal / SPIR-V / WGSL / CPU-SIMD**
  へ降ろす。`wgpu`/`cudarc` のような低レベルラッパと Burn のような高レベル
  フレームワークの**中間層**。2025 年時点の「Rust で書いて全ベンダーで
  走らせる」の実質的な最先端。
- **MoltenVK**([KhronosGroup/MoltenVK](https://github.com/KhronosGroup/MoltenVK)):
  Vulkan 1.4 のほぼ完全なサブセットを **Apple Metal の上に実装**
  (macOS/iOS/tvOS)。SPIR-V → MSL 変換器を**ランタイムに内蔵**。
  Compute シェーダ対応。→ **`opencuda-vulkan` は新規コード無しで macOS
  でも動く**(Vulkan ローダが `libMoltenVK` を ICD として拾えばよい)。
- **FP8 の移植性**([VK_EXT_shader_float8](https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_shader_float8.html)、
  [VK_KHR_cooperative_matrix](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_cooperative_matrix.html)):
  `VK_EXT_shader_float8`(E4M3/E5M2、OCP 準拠)は **`VK_KHR_cooperative_matrix`
  と組み合わせて ML 用に設計**されている。NVIDIA Turing 以降 / Qualcomm
  (`VK_QCOM_cooperative_matrix_conversion`)/ AMD で対応が広がる。
  → §10 残課題の「ベンダー FP8 GEMM」の**移植性の高い実装先は
  cuBLASLt/hipBLASLt ではなく SPIR-V + これらの Vulkan 拡張**。

### 11.2 open-cuda が既に正しくやっていること

| 設計判断 | 一次資料の裏付け | open-cuda の現状 |
|---|---|---|
| **SPIR-V を移植性カーネル IR にする** | rust-gpu / CubeCL / wgpu すべて SPIR-V を中核に置く | `KernelSource::SpirV`、`opencuda-vulkan` が単一コードパスでディスパッチ(§8.5) |
| **CPU を第一級バックエンドにして GPU 無し CI を可能に** | rust-gpu「カーネルは素の Rust なので GPU 無し CI でテストできる」 | `KernelSource::Native` + `opencuda-cpu`、全 crate のロジックテストが GPU 無しで走る |
| **ベンダー ID は"報告"に留め、実行経路に分岐を入れない** | wgpu/Vulkan の設計そのもの | `vendor_from_id` は報告用途のみ(§8.5) |
| **能力フラグで機能を交渉する** | Vulkan の `VkPhysicalDeviceFeatures` 方式 | `supports_spirv` / `supports_dxil` / `supports_fp8_tensor_core`(2026-09-02 追加) |

### 11.3 クロスOS マトリクス(目標と現状)

| OS | GPU 到達手段 | open-cuda の状態 | 必要作業 |
|---|---|---|---|
| **Linux** | Vulkan ICD(mesa RADV / ANV / NVIDIA)、`libvulkan.so` | `opencuda-vulkan`(`ash` `loaded` feature)で動く。NVIDIA GT 730 で実機検証済み | AMD/Intel 実機での列挙・実行検証(このマシンに無い) |
| **Windows** | Vulkan(`vulkan-1.dll`)+ **D3D12/DXIL フォールバック** | Vulkan・DirectX 両方実機検証済み(GT 730) | 統合 GPU 実機での再測 |
| **macOS / iOS** | **MoltenVK**(Vulkan→Metal、SPIR-V→MSL 内蔵) | **未検証**。`ash` の `loaded` は `libMoltenVK.dylib` を探せる設計なので**コード変更はおそらく不要** | (a) `MoltenVK` を同梱 or Vulkan SDK 依存を明記、(b) 実機 Mac で `vulkan_info` / `matmul_vulkan_real` をクロスビルド・実行(Android で 2026-08-15 にやったのと同じ手順)、(c) Portability Subset(`VK_KHR_portability_subset`)で欠ける機能(一部の `storageBuffer` レイアウト等)をカーネル側で回避 |
| **その他 Unix**(FreeBSD 等) | mesa Vulkan が動く範囲で Linux と同じ | 未検証 | ベストエフォート(mesa 次第) |
| **ブラウザ / WASM** | WebGPU | 対象外(§0 の立ち位置。必要なら `wgpu` バックエンドを別 crate で) | — |

### 11.4 設計判断(この調査を実装へ活かす)

1. **Vulkan + SPIR-V を「移植性の背骨」と正式に位置づける**。CUDA(PTX)/
   ROCm(HIP)/ oneAPI(Level Zero)/ Metal(MSL)は**任意の高速化経路**で
   あって前提ではない、と §8.5 の結論を OS 軸へ拡張して明文化した(本節)。
2. **macOS 対応は「新バックエンドを書く」のではなく「MoltenVK 経由の
   Vulkan を検証する」**。`opencuda-directx` 相当の新規 crate は不要。
   `README-VULKAN.md` に「macOS では Vulkan SDK / MoltenVK を入れれば
   `opencuda-vulkan` がそのまま動く(検証待ち)」を追記する。
3. **FP8 ベンダー GEMM(§10 残課題)の移植性実装先を訂正**:
   `sgemm_fp8_weight_vendor` の本命は、cuBLASLt/hipBLASLt/oneDNN の
   ベンダー分岐ではなく、**SPIR-V compute + `VK_EXT_shader_float8` +
   `VK_KHR_cooperative_matrix`**。能力フラグ `supports_fp8_tensor_core`
   はそのまま「Vulkan がこの 2 拡張を報告するか」で立てられる。
   (`device.rs` / blas スタブ文言は 2026-09-03 に更新済み。)
4. **CubeCL / rust-gpu の採用可否**: どちらも「open-cuda を置き換える」
   候補になり得るが、open-cuda は既に SPIR-V/DXIL/Native の 3 IR + CPU
   第一級 + 能力交渉という**同じ設計原則**で動いており、外部依存を
   増やさず現状路線を進めるのが妥当(§9 の「実用的サブセット」方針)。
   ただし `naga`(wgpu の SPIR-V↔MSL/WGSL/HLSL 翻訳器)は、将来
   macOS ネイティブ Metal や WebGPU へ広げる際の**第一候補**として記録。

### 11.5 正直な現状(誇張しない)

- **実機検証は依然 NVIDIA GT 730(Kepler)1 台のみ**。AMD/Intel/Apple/
  モバイル GPU での実行検証手段はこの開発機に無い。
- 本節は**設計の明文化と一次資料の裏取り**であり、macOS/AMD/Intel
  実機で動くことを新たに実証したものではない。
- FP8 の Vulkan 拡張経路(`VK_EXT_shader_float8` + cooperative matrix)は
  GT 730(Turing 未満)が非対応のため、この機ではコンパイル検証すら
  できない。

### 11.6 2026-09 更新(世界中の言語で Google/GitHub 再調査)

ユーザー指示「世界中の言語で Google/GitHub を再調査してから記録」。
英・日・中で再検索した結果、§11.1〜11.5 の設計方針は**変更不要**だが、
以下の一次資料の更新を反映する。

- **`EmbarkStudios/rust-gpu` は 2025-10-31 にアーカイブ(読み取り専用)化**
  ([rust-gpu ecosystem](https://rust-gpu.github.io/ecosystem/)、
  [HN 44692876](https://news.ycombinator.com/item?id=44692876))。
  「単一 Rust ソース → 全 GPU」の**メンテされている実装は
  [`tracel-ai/cubecl`](https://github.com/tracel-ai/cubecl)**(`#[cube]` →
  CUDA/HIP/Metal/SPIR-V/WGSL/CPU-SIMD、comptime 特殊化 + autotune +
  tensor-core 自動経路、Burn の計算バックエンド)と **wgpu+`naga`**。
  → §11.4-4 の「naga は将来の第一候補」を「**CubeCL と naga の 2 択**、
  ただし open-cuda は既に同じ設計原則(SPIR-V/DXIL/Native 3 IR + CPU
  第一級 + 能力交渉)なので外部依存を増やさず現状路線」へ更新。
- **`VK_EXT_shader_float8`(E4M3/E5M2)は 2026 時点で出荷ドライバ入り**:
  NVIDIA は 2025-06-08 ドライバ(Windows 573.38 / Linux 570.123.18)以降、
  **AMD は Adrenalin 25.10.2(2025-10-29)以降**で対応
  ([VK_EXT_shader_float8 proposal](https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_shader_float8.html)、
  [Khronos SIGGRAPH 2025](https://www.khronos.org/blog/vulkan-continuing-to-forge-ahead-siggraph-2025))。
  Intel の対応状況は未確認。→ §11.4-3 の「FP8 ベンダー GEMM の移植性
  実装先 = SPIR-V + Vulkan 拡張」は**もはや理論値ではなく、NVIDIA
  Ada/Blackwell + AMD RDNA4 の実ドライバで動く前提**として格上げ。
  `sgemm_fp8_weight_vendor` を実装する GPU が入手できたら、cuBLASLt
  ではなく **`VK_EXT_shader_float8` + `VK_KHR_cooperative_matrix` の
  SPIR-V compute** を第一実装とする。
- **`VK_KHR_cooperative_matrix`(ベンダー中立)** に加え、Vulkan 1.4.342
  で **`VK_QCOM_cooperative_matrix_conversion`**(shared memory を介さず
  cooperative matrix をロード/ストア)が追加
  ([Phoronix](https://www.phoronix.com/news/Vulkan-1.4.342-Released))。
  llama.cpp の Vulkan バックエンドは `VK_KHR_cooperative_matrix` +
  `VK_NV_cooperative_matrix2` + `VK_KHR_shader_integer_dot_product` +
  `VK_KHR_shader_bfloat16` を使う
  ([FOSDEM 2026: Vulkan API for ML](https://philpax.me/notes/talks/other-people/fosdem-2026/vulkan-api-for-machine-learning-competing-with-cuda-and-rocm-in-llamacpp/))。
  → open-cuda の Vulkan GEMM/Attention を tensor-core 相当へ最適化する
  際の拡張リストの正本。
- **llama.cpp は 2026-04 に「バックエンド非依存のテンソル並列」を導入**
  (演算単位で複数 GPU へ分割、ベンダーロックイン無し)。
  → §6「マルチGPU統合」を将来拡張する際、ベンダー混在(NVIDIA+AMD)の
  テンソル並列は SPIR-V 単一カーネル + デバイス列挙だけで組める、という
  裏付け。
- **DXIL→SPIR-V は 2026-02 に production SM 6.9 到達**
  ([dxil-spirv](https://github.com/HansKristian-Work/dxil-spirv)、
  [DXC Vulkan interop 2026](https://www.huuphan.com/2026/03/directx-shader-compiler-7-massive.html))。
  DXBC(SM4/5)も `dxbc-spirv` で扱える。→ §12.3 の「open-directx =
  DXBC/DXIL→SPIR-V フロントエンド」役割再定義の実現性が確定。
- **WebGPU は W3C Candidate Recommendation Draft(2026-05-21)**。
  WebLLM は同一デバイスでネイティブ比 ~80%、埋め込み生成は WebGPU が
  WASM 比 40〜75×
  ([WebLLM arXiv:2412.15803](https://arxiv.org/html/2412.15803v2)、
  [Llamas on the Web arXiv:2605.20706](https://arxiv.org/html/2605.20706v1))。
  → aruaru-llm の「WebGPU/wasm は将来オプション」(§12.3)の裏付け更新。

- **`naga`(wgpu の翻訳器)** は WGSL/SPIR-V を入力に取り、**SPIR-V /
  MSL / GLSL / HLSL / DXIL** を出力する(GitHub 調査で golden 出力
  parity: SPIR-V 87/87・MSL 91/91・HLSL 72/72 等)。→ open-cuda が
  将来 macOS ネイティブ Metal(MoltenVK を挟まない)や WebGPU へ広げる
  なら、`SpirV` カーネルを `naga` に通して MSL/WGSL を得るのが最短。
  新 IR は増やさない(§12.2-1)。
- **Intel**: `VK_EXT_shader_float8` の対応は未確認だが、Arc/Xe は
  Vulkan と SYCL の両方で llama.cpp が動く実績があり、Linux の
  オープンソース compute スタック(`intel/compute-runtime`、Level Zero)
  は成熟してきている
  ([Phoronix: Arc compute Q1](https://www.phoronix.com/review/arc-graphics-compute-q1))。
  → open-cuda は Intel も **Vulkan/SPIR-V の同一経路**で扱えばよい
  (Level Zero ネイティブ経路は "任意の高速化" 扱い、§12.2-1)。

**結論(誇張しない)**: 設計方針(§11.4)は 1 行も撤回しない。変わったのは
「rust-gpu が非メンテ → CubeCL/naga が後継」「Vulkan FP8 が理論 → 実
ドライバ(NVIDIA・AMD)」の 2 点で、いずれも **open-cuda の現状路線
(SPIR-V 背骨 + Vulkan 拡張で FP8)を強化する方向**。実機検証手段
(AMD/Intel/Apple/FP8 対応 GPU)がこの開発機に無い点は不変。

---

## 12. エコシステム横断の設計見直し(2026-09-03、dream-os / open-directx / open-cuda / aruaru-llm / aruaru-db)

ユーザー指示「§11 の調査を次の作業(dream-os・open-directx・open-cuda・
aruaru-llm・aruaru-db の新規設計・新規実装の見直し)に活かす」への対応。
§11 の一次資料に加え、DirectX 互換層 / LLM 推論エンジンの移植性設計を
追加調査し、**5 リポジトリの役割を「Vulkan + SPIR-V の単一移植性コア」に
収斂させる方針**をまとめる。実装はこの節の方針に沿って順次進める
(この節自体はコード変更を含まない設計文書)。

### 12.1 追加の一次資料調査(2025〜2026、英語)

- **vkd3d-proton / DXVK 3.0**([DXIL to SPIR-V, DeepWiki](https://deepwiki.com/HansKristian-Work/vkd3d-proton/4.2-dxil-to-spir-v)、
  [DXVK 3.0, Phoronix](https://www.phoronix.com/news/DXVK-3.0-Release)):
  DXVK と vkd3d-proton は**共通の DXBC フロントエンド**を持ち、DXBC(SM4/5)も
  DXIL(SM6)も外部ライブラリ **`dxil-spirv`** 経由で SPIR-V へ変換する。
  DXVK 3.0 の `DXBC-SPIRV` は D3D SM5.1+ 向けの **SSA ベースコンパイラ**で、
  ネイティブ翻訳より**コンパクトな SPIR-V** を吐き、翻訳はワーカースレッドへ
  オフロードされる。→ **DirectX シェーダを "並行バックエンド" にするのは
  もう筋が悪い。DXBC/DXIL → SPIR-V → 単一 Vulkan 経路が実証された道。**
- **llama.cpp のバックエンド行列**([Red Hat Developer, 2026](https://developers.redhat.com/articles/2026/06/15/llamacpp-vs-vllm-choosing-right-local-llm-inference-engine)):
  Metal(Apple)/ AVX・AVX2・AVX512・AMX(x86)/ RISC-V / CUDA / HIP(AMD)/
  MUSA / **Vulkan** / SYCL(Intel)。CUDA/Vulkan では int8 活性化量子化も。
  「互換性重視の推論エンジン(llama.cpp / MLC-LLM)はヘテロな個人デバイスと
  バックエンドへの互換性に集中する」——aruaru-llm と同じ立ち位置。
- **「Llamas on the Web」**([arXiv:2605.20706](https://arxiv.org/pdf/2605.20706)):
  **WebGPU** で省メモリ・性能移植性・多精度の LLM 推論。WebGPU/wgpu が
  実運用の LLM 推論に耐えることを示した(将来 aruaru-llm を wasm/WebGPU へ
  広げる際の裏付け)。
- **MLC-LLM**: ML コンパイル(TVM)で "一度コンパイルしてどこでも実行"。

**2026-09 更新(世界中の言語で再調査、§11.6 と対)**: (1) `rust-gpu` 非
メンテ化により「単一 Rust ソース → 全 GPU」のメンテ実装は **CubeCL /
wgpu+naga**。(2) `VK_EXT_shader_float8` が NVIDIA(2025-06)/ AMD
Adrenalin 25.10.2(2025-10)の**出荷ドライバ**へ。(3) llama.cpp が
2026-04 に**バックエンド非依存のテンソル並列**(ベンダー混在可)を導入。
(4) `dxil-spirv` が 2026-02 に **production SM 6.9**。(5) WebGPU が W3C
**Candidate Recommendation Draft**(2026-05)。→ いずれも §12.2〜12.3 の
方針を**強化する方向**で、撤回・変更は無い。

### 12.2 横断的な設計原則(全リポジトリ共通)

§11 と合わせ、以下を**エコシステム全体の設計哲学**として確定する。

1. **SPIR-V を唯一の実行時 GPU IR にする**。CUDA(PTX)/ HIP / Level Zero /
   Metal(MSL)は "任意の高速化経路"。並行して別 IR を実行系として持たない
   (rust-gpu / CubeCL / wgpu / vkd3d-proton すべてこの構造)。
2. **CPU を第一級バックエンドにして GPU 無し CI を可能にする**
   (`KernelSource::Native` + `opencuda-cpu`)。
3. **能力交渉**(`supports_spirv` / `supports_dxil` /
   `supports_fp8_tensor_core` / 将来 `supports_cooperative_matrix`)で
   機能を選び、ベンダー ID で実行経路を分岐しない。
4. **外向き互換を保ちつつ内部を刷新する**(aruaru-db の HLC P-HLC-3 で
   実証: u64 ワイヤ形式を維持したまま内部を案A へ全面移行)。他リポジトリの
   大きな設計変更にもこの原則を適用する。
5. **OS 差は "Vulkan ローダをどう見つけるか" に閉じ込める**
   (Linux: mesa/proprietary ICD、Windows: `vulkan-1.dll`、macOS/iOS:
   MoltenVK、その他 Unix: mesa)。OS 依存の実行経路分岐を書かない。

### 12.3 リポジトリ別の見直し方針

| リポジトリ | 現状 | 見直し方針(§11/§12 を活かす) |
|---|---|---|
| **open-cuda** | `KernelSource` に `Native` / `SpirV` / `Dxil` の 3 IR。Vulkan・DirectX 両バックエンド実機検証済み(GT 730)。CPU 第一級。能力フラグ 3 種。 | (a) **`Dxil` を "実行 IR" から降格**し、`dxil-spirv` 相当の DXIL→SPIR-V 変換を通して**単一 Vulkan 経路へ寄せる**(DirectX バックエンドは Vulkan が無い Windows 向けの純フォールバックに限定)。(b) **macOS 対応 = MoltenVK 経由 Vulkan の実機検証**(新バックエンドを書かない)。(c) FP8 ベンダー GEMM の実装先を **SPIR-V + `VK_EXT_shader_float8` + `VK_KHR_cooperative_matrix`** に確定(cuBLASLt/hipBLASLt 分岐は書かない)。(d) `naga` を将来の Metal ネイティブ / WebGPU 拡張時の翻訳器候補として記録。 |
| **open-directx** | DXBC/DXIL ↔ SPIR-V 変換 + Vulkan グラフィックス(独立リポジトリ。open-cuda 内蔵の `opencuda-directx` とは別物、§8.5 の混同注意)。 | **役割を「並行 GPU バックエンド」から「DXBC/DXIL → SPIR-V フロントエンド」へ再定義**。vkd3d-proton/DXVK が実証した通り、DirectX 由来のシェーダ・OS レベルのグラフィックス命令を **SPIR-V へ翻訳して open-cuda の Vulkan 経路へ流す**のが本筋。`dxil-spirv` の設計(共通 DXBC フロントエンド + SSA IR)を参照実装として、翻訳品質・コンパクトさを目標にする。 |
| **dream-os** | SBM(`sbm_ising`)・マイニング相当(`sha256d_mine`)カーネルは既に SPIR-V(open-cuda 経由)。OS レベルの GPU 利用構想。 | dream-os 固有の GPU 移植性コードは**追加しない**。open-cuda の `GpuDevice` trait + SPIR-V + 能力交渉を**そのまま継承**する。OS レベルのスケジューリング(どのプロセスにどれだけ GPU を割り当てるか)は dream-os の責務だが、実行経路は open-cuda に委譲。東芝 SBM の第3世代アルゴリズム("edge of chaos")は SPIR-V カーネルの改良として取り込める(実行系の変更は不要)。 |
| **aruaru-llm** | CPU-SIMD(AVX2+FMA3、実測 3.34x)/ Vulkan / DXIL 済み。`ARUARU_LLM_*` env で FP8・MLA 等を opt-in。 | (a) **バックエンド行列を llama.cpp に倣って明示**(CPU-SIMD=済 / Vulkan=済 / DXIL=済 / Metal=MoltenVK 経由で将来 / HIP・SYCL=optional)。(b) `ARUARU_LLM_ENABLE_*` の opt-in 群を、起動時の**能力交渉 → 自動選択**へ寄せる(env は override として残す)。(c) int8 活性化量子化(llama.cpp が CUDA/Vulkan で採用)を `opencuda-blas` の `dot_i8`(VNNI 経路、実装済み・未配線)と繋ぐ検討。(d) WebGPU 経路は「Llamas on the Web」を裏付けに、wasm ターゲットの将来オプションとして記録。 |
| **aruaru-db** | GPU とは直交。HLC は P-HLC-3 で CockroachDB 準拠のフル精度 2 フィールド + uncertainty interval へ。 | GPU 移植性の直接の対象外。ただし **§12.2-(4)「外向き互換を保ちつつ内部を刷新」は HLC P-HLC-3 が最初の実証例**であり、今後 aruaru-db が pgwire / GraphQL / Raft ワイヤ形式を保ったまま内部を刷新する際の設計原則として明文化する(`docs/CONTROL_PLANE_REDESIGN.md` の宣言的設計と同じ精神)。 |

### 12.4 実装順(この方針を反映する次の作業)

1. **open-cuda**: macOS/MoltenVK の実機検証(Android クロスビルドと同じ手順、
   §11.3)。→ クロスOS の "検証済み" 欄を Linux/Windows から macOS へ拡張。
2. **open-cuda**: `dxil-spirv` 相当の DXIL→SPIR-V 経路を調査し、
   `opencuda-directx` を Vulkan フォールバック専用へ縮退させる設計 PR。
3. **aruaru-llm**: バックエンド自動選択(能力交渉)+ バックエンド行列の
   README 明記。
4. **open-directx**(独立リポ): 役割再定義を CLAUDE.md/README へ明記し、
   SPIR-V フロントエンド化のロードマップを書く。
5. **dream-os**: 「GPU 実行系は open-cuda に委譲、dream-os は OS レベルの
   割り当てのみ」を CLAUDE.md へ明記。

### 12.5 正直な現状・非目標(誇張しない)

- 実機検証は依然 **NVIDIA GT 730(Kepler)1 台のみ**。macOS / AMD / Intel /
  モバイル / FP8 対応 GPU での検証手段はこの開発機に無い。
- 本節は**設計方針の確定と一次資料の裏取り**。上記の実装順はまだ着手して
  いない(次の作業)。
- 「CUDA 完全互換」は非目標(§9)。目標は「1 つの Rust コードが SPIR-V
  経由で NVIDIA/AMD/Intel の Vulkan 対応 GPU 上を、Windows/macOS/Linux/
  Unix で動く」実用的サブセット。

## 13. 精度(F16/F32/F64/F128) × 32GB級 VRAM ベンダー対応(2026-09-03)

ユーザー指示「今後は NVIDIA(RTX)/AMD/Intel の 32GB VRAM 級カードを前提に、
F16/F32/F64 を見据えて開発する」+ 追って「F128 まで対応して」。本節は
その設計方針と、このマシン(GT 730、Kepler、2GB)では実機検証できない
ことの正直な線引きを記録する。

### 13.1 前提とする 32GB 級カード(既知の範囲、確認できないものは明記)

- **NVIDIA**: RTX PRO 6000 Blackwell(96GB、参考。32GB 級としては
  RTX 5090 32GB)。データセンター向け H100/H200 は 80GB/141GB クラスで
  今回の「32GB 級」の枠外だが同系アーキテクチャとして参考にする。
- **AMD**: Radeon PRO(Instinct系データセンター GPU、または Radeon AI
  PRO R9700 32GB)。§11.4-3 の FP8 調査で言及した RDNA4 世代。
- **Intel**: Arc Pro シリーズの高VRAM構成。**正直な開示**: Arc Pro の
  32GB 単体カード構成の存在は本セッション内で一次資料の裏取りができて
  おらず未確認のまま記録する(過大主張を避けるため)。
- **共通の正直な開示**: これらいずれの実機もこの開発機には無い
  (§12.5 と同じ制約)。本節は「今後この階級のカードが来た場合に
  どう設計を対応させるか」の設計文書であり、実機検証済みという主張は
  一切していない。

### 13.2 F16/F32/F64/F128 のベンダー対応(定性、数値は書かない)

| 精度 | NVIDIA | AMD | Intel | open-cuda の実装状態 |
|---|---|---|---|---|
| F16 | Pascal 以降ネイティブ、Volta 以降 Tensor Core | RDNA/CDNA でネイティブ(世代により速度差、要個別確認) | Xe でネイティブ(世代により速度差、要個別確認) | `KernelArg::F16`/`ResolvedArg::F16`(`half::f16`)追加済み。`opencuda-blas::hgemm`(CPU参照実装)追加済み。GPU ディスパッチ配線は未着手(次の増分)。 |
| F32 | 全世代ネイティブ | 全世代ネイティブ | 全世代ネイティブ | 既存 `sgemm`(CPU SIMD + Vulkan/DirectX 実機検証済み、GT 730)。 |
| F64 | CUDA Core でネイティブだが、コンシューマ機は F32 比で大幅低スループット(世代・製品階級で大きく異なる、具体的倍率は個別に確認要) | 同様にネイティブだが製品階級依存 | 同様にネイティブだが製品階級依存 | `KernelArg::F64`/`ResolvedArg::F64`追加済み。`opencuda-blas::dgemm`(CPU参照実装)追加済み。GPU ディスパッチ配線は未着手。 |
| F128 | **ネイティブ命令なし(NVIDIA/AMD/Intel いずれの製品にも存在しない)** | 同左 | 同左 | `opencuda_core::DoubleDouble`(double-double ソフトウェアエミュレーション、Dekker/Knuth 方式)+ `opencuda-blas::qgemm`。**GPU 加速ではない**、CPU 側の高精度演算専用。 |

### 13.3 F128(ソフトウェア四倍精度)の位置づけ

- Rust `std` に安定版 `f128` は無い(nightly の実験的プリミティブのみ、
  本番ビルド〈stable〉では使えない)。
- 2026年時点で FP128 をネイティブ実行できる GPU は NVIDIA/AMD/Intel の
  コンシューマ・データセンター製品いずれにも存在しない(このマシンに
  限らず、業界一般の事実として)。
- そのため `crates/opencuda-core/src/f128.rs` に **double-double(倍々
  精度)** を自前実装した(Dekker 1971 の `two_sum`/`two_prod`、Knuth
  `TAOCP` vol.2 と同系統のアルゴリズム、2つの `f64` の組で仮数部
  約106ビット相当)。`qd`/`twofloat` のような専用 crate を追加依存
  させず自己完結させた(オフライン環境でも `cargo build` が壊れない
  ようにするため)。
- **用途は性能ではなく数値的正確性**——Kahan和のような桁落ちに敏感な
  縮約(reduction)処理向け。`opencuda-blas::qgemm`(GEMM 参照実装)に
  加え、実測テスト(`dd_summation_is_more_accurate_than_plain_f64_
  for_ill_conditioned_sum`・`qgemm_is_more_accurate_than_dgemm_for_
  ill_conditioned_dot_product`)で、病的に条件の悪い和/内積において
  f64 単体より実際に精度が高い(このケースでは厳密値と完全一致する)
  ことを確認済み。

### 13.4 正直な現状・未着手項目

- `hgemm`/`dgemm`/`qgemm` はいずれも **CPU 参照実装のみ**。既存の
  `sgemm` が持つ `GpuDevice::launch_kernel` 経由の実 GPU ディスパッチ
  (CPU SIMD・Vulkan・DirectX の階層フォールバック)は、この3関数には
  まだ配線していない——「精度ごとの正しさをまず確立する」ことを優先し、
  GPU ディスパッチは次の増分として切り出した。
- F16 の GPU ネイティブ実行(NVIDIA Tensor Core 等)は、この開発機の
  GPU(GT 730、Kepler、FP16 Tensor Core 非搭載)では原理的に検証不可能。
  32GB 級の対象カードが入手できた場合にのみ再検討する。
- 32GB VRAM 級カードでの実機検証は一切行っていない(§13.1 参照)。
