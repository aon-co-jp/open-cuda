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
