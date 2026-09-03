//! カーネル表現（設計確定4）。
//!
//! カーネルは「事前コンパイル済みバイナリ」と「実行時JIT」の両方がありうる。
//! 薄い enum で多形式を保持し、各バックエンドは対応形式だけ受け付ける。
//!
//! Phase 1 では `Native`（CPU）と `SpirV`（Vulkan）だけ実装し、
//! `Ptx` と `OmniIr` のJITは Phase 2 に回す。enum なので後から足しても
//! 既存コードは壊れない。

use std::sync::Arc;

use crate::f128::DoubleDouble;
use crate::memory::DevicePtr;

/// カーネルの供給形式。
pub enum KernelSource {
    /// CPUバックエンド用 Rust 関数（Phase 1 実装）。
    Native(NativeKernelFn),
    /// 事前コンパイル済み SPIR-V（Vulkan / Intel, Phase 1 後半）。
    SpirV(Vec<u8>),
    /// CUDA 互換層からの PTX（Phase 2）。
    Ptx(String),
    /// JIT変換用の共通中間表現。v0.2 は opencuda-ir の最小fixtureを格納する。
    OmniIr(Vec<u8>),
    /// 事前コンパイル済み DXIL（DirectX 12 Compute、`opencuda-directx`向け、
    /// 2026-07-23追加)。DXBCコンテナ形式(先頭4バイトが`b"DXBC"`)を想定。
    Dxil(Vec<u8>),
}

impl KernelSource {
    pub fn kind(&self) -> &'static str {
        match self {
            KernelSource::Native(_) => "Native",
            KernelSource::SpirV(_) => "SpirV",
            KernelSource::Ptx(_) => "Ptx",
            KernelSource::OmniIr(_) => "OmniIr",
            KernelSource::Dxil(_) => "Dxil",
        }
    }
}

/// CPUバックエンドのカーネル本体。
/// 1スレッド分の計算を、スレッド位置 `ThreadCtx` と「解決済み引数」列で表す。
///
/// デバイスポインタはバックエンドが実メモリアドレス（`*mut u8`）に解決してから
/// `ResolvedArg::Ptr` として渡す。カーネル側は生ポインタを安全に扱う責務を負う
/// （CUDA カーネルが生ポインタを受け取るのと同じ立場）。
pub type NativeKernelFn = Arc<dyn Fn(ThreadCtx, &[ResolvedArg]) + Send + Sync>;

/// バックエンドが解決した後の引数（ポインタは実アドレス）。
#[derive(Clone, Copy, Debug)]
pub enum ResolvedArg {
    /// 実メモリの先頭アドレスと確保バイト数。
    Ptr { addr: *mut u8, len: usize },
    U32(u32),
    I32(i32),
    F32(f32),
    Usize(usize),
    /// 半精度(2026-09-03追加)。NVIDIA(Pascal以降のFP16、Volta以降の
    /// Tensor Core)/AMD(RDNA/CDNA)/Intel(Xe)いずれも世代によりネイティブ
    /// FP16演算を持つ——ただしこのGPU一覧の「ネイティブ対応」は本フィールド
    /// 自体の値ではなく、実行時に`GpuDevice`側の能力フラグで判定する
    /// (詳細は`OmniGPU-Design.md`の精度/ベンダー対応節を参照)。
    F16(half::f16),
    /// 倍精度(2026-09-03追加)。CUDA CoreはF64もネイティブ対応するが、
    /// コンシューマ向けGPUではF32比で大幅に遅い(1/32〜1/64スループット)
    /// 機種が多い——用途に応じA/B比較すること。
    F64(f64),
    /// ソフトウェアエミュレーション四倍精度(2026-09-03追加)。
    /// **どのGPUベンダーもFP128ハードウェアを持たない**ため、これは
    /// CPU側のdouble-double演算専用(`crate::f128::DoubleDouble`参照)。
    /// GPUカーネル引数として使う想定ではなく、ホスト側の高精度リダクション
    /// 用に`ResolvedArg`/`KernelArg`の枠組みへ揃えて追加した。
    F128(DoubleDouble),
}

// カーネルは1スレッドずつ呼ばれ、各スレッドは自分の担当インデックスだけ触る前提。
unsafe impl Send for ResolvedArg {}
unsafe impl Sync for ResolvedArg {}

impl ResolvedArg {
    pub fn as_ptr(&self) -> Option<(*mut u8, usize)> {
        match self {
            ResolvedArg::Ptr { addr, len } => Some((*addr, *len)),
            _ => None,
        }
    }
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            ResolvedArg::Usize(v) => Some(*v),
            ResolvedArg::U32(v) => Some(*v as usize),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            ResolvedArg::F32(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_f16(&self) -> Option<half::f16> {
        match self {
            ResolvedArg::F16(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ResolvedArg::F64(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_f128(&self) -> Option<DoubleDouble> {
        match self {
            ResolvedArg::F128(v) => Some(*v),
            _ => None,
        }
    }
}

/// カーネルへ渡す引数。スカラとデバイスポインタを区別する。
#[derive(Clone, Copy, Debug)]
pub enum KernelArg {
    Ptr(DevicePtr),
    U32(u32),
    I32(i32),
    F32(f32),
    Usize(usize),
    /// 半精度(2026-09-03追加、`ResolvedArg::F16`と対)。
    F16(half::f16),
    /// 倍精度(2026-09-03追加、`ResolvedArg::F64`と対)。
    F64(f64),
    /// ソフトウェア四倍精度(2026-09-03追加、`ResolvedArg::F128`と対)。
    /// GPUハードウェア対応が無いことの開示は`ResolvedArg::F128`参照。
    F128(DoubleDouble),
}

impl KernelArg {
    pub fn as_ptr(&self) -> Option<DevicePtr> {
        match self {
            KernelArg::Ptr(p) => Some(*p),
            _ => None,
        }
    }
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            KernelArg::U32(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            KernelArg::Usize(v) => Some(*v),
            KernelArg::U32(v) => Some(*v as usize),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            KernelArg::F32(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_f16(&self) -> Option<half::f16> {
        match self {
            KernelArg::F16(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            KernelArg::F64(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_f128(&self) -> Option<DoubleDouble> {
        match self {
            KernelArg::F128(v) => Some(*v),
            _ => None,
        }
    }
}

/// カーネル起動時の実行単位の位置情報（CUDA threadIdx / blockIdx 等に相当）。
#[derive(Clone, Copy, Debug)]
pub struct ThreadCtx {
    pub block_idx: (u32, u32, u32),
    pub thread_idx: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub grid_dim: (u32, u32, u32),
}

impl ThreadCtx {
    /// 1次元グローバルインデックス（CUDA の
    /// `blockIdx.x * blockDim.x + threadIdx.x` 相当）。
    pub fn global_id_x(&self) -> u32 {
        self.block_idx.0 * self.block_dim.0 + self.thread_idx.0
    }
    pub fn global_id_y(&self) -> u32 {
        self.block_idx.1 * self.block_dim.1 + self.thread_idx.1
    }
}

/// コンパイル済み（または供給済み）カーネル。
pub struct CompiledKernel {
    pub name: String,
    pub source: KernelSource,
    pub entry: String,
}

impl CompiledKernel {
    /// CPUネイティブカーネルを作る簡便コンストラクタ。
    pub fn native(
        name: impl Into<String>,
        f: impl Fn(ThreadCtx, &[ResolvedArg]) + Send + Sync + 'static,
    ) -> Self {
        let name = name.into();
        Self {
            entry: name.clone(),
            name,
            source: KernelSource::Native(Arc::new(f)),
        }
    }

    /// 事前コンパイル済み SPIR-V カーネルを作る簡便コンストラクタ。
    ///
    /// 実Vulkanバックエンドでは、この `source` を `VkShaderModule` に渡す。
    /// v0.1.1 の Vulkan Mock では、GPUなしで `SpirV` 経路だけを検証する。
    pub fn spirv(
        name: impl Into<String>,
        entry: impl Into<String>,
        bytes: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            name: name.into(),
            entry: entry.into(),
            source: KernelSource::SpirV(bytes.into()),
        }
    }

    /// OmniIR カーネルを作る簡便コンストラクタ。
    ///
    /// v0.2 では `opencuda-ir` が作る最小バイナリfixtureを保持する。
    pub fn omniir(
        name: impl Into<String>,
        entry: impl Into<String>,
        bytes: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            name: name.into(),
            entry: entry.into(),
            source: KernelSource::OmniIr(bytes.into()),
        }
    }

    /// 事前コンパイル済み DXIL カーネルを作る簡便コンストラクタ。
    ///
    /// 実DirectX 12バックエンドでは、この `source` を`ID3D12PipelineState`
    /// (Compute PSO)に渡す。`opencuda-directx`のモック実装では、GPUなしで
    /// `Dxil`経路だけを検証する(Vulkanの`VulkanMockDevice`と同じ設計)。
    pub fn dxil(
        name: impl Into<String>,
        entry: impl Into<String>,
        bytes: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            name: name.into(),
            entry: entry.into(),
            source: KernelSource::Dxil(bytes.into()),
        }
    }

    pub fn source_kind(&self) -> &'static str {
        self.source.kind()
    }
}
