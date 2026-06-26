//! # opencuda-blas
//!
//! LLM推論に必要な高レベルカーネルを提供する（Phase 3）。
//!
//! 設計方針: 各ベンダーの最速ライブラリ（cuBLAS / rocBLAS / oneMKL）を
//! 自動選択し、無ければ汎用カーネル（CPU / Vulkan）にフォールバックする。
//! LLM推論は GEMM と Attention が計算時間の大半を占めるため、この少数の
//! カーネルを各バックエンドで最適化することが「実用的なフル機能」への近道。
//!
//! Phase 1 時点ではディスパッチの骨格のみ。中身は段階的に実装する。

use opencuda_core::{GpuDevice, GpuVendor, Result};

/// GEMM のバックエンド選択。ベンダーごとに最速経路へ振り分ける。
pub fn select_gemm_path(device: &dyn GpuDevice) -> GemmPath {
    match &device.info().vendor {
        GpuVendor::Nvidia { .. } => GemmPath::CuBlas,
        GpuVendor::Amd { .. } => GemmPath::RocBlas,
        GpuVendor::Intel { .. } => GemmPath::OneMkl,
        GpuVendor::Cpu => GemmPath::CpuNaive,
        GpuVendor::Unknown => GemmPath::VulkanGeneric,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmPath {
    CuBlas,         // NVIDIA   (Phase 2/3)
    RocBlas,        // AMD      (Phase 2/3)
    OneMkl,         // Intel    (Phase 4)
    VulkanGeneric,  // 汎用     (Phase 1 後半)
    CpuNaive,       // CPU      (Phase 1, examples/matmul に実装済みの形)
}

/// 単精度 GEMM: C = alpha * A·B + beta * C。
///
/// TODO(Phase 3): 各 path の実装を埋める。現状はディスパッチのみ。
pub fn sgemm(device: &dyn GpuDevice, _m: usize, _k: usize, _n: usize) -> Result<()> {
    let path = select_gemm_path(device);
    tracing::debug!("sgemm path = {path:?}");
    match path {
        GemmPath::CpuNaive => {
            // examples/matmul と同じ naive カーネルをここに移植予定。
            anyhow::bail!("sgemm: CpuNaive not yet wired into blas crate (see examples/matmul)")
        }
        other => anyhow::bail!("sgemm: {other:?} backend not yet implemented (Phase 3)"),
    }
}

/// Flash Attention（Phase 3）。online softmax + タイル化で実装予定。
pub fn flash_attention(_device: &dyn GpuDevice) -> Result<()> {
    anyhow::bail!("flash_attention: not yet implemented (Phase 3)")
}

/// 量子化（INT4 / INT8、Phase 3）。aruaru-llm の Q4_K_M 系に対応予定。
pub fn quantize_int4(_device: &dyn GpuDevice) -> Result<()> {
    anyhow::bail!("quantize_int4: not yet implemented (Phase 3)")
}
