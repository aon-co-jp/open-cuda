//! エラー型。
//!
//! 方針（設計確定3）: ふだんは `anyhow` で気軽に書く。ただし `opencuda-compat`
//! （CUDA互換層）は `cudaMalloc` 等が整数のCUDAエラーコードを返す義務があるため、
//! 変換が必要な代表的失敗だけ `GpuError` enum に切り出す。
//!
//! compat層では `err.downcast_ref::<GpuError>()` でこの数種だけ拾い、
//! `cudaError_t` に対応させる。

/// ライブラリ全体の Result 型。
pub type Result<T> = anyhow::Result<T>;

/// CUDAコードへ変換が必要な代表的失敗のみを表す。
///
/// 右側コメントは将来の compat 層での対応先 `cudaError_t`。
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("out of memory: requested {0} bytes")]
    OutOfMemory(usize), // → cudaErrorMemoryAllocation (2)

    #[error("invalid device pointer: {0:?}")]
    InvalidPtr(crate::memory::DevicePtr), // → cudaErrorInvalidValue (1)

    #[error("no device found")]
    NoDevice, // → cudaErrorNoDevice (100)

    #[error("kernel launch failed: {0}")]
    LaunchFailed(String), // → cudaErrorInvalidDeviceFunction (8)

    #[error("kernel source not supported by this backend: {0}")]
    UnsupportedKernel(&'static str),

    #[error("device id {requested} out of range (have {available} devices)")]
    DeviceOutOfRange { requested: usize, available: usize },
}

impl GpuError {
    /// 将来の compat 層用: CUDA ランタイムのエラーコードへ変換する。
    pub fn to_cuda_code(&self) -> i32 {
        match self {
            GpuError::OutOfMemory(_) => 2,
            GpuError::InvalidPtr(_) => 1,
            GpuError::NoDevice => 100,
            GpuError::LaunchFailed(_) => 8,
            GpuError::UnsupportedKernel(_) => 8,
            GpuError::DeviceOutOfRange { .. } => 1,
        }
    }
}
