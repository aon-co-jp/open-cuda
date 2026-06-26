//! # opencuda-core
//!
//! OpenCUDA の中核抽象。クロスベンダーGPUランタイムの「背骨」を定義する。
//!
//! 設計の確定事項（Phase 1）:
//! 1. メモリ所有モデルは二層（内部生ポインタ [`DevicePtr`] + RAII [`DeviceBuffer`]）。
//! 2. [`DeviceBuffer`] は `Arc<dyn GpuDevice>` を保持し Drop で自動解放。
//! 3. エラーは [`Result`]（anyhow基調）+ 変換用の最小 [`GpuError`] enum。
//! 4. カーネル表現は [`KernelSource`] enum（Phase 1 は Native / SpirV のみ実装）。
//!
//! このクレートはどのバックエンドにも依存しない。バックエンド
//! （`opencuda-cpu` など）が [`GpuDevice`] を実装し、[`DeviceRegistry`] に登録する。

pub mod device;
pub mod error;
pub mod kernel;
pub mod memory;
pub mod registry;

pub use device::{DeviceInfo, GpuDevice, GpuVendor, LaunchConfig};
pub use error::{GpuError, Result};
pub use kernel::{
    CompiledKernel, KernelArg, KernelSource, NativeKernelFn, ResolvedArg, ThreadCtx,
};
pub use memory::{alloc_buffer, DeviceBuffer, DevicePtr};
pub use registry::DeviceRegistry;
