//! GpuDevice trait — プロジェクトの背骨。
//!
//! すべてのバックエンド（CPU / Vulkan / CUDA / ROCm / oneAPI）がこれを実装する。
//! この契約が固まっていれば、バックエンドは後から好きな順で足せる。

use crate::error::Result;
use crate::kernel::{CompiledKernel, KernelArg};
use crate::memory::DevicePtr;

/// GPU（または CPU）ベンダー識別。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuVendor {
    Nvidia { compute_capability: (u32, u32) },
    Amd { gfx_version: String },
    Intel { architecture: String },
    /// CPUバックエンド（Phase 1 の最初の実行ターゲット）。
    Cpu,
    Unknown,
}

/// デバイス情報。
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub id: usize,
    pub vendor: GpuVendor,
    pub name: String,
    pub total_memory: u64,
    pub compute_units: u32,
}

/// カーネル起動設定（CUDA の <<<grid, block, smem>>> に相当）。
#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem: u32,
}

impl LaunchConfig {
    /// 1次元の簡便コンストラクタ。
    pub fn linear(total_threads: u32, block_size: u32) -> Self {
        let blocks = total_threads.div_ceil(block_size.max(1));
        Self {
            grid: (blocks, 1, 1),
            block: (block_size, 1, 1),
            shared_mem: 0,
        }
    }

    pub fn total_threads(&self) -> u64 {
        let g = self.grid.0 as u64 * self.grid.1 as u64 * self.grid.2 as u64;
        let b = self.block.0 as u64 * self.block.1 as u64 * self.block.2 as u64;
        g * b
    }
}

/// 全バックエンドが実装する契約。
pub trait GpuDevice: Send + Sync {
    fn info(&self) -> &DeviceInfo;

    // --- メモリ管理（CUDA Runtime API 互換のセマンティクス） ---
    fn alloc(&self, bytes: usize) -> Result<DevicePtr>;
    fn free(&self, ptr: DevicePtr) -> Result<()>;
    fn memcpy_h2d(&self, dst: DevicePtr, src: &[u8]) -> Result<()>;
    fn memcpy_d2h(&self, dst: &mut [u8], src: DevicePtr) -> Result<()>;
    fn memcpy_d2d(&self, dst: DevicePtr, src: DevicePtr, bytes: usize) -> Result<()>;

    // --- カーネル実行 ---
    fn launch_kernel(
        &self,
        kernel: &CompiledKernel,
        cfg: &LaunchConfig,
        args: &[KernelArg],
    ) -> Result<()>;

    // --- 同期 ---
    fn synchronize(&self) -> Result<()>;
}
