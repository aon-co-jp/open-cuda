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
    /// Qualcomm Adreno(モバイルGPU、PCI/VulkanベンダーID`0x5143`、
    /// pci-ids.ucw.czで"Qualcomm Inc"と確認済み——2026-07-25、
    /// Android-Vulkan監査の続きとして追加)。
    Qualcomm { architecture: String },
    /// ARM Mali(モバイルGPU、PCI/VulkanベンダーID`0x13B5`、
    /// pci-ids.ucw.czで"ARM"と確認済み)。
    Arm { architecture: String },
    /// Imagination Technologies PowerVR(モバイル/組込みGPU、PCI/Vulkan
    /// ベンダーID`0x1010`。pci-ids.ucw.czでは"Video Logic, Ltd."名義だが、
    /// これはImagination Technologies設立前の旧社名(PowerVR部門の前身、
    /// Wikipedia "PowerVR"項目で"formerly VideoLogic"と裏付け済み)であり
    /// 同一のPCIベンダーIDが引き継がれている)。
    ImaginationPowerVr { architecture: String },
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

    /// 2次元の簡便コンストラクタ(matmul等、行×列の出力を持つカーネル向け)。
    ///
    /// `cols`/`rows` は出力全体の要素数(スレッド総数)、`block_x`/`block_y` は
    /// 1ワークグループ/1スレッドブロックあたりのスレッド数。
    /// Vulkanバックエンドでは `grid` の値がそのまま `vkCmdDispatch` の
    /// ワークグループ数になるため、シェーダの `local_size_x/y` と `block_x/y` を
    /// 一致させる契約になっている。
    pub fn grid2d(rows: u32, cols: u32, block_x: u32, block_y: u32) -> Self {
        let groups_x = cols.div_ceil(block_x.max(1));
        let groups_y = rows.div_ceil(block_y.max(1));
        Self {
            grid: (groups_x, groups_y, 1),
            block: (block_x, block_y, 1),
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

    /// この `GpuDevice` 実装が `CompiledKernel::spirv`（SPIR-V/Vulkan Compute
    /// カーネル）の `launch_kernel` をサポートするかどうか。
    ///
    /// `DeviceInfo::vendor`（`GpuVendor::Nvidia`等）はハードウェアベンダーを
    /// 表すだけで、その情報だけでは「Vulkan経由でアクセスしているのか、
    /// 将来のCUDA直叩き実装なのか」を区別できない（`opencuda-blas`の
    /// `select_gemm_path`がベンダー別スタブ経路とVulkan汎用経路のどちらを
    /// 自動選択すべきか判断するのに、ベンダー情報だけでは不十分だった）。
    /// このメソッドはその区別を明示的に行うための能力フラグ。
    /// デフォルトは `false`（`CompiledKernel::native`のみを想定する既存の
    /// バックエンドは変更不要）。SPIR-Vカーネルを実行できるバックエンド
    /// （`opencuda-vulkan::real::VulkanDevice`）だけが `true` を返す。
    fn supports_spirv(&self) -> bool {
        false
    }

    /// この `GpuDevice` 実装が `CompiledKernel::dxil`（DXIL/DirectX 12
    /// Computeカーネル)の`launch_kernel`をサポートするかどうか。
    /// `supports_spirv`と同じ設計判断(2026-07-23、DirectXバックエンド
    /// 追加時に追加)。デフォルトは`false`。
    fn supports_dxil(&self) -> bool {
        false
    }

    /// このデバイスがネイティブ FP8(E4M3/E5M2)行列演算を実行できるか。
    /// FP8 世代: NVIDIA Hopper(SM90)/ Ada(SM89)/ Blackwell(SM100・
    /// SM120、RTX 5080/5090 含む)、AMD RDNA4(Radeon AI PRO R9700 等、
    /// FP8 WMMA)、Intel Arc B/Xe2。移植性の高い経路は Vulkan
    /// `VK_KHR_shader_float8` + cooperative matrix。デフォルトは `false`
    /// ——`opencuda-blas` の `select_gemm_path` が、FP8 量子化済み重みの
    /// GEMM をベンダー FP8 経路(`GemmPath::Fp8Tensor`)へ振るべきか、
    /// ソフトウェア dequant 経路へフォールバックすべきかを判断するための
    /// 能力フラグ。現状 `true` を返す実バックエンドは無い(FP8 対応 GPU が
    /// 本開発環境に無く未検証、AVX-512 経路と同じ「コードは用意して
    /// 機能フラグで有効化」の方針)。
    fn supports_fp8_tensor_core(&self) -> bool {
        false
    }

    /// このデバイスがVulkan Computeシェーダ内でネイティブ`double`
    /// (SPIR-V `Float64` capability、Vulkanの`shaderFloat64`機能)を
    /// 実行できるか(2026-09-05新設、dgemmのGPUディスパッチ配線向け)。
    /// `supports_spirv`(SpirVカーネル自体を受理できるか)とは別軸——
    /// SpirVカーネルを実行できるバックエンドでも、物理デバイス/ドライバが
    /// `shaderFloat64`をサポートしない場合はこちらが`false`のままになる。
    /// デフォルトは`false`。`opencuda-vulkan::real::VulkanDevice`は
    /// 実際に`vkGetPhysicalDeviceFeatures`で確認した結果をそのまま返す。
    fn supports_f64_shader(&self) -> bool {
        false
    }
}
