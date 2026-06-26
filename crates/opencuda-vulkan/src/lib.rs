//! # opencuda-vulkan
//!
//! Vulkan Compute バックエンドの準備用クレート。
//!
//! v0.3.5 では、実GPUや Vulkan SDK が無い環境でも開発を進めるため、
//! `VulkanMockDevice` を提供する。これは実Vulkanではない。目的は次の3つ。
//!
//! 1. Vulkan系バックエンドでは `Native` カーネルを拒否し、`SpirV` だけを受ける契約を固定する。
//! 2. `KernelSource::SpirV` と `KernelSource::OmniIr` の経路をGPUなしで検証する。
//! 3. `vector_add_f32` を OmniIR → SPIR-V 相当経路のシミュレーションとして実行し、CPU参照値と比較できるようにする。
//!
//! 本物の Vulkan 実装では、ここを `ash` / `wgpu-hal` 等に置き換え、
//! `KernelSource::SpirV` を `VkShaderModule` → Compute Pipeline → Dispatch に流す。

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail};
use opencuda_core::{
    CompiledKernel, DeviceInfo, DevicePtr, GpuDevice, GpuError, GpuVendor, KernelArg, KernelSource,
    LaunchConfig, Result,
};
use opencuda_ir::IrModule;

const SPIRV_MAGIC_LE: [u8; 4] = [0x03, 0x02, 0x23, 0x07];
const SPIRV_VECTOR_ADD_FIXTURE: &[u8] = &[0x03, 0x02, 0x23, 0x07, 0x00, 0x00, 0x02, 0x00];

#[derive(Default)]
struct Allocation {
    bytes: Vec<u8>,
}

/// GPUなしで Vulkan/SPIR-V 経路をテストするための代替デバイス。
///
/// 名前に `Mock` を入れている通り、これは実Vulkanバックエンドではない。
/// 実機バックエンドを実装する前の Phase 1.5 テスト用である。
pub struct VulkanMockDevice {
    info: DeviceInfo,
    allocations: Mutex<HashMap<u64, Allocation>>,
    next_handle: AtomicU64,
}

impl VulkanMockDevice {
    pub fn new(id: usize) -> Arc<Self> {
        Arc::new(Self {
            info: DeviceInfo {
                id,
                vendor: GpuVendor::Unknown,
                name: "OpenCUDA Vulkan Mock Device (SPIR-V path simulator, no GPU)".to_string(),
                total_memory: 512 * 1024 * 1024,
                compute_units: 1,
            },
            allocations: Mutex::new(HashMap::new()),
            next_handle: AtomicU64::new(1),
        })
    }

    fn check_ptr(&self, ptr: DevicePtr) -> Result<()> {
        if ptr.device_id as usize != self.info.id {
            return Err(GpuError::InvalidPtr(ptr).into());
        }
        if !self.allocations.lock().unwrap().contains_key(&ptr.addr) {
            return Err(GpuError::InvalidPtr(ptr).into());
        }
        Ok(())
    }

    fn validate_spirv(bytes: &[u8]) -> Result<()> {
        if bytes.len() < 4 {
            bail!("invalid SPIR-V: buffer is shorter than 4 bytes");
        }
        if bytes[..4] != SPIRV_MAGIC_LE {
            bail!("invalid SPIR-V: missing little-endian magic 0x07230203");
        }
        Ok(())
    }

    fn read_f32_vec(&self, ptr: DevicePtr, n: usize) -> Result<Vec<f32>> {
        self.check_ptr(ptr)?;
        let map = self.allocations.lock().unwrap();
        let alloc = map.get(&ptr.addr).unwrap();
        let bytes = n
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow!("byte size overflow"))?;
        if bytes > alloc.bytes.len() {
            return Err(GpuError::InvalidPtr(ptr).into());
        }
        let mut out = Vec::with_capacity(n);
        for chunk in alloc.bytes[..bytes].chunks_exact(4) {
            out.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(out)
    }

    fn write_f32_vec(&self, ptr: DevicePtr, values: &[f32]) -> Result<()> {
        self.check_ptr(ptr)?;
        let mut map = self.allocations.lock().unwrap();
        let alloc = map.get_mut(&ptr.addr).unwrap();
        let bytes = values
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow!("byte size overflow"))?;
        if bytes > alloc.bytes.len() {
            return Err(GpuError::InvalidPtr(ptr).into());
        }
        for (chunk, value) in alloc.bytes[..bytes].chunks_exact_mut(4).zip(values.iter()) {
            chunk.copy_from_slice(&value.to_ne_bytes());
        }
        Ok(())
    }

    /// v0.3.5 のSPIR-V/OmniIR経路シミュレーション。
    ///
    /// 実装済み範囲を意図的に限定し、誇大に見えないよう vector_add 系のみ受け付ける。
    fn run_vector_add_simulation(&self, args: &[KernelArg]) -> Result<()> {
        if args.len() != 4 {
            bail!("vector_add expects 4 args: a, b, c, n");
        }
        let a = args[0].as_ptr().ok_or_else(|| anyhow!("arg0 must be pointer"))?;
        let b = args[1].as_ptr().ok_or_else(|| anyhow!("arg1 must be pointer"))?;
        let c = args[2].as_ptr().ok_or_else(|| anyhow!("arg2 must be pointer"))?;
        let n = args[3].as_usize().ok_or_else(|| anyhow!("arg3 must be usize"))?;

        let av = self.read_f32_vec(a, n)?;
        let bv = self.read_f32_vec(b, n)?;
        let mut cv = Vec::with_capacity(n);
        for i in 0..n {
            cv.push(av[i] + bv[i]);
        }
        self.write_f32_vec(c, &cv)
    }
}

impl GpuDevice for VulkanMockDevice {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        if bytes == 0 {
            return Err(GpuError::OutOfMemory(0).into());
        }
        let handle = self.next_handle.fetch_add(1, Ordering::Relaxed);
        self.allocations
            .lock()
            .unwrap()
            .insert(handle, Allocation { bytes: vec![0; bytes] });
        Ok(DevicePtr::new(handle, self.info.id as u32))
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        self.check_ptr(ptr)?;
        self.allocations.lock().unwrap().remove(&ptr.addr);
        Ok(())
    }

    fn memcpy_h2d(&self, dst: DevicePtr, src: &[u8]) -> Result<()> {
        self.check_ptr(dst)?;
        let mut map = self.allocations.lock().unwrap();
        let alloc = map.get_mut(&dst.addr).unwrap();
        if src.len() > alloc.bytes.len() {
            return Err(GpuError::OutOfMemory(src.len()).into());
        }
        alloc.bytes[..src.len()].copy_from_slice(src);
        Ok(())
    }

    fn memcpy_d2h(&self, dst: &mut [u8], src: DevicePtr) -> Result<()> {
        self.check_ptr(src)?;
        let map = self.allocations.lock().unwrap();
        let alloc = map.get(&src.addr).unwrap();
        if dst.len() > alloc.bytes.len() {
            return Err(GpuError::InvalidPtr(src).into());
        }
        dst.copy_from_slice(&alloc.bytes[..dst.len()]);
        Ok(())
    }

    fn memcpy_d2d(&self, dst: DevicePtr, src: DevicePtr, bytes: usize) -> Result<()> {
        self.check_ptr(dst)?;
        self.check_ptr(src)?;
        let mut map = self.allocations.lock().unwrap();
        let tmp = {
            let s = map.get(&src.addr).unwrap();
            if bytes > s.bytes.len() {
                return Err(GpuError::InvalidPtr(src).into());
            }
            s.bytes[..bytes].to_vec()
        };
        let d = map.get_mut(&dst.addr).unwrap();
        if bytes > d.bytes.len() {
            return Err(GpuError::InvalidPtr(dst).into());
        }
        d.bytes[..bytes].copy_from_slice(&tmp);
        Ok(())
    }

    fn launch_kernel(
        &self,
        kernel: &CompiledKernel,
        _cfg: &LaunchConfig,
        args: &[KernelArg],
    ) -> Result<()> {
        match &kernel.source {
            KernelSource::SpirV(bytes) => Self::validate_spirv(bytes)?,
            KernelSource::OmniIr(bytes) => {
                let module = IrModule::decode(bytes)?;
                let spirv = compile_omniir_to_spirv_fixture(&module)?;
                Self::validate_spirv(&spirv)?;
            }
            other => return Err(GpuError::UnsupportedKernel(other.kind()).into()),
        }

        match kernel.name.as_str() {
            "vector_add" | "vector_add_f32" => self.run_vector_add_simulation(args),
            other => bail!(
                "VulkanMockDevice only simulates vector_add/vector_add_f32 in v0.3.5; got kernel `{other}`"
            ),
        }
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

pub fn enumerate(start_id: usize) -> Vec<Arc<dyn GpuDevice>> {
    vec![VulkanMockDevice::new(start_id)]
}


/// v0.3.5 の OmniIR → SPIR-V 代替コンパイラ。
///
/// これは本物のSPIR-V生成器ではなく、Vulkan実装前にパイプライン契約を固定するfixture生成器。
/// 本物の実装では `IrModule` を SPIR-V 命令列へ下げ、`VkShaderModule` に渡す。
pub fn compile_omniir_to_spirv_fixture(module: &IrModule) -> Result<Vec<u8>> {
    if module.name != "vector_add_f32" {
        bail!("opencuda-vulkan v0.3.5 only compiles vector_add_f32 OmniIR fixture");
    }
    Ok(SPIRV_VECTOR_ADD_FIXTURE.to_vec())
}

#[cfg(feature = "real-vulkan")]
pub mod real;

#[cfg(feature = "real-vulkan")]
pub use real::{enumerate_real, VulkanDevice};
