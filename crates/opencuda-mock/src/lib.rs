//! # opencuda-mock
//!
//! GPU 実機なしで OpenCUDA のデバイス抽象・メモリ転送・カーネル起動の形を検証する
//! テスト用バックエンド。
//!
//! `MockDevice` は計算を実行しない。代わりに、どのカーネル形式が、どの起動設定と
//! 引数数で呼ばれたかを記録する。CI や設計テストで「呼び出し経路が正しいか」を
//! 検証するための代替デバイスであり、性能測定やGPU実行の代替ではない。

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use opencuda_core::{
    CompiledKernel, DeviceInfo, DevicePtr, GpuDevice, GpuError, GpuVendor, KernelArg,
    LaunchConfig, Result,
};

#[derive(Debug, Clone)]
pub struct LaunchRecord {
    pub kernel_name: String,
    pub source_kind: &'static str,
    pub entry: String,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub arg_count: usize,
}

#[derive(Default)]
struct Allocation {
    bytes: Vec<u8>,
}

pub struct MockDevice {
    info: DeviceInfo,
    allocations: Mutex<HashMap<u64, Allocation>>,
    launches: Mutex<Vec<LaunchRecord>>,
    next_handle: AtomicU64,
}

impl MockDevice {
    pub fn new(id: usize) -> Arc<Self> {
        Arc::new(Self {
            info: DeviceInfo {
                id,
                vendor: GpuVendor::Unknown,
                name: "OpenCUDA Mock Device (no GPU, records launches)".to_string(),
                total_memory: 512 * 1024 * 1024,
                compute_units: 1,
            },
            allocations: Mutex::new(HashMap::new()),
            launches: Mutex::new(Vec::new()),
            next_handle: AtomicU64::new(1),
        })
    }

    pub fn launches(&self) -> Vec<LaunchRecord> {
        self.launches.lock().unwrap().clone()
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
}

impl GpuDevice for MockDevice {
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
        cfg: &LaunchConfig,
        args: &[KernelArg],
    ) -> Result<()> {
        for arg in args {
            if let KernelArg::Ptr(ptr) = arg {
                self.check_ptr(*ptr)?;
            }
        }

        self.launches.lock().unwrap().push(LaunchRecord {
            kernel_name: kernel.name.clone(),
            source_kind: kernel.source_kind(),
            entry: kernel.entry.clone(),
            grid: cfg.grid,
            block: cfg.block,
            arg_count: args.len(),
        });
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

pub fn enumerate(start_id: usize) -> Vec<Arc<dyn GpuDevice>> {
    vec![MockDevice::new(start_id)]
}
