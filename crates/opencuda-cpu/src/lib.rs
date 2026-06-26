//! CPU バックエンド。
//!
//! Phase 1 の最初の実行ターゲット。GPU を一切持たなくても、`rayon` で
//! カーネルをマルチスレッド実行して設計の正しさを検証できる。
//!
//! 「デバイスメモリ」はホスト上のヒープ確保で代用する。`DevicePtr.addr` には
//! 確保したメモリの生ポインタ（`*mut u8` を `u64` 化したもの）を入れ、
//! `memcpy_*` は実体としては `memcpy`。GPU と API の形を揃えることに意味がある。

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use opencuda_core::{
    CompiledKernel, DeviceInfo, DevicePtr, GpuDevice, GpuError, GpuVendor, KernelArg, KernelSource,
    LaunchConfig, ResolvedArg, Result, ThreadCtx,
};

/// 1つの確保領域の記録。
struct Allocation {
    ptr: *mut u8,
    layout: std::alloc::Layout,
    len: usize,
}

// 生ポインタを跨スレッドで持つため、安全性は Mutex とアクセス規律で担保する。
unsafe impl Send for Allocation {}

pub struct CpuDevice {
    info: DeviceInfo,
    /// addr -> Allocation。確保・解放・範囲チェックに使う。
    allocations: Mutex<HashMap<u64, Allocation>>,
    /// 0 をヌル扱いにするため 1 から開始する論理アドレス採番。
    next_handle: AtomicU64,
}

impl CpuDevice {
    pub fn new(id: usize) -> Arc<Self> {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1);

        let info = DeviceInfo {
            id,
            vendor: GpuVendor::Cpu,
            name: format!("OpenCUDA CPU Device (rayon, {threads} threads)"),
            // 物理RAMの正確な取得はOS依存なので、当面は控えめな固定値で申告。
            total_memory: 0,
            compute_units: threads,
        };

        Arc::new(Self {
            info,
            allocations: Mutex::new(HashMap::new()),
            next_handle: AtomicU64::new(1),
        })
    }

    fn resolve(&self, ptr: DevicePtr) -> Result<*mut u8> {
        if ptr.device_id as usize != self.info.id {
            return Err(GpuError::InvalidPtr(ptr).into());
        }
        let map = self.allocations.lock().unwrap();
        map.get(&ptr.addr)
            .map(|a| a.ptr)
            .ok_or_else(|| GpuError::InvalidPtr(ptr).into())
    }

    fn len_of(&self, ptr: DevicePtr) -> Result<usize> {
        let map = self.allocations.lock().unwrap();
        map.get(&ptr.addr)
            .map(|a| a.len)
            .ok_or_else(|| GpuError::InvalidPtr(ptr).into())
    }
}

impl GpuDevice for CpuDevice {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        if bytes == 0 {
            return Err(GpuError::OutOfMemory(0).into());
        }
        let layout = std::alloc::Layout::from_size_align(bytes, 16)
            .map_err(|_| GpuError::OutOfMemory(bytes))?;
        // SAFETY: layout の size>0 を保証済み。
        let raw = unsafe { std::alloc::alloc(layout) };
        if raw.is_null() {
            return Err(GpuError::OutOfMemory(bytes).into());
        }
        let handle = self.next_handle.fetch_add(1, Ordering::Relaxed);
        let mut map = self.allocations.lock().unwrap();
        map.insert(
            handle,
            Allocation {
                ptr: raw,
                layout,
                len: bytes,
            },
        );
        Ok(DevicePtr::new(handle, self.info.id as u32))
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        let mut map = self.allocations.lock().unwrap();
        match map.remove(&ptr.addr) {
            Some(a) => {
                // SAFETY: 確保時と同じ layout で解放。
                unsafe { std::alloc::dealloc(a.ptr, a.layout) };
                Ok(())
            }
            None => Err(GpuError::InvalidPtr(ptr).into()),
        }
    }

    fn memcpy_h2d(&self, dst: DevicePtr, src: &[u8]) -> Result<()> {
        let cap = self.len_of(dst)?;
        if src.len() > cap {
            return Err(GpuError::OutOfMemory(src.len()).into());
        }
        let d = self.resolve(dst)?;
        // SAFETY: 範囲チェック済み、d は有効な確保領域の先頭。
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), d, src.len()) };
        Ok(())
    }

    fn memcpy_d2h(&self, dst: &mut [u8], src: DevicePtr) -> Result<()> {
        let cap = self.len_of(src)?;
        if dst.len() > cap {
            return Err(GpuError::InvalidPtr(src).into());
        }
        let s = self.resolve(src)?;
        // SAFETY: 範囲チェック済み。
        unsafe { std::ptr::copy_nonoverlapping(s, dst.as_mut_ptr(), dst.len()) };
        Ok(())
    }

    fn memcpy_d2d(&self, dst: DevicePtr, src: DevicePtr, bytes: usize) -> Result<()> {
        if self.len_of(dst)? < bytes || self.len_of(src)? < bytes {
            return Err(GpuError::InvalidPtr(dst).into());
        }
        let d = self.resolve(dst)?;
        let s = self.resolve(src)?;
        // SAFETY: 両者とも範囲チェック済み、別個の確保領域。
        unsafe { std::ptr::copy_nonoverlapping(s, d, bytes) };
        Ok(())
    }

    fn launch_kernel(
        &self,
        kernel: &CompiledKernel,
        cfg: &LaunchConfig,
        args: &[KernelArg],
    ) -> Result<()> {
        let f = match &kernel.source {
            KernelSource::Native(f) => f.clone(),
            other => return Err(GpuError::UnsupportedKernel(other.kind()).into()),
        };

        // KernelArg を ResolvedArg に解決する（DevicePtr → 実アドレス + len）。
        let resolved: Vec<ResolvedArg> = args
            .iter()
            .map(|a| -> Result<ResolvedArg> {
                Ok(match a {
                    KernelArg::Ptr(p) => {
                        let addr = self.resolve(*p)?;
                        let len = self.len_of(*p)?;
                        ResolvedArg::Ptr { addr, len }
                    }
                    KernelArg::U32(v) => ResolvedArg::U32(*v),
                    KernelArg::I32(v) => ResolvedArg::I32(*v),
                    KernelArg::F32(v) => ResolvedArg::F32(*v),
                    KernelArg::Usize(v) => ResolvedArg::Usize(*v),
                })
            })
            .collect::<Result<_>>()?;

        let (gx, gy, _gz) = cfg.grid;
        let (bx, by, _bz) = cfg.block;
        let grid_dim = cfg.grid;
        let block_dim = cfg.block;

        let total_blocks = (gx as u64) * (gy as u64) * (cfg.grid.2 as u64);
        let threads_per_block = (bx as u64) * (by as u64) * (cfg.block.2 as u64);

        // 各ブロックを rayon で並列に。ブロック内スレッドは逐次。
        (0..total_blocks).into_par_iter().for_each(|blk| {
            let plane = gx as u64 * gy as u64;
            let biz = (blk / plane) as u32;
            let rem = blk % plane;
            let biy = (rem / gx as u64) as u32;
            let bix = (rem % gx as u64) as u32;

            for t in 0..threads_per_block {
                let tplane = bx as u64 * by as u64;
                let tiz = (t / tplane) as u32;
                let trem = t % tplane;
                let tiy = (trem / bx as u64) as u32;
                let tix = (trem % bx as u64) as u32;

                let ctx = ThreadCtx {
                    block_idx: (bix, biy, biz),
                    thread_idx: (tix, tiy, tiz),
                    block_dim,
                    grid_dim,
                };
                f(ctx, &resolved);
            }
        });

        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        // CPUバックエンドの launch は同期実行なので何もしない。
        Ok(())
    }
}

/// CPUデバイスを生成して返す。バックエンド検出のエントリポイント。
pub fn enumerate(start_id: usize) -> Vec<Arc<dyn GpuDevice>> {
    vec![CpuDevice::new(start_id)]
}
