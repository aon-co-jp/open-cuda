//! 実DirectX 12実装(Phase 1)。
//!
//! `D3D12CreateDevice`での実デバイス作成、UPLOADヒープ上のコミット
//! リソース経由での実メモリ確保・h2d/d2h/d2dコピーまでを実装する。
//! **カーネルディスパッチ(ルートシグネチャ・Compute PSO・ディスクリプタ
//! ヒープ・コマンドリスト記録)はPhase 2として未実装**——`launch_kernel`
//! は`GpuError::UnsupportedKernel`を返す(モジュールdoc・`lib.rs`のdoc
//! 参照、誇大な実装済み表示をしない)。

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Context;
use opencuda_core::{
    CompiledKernel, DeviceInfo, DevicePtr, GpuDevice, GpuError, GpuVendor, KernelArg, LaunchConfig, Result,
};

use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::{
    D3D12CreateDevice, ID3D12Device, ID3D12Resource, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES,
    D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_FLAG_NONE,
    D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};

/// 確保済みバッファ1個分。生ポインタはリソース作成時に一度`Map`した
/// ままにする(Vulkanバックエンドの「常時マップ」設計と同じ)。
struct MappedResource {
    /// Dropで解放されるようにリソース自体を保持し続ける。
    _resource: ID3D12Resource,
    ptr: *mut u8,
    len: usize,
}

// SAFETY: `ptr`はD3D12がプロセスに割り当てたUPLOADヒープのマップ済み
// アドレスであり、`DirectXDevice`が`Mutex`で全アクセスを排他制御する
// ため、複数スレッドから安全に共有できる。COMインターフェース自体も
// windows-rsではSend/Sync実装済みだが、生ポインタ`ptr`の分だけ明示。
unsafe impl Send for MappedResource {}

/// 実DirectX 12デバイス(Phase 1: デバイス作成・メモリ管理のみ)。
pub struct DirectXDevice {
    info: DeviceInfo,
    #[allow(dead_code)]
    device: ID3D12Device,
    allocations: Mutex<HashMap<u64, MappedResource>>,
    next_handle: AtomicU64,
}

// SAFETY: `device`・`allocations`いずれも内部で排他制御されるか、
// windows-rsのCOMラッパー自体がSend/Syncを提供するため安全。
unsafe impl Sync for DirectXDevice {}

#[derive(Debug, Clone)]
pub struct DirectXDiagnostics {
    pub feature_level: &'static str,
}

impl DirectXDevice {
    /// 実`D3D12CreateDevice`でデフォルトアダプタ上にデバイスを作成する。
    /// DirectX 12ランタイム・互換GPU/ドライバが無い環境ではエラーを返す
    /// (呼び出し側で握りつぶしてスキップする設計は`opencuda-vulkan`と
    /// 同じ、テスト側の責務)。
    pub fn new(id: usize) -> anyhow::Result<Arc<Self>> {
        let device: ID3D12Device = unsafe {
            let mut result: Option<ID3D12Device> = None;
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut result)
                .context("D3D12CreateDevice failed (no DirectX 12 capable GPU/driver, or runtime unavailable)")?;
            result.ok_or_else(|| anyhow::anyhow!("D3D12CreateDevice succeeded but returned no device"))?
        };

        Ok(Arc::new(Self {
            info: DeviceInfo {
                id,
                // D3D12は`GetAdapterLuid`はあるがベンダーIDはDXGIアダプタ
                // 列挙経由でないと取れない(Phase 1ではデフォルトアダプタ
                // 決め打ちのため未取得、Unknownとして正直に扱う)。
                vendor: GpuVendor::Unknown,
                name: "DirectX 12 Device (default adapter, feature level 11_0+)".to_string(),
                total_memory: 0,
                compute_units: 0,
            },
            device,
            allocations: Mutex::new(HashMap::new()),
            next_handle: AtomicU64::new(1),
        }))
    }

    pub fn diagnostics(&self) -> DirectXDiagnostics {
        DirectXDiagnostics { feature_level: "11_0+" }
    }

    fn create_upload_buffer(&self, bytes: usize) -> anyhow::Result<(ID3D12Resource, *mut u8)> {
        let heap_props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: bytes as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };

        let resource: ID3D12Resource = unsafe {
            let mut result: Option<ID3D12Resource> = None;
            self.device
                .CreateCommittedResource(
                    &heap_props,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut result,
                )
                .context("ID3D12Device::CreateCommittedResource failed")?;
            result.ok_or_else(|| anyhow::anyhow!("CreateCommittedResource succeeded but returned no resource"))?
        };

        let mut mapped: *mut c_void = std::ptr::null_mut();
        unsafe {
            resource.Map(0, None, Some(&mut mapped)).context("ID3D12Resource::Map failed")?;
        }
        Ok((resource, mapped as *mut u8))
    }
}

impl GpuDevice for DirectXDevice {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        if bytes == 0 {
            return Err(GpuError::OutOfMemory(0).into());
        }
        let (resource, ptr) = self.create_upload_buffer(bytes)?;
        let handle = self.next_handle.fetch_add(1, Ordering::Relaxed);
        self.allocations
            .lock()
            .unwrap()
            .insert(handle, MappedResource { _resource: resource, ptr, len: bytes });
        Ok(DevicePtr::new(handle, self.info.id as u32))
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        if ptr.device_id as usize != self.info.id {
            return Err(GpuError::InvalidPtr(ptr).into());
        }
        let mut map = self.allocations.lock().unwrap();
        if map.remove(&ptr.addr).is_none() {
            return Err(GpuError::InvalidPtr(ptr).into());
        }
        // MappedResourceのDropで_resourceが解放される(Mapしたままだが、
        // D3D12はUnmapせずリソース解放しても実害はない——COM参照カウント
        // がゼロになった時点でドライバ側が回収する)。
        Ok(())
    }

    fn memcpy_h2d(&self, dst: DevicePtr, src: &[u8]) -> Result<()> {
        if dst.device_id as usize != self.info.id {
            return Err(GpuError::InvalidPtr(dst).into());
        }
        let map = self.allocations.lock().unwrap();
        let alloc = map.get(&dst.addr).ok_or(GpuError::InvalidPtr(dst))?;
        if src.len() > alloc.len {
            return Err(GpuError::OutOfMemory(src.len()).into());
        }
        // SAFETY: `ptr`はDirectX 12がこのプロセスへマップした、`alloc.len`
        // バイト分だけ有効なUPLOADヒープの書き込み可能領域。
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), alloc.ptr, src.len());
        }
        Ok(())
    }

    fn memcpy_d2h(&self, dst: &mut [u8], src: DevicePtr) -> Result<()> {
        if src.device_id as usize != self.info.id {
            return Err(GpuError::InvalidPtr(src).into());
        }
        let map = self.allocations.lock().unwrap();
        let alloc = map.get(&src.addr).ok_or(GpuError::InvalidPtr(src))?;
        if dst.len() > alloc.len {
            return Err(GpuError::InvalidPtr(src).into());
        }
        unsafe {
            std::ptr::copy_nonoverlapping(alloc.ptr, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    fn memcpy_d2d(&self, dst: DevicePtr, src: DevicePtr, bytes: usize) -> Result<()> {
        if dst.device_id as usize != self.info.id || src.device_id as usize != self.info.id {
            return Err(GpuError::InvalidPtr(dst).into());
        }
        let map = self.allocations.lock().unwrap();
        let src_alloc = map.get(&src.addr).ok_or(GpuError::InvalidPtr(src))?;
        if bytes > src_alloc.len {
            return Err(GpuError::InvalidPtr(src).into());
        }
        let src_ptr = src_alloc.ptr;
        let dst_alloc = map.get(&dst.addr).ok_or(GpuError::InvalidPtr(dst))?;
        if bytes > dst_alloc.len {
            return Err(GpuError::InvalidPtr(dst).into());
        }
        unsafe {
            std::ptr::copy(src_ptr, dst_alloc.ptr, bytes);
        }
        Ok(())
    }

    fn launch_kernel(&self, _kernel: &CompiledKernel, _cfg: &LaunchConfig, _args: &[KernelArg]) -> Result<()> {
        // Phase 2で実装予定(ルートシグネチャ/Compute PSO/ディスクリプタ
        // ヒープ/コマンドリスト記録)。誇大に「動く」と見せないため
        // 明示的に未対応エラーを返す。
        Err(GpuError::UnsupportedKernel("Dxil kernel dispatch not yet implemented (Phase 2)").into())
    }

    fn synchronize(&self) -> Result<()> {
        // Phase 1はコマンドキューへ何も積んでいない(CPU側memcpyのみ)ため、
        // 待つべきGPU作業自体が無い。Phase 2でコマンドキュー+フェンスを
        // 導入した際にここを実装する。
        Ok(())
    }

    fn supports_dxil(&self) -> bool {
        // Phase 1時点ではlaunch_kernelが常にエラーを返すため、正直に
        // falseとする(「対応している」という誤ったシグナルを出さない)。
        false
    }
}

pub fn enumerate_real(start_id: usize) -> anyhow::Result<Vec<Arc<dyn GpuDevice>>> {
    let device = DirectXDevice::new(start_id)?;
    Ok(vec![device as Arc<dyn GpuDevice>])
}
