//! 実DirectX 12実装。
//!
//! - **Phase 1(完了)**: `D3D12CreateDevice`での実デバイス作成・実メモリ
//!   確保・h2d/d2h/d2dコピー。
//! - **Phase 2(本ファイルで実装)**: DXILカーネルディスパッチ。
//!   ルートシグネチャをHLSL内へ`[RootSignature(...)]`属性で埋め込み、
//!   dxcがコンパイルしたDXILバイト列に同梱させることで、Rust側は
//!   `ID3D12Device::CreateRootSignature`へそのバイト列をそのまま渡す
//!   だけでよい設計にした(手動でのルートシグネチャ記述子構築を回避)。
//!   3つのUAVバッファはディスクリプタヒープを経由せず、ルート
//!   ディスクリプタとして直接バインドする(`SetComputeRootUnorderedAccessView`)
//!   ——ディスクリプタヒープ管理という別のバグの温床を避けるための
//!   設計判断。
//!
//! Phase 2に合わせて、Phase 1で採用していた「UPLOADヒープに常時マップ」
//! というメモリ管理方式は廃止し、**全バッファをDEFAULTヒープ(UAV対応)
//! で確保し、h2d/d2hは都度UPLOAD/READBACKヒープの一時ステージング
//! バッファ+コマンドリストでのコピーを介する**方式へ変更した(UAV
//! バインドにはDEFAULTヒープが必須、UPLOADヒープはUAVを許可しない
//! というD3D12の制約による)。
//!
//! **正直な開示・既知の単純化**: 毎回の操作(h2d/d2h/d2d/dispatch)を
//! コマンドキュー投入→フェンス待機で同期的に完結させており、複数
//! 操作をまたいだコマンドリストのバッチ化はしていない(スループット
//! より正しさを優先したPhase 2 MVPの設計)。またバッファの明示的な
//! `ResourceBarrier`は発行していない——D3D12のバッファに対する暗黙的
//! 状態昇格(implicit state promotion、COMMONから読み取り状態や単一の
//! 書き込み状態への自動昇格、コマンドリスト実行後のCOMMONへの暗黙的
//! 降格)に依存している。これは公式に文書化された挙動だが、テクス
//! チャや複数回にわたる複雑な状態遷移には昇格が効かないため、将来
//! バッファ以外のリソースを扱う場合は明示的なバリアの追加が必要になる。

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Context;
use opencuda_core::{
    CompiledKernel, DeviceInfo, DevicePtr, GpuDevice, GpuError, GpuVendor, KernelArg, KernelSource, LaunchConfig,
    Result,
};

use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::{
    D3D12CreateDevice, ID3D12CommandAllocator, ID3D12CommandQueue, ID3D12Device, ID3D12Fence,
    ID3D12GraphicsCommandList, ID3D12PipelineState, ID3D12Resource, ID3D12RootSignature,
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12_COMPUTE_PIPELINE_STATE_DESC,
    D3D12_FENCE_FLAG_NONE, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT,
    D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER,
    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON,
    D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_SHADER_BYTECODE,
    D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::core::Interface;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject, INFINITE};

/// DEFAULTヒープ上のGPU専用バッファ(UAVディスパッチに直接使える)。
struct GpuBuffer {
    resource: ID3D12Resource,
    len: usize,
}

// SAFETY: `resource`(COMインターフェース)自体はwindows-rsでSend/Sync。
// 生ポインタは保持しない(Phase 1と異なり常時マップしない設計のため)。
unsafe impl Send for GpuBuffer {}

/// カーネル名ごとにキャッシュするルートシグネチャ+Compute PSO。
struct Pipeline {
    root_signature: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

unsafe impl Send for Pipeline {}

/// 実DirectX 12デバイス。
pub struct DirectXDevice {
    info: DeviceInfo,
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    command_list: Mutex<ID3D12GraphicsCommandList>,
    fence: ID3D12Fence,
    fence_event: HANDLE,
    fence_value: AtomicU64,
    allocations: Mutex<HashMap<u64, GpuBuffer>>,
    next_handle: AtomicU64,
    pipelines: Mutex<HashMap<String, Pipeline>>,
}

// SAFETY: 全可変状態は`Mutex`または`Atomic*`で保護されている。
unsafe impl Sync for DirectXDevice {}
unsafe impl Send for DirectXDevice {}

#[derive(Debug, Clone)]
pub struct DirectXDiagnostics {
    pub feature_level: &'static str,
}

impl DirectXDevice {
    /// 実`D3D12CreateDevice`でデフォルトアダプタ上にデバイスを作成し、
    /// コマンドキュー・コマンドアロケータ・コマンドリスト・フェンスまで
    /// 一式初期化する。
    pub fn new(id: usize) -> anyhow::Result<Arc<Self>> {
        let device: ID3D12Device = unsafe {
            let mut result: Option<ID3D12Device> = None;
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut result)
                .context("D3D12CreateDevice failed (no DirectX 12 capable GPU/driver, or runtime unavailable)")?;
            result.ok_or_else(|| anyhow::anyhow!("D3D12CreateDevice succeeded but returned no device"))?
        };

        let queue: ID3D12CommandQueue = unsafe {
            device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() })
                .context("CreateCommandQueue failed")?
        };

        let allocator: ID3D12CommandAllocator = unsafe {
            device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).context("CreateCommandAllocator failed")?
        };

        let command_list: ID3D12GraphicsCommandList = unsafe {
            device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .context("CreateCommandList failed")?
        };
        // 作成直後のコマンドリストはオープン状態。以後は
        // execute_and_wait内でClose→実行→Resetのサイクルを回すため、
        // 初期状態を一致させるためここで一度閉じておく。
        unsafe {
            command_list.Close().context("initial CommandList::Close failed")?;
        }

        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE).context("CreateFence failed")? };
        let fence_event = unsafe { CreateEventW(None, false, false, None).context("CreateEventW failed")? };

        Ok(Arc::new(Self {
            info: DeviceInfo {
                id,
                vendor: GpuVendor::Unknown,
                name: "DirectX 12 Device (default adapter, feature level 11_0+)".to_string(),
                total_memory: 0,
                compute_units: 0,
            },
            device,
            queue,
            allocator,
            command_list: Mutex::new(command_list),
            fence,
            fence_event,
            fence_value: AtomicU64::new(0),
            allocations: Mutex::new(HashMap::new()),
            next_handle: AtomicU64::new(1),
            pipelines: Mutex::new(HashMap::new()),
        }))
    }

    pub fn diagnostics(&self) -> DirectXDiagnostics {
        DirectXDiagnostics { feature_level: "11_0+" }
    }

    /// コマンドリストを記録用に開き、`record`で内容を積んだ後に
    /// クローズ・実行・フェンス待機まで同期的に行う。
    fn execute_and_wait(&self, record: impl FnOnce(&ID3D12GraphicsCommandList) -> anyhow::Result<()>) -> anyhow::Result<()> {
        let command_list = self.command_list.lock().unwrap();
        unsafe {
            self.allocator.Reset().context("CommandAllocator::Reset failed")?;
            command_list.Reset(&self.allocator, None).context("CommandList::Reset failed")?;
        }

        record(&command_list)?;

        unsafe {
            command_list.Close().context("CommandList::Close failed")?;
        }

        let list_unknown: windows::Win32::Graphics::Direct3D12::ID3D12CommandList = command_list.cast()?;
        unsafe {
            self.queue.ExecuteCommandLists(&[Some(list_unknown)]);
        }

        let value = self.fence_value.fetch_add(1, Ordering::SeqCst) + 1;
        unsafe {
            self.queue.Signal(&self.fence, value).context("CommandQueue::Signal failed")?;
            if self.fence.GetCompletedValue() < value {
                self.fence.SetEventOnCompletion(value, self.fence_event).context("SetEventOnCompletion failed")?;
                WaitForSingleObject(self.fence_event, INFINITE);
            }
        }
        Ok(())
    }

    fn create_default_buffer(&self, bytes: usize) -> anyhow::Result<ID3D12Resource> {
        let heap_props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let desc = buffer_desc(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        unsafe {
            let mut result: Option<ID3D12Resource> = None;
            self.device
                .CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut result)
                .context("CreateCommittedResource(DEFAULT) failed")?;
            result.ok_or_else(|| anyhow::anyhow!("CreateCommittedResource(DEFAULT) returned no resource"))
        }
    }

    fn create_staging_buffer(&self, bytes: usize, heap_type: windows::Win32::Graphics::Direct3D12::D3D12_HEAP_TYPE) -> anyhow::Result<(ID3D12Resource, *mut u8)> {
        let heap_props = D3D12_HEAP_PROPERTIES { Type: heap_type, ..Default::default() };
        let desc = buffer_desc(bytes, D3D12_RESOURCE_FLAG_NONE);
        // D3D12の要件: UPLOADヒープの初期状態はGENERIC_READ固定、READBACK
        // ヒープの初期状態はCOPY_DEST固定(それ以外を渡すと
        // CreateCommittedResourceがE_INVALIDARG=0x80070057を返す)。
        let initial_state = if heap_type == D3D12_HEAP_TYPE_READBACK {
            D3D12_RESOURCE_STATE_COPY_DEST
        } else {
            D3D12_RESOURCE_STATE_GENERIC_READ
        };
        let resource: ID3D12Resource = unsafe {
            let mut result: Option<ID3D12Resource> = None;
            self.device
                .CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &desc, initial_state, None, &mut result)
                .context("CreateCommittedResource(staging) failed")?;
            result.ok_or_else(|| anyhow::anyhow!("CreateCommittedResource(staging) returned no resource"))?
        };
        let mut mapped: *mut c_void = std::ptr::null_mut();
        unsafe {
            resource.Map(0, None, Some(&mut mapped)).context("staging Resource::Map failed")?;
        }
        Ok((resource, mapped as *mut u8))
    }

    /// カーネル名からルートシグネチャ+Compute PSOを取得する。未作成
    /// なら`dxil`から新規作成しキャッシュする。
    fn pipeline_for(&self, name: &str, dxil: &[u8]) -> anyhow::Result<()> {
        let mut pipelines = self.pipelines.lock().unwrap();
        if pipelines.contains_key(name) {
            return Ok(());
        }

        let root_signature: ID3D12RootSignature =
            unsafe { self.device.CreateRootSignature(0, dxil).context("CreateRootSignature (from embedded DXIL) failed")? };

        let pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(root_signature.clone())),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as *const c_void, BytecodeLength: dxil.len() },
            ..Default::default()
        };
        let pso: ID3D12PipelineState =
            unsafe { self.device.CreateComputePipelineState(&pso_desc).context("CreateComputePipelineState failed")? };

        pipelines.insert(name.to_string(), Pipeline { root_signature, pso });
        Ok(())
    }
}

fn buffer_desc(bytes: usize, flags: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: bytes as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
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
        let resource = self.create_default_buffer(bytes)?;
        let handle = self.next_handle.fetch_add(1, Ordering::Relaxed);
        self.allocations.lock().unwrap().insert(handle, GpuBuffer { resource, len: bytes });
        Ok(DevicePtr::new(handle, self.info.id as u32))
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        if ptr.device_id as usize != self.info.id {
            return Err(GpuError::InvalidPtr(ptr).into());
        }
        if self.allocations.lock().unwrap().remove(&ptr.addr).is_none() {
            return Err(GpuError::InvalidPtr(ptr).into());
        }
        Ok(())
    }

    fn memcpy_h2d(&self, dst: DevicePtr, src: &[u8]) -> Result<()> {
        if dst.device_id as usize != self.info.id {
            return Err(GpuError::InvalidPtr(dst).into());
        }
        let dst_resource = {
            let map = self.allocations.lock().unwrap();
            let alloc = map.get(&dst.addr).ok_or(GpuError::InvalidPtr(dst))?;
            if src.len() > alloc.len {
                return Err(GpuError::OutOfMemory(src.len()).into());
            }
            alloc.resource.clone()
        };

        let (staging, mapped_ptr) = self.create_staging_buffer(src.len(), D3D12_HEAP_TYPE_UPLOAD)?;
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), mapped_ptr, src.len());
        }
        self.execute_and_wait(|cl| {
            unsafe { cl.CopyBufferRegion(&dst_resource, 0, &staging, 0, src.len() as u64) };
            Ok(())
        })?;
        Ok(())
    }

    fn memcpy_d2h(&self, dst: &mut [u8], src: DevicePtr) -> Result<()> {
        if src.device_id as usize != self.info.id {
            return Err(GpuError::InvalidPtr(src).into());
        }
        let src_resource = {
            let map = self.allocations.lock().unwrap();
            let alloc = map.get(&src.addr).ok_or(GpuError::InvalidPtr(src))?;
            if dst.len() > alloc.len {
                return Err(GpuError::InvalidPtr(src).into());
            }
            alloc.resource.clone()
        };

        let (staging, mapped_ptr) = self.create_staging_buffer(dst.len(), D3D12_HEAP_TYPE_READBACK)?;
        self.execute_and_wait(|cl| {
            unsafe { cl.CopyBufferRegion(&staging, 0, &src_resource, 0, dst.len() as u64) };
            Ok(())
        })?;
        unsafe {
            std::ptr::copy_nonoverlapping(mapped_ptr, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    fn memcpy_d2d(&self, dst: DevicePtr, src: DevicePtr, bytes: usize) -> Result<()> {
        if dst.device_id as usize != self.info.id || src.device_id as usize != self.info.id {
            return Err(GpuError::InvalidPtr(dst).into());
        }
        let (src_resource, dst_resource) = {
            let map = self.allocations.lock().unwrap();
            let src_alloc = map.get(&src.addr).ok_or(GpuError::InvalidPtr(src))?;
            if bytes > src_alloc.len {
                return Err(GpuError::InvalidPtr(src).into());
            }
            let dst_alloc = map.get(&dst.addr).ok_or(GpuError::InvalidPtr(dst))?;
            if bytes > dst_alloc.len {
                return Err(GpuError::InvalidPtr(dst).into());
            }
            (src_alloc.resource.clone(), dst_alloc.resource.clone())
        };
        self.execute_and_wait(|cl| {
            unsafe { cl.CopyBufferRegion(&dst_resource, 0, &src_resource, 0, bytes as u64) };
            Ok(())
        })?;
        Ok(())
    }

    fn launch_kernel(&self, kernel: &CompiledKernel, cfg: &LaunchConfig, args: &[KernelArg]) -> Result<()> {
        let dxil = match &kernel.source {
            KernelSource::Dxil(bytes) => bytes,
            other => return Err(GpuError::UnsupportedKernel(other.kind()).into()),
        };
        if !matches!(kernel.name.as_str(), "vector_add" | "vector_add_f32") {
            return Err(GpuError::UnsupportedKernel("DirectXDevice only dispatches vector_add/vector_add_f32").into());
        }
        if args.len() != 4 {
            return Err(GpuError::LaunchFailed("vector_add expects 4 args: a, b, c, n".to_string()).into());
        }
        let a = args[0].as_ptr().ok_or_else(|| GpuError::LaunchFailed("arg0 must be a device pointer".to_string()))?;
        let b = args[1].as_ptr().ok_or_else(|| GpuError::LaunchFailed("arg1 must be a device pointer".to_string()))?;
        let c = args[2].as_ptr().ok_or_else(|| GpuError::LaunchFailed("arg2 must be a device pointer".to_string()))?;
        let n = args[3].as_usize().ok_or_else(|| GpuError::LaunchFailed("arg3 must be usize/u32".to_string()))?;

        self.pipeline_for(&kernel.name, dxil)?;

        let (a_res, b_res, c_res) = {
            let map = self.allocations.lock().unwrap();
            let a_res = map.get(&a.addr).ok_or(GpuError::InvalidPtr(a))?.resource.clone();
            let b_res = map.get(&b.addr).ok_or(GpuError::InvalidPtr(b))?.resource.clone();
            let c_res = map.get(&c.addr).ok_or(GpuError::InvalidPtr(c))?.resource.clone();
            (a_res, b_res, c_res)
        };

        let pipelines = self.pipelines.lock().unwrap();
        let pipeline = pipelines.get(&kernel.name).expect("pipeline_for just inserted this key");
        let root_signature = pipeline.root_signature.clone();
        let pso = pipeline.pso.clone();
        drop(pipelines);

        let group_count_x = cfg.grid.0.max(1);
        let n_u32 = u32::try_from(n).map_err(|_| GpuError::LaunchFailed("n does not fit in u32".to_string()))?;

        self.execute_and_wait(|cl| {
            unsafe {
                cl.SetPipelineState(&pso);
                cl.SetComputeRootSignature(&root_signature);
                cl.SetComputeRootUnorderedAccessView(0, a_res.GetGPUVirtualAddress());
                cl.SetComputeRootUnorderedAccessView(1, b_res.GetGPUVirtualAddress());
                cl.SetComputeRootUnorderedAccessView(2, c_res.GetGPUVirtualAddress());
                cl.SetComputeRoot32BitConstant(3, n_u32, 0);
                cl.Dispatch(group_count_x, 1, 1);
            }
            Ok(())
        })?;
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        // execute_and_waitが操作ごとに同期しているため、追加で待つべき
        // 未完了のGPU作業は無い。
        Ok(())
    }

    fn supports_dxil(&self) -> bool {
        true
    }
}

impl Drop for DirectXDevice {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(self.fence_event);
        }
    }
}

pub fn enumerate_real(start_id: usize) -> anyhow::Result<Vec<Arc<dyn GpuDevice>>> {
    let device = DirectXDevice::new(start_id)?;
    Ok(vec![device as Arc<dyn GpuDevice>])
}
