//! メモリの二層構造（設計確定1・2）。
//!
//! - 内部・生ポインタ層 `DevicePtr`: compat層とバックエンドが使う。Copy。
//!   `device_id` を埋め込むことで「どのGPU上のメモリか」を型で持ち、
//!   RTX 4090 と RX 7900 XTX の混在時にポインタの取り違えを早期に防ぐ。
//! - ネイティブAPI層 `DeviceBuffer`: ユーザーが使う。`Arc<dyn GpuDevice>` を
//!   保持し、Drop で自動解放（RAII）。

use std::sync::Arc;

use crate::device::GpuDevice;
use crate::error::Result;

/// 内部・生ポインタ層。どのデバイス上のアドレスかを型で持つ。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DevicePtr {
    pub addr: u64,
    pub device_id: u32,
}

impl DevicePtr {
    pub fn new(addr: u64, device_id: u32) -> Self {
        Self { addr, device_id }
    }

    /// ヌル相当（未割り当て）の判定に使う。
    pub fn is_null(&self) -> bool {
        self.addr == 0
    }
}

/// ネイティブAPI層。Drop で自動解放される安全なバッファ。
///
/// `Arc<dyn GpuDevice>` を保持するぶんバッファはやや重いが、安全側に倒す。
/// 速度が要る高速経路は compat 層が `DevicePtr` を直接触れるのでそちらで稼ぐ。
pub struct DeviceBuffer {
    ptr: DevicePtr,
    len: usize,
    device: Arc<dyn GpuDevice>,
}

impl DeviceBuffer {
    /// `alloc_buffer` 経由でのみ作る。バックエンドからは直接生成しない。
    pub(crate) fn from_parts(ptr: DevicePtr, len: usize, device: Arc<dyn GpuDevice>) -> Self {
        Self { ptr, len, device }
    }

    pub fn as_ptr(&self) -> DevicePtr {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn device(&self) -> &Arc<dyn GpuDevice> {
        &self.device
    }

    /// ホスト → このバッファへコピー。
    pub fn copy_from_host(&self, src: &[u8]) -> Result<()> {
        self.device.memcpy_h2d(self.ptr, src)
    }

    /// このバッファ → ホストへコピー。
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
        self.device.memcpy_d2h(dst, self.ptr)
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // Drop はパニック不可。解放失敗はログのみ。
        if let Err(e) = self.device.free(self.ptr) {
            tracing::warn!("DeviceBuffer drop: free failed for {:?}: {e}", self.ptr);
        }
    }
}

/// ネイティブ層のヘルパー: デバイスからバッファを確保する。
/// `Arc<dyn GpuDevice>` を渡すことで、解放先がバッファに紐づく。
pub fn alloc_buffer(device: &Arc<dyn GpuDevice>, len: usize) -> Result<DeviceBuffer> {
    let ptr = device.alloc(len)?;
    Ok(DeviceBuffer::from_parts(ptr, len, Arc::clone(device)))
}
