//! デバイスの登録・検出・選択。
//!
//! バックエンドは feature でビルド時に切り離されるため、Registry 自体は
//! バックエンドに依存しない。各バックエンドが自分を Registry に push する形にする
//! ことで、core はどのバックエンドにも依存しない（依存方向が一方向に保たれる）。

use std::sync::Arc;

use crate::device::{GpuDevice, GpuVendor};
use crate::error::{GpuError, Result};

/// 利用可能なデバイス群を保持する。
#[derive(Default)]
pub struct DeviceRegistry {
    devices: Vec<Arc<dyn GpuDevice>>,
}

impl DeviceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// バックエンドが検出したデバイスを登録する。
    pub fn register(&mut self, device: Arc<dyn GpuDevice>) {
        self.devices.push(device);
    }

    pub fn len(&self) -> usize {
        self.devices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.devices.is_empty()
    }

    pub fn get(&self, id: usize) -> Result<Arc<dyn GpuDevice>> {
        self.devices
            .get(id)
            .cloned()
            .ok_or_else(|| {
                GpuError::DeviceOutOfRange {
                    requested: id,
                    available: self.devices.len(),
                }
                .into()
            })
    }

    pub fn all(&self) -> &[Arc<dyn GpuDevice>] {
        &self.devices
    }

    /// 推論向けに最良のデバイスを選ぶ。
    /// 当面は「メモリ最大」を基準にする（マルチGPUでは将来 topology も考慮）。
    pub fn best_for_inference(&self) -> Result<Arc<dyn GpuDevice>> {
        self.devices
            .iter()
            .max_by_key(|d| d.info().total_memory)
            .cloned()
            .ok_or_else(|| GpuError::NoDevice.into())
    }

    /// 指定ベンダーのデバイスだけ取り出す。
    pub fn by_vendor<'a>(
        &'a self,
        pred: impl Fn(&GpuVendor) -> bool + 'a,
    ) -> impl Iterator<Item = &'a Arc<dyn GpuDevice>> {
        self.devices.iter().filter(move |d| pred(&d.info().vendor))
    }
}
