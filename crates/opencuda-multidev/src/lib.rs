//! # opencuda-multidev
//!
//! 複数デバイスにまたがる実行を司る（Phase 3）。主用途は LLM のレイヤー分割
//! （Pipeline Parallelism）。例: RTX 4090 24GB + RX 7900 XTX 24GB = 統合48GB に
//! Qwen3-14B を展開する。
//!
//! 重要な制約: NVIDIA↔AMD 間は NVLink/Infinity Fabric の直結が使えないため、
//! デバイス間転送は PCIe 経由（ホストメモリ経由のステージング）になる。転送量の
//! 少ないパイプライン分割がテンソル並列より有利。KV Cache は後段デバイス側に
//! 置くと転送が減る。

use std::sync::Arc;

use opencuda_core::{GpuDevice, Result};

/// モデルのレイヤーをどのデバイスに載せるかの割り当て。
#[derive(Debug, Clone)]
pub struct DevicePartition {
    pub device_id: usize,
    pub layer_start: usize,
    pub layer_end: usize, // exclusive
}

/// レイヤーを VRAM 容量比で各デバイスに分割する。
///
/// 当面は「メモリ比に応じた均等割り」。将来は層ごとの重みサイズや
/// 転送コストを考慮した最適化に置き換える。
pub fn partition_layers(
    devices: &[Arc<dyn GpuDevice>],
    total_layers: usize,
) -> Result<Vec<DevicePartition>> {
    if devices.is_empty() {
        anyhow::bail!("no devices to partition across");
    }

    let total_mem: u128 = devices
        .iter()
        .map(|d| d.info().total_memory as u128)
        .sum();

    // total_memory が未申告(0)のバックエンド（CPU等）だけのときは均等割り。
    let mut partitions = Vec::with_capacity(devices.len());
    let mut offset = 0usize;

    for (i, d) in devices.iter().enumerate() {
        let n = if total_mem == 0 {
            // 均等割り（端数は最後のデバイスに寄せる）。
            if i + 1 == devices.len() {
                total_layers - offset
            } else {
                total_layers / devices.len()
            }
        } else {
            let ratio = d.info().total_memory as u128;
            if i + 1 == devices.len() {
                total_layers - offset
            } else {
                ((total_layers as u128 * ratio) / total_mem) as usize
            }
        };

        let end = (offset + n).min(total_layers);
        partitions.push(DevicePartition {
            device_id: i,
            layer_start: offset,
            layer_end: end,
        });
        offset = end;
    }

    Ok(partitions)
}

/// デバイス間転送（PCIe ステージング）。
/// TODO(Phase 3): ホストメモリ経由の d2h → h2d 実装。
pub fn transfer_between_devices(
    _src: &Arc<dyn GpuDevice>,
    _dst: &Arc<dyn GpuDevice>,
) -> Result<()> {
    anyhow::bail!("transfer_between_devices: not yet implemented (Phase 3)")
}

#[cfg(test)]
mod tests {
    // 分割ロジックのユニットテストは PC 上で `cargo test` で回せる。
    // （デバイス実体が要るため、ここではロジックの形だけ示す。）
}
