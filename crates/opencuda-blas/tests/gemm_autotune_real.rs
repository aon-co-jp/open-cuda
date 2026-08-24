//! `autotune::decide_dxil_offload`の実機検証(2026-08-24新設)。
//!
//! 実D3D12デバイスが無い環境ではスキップする(CIやLinuxでも壊れない)。

#![cfg(windows)]

use opencuda_blas::autotune::{decide_dxil_offload, OffloadPolicy};
use opencuda_core::GpuDevice;
use std::sync::Arc;

const MATMUL_DXIL: &[u8] = include_bytes!("../../opencuda-directx/shaders/matmul.dxil");

#[test]
fn decide_dxil_offload_measures_and_reports_on_real_hardware() {
    let device: Arc<dyn GpuDevice> = match opencuda_directx::real::DirectXDevice::new(0) {
        Ok(d) => d,
        Err(err) => {
            eprintln!("skip: no real D3D12 device available ({err})");
            return;
        }
    };
    println!("D3D12 device: {}", device.info().name);

    // 小さめの形状で実測(テスト時間を抑えるため`lm_head`相当は含めない)。
    let shapes = [(1usize, 256usize, 512usize), (1, 512, 256)];
    let d = decide_dxil_offload(&*device, MATMUL_DXIL, &shapes, 2).expect("decide");
    println!("{}", d.summary());

    assert_eq!(d.probes.len(), shapes.len());
    assert!(d.probes.iter().all(|p| p.gpu_ms > 0.0 && p.cpu_ms > 0.0), "both paths must be actually measured");
    // 数値検証が通っていること(GPUカーネルが壊れていない)。
    assert!(d.numerics_ok, "GPU result must match the CPU reference: {}", d.reason);
    // 判定は実測に従うこと(policy=Autoのとき)。
    if d.policy == OffloadPolicy::Auto {
        assert_eq!(d.use_gpu, d.gpu_total_ms < d.cpu_total_ms, "auto policy must follow the measurement");
    }
}
