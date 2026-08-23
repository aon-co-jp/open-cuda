//! `opencuda_blas::sgemm_directx_generic`(2026-08-23新設)の実ハードウェア
//! 検証。実際のD3D12デバイス上で`matmul.dxil`をディスパッチし、CPU参照
//! 実装(`sgemm`の`GemmPath::CpuNaive`経路)とビット単位ではなく数値許容差
//! 内で一致することを確認する。
//!
//! Windows以外、またはD3D12デバイスを構築できない環境では
//! `DirectXDevice::new`が失敗するので、テストは**失敗ではなくスキップ**
//! する(CIのLinuxランナーを壊さないため。既存のVulkan実機テストと同じ方針)。

#![cfg(windows)]

use opencuda_core::GpuDevice;
use std::sync::Arc;

const MATMUL_DXIL: &[u8] = include_bytes!("../../opencuda-directx/shaders/matmul.dxil");

#[test]
fn sgemm_directx_generic_matches_cpu_reference_on_real_d3d12() {
    let device: Arc<dyn GpuDevice> = match opencuda_directx::real::DirectXDevice::new(0) {
        Ok(d) => d,
        Err(err) => {
            eprintln!("skip: no real D3D12 device available ({err})");
            return;
        }
    };

    let (m, k, n) = (16usize, 24usize, 32usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.25 - 0.5).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.5 - 1.0).collect();

    let gpu = opencuda_blas::sgemm_directx_generic(&*device, m, k, n, &a, &b, MATMUL_DXIL).expect("directx sgemm");

    let cpu_device: Arc<dyn GpuDevice> = opencuda_cpu::CpuDevice::new(0);
    let mut cpu = vec![0.0f32; m * n];
    opencuda_blas::sgemm(&*cpu_device, m, k, n, 1.0, &a, &b, 0.0, &mut cpu, None).expect("cpu sgemm");

    assert_eq!(gpu.len(), cpu.len());
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert!((g - c).abs() < 1e-3, "element {i}: gpu={g} cpu={c}");
    }
}

#[test]
fn select_gemm_path_prefers_directx_when_device_only_supports_dxil() {
    let device: Arc<dyn GpuDevice> = match opencuda_directx::real::DirectXDevice::new(0) {
        Ok(d) => d,
        Err(err) => {
            eprintln!("skip: no real D3D12 device available ({err})");
            return;
        }
    };
    assert!(device.supports_dxil());
    assert!(!device.supports_spirv());
    assert_eq!(opencuda_blas::select_gemm_path(&*device), opencuda_blas::GemmPath::DirectXGeneric);
}
