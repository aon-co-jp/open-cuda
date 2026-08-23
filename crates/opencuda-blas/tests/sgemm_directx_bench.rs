//! `sgemm_directx_generic`(D3D12 Compute)とCPU経路(`sgemm`の
//! `GemmPath::CpuNaive`、`open-cpu`のSIMDディスパッチが効く)の実測比較。
//!
//! ベンチマークであって正しさの検証ではないので、`--nocapture`付きで
//! 実行して数値を読むためのテスト。GPT-2 124Mの実際のGEMM形状
//! (seq_len x hidden x {3*hidden, hidden, 4*hidden, vocab})を使う。

#![cfg(windows)]

use opencuda_core::GpuDevice;
use std::sync::Arc;
use std::time::Instant;

const MATMUL_DXIL: &[u8] = include_bytes!("../../opencuda-directx/shaders/matmul.dxil");

#[test]
fn bench_sgemm_directx_vs_cpu_on_gpt2_shapes() {
    let device: Arc<dyn GpuDevice> = match opencuda_directx::real::DirectXDevice::new(0) {
        Ok(d) => d,
        Err(err) => {
            eprintln!("skip: no real D3D12 device available ({err})");
            return;
        }
    };
    let cpu: Arc<dyn GpuDevice> = opencuda_cpu::CpuDevice::new(0);
    println!("D3D12 device: {}", device.info().name);

    // (m, k, n): m=seq_len(デコード時は1)、GPT-2 124M の hidden=768, vocab=50257
    let shapes = [(1usize, 768usize, 2304usize), (1, 768, 768), (1, 768, 3072), (1, 3072, 768), (1, 768, 50257), (64, 768, 3072)];

    for (m, k, n) in shapes {
        let a: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.01).collect();

        // ウォームアップ(PSOキャッシュ生成をタイミングから除く)
        let _ = opencuda_blas::sgemm_directx_generic(&*device, m, k, n, &a, &b, MATMUL_DXIL).expect("dx sgemm");
        let t = Instant::now();
        let _ = opencuda_blas::sgemm_directx_generic(&*device, m, k, n, &a, &b, MATMUL_DXIL).expect("dx sgemm");
        let dx_ms = t.elapsed().as_secs_f64() * 1000.0;

        // 重み常駐版(Bを1度だけ転送)
        let b_ptr = opencuda_blas::upload_resident_matrix(&*device, &b).expect("upload b");
        let _ = opencuda_blas::sgemm_directx_resident_b(&*device, m, k, n, &a, b_ptr, MATMUL_DXIL).expect("dx resident");
        let t = Instant::now();
        let _ = opencuda_blas::sgemm_directx_resident_b(&*device, m, k, n, &a, b_ptr, MATMUL_DXIL).expect("dx resident");
        let dxres_ms = t.elapsed().as_secs_f64() * 1000.0;
        device.free(b_ptr).expect("free b");

        let mut c = vec![0.0f32; m * n];
        let t = Instant::now();
        opencuda_blas::sgemm(&*cpu, m, k, n, 1.0, &a, &b, 0.0, &mut c, None).expect("cpu sgemm");
        let cpu_ms = t.elapsed().as_secs_f64() * 1000.0;

        println!(
            "m={m:<3} k={k:<5} n={n:<6} directx={dx_ms:>9.3}ms  directx_resident_b={dxres_ms:>9.3}ms  cpu={cpu_ms:>9.3}ms  \
             speedup(naive)={:.2}x speedup(resident)={:.2}x",
            cpu_ms / dx_ms,
            cpu_ms / dxres_ms
        );
    }
}
