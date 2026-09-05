//! `hgemm_vulkan_generic`/`dgemm_vulkan_generic`(Vulkan Compute)と
//! CPU経路(`hgemm`/`dgemm`)の実測比較(2026-09-05新設)。
//!
//! ベンチマークであって正しさの検証ではないので、`--nocapture`付きで
//! 実行して数値を読むためのテスト。GPT-2 124Mの実際のGEMM形状
//! (seq_len x hidden x {3*hidden, hidden, 4*hidden, vocab})を使うが、
//! `hgemm`のパッキング制約(k・nが偶数)に合わせてvocab_sizeのみ
//! 50256(偶数、50257から1減らした近似値)へ調整する。
//!
//! 2026-09-05時点のHANDOFFで「動くことは実証したが速度は未計測」と
//! 記録されていたギャップへの対応。

use half::f16;
use opencuda_core::GpuDevice;
use std::time::Instant;

const HGEMM_SPIRV: &[u8] = include_bytes!("../../../examples/hgemm_vulkan_real/shaders/hgemm.spv");
const DGEMM_SPIRV: &[u8] = include_bytes!("../../../examples/dgemm_vulkan_real/shaders/dgemm.spv");

#[test]
fn bench_hgemm_vulkan_vs_cpu_on_gpt2_shapes() {
    let device = match opencuda_vulkan::VulkanDevice::new(0) {
        Ok(d) => d,
        Err(err) => {
            eprintln!("skip: no real Vulkan device available ({err})");
            return;
        }
    };
    println!("Vulkan device: {}", device.info().name);

    // (m, k, n): m=seq_len(デコード時は1)、hidden=768。nは偶数のみ
    // (hgemmのパッキング制約)、vocab_sizeは50257→50256(偶数)へ近似。
    let shapes = [(1usize, 768usize, 2304usize), (1, 768, 768), (1, 768, 3072), (1, 3072, 768), (1, 768, 50256), (64, 768, 3072)];

    for (m, k, n) in shapes {
        let a: Vec<f16> = (0..m * k).map(|i| f16::from_f32((i % 13) as f32 * 0.01)).collect();
        let b: Vec<f16> = (0..k * n).map(|i| f16::from_f32((i % 11) as f32 * 0.01)).collect();

        // ウォームアップ(パイプラインキャッシュ生成をタイミングから除く)
        let _ = opencuda_blas::hgemm_vulkan_generic(&*device, m, k, n, &a, &b, HGEMM_SPIRV).expect("vulkan hgemm");
        let t = Instant::now();
        let _ = opencuda_blas::hgemm_vulkan_generic(&*device, m, k, n, &a, &b, HGEMM_SPIRV).expect("vulkan hgemm");
        let vulkan_ms = t.elapsed().as_secs_f64() * 1000.0;

        let mut c = vec![f16::from_f32(0.0); m * n];
        let t = Instant::now();
        opencuda_blas::hgemm(m, k, n, &a, &b, &mut c).expect("cpu hgemm");
        let cpu_ms = t.elapsed().as_secs_f64() * 1000.0;

        println!(
            "hgemm m={m:<3} k={k:<5} n={n:<6} vulkan={vulkan_ms:>9.3}ms  cpu={cpu_ms:>9.3}ms  speedup={:.3}x",
            cpu_ms / vulkan_ms
        );
    }
}

#[test]
fn bench_dgemm_vulkan_vs_cpu_on_gpt2_shapes() {
    let device = match opencuda_vulkan::VulkanDevice::new(0) {
        Ok(d) => d,
        Err(err) => {
            eprintln!("skip: no real Vulkan device available ({err})");
            return;
        }
    };
    println!("Vulkan device: {}", device.info().name);

    if !device.supports_f64_shader() {
        eprintln!("skip: this device/driver does not report shaderFloat64 support");
        return;
    }

    let shapes = [(1usize, 768usize, 2304usize), (1, 768, 768), (1, 768, 3072), (1, 3072, 768), (1, 768, 50257), (64, 768, 3072)];

    for (m, k, n) in shapes {
        let a: Vec<f64> = (0..m * k).map(|i| (i % 13) as f64 * 0.01).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i % 11) as f64 * 0.01).collect();

        let _ = opencuda_blas::dgemm_vulkan_generic(&*device, m, k, n, &a, &b, DGEMM_SPIRV).expect("vulkan dgemm");
        let t = Instant::now();
        let _ = opencuda_blas::dgemm_vulkan_generic(&*device, m, k, n, &a, &b, DGEMM_SPIRV).expect("vulkan dgemm");
        let vulkan_ms = t.elapsed().as_secs_f64() * 1000.0;

        let mut c = vec![0.0f64; m * n];
        let t = Instant::now();
        opencuda_blas::dgemm(m, k, n, 1.0, &a, &b, 0.0, &mut c).expect("cpu dgemm");
        let cpu_ms = t.elapsed().as_secs_f64() * 1000.0;

        println!(
            "dgemm m={m:<3} k={k:<5} n={n:<6} vulkan={vulkan_ms:>9.3}ms  cpu={cpu_ms:>9.3}ms  speedup={:.3}x",
            cpu_ms / vulkan_ms
        );
    }
}
