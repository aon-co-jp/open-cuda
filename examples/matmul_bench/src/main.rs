//! matmul_bench: CPU/GPU(Vulkan)の実行時間を個別計測する専用ベンチマーク。
//!
//! `matmul_vulkan_real`は正しさの検証(数値一致)のみを目的とし、速度計測の
//! 仕組みを持たない——小さい行列(64x64)を1回実行するだけなので、デバイス
//! 初期化オーバーヘッドが支配的になり速度比較の材料にならなかった
//! (2026-08-15、Android実機検証時に判明)。本クレートはこれを解消し、
//! 「デバイス初期化を含む合計時間」と「デバイス初期化を除いた純計算時間
//! (複数回実行の中央値)」の両方を計測・報告する。
//!
//! 使い方: `matmul_bench [size] [iterations]`(省略時 size=256, iterations=5)

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use opencuda_core::{
    alloc_buffer, CompiledKernel, GpuDevice, KernelArg, LaunchConfig, ResolvedArg, ThreadCtx,
};
use opencuda_cpu::CpuDevice;
use opencuda_vulkan::VulkanDevice;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let size: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(256);
    let iterations: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);

    println!("matmul_bench: size={size}x{size}, iterations={iterations}");

    let a: Vec<f32> = (0..size * size).map(|i| (i % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..size * size).map(|i| (i % 5) as f32 * 0.1).collect();

    let spv_path = shader_path()?;
    let spirv = std::fs::read(&spv_path).with_context(|| {
        format!(
            "failed to read {}. Compile it first, for example: glslc shaders/matmul.comp -o shaders/matmul.spv",
            spv_path.display()
        )
    })?;

    // --- CPU: デバイス構築(初期化含む)+計算 の合計時間、および計算のみの時間 ---
    let cpu_setup_start = Instant::now();
    let cpu_device: Arc<dyn GpuDevice> = CpuDevice::new(0);
    println!("cpu device: {}", cpu_device.info().name);
    let cpu_setup_elapsed = cpu_setup_start.elapsed();

    let mut cpu_compute_times = Vec::with_capacity(iterations);
    let mut last_cpu_result = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        last_cpu_result = run_matmul_native(&cpu_device, size, &a, &b)?;
        cpu_compute_times.push(start.elapsed());
    }

    // --- GPU(Vulkan): デバイス構築(実機初期化含む、GPU側は通常CPUより
    // 重い)+計算 の合計時間、および計算のみの時間 ---
    let gpu_setup_start = Instant::now();
    let gpu_device: Arc<dyn GpuDevice> = VulkanDevice::new(0)?;
    println!("gpu device: {}", gpu_device.info().name);
    let gpu_setup_elapsed = gpu_setup_start.elapsed();

    let mut gpu_compute_times = Vec::with_capacity(iterations);
    let mut last_gpu_result = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        last_gpu_result = run_matmul_spirv(&gpu_device, size, &a, &b, &spirv)?;
        gpu_compute_times.push(start.elapsed());
    }

    // 正しさの確認(誇張しない速度比較のため、数値が一致しない場合は
    // ベンチマーク結果そのものを信頼できないと明示して失敗させる)。
    for (idx, (&c, &g)) in last_cpu_result.iter().zip(last_gpu_result.iter()).enumerate() {
        if (c - g).abs() > 1e-2 {
            anyhow::bail!("CPU/GPU結果が不一致(idx={idx}, cpu={c}, gpu={g})——ベンチマーク結果は無効");
        }
    }
    println!("OK: CPU/GPU results match (results are trustworthy for timing comparison)");

    report("CPU", cpu_setup_elapsed, &cpu_compute_times);
    report("GPU (Vulkan)", gpu_setup_elapsed, &gpu_compute_times);

    let cpu_median = median(&cpu_compute_times);
    let gpu_median = median(&gpu_compute_times);
    if gpu_median < cpu_median {
        println!(
            "=> GPUの方が計算のみで{:.2}倍速い(初期化時間は含まない、複数回実行の中央値比較)",
            cpu_median.as_secs_f64() / gpu_median.as_secs_f64()
        );
    } else {
        println!(
            "=> CPUの方が計算のみで{:.2}倍速い(初期化時間は含まない、複数回実行の中央値比較)",
            gpu_median.as_secs_f64() / cpu_median.as_secs_f64()
        );
    }

    Ok(())
}

fn report(label: &str, setup: Duration, compute_times: &[Duration]) {
    let median = median(compute_times);
    let min = compute_times.iter().min().copied().unwrap_or_default();
    let max = compute_times.iter().max().copied().unwrap_or_default();
    println!(
        "{label}: setup(init)={:.3}ms, compute median={:.3}ms (min={:.3}ms, max={:.3}ms, n={})",
        setup.as_secs_f64() * 1000.0,
        median.as_secs_f64() * 1000.0,
        min.as_secs_f64() * 1000.0,
        max.as_secs_f64() * 1000.0,
        compute_times.len(),
    );
}

fn median(durations: &[Duration]) -> Duration {
    let mut sorted: Vec<Duration> = durations.to_vec();
    sorted.sort();
    sorted.get(sorted.len() / 2).copied().unwrap_or_default()
}

fn run_matmul_native(device: &Arc<dyn GpuDevice>, size: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    let m = size;
    let k = size;
    let n = size;

    let da = alloc_buffer(device, m * k * 4)?;
    let db = alloc_buffer(device, k * n * 4)?;
    let dc = alloc_buffer(device, m * n * 4)?;
    da.copy_from_host(cast_f32_to_u8(a))?;
    db.copy_from_host(cast_f32_to_u8(b))?;

    let kernel = CompiledKernel::native("matmul_naive", |ctx: ThreadCtx, args: &[ResolvedArg]| {
        let idx = ctx.global_id_x() as usize;
        let m = args[3].as_usize().unwrap();
        let k = args[4].as_usize().unwrap();
        let n = args[5].as_usize().unwrap();
        if idx >= m * n {
            return;
        }
        let row = idx / n;
        let col = idx % n;
        let (a_ptr, _) = args[0].as_ptr().unwrap();
        let (b_ptr, _) = args[1].as_ptr().unwrap();
        let (c_ptr, _) = args[2].as_ptr().unwrap();

        let mut acc = 0.0f32;
        unsafe {
            let a = a_ptr as *const f32;
            let b = b_ptr as *const f32;
            let c = c_ptr as *mut f32;
            for kk in 0..k {
                acc += a.add(row * k + kk).read() * b.add(kk * n + col).read();
            }
            c.add(idx).write(acc);
        }
    });

    let cfg = LaunchConfig::linear((m * n) as u32, 256);
    device.launch_kernel(
        &kernel,
        &cfg,
        &[
            KernelArg::Ptr(da.as_ptr()),
            KernelArg::Ptr(db.as_ptr()),
            KernelArg::Ptr(dc.as_ptr()),
            KernelArg::Usize(m),
            KernelArg::Usize(k),
            KernelArg::Usize(n),
        ],
    )?;
    device.synchronize()?;

    let mut c = vec![0.0f32; m * n];
    dc.copy_to_host(cast_f32_to_u8_mut(&mut c))?;
    Ok(c)
}

fn run_matmul_spirv(
    device: &Arc<dyn GpuDevice>,
    size: usize,
    a: &[f32],
    b: &[f32],
    spirv: &[u8],
) -> Result<Vec<f32>> {
    let m = size;
    let k = size;
    let n = size;

    let da = alloc_buffer(device, m * k * 4)?;
    let db = alloc_buffer(device, k * n * 4)?;
    let dc = alloc_buffer(device, m * n * 4)?;
    da.copy_from_host(cast_f32_to_u8(a))?;
    db.copy_from_host(cast_f32_to_u8(b))?;

    let cfg = LaunchConfig::grid2d(m as u32, n as u32, 16, 16);
    let kernel = CompiledKernel::spirv("matmul", "main", spirv);

    device.launch_kernel(
        &kernel,
        &cfg,
        &[
            KernelArg::Ptr(da.as_ptr()),
            KernelArg::Ptr(db.as_ptr()),
            KernelArg::Ptr(dc.as_ptr()),
            KernelArg::Usize(m),
            KernelArg::Usize(k),
            KernelArg::Usize(n),
        ],
    )?;
    device.synchronize()?;

    let mut c = vec![0.0f32; m * n];
    dc.copy_to_host(cast_f32_to_u8_mut(&mut c))?;
    Ok(c)
}

fn cast_f32_to_u8(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn cast_f32_to_u8_mut(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}

/// 実行ファイルと同じディレクトリの`shaders/matmul.spv`を優先し
/// (Android実機等へ`adb push`する運用向け)、無ければ開発機のビルド時
/// パスへフォールバックする(`matmul_vulkan_real`と同じパターン)。
fn shader_path() -> Result<PathBuf> {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let candidate = exe_dir.join("shaders").join("matmul.spv");
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    Ok(manifest_dir.join("shaders").join("matmul.spv"))
}
