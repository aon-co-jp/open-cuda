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

    // 2026-08-15追加(ユーザー指示: スマホの「実システムメモリ+仮想化
    // メモリ」はPCと事情が違うので活かして欲しい): デスクトップPC
    // (大容量RAM+ページファイル、確保しすぎてもOSが緩やかに対応)とは
    // 異なり、Androidは物理メモリが少なく(実機で確認済み: 総容量
    // 3.5GB、実際に空いているのは1.3GB程度——残りは他アプリ・
    // キャッシュが使用中)、メモリ逼迫時はページングではなく
    // Low Memory Killer(LMK)がプロセスを強制終了させる。統合GPU
    // (UMA)はCPU/GPUで同じ物理メモリを共有するため、GPU用に確保した
    // 分だけ他プロセスから見える空きメモリも減る——この特性を無視して
    // PC感覚で大きな行列を確保すると、LMKに問答無用でkillされうる。
    // `/proc/meminfo`(Linux/Android共通)が読めれば確保前に警告する
    // (Windows等/procが無い環境では単にスキップし、確保自体は試みる)。
    if let Some(mem) = read_mem_status() {
        let matrices_bytes = 3 * size * size * std::mem::size_of::<f32>(); // a, b, c
        // 物理RAMの空き(MemAvailable)に加え、Androidは仮想メモリ層
        // (zram圧縮スワップ、実機確認: SwapTotal約2.3GB)を持つ——
        // 実測(このmoto g53y 5G)では物理RAM約3.5GB+スワップ約2.3GB
        // ≈合計5.8GBが実質的な確保余地。物理側だけを見て安全域を
        // 判定すると、実際にはスワップに余裕があるケースを過度に
        // 警告してしまう(逆に、スワップは書き込み速度がRAMより遅く、
        // 使用中はアプリの体感速度低下・バッテリー消費増につながる
        // ため、「使えるから使う」のではなく参考情報として両方を
        // 提示するに留める)。
        let physical_available_bytes = mem.available_kb * 1024;
        let swap_free_bytes = mem.swap_free_kb.unwrap_or(0) * 1024;
        let total_headroom_bytes = physical_available_bytes + swap_free_bytes;
        let usage_ratio_physical = matrices_bytes as f64 / physical_available_bytes as f64;
        println!(
            "メモリ状況: 物理RAM空き約{:.1}MB + スワップ空き約{:.1}MB(仮想メモリ、\
             実RAMより低速) = 合計余地約{:.1}MB。本ベンチの推定使用量約{:.1}MB\
             (物理RAM空きの{:.1}%)",
            physical_available_bytes as f64 / 1_048_576.0,
            swap_free_bytes as f64 / 1_048_576.0,
            total_headroom_bytes as f64 / 1_048_576.0,
            matrices_bytes as f64 / 1_048_576.0,
            usage_ratio_physical * 100.0,
        );
        if usage_ratio_physical > 0.5 {
            eprintln!(
                "警告: 推定使用量が物理RAM空きの50%を超えています。Android実機ではLow \
                 Memory Killerに強制終了させられる可能性があります(CPU/GPUは同じ物理\
                 メモリを共有するUMA構成のため、GPU用の確保分も他プロセスの空きメモリを\
                 圧迫します)。スワップに余裕があっても、スワップは低速でありLMKの判断は\
                 主に物理メモリ圧迫を見るため、物理RAM空きに収まるサイズを推奨します。"
            );
        }
    }

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

struct MemStatus {
    available_kb: u64,
    swap_free_kb: Option<u64>,
}

/// `/proc/meminfo`から`MemAvailable`(物理RAM空き)と`SwapFree`
/// (仮想メモリ/スワップ空き、Android機種によってはzram圧縮スワップ)を
/// 読む。Linux/Android共通のインターフェースで、Windows等には存在
/// しないため`None`を返す(呼び出し側はチェックをスキップするだけで、
/// 確保自体は試みる)。
fn read_mem_status() -> Option<MemStatus> {
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut available_kb = None;
    let mut swap_free_kb = None;
    for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            available_kb = rest.chars().filter(|c| c.is_ascii_digit()).collect::<String>().parse().ok();
        } else if let Some(rest) = line.strip_prefix("SwapFree:") {
            swap_free_kb = rest.chars().filter(|c| c.is_ascii_digit()).collect::<String>().parse().ok();
        }
    }
    available_kb.map(|available_kb| MemStatus { available_kb, swap_free_kb })
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
