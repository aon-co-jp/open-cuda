//! matmul_vulkan_real: v0.4.0 の実Vulkan matmul最小サンプル。
//!
//! CPUバックエンド(rayon)の naive matmul と、実Vulkan Compute(SPIR-V)の naive matmul を
//! 同じ入力行列で実行し、両者がホスト側で計算したリファレンス値と一致することを確認する。
//! v0.4.0の方針(DEVELOPMENT-NEXT.md)通り、性能ではなく正確性を最優先する。
//!
//! 事前に Vulkan SDK の `glslc` などで `shaders/matmul.comp` を
//! `shaders/matmul.spv` にコンパイルしてから実行する
//! (`tools/compile-vulkan-shaders.{ps1,cmd,sh}` が両方の .comp をまとめてコンパイルする)。

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use opencuda_core::{
    alloc_buffer, CompiledKernel, GpuDevice, KernelArg, LaunchConfig, ResolvedArg, ThreadCtx,
};
use opencuda_cpu::CpuDevice;
use opencuda_vulkan::VulkanDevice;

const M: usize = 64;
const K: usize = 64;
const N: usize = 64;
const EPS: f32 = 1e-3;

fn main() -> Result<()> {
    let spv_path = shader_path()?;
    let spirv = std::fs::read(&spv_path).with_context(|| {
        format!(
            "failed to read {}. Compile it first, for example: glslc shaders/matmul.comp -o shaders/matmul.spv",
            spv_path.display()
        )
    })?;

    let a: Vec<f32> = (0..M * K).map(|i| (i % 7) as f32).collect();
    let b: Vec<f32> = (0..K * N).map(|i| (i % 5) as f32).collect();

    let c_ref = matmul_reference(&a, &b);
    let c_cpu = run_cpu(&a, &b)?;
    let c_vulkan = run_vulkan(&a, &b, &spirv)?;

    compare("CPU vs reference", &c_cpu, &c_ref)?;
    compare("Vulkan vs reference", &c_vulkan, &c_ref)?;
    compare("Vulkan vs CPU", &c_vulkan, &c_cpu)?;

    println!("OK: matmul {M}x{K} * {K}x{N} verified: CPU and real Vulkan Compute agree with the reference");
    Ok(())
}

fn matmul_reference(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; M * N];
    for row in 0..M {
        for col in 0..N {
            let mut acc = 0.0f32;
            for kk in 0..K {
                acc += a[row * K + kk] * b[kk * N + col];
            }
            c[row * N + col] = acc;
        }
    }
    c
}

fn run_cpu(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    let device: Arc<dyn GpuDevice> = CpuDevice::new(0);

    let da = alloc_buffer(&device, M * K * 4)?;
    let db = alloc_buffer(&device, K * N * 4)?;
    let dc = alloc_buffer(&device, M * N * 4)?;
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

    let cfg = LaunchConfig::linear((M * N) as u32, 256);
    device.launch_kernel(
        &kernel,
        &cfg,
        &[
            KernelArg::Ptr(da.as_ptr()),
            KernelArg::Ptr(db.as_ptr()),
            KernelArg::Ptr(dc.as_ptr()),
            KernelArg::Usize(M),
            KernelArg::Usize(K),
            KernelArg::Usize(N),
        ],
    )?;
    device.synchronize()?;

    let mut c = vec![0.0f32; M * N];
    dc.copy_to_host(cast_f32_to_u8_mut(&mut c))?;
    Ok(c)
}

fn run_vulkan(a: &[f32], b: &[f32], spirv: &[u8]) -> Result<Vec<f32>> {
    let device: Arc<dyn GpuDevice> = VulkanDevice::new(0)?;
    println!("device: {}", device.info().name);

    let da = alloc_buffer(&device, M * K * 4)?;
    let db = alloc_buffer(&device, K * N * 4)?;
    let dc = alloc_buffer(&device, M * N * 4)?;
    da.copy_from_host(cast_f32_to_u8(a))?;
    db.copy_from_host(cast_f32_to_u8(b))?;

    // シェーダの local_size_x/y = 16 と一致させる (opencuda-vulkan real.rs の
    // dispatch_spirv はワークグループ数をそのまま vkCmdDispatch に渡すため)。
    let cfg = LaunchConfig::grid2d(M as u32, N as u32, 16, 16);
    let kernel = CompiledKernel::spirv("matmul", "main", spirv);

    device.launch_kernel(
        &kernel,
        &cfg,
        &[
            KernelArg::Ptr(da.as_ptr()),
            KernelArg::Ptr(db.as_ptr()),
            KernelArg::Ptr(dc.as_ptr()),
            KernelArg::Usize(M),
            KernelArg::Usize(K),
            KernelArg::Usize(N),
        ],
    )?;
    device.synchronize()?;

    let mut c = vec![0.0f32; M * N];
    dc.copy_to_host(cast_f32_to_u8_mut(&mut c))?;
    Ok(c)
}

fn compare(label: &str, got: &[f32], expected: &[f32]) -> Result<()> {
    for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        if (g - e).abs() > EPS {
            anyhow::bail!("{label} mismatch at {idx}: got {g}, expected {e}");
        }
    }
    println!("OK: {label} matches within {EPS}");
    Ok(())
}

fn shader_path() -> Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    Ok(manifest_dir.join("shaders").join("matmul.spv"))
}

fn cast_f32_to_u8(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn cast_f32_to_u8_mut(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}
