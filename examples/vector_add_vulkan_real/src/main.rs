//! vector_add_vulkan_real: v0.3.5 の実Vulkan Compute最小サンプル。
//!
//! 事前に Vulkan SDK の `glslc` などで `shaders/vector_add.comp` を
//! `shaders/vector_add.spv` にコンパイルしてから実行する。

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use opencuda_core::{alloc_buffer, CompiledKernel, GpuDevice, KernelArg, LaunchConfig};
use opencuda_vulkan::VulkanDevice;

const N: usize = 1_000_000;

fn main() -> Result<()> {
    let spv_path = shader_path()?;
    let spirv = std::fs::read(&spv_path).with_context(|| {
        format!(
            "failed to read {}. Compile it first, for example: glslc shaders/vector_add.comp -o shaders/vector_add.spv",
            spv_path.display()
        )
    })?;

    let device: Arc<dyn GpuDevice> = VulkanDevice::new(0)?;
    println!("device: {}", device.info().name);

    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| (N - i) as f32).collect();
    let bytes = N * std::mem::size_of::<f32>();

    let da = alloc_buffer(&device, bytes)?;
    let db = alloc_buffer(&device, bytes)?;
    let dc = alloc_buffer(&device, bytes)?;

    da.copy_from_host(cast_f32_to_u8(&a))?;
    db.copy_from_host(cast_f32_to_u8(&b))?;

    let cfg = LaunchConfig::linear(N as u32, 256);
    let kernel = CompiledKernel::spirv("vector_add", "main", spirv);

    device.launch_kernel(
        &kernel,
        &cfg,
        &[
            KernelArg::Ptr(da.as_ptr()),
            KernelArg::Ptr(db.as_ptr()),
            KernelArg::Ptr(dc.as_ptr()),
            KernelArg::Usize(N),
        ],
    )?;
    device.synchronize()?;

    let mut c = vec![0.0f32; N];
    dc.copy_to_host(cast_f32_to_u8_mut(&mut c))?;

    let expected = N as f32;
    for (idx, &v) in c.iter().enumerate() {
        if (v - expected).abs() > 1e-3 {
            anyhow::bail!("mismatch at {idx}: got {v}, expected {expected}");
        }
    }

    println!("OK: real Vulkan Compute produced correct vector_add result");
    println!("c[0]={}, c[{}]={}", c[0], N - 1, c[N - 1]);
    Ok(())
}

fn shader_path() -> Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    Ok(manifest_dir.join("shaders").join("vector_add.spv"))
}

fn cast_f32_to_u8(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn cast_f32_to_u8_mut(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}
