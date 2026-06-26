//! vector_add_omniir: OpenCUDA v0.2 の最小 OmniIR サンプル。
//!
//! 同じ `IrModule::vector_add_f32()` から、CPU Native 経路と VulkanMock の OmniIR 経路を実行する。
//! 実Vulkan/GPU実行ではないが、IR → backend という Phase 2 入口の契約をGPUなしで検証できる。

use std::sync::Arc;

use anyhow::Result;
use opencuda_core::{alloc_buffer, GpuDevice, KernelArg, LaunchConfig};
use opencuda_cpu::CpuDevice;
use opencuda_ir::{lower_to_native, IrModule};
use opencuda_vulkan::VulkanMockDevice;

const N: usize = 1_000_000;

fn main() -> Result<()> {
    let module = IrModule::vector_add_f32();
    println!("OmniIR module: {} / entry {}", module.name, module.entry);

    run_on_cpu_from_omniir(&module)?;
    run_on_vulkan_mock_from_omniir(&module)?;

    println!("OK: same OmniIR vector_add_f32 verified on CPU Native and VulkanMock OmniIR paths");
    Ok(())
}

fn run_on_cpu_from_omniir(module: &IrModule) -> Result<()> {
    let device: Arc<dyn GpuDevice> = CpuDevice::new(0);
    let kernel = lower_to_native(module)?;
    run_vector_add(device, kernel, "CPU Native lowered from OmniIR")
}

fn run_on_vulkan_mock_from_omniir(module: &IrModule) -> Result<()> {
    let device: Arc<dyn GpuDevice> = VulkanMockDevice::new(0);
    let kernel = module.to_compiled_omniir();
    run_vector_add(device, kernel, "VulkanMock OmniIR -> SPIR-V fixture")
}

fn run_vector_add(
    device: Arc<dyn GpuDevice>,
    kernel: opencuda_core::CompiledKernel,
    label: &str,
) -> Result<()> {
    println!("device: {} ({label})", device.info().name);

    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| (N - i) as f32).collect();
    let bytes = N * std::mem::size_of::<f32>();

    let da = alloc_buffer(&device, bytes)?;
    let db = alloc_buffer(&device, bytes)?;
    let dc = alloc_buffer(&device, bytes)?;

    da.copy_from_host(cast_f32_to_u8(&a))?;
    db.copy_from_host(cast_f32_to_u8(&b))?;

    let cfg = LaunchConfig::linear(N as u32, 256);
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
            anyhow::bail!("{label}: mismatch at {idx}: got {v}, expected {expected}");
        }
    }
    println!("OK: {label}: c[0]={}, c[{}]={}", c[0], N - 1, c[N - 1]);
    Ok(())
}

fn cast_f32_to_u8(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn cast_f32_to_u8_mut(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}
