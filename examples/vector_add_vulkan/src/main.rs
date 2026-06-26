//! vector_add_vulkan: GPUなしで SPIR-V 経路を検証する Phase 1.5 サンプル。
//!
//! 重要: このサンプルは `VulkanMockDevice` を使うため、実Vulkan/GPU実行ではない。
//! 目的は、Vulkan系バックエンドが `Native` カーネルを拒否し、`SpirV` カーネルだけを
//! 受け付ける設計を、実機なしでテスト可能にすること。

use std::sync::Arc;

use anyhow::Result;
use opencuda_core::{
    alloc_buffer, CompiledKernel, GpuDevice, KernelArg, LaunchConfig, ResolvedArg, ThreadCtx,
};
use opencuda_vulkan::VulkanMockDevice;

const N: usize = 1_000_000;

// SPIR-V の little-endian magic number 0x07230203 を含む最小fixture。
// 実GPU用の完全なシェーダーバイナリではない。VulkanMockDevice の経路テスト用。
const SPIRV_VECTOR_ADD_FIXTURE: &[u8] = &[
    0x03, 0x02, 0x23, 0x07, // magic
    0x00, 0x00, 0x01, 0x00, // version-like placeholder
];

fn main() -> Result<()> {
    let device: Arc<dyn GpuDevice> = VulkanMockDevice::new(0);
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

    // 1. Vulkan系バックエンドでは Native は動かないことを明示的に検証。
    let native_kernel = CompiledKernel::native("vector_add", |_ctx: ThreadCtx, _args: &[ResolvedArg]| {});
    match device.launch_kernel(&native_kernel, &cfg, &[]) {
        Ok(_) => anyhow::bail!("BUG: VulkanMockDevice accepted Native kernel"),
        Err(e) => println!("OK: Native kernel rejected on Vulkan path: {e}"),
    }

    // 2. SPIR-V経路で vector_add を実行。v0.1.1 ではMock内の限定シミュレーション。
    let spirv_kernel = CompiledKernel::spirv(
        "vector_add",
        "main",
        SPIRV_VECTOR_ADD_FIXTURE.to_vec(),
    );

    device.launch_kernel(
        &spirv_kernel,
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

    println!("OK: SPIR-V path simulation produced correct vector_add result");
    println!("c[0]={}, c[{}]={}", c[0], N - 1, c[N - 1]);
    Ok(())
}

fn cast_f32_to_u8(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn cast_f32_to_u8_mut(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}
