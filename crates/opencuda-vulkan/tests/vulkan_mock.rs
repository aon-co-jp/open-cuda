use std::sync::Arc;

use opencuda_core::{CompiledKernel, GpuDevice, LaunchConfig, ResolvedArg, ThreadCtx};
use opencuda_vulkan::VulkanMockDevice;

#[test]
fn vulkan_mock_rejects_native_kernel() {
    let device: Arc<dyn GpuDevice> = VulkanMockDevice::new(0);
    let cfg = LaunchConfig::linear(1, 1);
    let native = CompiledKernel::native("noop", |_ctx: ThreadCtx, _args: &[ResolvedArg]| {});

    let err = device.launch_kernel(&native, &cfg, &[]).unwrap_err();
    assert!(err.to_string().contains("Native"));
}

#[test]
fn vulkan_mock_rejects_invalid_spirv() {
    let device: Arc<dyn GpuDevice> = VulkanMockDevice::new(0);
    let cfg = LaunchConfig::linear(1, 1);
    let spirv = CompiledKernel::spirv("vector_add", "main", vec![0, 1, 2, 3]);

    let err = device.launch_kernel(&spirv, &cfg, &[]).unwrap_err();
    assert!(err.to_string().contains("invalid SPIR-V"));
}
