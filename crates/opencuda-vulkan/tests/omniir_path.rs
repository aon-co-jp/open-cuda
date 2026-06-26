use std::sync::Arc;

use opencuda_core::{alloc_buffer, CompiledKernel, GpuDevice, KernelArg, LaunchConfig, ResolvedArg, ThreadCtx};
use opencuda_ir::IrModule;
use opencuda_vulkan::VulkanMockDevice;

#[test]
fn native_is_rejected_but_omniir_vector_add_runs() {
    let device: Arc<dyn GpuDevice> = VulkanMockDevice::new(0);
    let cfg = LaunchConfig::linear(4, 4);

    let native = CompiledKernel::native("vector_add_f32", |_ctx: ThreadCtx, _args: &[ResolvedArg]| {});
    assert!(device.launch_kernel(&native, &cfg, &[]).is_err());

    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [10.0f32, 20.0, 30.0, 40.0];
    let mut c = [0.0f32; 4];
    let bytes = 4 * std::mem::size_of::<f32>();

    let da = alloc_buffer(&device, bytes).unwrap();
    let db = alloc_buffer(&device, bytes).unwrap();
    let dc = alloc_buffer(&device, bytes).unwrap();
    da.copy_from_host(as_bytes(&a)).unwrap();
    db.copy_from_host(as_bytes(&b)).unwrap();

    let module = IrModule::vector_add_f32();
    let kernel = module.to_compiled_omniir();
    device.launch_kernel(&kernel, &cfg, &[
        KernelArg::Ptr(da.as_ptr()),
        KernelArg::Ptr(db.as_ptr()),
        KernelArg::Ptr(dc.as_ptr()),
        KernelArg::Usize(4),
    ]).unwrap();
    dc.copy_to_host(as_bytes_mut(&mut c)).unwrap();

    assert_eq!(c, [11.0, 22.0, 33.0, 44.0]);
}

fn as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn as_bytes_mut(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}
