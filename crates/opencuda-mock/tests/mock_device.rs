use std::sync::Arc;

use opencuda_core::{CompiledKernel, GpuDevice, KernelArg, LaunchConfig};
use opencuda_mock::MockDevice;

#[test]
fn mock_device_records_launches() {
    let device = MockDevice::new(0);
    let dyn_device: Arc<dyn GpuDevice> = device.clone();
    let cfg = LaunchConfig::linear(256, 64);
    let spirv = CompiledKernel::spirv("dummy", "main", vec![0x03, 0x02, 0x23, 0x07]);
    let buf = dyn_device.alloc(16).unwrap();

    dyn_device
        .launch_kernel(&spirv, &cfg, &[KernelArg::Ptr(buf)])
        .unwrap();

    let launches = device.launches();
    assert_eq!(launches.len(), 1);
    assert_eq!(launches[0].kernel_name, "dummy");
    assert_eq!(launches[0].source_kind, "SpirV");
    assert_eq!(launches[0].arg_count, 1);
}
