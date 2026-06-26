//! vulkan_info: real-vulkan feature の最小診断ツール。
//!
//! 実Vulkanの vector_add を動かす前に、Vulkan loader / driver / compute queue が
//! 使えるかを確認するための小さな実行ファイル。

use anyhow::Result;
use opencuda_core::GpuDevice;
use opencuda_vulkan::VulkanDevice;

fn main() -> Result<()> {
    let device = VulkanDevice::new(0)?;
    let info = device.info();

    println!("OpenCUDA Vulkan info");
    println!("device: {}", info.name);
    println!("vendor: {:?}", info.vendor);
    println!("total_memory_bytes: {}", info.total_memory);
    println!("compute_units_reported: {}", info.compute_units);
    println!("OK: Vulkan loader, physical device, logical device, and compute queue are available");
    Ok(())
}
