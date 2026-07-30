//! raid6_xor_parity_vulkan_real: RAID6のP-parity(XOR)計算を実Vulkan Compute
//! (SPIR-V)でオフロードする最小サンプル(2026-07-30追記)。
//!
//! open-raid-zのCLAUDE.mdに記録済みのロードマップ(NVMe SSD RAID6の
//! ランダムアクセス低速化=parity write penaltyを、GPU/ASICアクセラレーター
//! でのパリティ計算オフロードで緩和する)の第一段。CPU(rayonなし、素朴な
//! XORループ)のリファレンス実装と、実Vulkan Compute実装が同じ入力ブロックで
//! 一致することを検証する。性能ではなく正確性優先(既存のmatmul/vector_add
//! サンプルと同じ方針)。
//!
//! 事前に `shaders/raid6_xor_parity.comp` を `.spv` へコンパイルしておく
//! (`tools/compile-vulkan-shaders.{ps1,cmd,sh}` が対応済み)。

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use opencuda_core::{alloc_buffer, CompiledKernel, GpuDevice, KernelArg, LaunchConfig, ResolvedArg, ThreadCtx};
use opencuda_cpu::CpuDevice;
use opencuda_vulkan::VulkanDevice;

// RAID6を模した4本のデータディスク、1ブロック=4096バイト(NVMeの典型的なセクタ/ページ単位に近い値)。
const NUM_DISKS: usize = 4;
const BLOCK_WORDS: usize = 1024; // 4096 bytes / 4 bytes-per-word

fn main() -> Result<()> {
    let spv_path = shader_path()?;
    let spirv = std::fs::read(&spv_path).with_context(|| {
        format!(
            "failed to read {}. Compile it first, for example: glslc shaders/raid6_xor_parity.comp -o shaders/raid6_xor_parity.spv",
            spv_path.display()
        )
    })?;

    // 決定的な疑似データ(ディスクごとに異なるパターン、全ゼロや全同一値だと
    // XORの取りこぼしバグを見逃しやすいため、ディスクとwordの両方に依存する値にする)。
    let data: Vec<u32> = (0..NUM_DISKS * BLOCK_WORDS)
        .map(|idx| {
            let disk = (idx / BLOCK_WORDS) as u32;
            let word = (idx % BLOCK_WORDS) as u32;
            word.wrapping_mul(2654435761).wrapping_add(disk.wrapping_mul(40503))
        })
        .collect();

    let parity_ref = xor_parity_reference(&data);
    let parity_cpu = run_cpu(&data)?;
    let parity_vulkan = run_vulkan(&data, &spirv)?;

    compare("CPU vs reference", &parity_cpu, &parity_ref)?;
    compare("Vulkan vs reference", &parity_vulkan, &parity_ref)?;
    compare("Vulkan vs CPU", &parity_vulkan, &parity_cpu)?;

    println!(
        "OK: RAID6 P-parity (XOR) over {NUM_DISKS} disks x {BLOCK_WORDS} words verified: \
         CPU and real Vulkan Compute agree with the reference"
    );
    Ok(())
}

fn xor_parity_reference(data: &[u32]) -> Vec<u32> {
    let mut parity = vec![0u32; BLOCK_WORDS];
    for (idx, &word) in data.iter().enumerate() {
        parity[idx % BLOCK_WORDS] ^= word;
    }
    parity
}

fn run_cpu(data: &[u32]) -> Result<Vec<u32>> {
    let device: Arc<dyn GpuDevice> = CpuDevice::new(0);

    let d_data = alloc_buffer(&device, data.len() * 4)?;
    let d_parity = alloc_buffer(&device, BLOCK_WORDS * 4)?;
    d_data.copy_from_host(cast_u32_to_u8(data))?;

    let kernel = CompiledKernel::native("raid6_xor_parity_naive", |ctx: ThreadCtx, args: &[ResolvedArg]| {
        let idx = ctx.global_id_x() as usize;
        let num_disks = args[2].as_usize().unwrap();
        let block_words = args[3].as_usize().unwrap();
        if idx >= block_words {
            return;
        }
        let (data_ptr, _) = args[0].as_ptr().unwrap();
        let (parity_ptr, _) = args[1].as_ptr().unwrap();
        unsafe {
            let data = data_ptr as *const u32;
            let parity = parity_ptr as *mut u32;
            let mut acc = 0u32;
            for d in 0..num_disks {
                acc ^= data.add(d * block_words + idx).read();
            }
            parity.add(idx).write(acc);
        }
    });

    let cfg = LaunchConfig::linear(BLOCK_WORDS as u32, 256);
    device.launch_kernel(
        &kernel,
        &cfg,
        &[
            KernelArg::Ptr(d_data.as_ptr()),
            KernelArg::Ptr(d_parity.as_ptr()),
            KernelArg::Usize(NUM_DISKS),
            KernelArg::Usize(BLOCK_WORDS),
        ],
    )?;
    device.synchronize()?;

    let mut parity = vec![0u32; BLOCK_WORDS];
    d_parity.copy_to_host(cast_u32_to_u8_mut(&mut parity))?;
    Ok(parity)
}

fn run_vulkan(data: &[u32], spirv: &[u8]) -> Result<Vec<u32>> {
    let device: Arc<dyn GpuDevice> = VulkanDevice::new(0)?;
    println!("device: {}", device.info().name);

    let d_data = alloc_buffer(&device, data.len() * 4)?;
    let d_parity = alloc_buffer(&device, BLOCK_WORDS * 4)?;
    d_data.copy_from_host(cast_u32_to_u8(data))?;

    // シェーダの local_size_x = 256 と一致させる。
    let cfg = LaunchConfig::linear(BLOCK_WORDS as u32, 256);
    let kernel = CompiledKernel::spirv("raid6_xor_parity", "main", spirv);

    device.launch_kernel(
        &kernel,
        &cfg,
        &[
            KernelArg::Ptr(d_data.as_ptr()),
            KernelArg::Ptr(d_parity.as_ptr()),
            KernelArg::Usize(NUM_DISKS),
            KernelArg::Usize(BLOCK_WORDS),
        ],
    )?;
    device.synchronize()?;

    let mut parity = vec![0u32; BLOCK_WORDS];
    d_parity.copy_to_host(cast_u32_to_u8_mut(&mut parity))?;
    Ok(parity)
}

fn compare(label: &str, got: &[u32], expected: &[u32]) -> Result<()> {
    for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        if g != e {
            anyhow::bail!("{label} mismatch at word {idx}: got {g:#010x}, expected {e:#010x}");
        }
    }
    println!("OK: {label} matches exactly (XOR parity is bit-exact, no epsilon tolerance needed)");
    Ok(())
}

fn shader_path() -> Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    Ok(manifest_dir.join("shaders").join("raid6_xor_parity.spv"))
}

fn cast_u32_to_u8(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn cast_u32_to_u8_mut(v: &mut [u32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}
