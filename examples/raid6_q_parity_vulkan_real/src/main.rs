//! raid6_q_parity_vulkan_real: RAID6のQ-parity(Reed-Solomon、GF(2^8))計算を
//! 実Vulkan Compute(SPIR-V)でオフロードする最小サンプル(2026-07-30続き追記)。
//!
//! RAID6 GPUオフロード計画の第二段(第一段の`raid6_xor_parity_vulkan_real`=
//! P-parityに続く)。Linuxカーネル/H. Peter Anvinの論文
//! ("The mathematics of RAID-6")と同じ既約多項式`x^8+x^4+x^3+x^2+1`
//! (バイト表現`0x11D`、還元バイト`0x1D`)・生成元`0x02`を採用し、
//! CPU側のGF(2^8)乗算リファレンス実装とGPU実装の両方がbit-exactに一致
//! することを実機で検証する。性能ではなく正確性優先(既存のP-parity/
//! matmul/vector_addサンプルと同じ方針)。
//!
//! 事前に `shaders/raid6_q_parity.comp` を `.spv` へコンパイルしておく
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

/// GF(2^8)乗算(RAID6標準の既約多項式 `x^8+x^4+x^3+x^2+1`、還元バイト`0x1D`)。
/// シェーダ側`gf_mul`と全く同じアルゴリズム(キャリーレス乗算+条件付き還元を
/// 8回繰り返す"Russian peasant"乗算)をCPU側リファレンスとして独立実装する
/// (シェーダのコンパイル/実行結果と、シェーダを一切経由しないCPU実装の
/// 両方が同じ数式から独立に導出されることで、検証の意味を持たせる)。
fn gf_mul(a: u8, b: u8) -> u8 {
    let mut p: u8 = 0;
    let mut aa = a;
    let mut bb = b;
    for _ in 0..8 {
        if bb & 1 != 0 {
            p ^= aa;
        }
        let hi = aa & 0x80;
        aa <<= 1;
        if hi != 0 {
            aa ^= 0x1D;
        }
        bb >>= 1;
    }
    p
}

/// ディスク`d`のRAID6標準係数 `g^d`(生成元 g=2)。
fn gf_pow2(exp: usize) -> u8 {
    let mut result: u8 = 1;
    for _ in 0..exp {
        result = gf_mul(result, 2);
    }
    result
}

fn main() -> Result<()> {
    let spv_path = shader_path()?;
    let spirv = std::fs::read(&spv_path).with_context(|| {
        format!(
            "failed to read {}. Compile it first, for example: glslc shaders/raid6_q_parity.comp -o shaders/raid6_q_parity.spv",
            spv_path.display()
        )
    })?;

    // 決定的な疑似データ(P-parityサンプルと同じ生成式)。
    let data: Vec<u32> = (0..NUM_DISKS * BLOCK_WORDS)
        .map(|idx| {
            let disk = (idx / BLOCK_WORDS) as u32;
            let word = (idx % BLOCK_WORDS) as u32;
            word.wrapping_mul(2654435761).wrapping_add(disk.wrapping_mul(40503))
        })
        .collect();

    let coeffs: Vec<u32> = (0..NUM_DISKS).map(|d| gf_pow2(d) as u32).collect();

    let parity_ref = q_parity_reference(&data, &coeffs);
    let parity_cpu = run_cpu(&data, &coeffs)?;
    let parity_vulkan = run_vulkan(&data, &coeffs, &spirv)?;

    compare("CPU vs reference", &parity_cpu, &parity_ref)?;
    compare("Vulkan vs reference", &parity_vulkan, &parity_ref)?;
    compare("Vulkan vs CPU", &parity_vulkan, &parity_cpu)?;

    println!(
        "OK: RAID6 Q-parity (Reed-Solomon, GF(2^8)) over {NUM_DISKS} disks x {BLOCK_WORDS} words verified: \
         CPU and real Vulkan Compute agree with the reference"
    );
    Ok(())
}

fn q_parity_reference(data: &[u32], coeffs: &[u32]) -> Vec<u32> {
    let mut parity = vec![0u32; BLOCK_WORDS];
    for (i, parity_word) in parity.iter_mut().enumerate() {
        let mut acc = [0u8; 4];
        for (d, &coeff) in coeffs.iter().enumerate() {
            let word = data[d * BLOCK_WORDS + i];
            let bytes = word.to_le_bytes();
            for (b, acc_b) in acc.iter_mut().enumerate() {
                *acc_b ^= gf_mul(bytes[b], coeff as u8);
            }
        }
        *parity_word = u32::from_le_bytes(acc);
    }
    parity
}

fn run_cpu(data: &[u32], coeffs: &[u32]) -> Result<Vec<u32>> {
    let device: Arc<dyn GpuDevice> = CpuDevice::new(0);

    let d_data = alloc_buffer(&device, data.len() * 4)?;
    let d_coeffs = alloc_buffer(&device, coeffs.len() * 4)?;
    let d_parity = alloc_buffer(&device, BLOCK_WORDS * 4)?;
    d_data.copy_from_host(cast_u32_to_u8(data))?;
    d_coeffs.copy_from_host(cast_u32_to_u8(coeffs))?;

    let kernel = CompiledKernel::native("raid6_q_parity_naive", |ctx: ThreadCtx, args: &[ResolvedArg]| {
        let idx = ctx.global_id_x() as usize;
        let num_disks = args[3].as_usize().unwrap();
        let block_words = args[4].as_usize().unwrap();
        if idx >= block_words {
            return;
        }
        let (data_ptr, _) = args[0].as_ptr().unwrap();
        let (coeffs_ptr, _) = args[1].as_ptr().unwrap();
        let (parity_ptr, _) = args[2].as_ptr().unwrap();
        unsafe {
            let data = data_ptr as *const u32;
            let coeffs = coeffs_ptr as *const u32;
            let parity = parity_ptr as *mut u32;
            let mut acc = [0u8; 4];
            for d in 0..num_disks {
                let word = data.add(d * block_words + idx).read();
                let coeff = coeffs.add(d).read() as u8;
                let bytes = word.to_le_bytes();
                for (b, acc_b) in acc.iter_mut().enumerate() {
                    *acc_b ^= gf_mul(bytes[b], coeff);
                }
            }
            parity.add(idx).write(u32::from_le_bytes(acc));
        }
    });

    let cfg = LaunchConfig::linear(BLOCK_WORDS as u32, 256);
    device.launch_kernel(
        &kernel,
        &cfg,
        &[
            KernelArg::Ptr(d_data.as_ptr()),
            KernelArg::Ptr(d_coeffs.as_ptr()),
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

fn run_vulkan(data: &[u32], coeffs: &[u32], spirv: &[u8]) -> Result<Vec<u32>> {
    let device: Arc<dyn GpuDevice> = VulkanDevice::new(0)?;
    println!("device: {}", device.info().name);

    let d_data = alloc_buffer(&device, data.len() * 4)?;
    let d_coeffs = alloc_buffer(&device, coeffs.len() * 4)?;
    let d_parity = alloc_buffer(&device, BLOCK_WORDS * 4)?;
    d_data.copy_from_host(cast_u32_to_u8(data))?;
    d_coeffs.copy_from_host(cast_u32_to_u8(coeffs))?;

    // シェーダの local_size_x = 256 と一致させる。
    let cfg = LaunchConfig::linear(BLOCK_WORDS as u32, 256);
    let kernel = CompiledKernel::spirv("raid6_q_parity", "main", spirv);

    device.launch_kernel(
        &kernel,
        &cfg,
        &[
            KernelArg::Ptr(d_data.as_ptr()),
            KernelArg::Ptr(d_coeffs.as_ptr()),
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
    println!("OK: {label} matches exactly (GF(2^8) arithmetic is bit-exact, no epsilon tolerance needed)");
    Ok(())
}

fn shader_path() -> Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    Ok(manifest_dir.join("shaders").join("raid6_q_parity.spv"))
}

fn cast_u32_to_u8(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn cast_u32_to_u8_mut(v: &mut [u32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}
