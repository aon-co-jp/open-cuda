//! softmax_vulkan_real: 行ごと(row-wise) softmaxの実Vulkan最小サンプル(2026-08-06新設)。
//!
//! CLAUDE.md HANDOFF(2026-08-05)の「次にすべきこと(1) softmax専用のSPIR-Vカーネル」
//! への着手。CPU素朴実装と実Vulkan Compute(SPIR-V, 1ワークグループ=1行+共有メモリ
//! リダクション)を同じ入力行列で実行し、両者が一致することを確認する。
//!
//! 事前に `shaders/softmax.comp` を `shaders/softmax.spv` にコンパイルしてから
//! 実行する(`tools/compile-vulkan-shaders.{ps1,cmd,sh}`が対応)。

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use opencuda_core::{alloc_buffer, CompiledKernel, GpuDevice, KernelArg, LaunchConfig};
use opencuda_vulkan::VulkanDevice;

const ROWS: usize = 8;
const COLS: usize = 37; // 256を割り切らない値でループ境界を確認する。
const EPS: f32 = 1e-4;

fn main() -> Result<()> {
    let spv_path = shader_path()?;
    let spirv = std::fs::read(&spv_path).with_context(|| {
        format!(
            "failed to read {}. Compile it first, for example: glslc shaders/softmax.comp -o shaders/softmax.spv",
            spv_path.display()
        )
    })?;

    // 行・列両方に依存する疑似ランダム入力(全ゼロ・全同一値では検出できない
    // 取りこぼしバグを避ける設計、raid6_xor_parity_vulkan_realと同じ考え方)。
    let input: Vec<f32> = (0..ROWS * COLS)
        .map(|i| {
            let row = i / COLS;
            let col = i % COLS;
            ((row * 13 + col * 7) % 23) as f32 - 11.0
        })
        .collect();

    let expected = softmax_reference(&input);
    let vulkan_out = run_vulkan(&input, &spirv)?;

    compare("Vulkan vs reference", &vulkan_out, &expected)?;
    for row in 0..ROWS {
        let sum: f32 = vulkan_out[row * COLS..(row + 1) * COLS].iter().sum();
        if (sum - 1.0).abs() > EPS {
            anyhow::bail!("row {row} does not sum to 1.0: got {sum}");
        }
    }

    println!(
        "OK: softmax {ROWS}x{COLS} verified: real Vulkan Compute matches the CPU reference and each row sums to 1.0"
    );
    Ok(())
}

fn softmax_reference(data: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for row in 0..ROWS {
        let slice = &data[row * COLS..(row + 1) * COLS];
        let m = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = slice.iter().map(|&x| (x - m).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (col, e) in exps.into_iter().enumerate() {
            out[row * COLS + col] = e / sum;
        }
    }
    out
}

fn run_vulkan(input: &[f32], spirv: &[u8]) -> Result<Vec<f32>> {
    let device: Arc<dyn GpuDevice> = VulkanDevice::new(0)?;
    println!("device: {}", device.info().name);

    let d = alloc_buffer(&device, ROWS * COLS * 4)?;
    d.copy_from_host(cast_f32_to_u8(input))?;

    // 1ワークグループ=1行(シェーダのlocal_size_x=256)。grid.x=ROWSにするため
    // LaunchConfig::linear(ROWS*256, 256)を使う(rows*block_size / block_size = rows)。
    let cfg = LaunchConfig::linear((ROWS * 256) as u32, 256);
    let kernel = CompiledKernel::spirv("softmax", "main", spirv);

    device.launch_kernel(
        &kernel,
        &cfg,
        &[
            KernelArg::Ptr(d.as_ptr()),
            KernelArg::Usize(ROWS),
            KernelArg::Usize(COLS),
        ],
    )?;
    device.synchronize()?;

    let mut out = vec![0.0f32; ROWS * COLS];
    d.copy_to_host(cast_f32_to_u8_mut(&mut out))?;
    Ok(out)
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
    Ok(manifest_dir.join("shaders").join("softmax.spv"))
}

fn cast_f32_to_u8(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn cast_f32_to_u8_mut(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}
