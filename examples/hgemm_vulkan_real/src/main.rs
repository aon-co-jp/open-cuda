//! hgemm_vulkan_real: F16 GEMM(`hgemm`)の実Vulkan Compute検証(2026-09-05新設)。
//!
//! CPU参照実装(`opencuda_blas::hgemm`)と、実Vulkan Compute(SPIR-V、
//! `shaders/hgemm.comp`)のF16 GEMMを同じ入力行列で実行し、両者が
//! 一致することを確認する(`matmul_vulkan_real`と同じ検証パターン)。
//!
//! 事前に Vulkan SDK の `glslc` などで `shaders/hgemm.comp` を
//! `shaders/hgemm.spv` にコンパイルしてから実行する
//! (`tools/compile-vulkan-shaders.{ps1,cmd,sh}` が他の .comp と
//! まとめてコンパイルする)。

use std::path::PathBuf;

use anyhow::{Context, Result};
use half::f16;
use opencuda_blas::{hgemm, hgemm_vulkan_generic};
use opencuda_core::GpuDevice as _;
use opencuda_vulkan::VulkanDevice;

const M: usize = 8;
const K: usize = 6;
const N: usize = 4;
const EPS: f32 = 5e-2; // half精度の丸め誤差を許容(f32よりゆるい許容誤差)

fn main() -> Result<()> {
    let spv_path = shader_path()?;
    let spirv = std::fs::read(&spv_path).with_context(|| {
        format!(
            "failed to read {}. Compile it first, for example: glslc shaders/hgemm.comp -o shaders/hgemm.spv",
            spv_path.display()
        )
    })?;

    let a: Vec<f16> = (0..M * K).map(|i| f16::from_f32((i % 7) as f32 - 3.0)).collect();
    let b: Vec<f16> = (0..K * N).map(|i| f16::from_f32((i % 5) as f32 - 2.0)).collect();

    let mut c_cpu = vec![f16::from_f32(0.0); M * N];
    hgemm(M, K, N, &a, &b, &mut c_cpu)?;

    let vulkan_device = VulkanDevice::new(0)?;
    println!("device: {}", vulkan_device.info().name);
    let c_vulkan = hgemm_vulkan_generic(vulkan_device.as_ref(), M, K, N, &a, &b, &spirv)?;

    compare("Vulkan vs CPU", &c_vulkan, &c_cpu)?;

    println!("OK: hgemm(F16) {M}x{K} * {K}x{N} verified: real Vulkan Compute agrees with the CPU reference");
    Ok(())
}

fn compare(label: &str, got: &[f16], expected: &[f16]) -> Result<()> {
    for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g.to_f32() - e.to_f32()).abs();
        if diff > EPS {
            anyhow::bail!("{label} mismatch at {idx}: got {g}, expected {e} (diff={diff})");
        }
    }
    println!("OK: {label} matches within {EPS}");
    Ok(())
}

/// `matmul_vulkan_real`と同じ実機/クロス環境向けのパス解決
/// (実行ファイルと同じディレクトリの`shaders/hgemm.spv`を優先、
/// 無ければビルド時の`CARGO_MANIFEST_DIR`へフォールバック)。
fn shader_path() -> Result<PathBuf> {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let candidate = exe_dir.join("shaders").join("hgemm.spv");
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    Ok(manifest_dir.join("shaders").join("hgemm.spv"))
}
