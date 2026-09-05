//! dgemm_vulkan_real: F64 GEMM(`dgemm`)の実Vulkan Compute検証(2026-09-05新設)。
//!
//! CPU参照実装(`opencuda_blas::dgemm`)と、実Vulkan Compute(SPIR-V、
//! `shaders/dgemm.comp`、GLSLネイティブ`double`)のF64 GEMMを同じ入力
//! 行列で実行し、両者が一致することを確認する(`hgemm_vulkan_real`と
//! 同じ検証パターン)。
//!
//! **正直な開示**: 実行には物理デバイス/ドライバの`shaderFloat64`
//! サポートが必須。`device.supports_f64_shader()`が`false`を返す環境
//! (例: このプロジェクトの開発機であるGT730がもし非対応なら)では、
//! 誤魔化さずスキップメッセージを出して終了する。
//!
//! 事前に Vulkan SDK の `glslc` などで `shaders/dgemm.comp` を
//! `shaders/dgemm.spv` にコンパイルしてから実行する
//! (`tools/compile-vulkan-shaders.{ps1,cmd,sh}` が他の .comp と
//! まとめてコンパイルする)。

use std::path::PathBuf;

use anyhow::{Context, Result};
use opencuda_blas::{dgemm, dgemm_vulkan_generic};
use opencuda_core::GpuDevice as _;
use opencuda_vulkan::VulkanDevice;

const M: usize = 8;
const K: usize = 6;
const N: usize = 5;
const EPS: f64 = 1e-9;

fn main() -> Result<()> {
    let spv_path = shader_path()?;
    let spirv = std::fs::read(&spv_path).with_context(|| {
        format!(
            "failed to read {}. Compile it first, for example: glslc shaders/dgemm.comp -o shaders/dgemm.spv",
            spv_path.display()
        )
    })?;

    let vulkan_device = VulkanDevice::new(0)?;
    println!("device: {}", vulkan_device.info().name);

    if !vulkan_device.supports_f64_shader() {
        println!(
            "skipping: this device/driver does not report shaderFloat64 support \
             (vkGetPhysicalDeviceFeatures().shaderFloat64 == VK_FALSE); native double compute shaders \
             cannot run here — this is an honest environment limitation, not a bug"
        );
        return Ok(());
    }

    let a: Vec<f64> = (0..M * K).map(|i| (i % 7) as f64 - 3.0).collect();
    let b: Vec<f64> = (0..K * N).map(|i| (i % 5) as f64 - 2.0).collect();

    let mut c_cpu = vec![0.0f64; M * N];
    dgemm(M, K, N, 1.0, &a, &b, 0.0, &mut c_cpu)?;

    let c_vulkan = dgemm_vulkan_generic(vulkan_device.as_ref(), M, K, N, &a, &b, &spirv)?;

    compare("Vulkan vs CPU", &c_vulkan, &c_cpu)?;

    println!("OK: dgemm(F64) {M}x{K} * {K}x{N} verified: real Vulkan Compute agrees with the CPU reference");
    Ok(())
}

fn compare(label: &str, got: &[f64], expected: &[f64]) -> Result<()> {
    for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        if (g - e).abs() > EPS {
            anyhow::bail!("{label} mismatch at {idx}: got {g}, expected {e}");
        }
    }
    println!("OK: {label} matches within {EPS}");
    Ok(())
}

/// `hgemm_vulkan_real`と同じ実機/クロス環境向けのパス解決。
fn shader_path() -> Result<PathBuf> {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let candidate = exe_dir.join("shaders").join("dgemm.spv");
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    Ok(manifest_dir.join("shaders").join("dgemm.spv"))
}
