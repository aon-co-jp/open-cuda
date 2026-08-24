//! GEMMバックエンドの実測ベース自動選択(2026-08-24新設)。
//!
//! 背景: 2026-08-23のHANDOFFで、D3D12 Compute(DXIL)経由のGEMM
//! オフロードを実装したものの、この開発機(NVIDIA GeForce GT 730 +
//! Ryzen 9 3950X〈AVX2+FMA3〉)では**CPUより3〜30倍遅い**ことが実測で
//! 判明した。一方、過去HANDOFFのQualcomm Adreno 619実機では逆にGPUの方が
//! 最大5.99倍速かった。つまり「GPUへオフロードすべきかどうか」は
//! マシン依存であり、静的に決め打ちできない。
//!
//! このモジュールは、実際に走らせるモデルのGEMM形状で**その場で計測して
//! 速い方を選ぶ**(遅ければ黙ってCPUのままにする)ための小さな
//! オートチューナ。誇張を避けるため、判定に使った実測値をそのまま
//! [`OffloadDecision`]として呼び出し側へ返し、ログにも出せるようにしてある。
//!
//! **正しさのチェックも兼ねる**: 最小の形状1つについてGPU結果とCPU参照
//! 実装を突き合わせ、数値が一致しない場合は(速くても)GPUを選ばない。

use crate::{sgemm_directx_resident_b, upload_resident_matrix};
use opencuda_core::{GpuDevice, Result};

/// オフロードするかどうかの方針。環境変数`OPEN_CUDA_GEMM_OFFLOAD`で
/// 上書きできる([`policy_from_env`])。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffloadPolicy {
    /// 実測して速い方を選ぶ(既定)。
    Auto,
    /// 計測結果によらずGPUを使う(数値検証には引き続き失敗しうる)。
    ForceGpu,
    /// 計測すらせずCPUのままにする。
    ForceCpu,
}

/// `OPEN_CUDA_GEMM_OFFLOAD` を読む(`auto`(既定) / `gpu` / `cpu`、
/// 大文字小文字は区別しない。未知の値は`auto`扱いで警告ログを出す)。
pub fn policy_from_env() -> OffloadPolicy {
    match std::env::var("OPEN_CUDA_GEMM_OFFLOAD") {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => OffloadPolicy::Auto,
            "gpu" | "force" | "1" => OffloadPolicy::ForceGpu,
            "cpu" | "off" | "0" => OffloadPolicy::ForceCpu,
            other => {
                tracing::warn!("OPEN_CUDA_GEMM_OFFLOAD='{other}' is not one of auto/gpu/cpu; using auto");
                OffloadPolicy::Auto
            }
        },
        Err(_) => OffloadPolicy::Auto,
    }
}

/// 1形状分の実測結果(ミリ秒)。
#[derive(Debug, Clone, Copy)]
pub struct GemmProbe {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    /// GPU(重み常駐版`sgemm_directx_resident_b`)の実測時間。
    pub gpu_ms: f64,
    /// CPU(`simd::sgemm_cpu`、AVX-512/AVX2+FMA3のランタイム検出付き)の実測時間。
    pub cpu_ms: f64,
}

impl GemmProbe {
    /// CPU時間 / GPU時間。1.0より大きければGPUの方が速い。
    pub fn speedup(&self) -> f64 {
        if self.gpu_ms > 0.0 {
            self.cpu_ms / self.gpu_ms
        } else {
            f64::INFINITY
        }
    }
}

/// [`decide_dxil_offload`]の判定結果。
#[derive(Debug, Clone)]
pub struct OffloadDecision {
    /// GPUオフロードを有効にすべきか。
    pub use_gpu: bool,
    pub policy: OffloadPolicy,
    pub probes: Vec<GemmProbe>,
    /// 全形状合計のGPU時間(ms)。`ForceCpu`のときは0.0。
    pub gpu_total_ms: f64,
    /// 全形状合計のCPU時間(ms)。`ForceCpu`のときは0.0。
    pub cpu_total_ms: f64,
    /// 数値検証(GPU結果とCPU参照の一致)に通ったか。
    /// `ForceCpu`で計測を省いた場合は`false`のまま。
    pub numerics_ok: bool,
    /// 人間向けの判定理由(そのままログ・HANDOFFに貼れる短文)。
    pub reason: String,
}

impl OffloadDecision {
    /// ログ1行にまとめた要約。
    pub fn summary(&self) -> String {
        let mut s = format!(
            "gemm offload decision: use_gpu={} policy={:?} numerics_ok={} gpu_total={:.3}ms cpu_total={:.3}ms ({})",
            self.use_gpu, self.policy, self.numerics_ok, self.gpu_total_ms, self.cpu_total_ms, self.reason
        );
        for p in &self.probes {
            s.push_str(&format!("\n  m={} k={} n={}: gpu={:.3}ms cpu={:.3}ms speedup={:.2}x", p.m, p.k, p.n, p.gpu_ms, p.cpu_ms, p.speedup()));
        }
        s
    }
}

/// 計測に使う既定の形状(GPT-2 124M相当。`hidden=768`、`vocab=50257`)。
/// 呼び出し側が実際のモデル形状を知っている場合はそちらを渡すこと。
pub const DEFAULT_PROBE_SHAPES: &[(usize, usize, usize)] = &[(1, 768, 2304), (1, 768, 3072), (1, 768, 50257)];

/// D3D12(DXIL)経由の密GEMMオフロードを有効にすべきかを**実測して**判定する。
///
/// - `shapes`が空なら[`DEFAULT_PROBE_SHAPES`]を使う。
/// - 各形状につき、GPU側は[`sgemm_directx_resident_b`](実際のオフロード
///   経路と同じ、重みVRAM常駐版)、CPU側は[`crate::simd::sgemm_cpu`]を
///   計測する。いずれもウォームアップ1回のあと`reps`回まわして
///   **最小値**を採る(他プロセスの影響を受けにくくするため)。
/// - 最小形状で数値の一致(相対誤差1e-3以内)も検証する。一致しない場合は
///   速度によらず`use_gpu=false`。
///
/// この関数自体は失敗してもエラーを返さず「CPUを使う」判定に倒す設計には
/// **していない**——デバイス側の実エラー(VRAM不足等)は呼び出し側が
/// 認識できるべきなので`Err`をそのまま返す。呼び出し側で
/// `unwrap_or_else(|_| cpu)`のように倒すかどうかを決めること。
pub fn decide_dxil_offload(device: &dyn GpuDevice, dxil: &[u8], shapes: &[(usize, usize, usize)], reps: usize) -> Result<OffloadDecision> {
    let policy = policy_from_env();
    if policy == OffloadPolicy::ForceCpu {
        return Ok(OffloadDecision {
            use_gpu: false,
            policy,
            probes: Vec::new(),
            gpu_total_ms: 0.0,
            cpu_total_ms: 0.0,
            numerics_ok: false,
            reason: "OPEN_CUDA_GEMM_OFFLOAD=cpu (measurement skipped)".to_string(),
        });
    }
    if !device.supports_dxil() {
        anyhow::bail!("decide_dxil_offload: device '{}' does not support DXIL kernels", device.info().name);
    }

    let shapes: Vec<(usize, usize, usize)> = if shapes.is_empty() { DEFAULT_PROBE_SHAPES.to_vec() } else { shapes.to_vec() };
    let reps = reps.max(1);

    let mut probes = Vec::with_capacity(shapes.len());
    let mut numerics_ok = true;
    let mut numerics_checked = false;

    // 数値検証は最小(=一番安い)形状で行う。
    let smallest = shapes.iter().copied().min_by_key(|(m, k, n)| m * k + k * n).expect("shapes is non-empty");

    for (m, k, n) in shapes {
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 23) as f32 - 11.0) * 0.01).collect();

        let b_ptr = upload_resident_matrix(device, &b)?;
        let gpu_result = (|| -> Result<(f64, Vec<f32>)> {
            // ウォームアップ(PSO生成・初回アロケーションを計測から外す)。
            let warm = sgemm_directx_resident_b(device, m, k, n, &a, b_ptr, dxil)?;
            let mut best = f64::INFINITY;
            for _ in 0..reps {
                let t = std::time::Instant::now();
                let _ = sgemm_directx_resident_b(device, m, k, n, &a, b_ptr, dxil)?;
                best = best.min(t.elapsed().as_secs_f64() * 1000.0);
            }
            Ok((best, warm))
        })();
        // 計測の成否によらず常駐バッファは解放する。
        if let Err(e) = device.free(b_ptr) {
            tracing::warn!("decide_dxil_offload: free(b_ptr) failed: {e}");
        }
        let (gpu_ms, gpu_out) = gpu_result?;

        let mut c = vec![0.0f32; m * n];
        let mut cpu_ms = f64::INFINITY;
        for _ in 0..reps {
            let t = std::time::Instant::now();
            crate::simd::sgemm_cpu(m, k, n, 1.0, &a, &b, false, 0.0, &mut c);
            cpu_ms = cpu_ms.min(t.elapsed().as_secs_f64() * 1000.0);
        }

        if (m, k, n) == smallest {
            numerics_checked = true;
            let tol = 1e-3f32;
            for (i, (g, r)) in gpu_out.iter().zip(c.iter()).enumerate() {
                let denom = r.abs().max(1.0);
                if ((g - r).abs() / denom) > tol {
                    numerics_ok = false;
                    tracing::warn!("decide_dxil_offload: numeric mismatch at index {i}: gpu={g} cpu={r}");
                    break;
                }
            }
        }

        probes.push(GemmProbe { m, k, n, gpu_ms, cpu_ms });
    }

    let gpu_total_ms: f64 = probes.iter().map(|p| p.gpu_ms).sum();
    let cpu_total_ms: f64 = probes.iter().map(|p| p.cpu_ms).sum();
    let numerics_ok = numerics_ok && numerics_checked;

    let (use_gpu, reason) = if !numerics_ok {
        (false, "GPU result did not match the CPU reference within 1e-3; staying on CPU".to_string())
    } else if policy == OffloadPolicy::ForceGpu {
        (true, "OPEN_CUDA_GEMM_OFFLOAD=gpu (measurement ignored)".to_string())
    } else if gpu_total_ms < cpu_total_ms {
        (true, format!("measured GPU {:.2}x faster than CPU on the probe shapes", cpu_total_ms / gpu_total_ms))
    } else {
        (false, format!("measured GPU {:.2}x SLOWER than CPU on the probe shapes; staying on CPU", gpu_total_ms / cpu_total_ms.max(f64::MIN_POSITIVE)))
    };

    let decision = OffloadDecision { use_gpu, policy, probes, gpu_total_ms, cpu_total_ms, numerics_ok, reason };
    tracing::info!("{}", decision.summary());
    Ok(decision)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_from_env_defaults_to_auto_when_unset() {
        // 環境変数を触らずに読める既定値の確認(他テストと並列でも安全)。
        if std::env::var("OPEN_CUDA_GEMM_OFFLOAD").is_err() {
            assert_eq!(policy_from_env(), OffloadPolicy::Auto);
        }
    }

    #[test]
    fn speedup_reports_ratio() {
        let p = GemmProbe { m: 1, k: 2, n: 3, gpu_ms: 2.0, cpu_ms: 4.0 };
        assert!((p.speedup() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn default_probe_shapes_are_gpt2_like() {
        assert!(!DEFAULT_PROBE_SHAPES.is_empty());
        assert!(DEFAULT_PROBE_SHAPES.iter().all(|(m, k, n)| *m > 0 && *k > 0 && *n > 0));
    }
}
