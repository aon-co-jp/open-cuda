//! CPU SIMD カーネル(AI推論のCPUフォールバック経路の高速化)。
//!
//! 【位置づけ】
//! `open-cuda`の主役はGPU(Vulkan Compute/DirectX)だが、過去の実測
//! (CLAUDE.md 2026-08-15 HANDOFF)ではデスクトップのGT730はGEMMでCPUより
//! 遅く、`aruaru-llm`の1トークンデコードでもディスパッチ固定オーバーヘッドが
//! 支配的だった。つまり**このエコシステムでは「GPUが使えない/割に合わない
//! 環境でのCPU実行」が現実の主経路**であり、そのCPU経路をSIMDで速くする
//! ことが直接的に効く。
//!
//! 【多段ディスパッチ設計(実行時CPU機能検出)】
//! `AVX-512F → AVX2+FMA3 → SSE2 → スカラー`。非対応CPU・非x86では必ず
//! スカラーへ落ちるため、ビルドも実行も壊れない。将来AVX-512搭載機へ
//! 載せ替えた場合、**コードの書き足し無しに**`CpuFeatures::detect()`が
//! `avx512f=true`を返して64バイト幅の経路が自動的に使われる。
//!
//! 【正直な開示(2026-08-22時点の開発機: AMD Ryzen 9 3950X / Zen 2)】
//! - 実機で実行・ベンチマークできるのは **AVX2 + FMA3** 経路まで。
//! - **AVX-512F 経路・AVX-VNNI / AVX-512 VNNI 経路はコンパイル確認のみで
//!   実機未検証**(Zen 2はいずれも非搭載)。VNNI経路は将来のCPUで
//!   自動的に有効化されるよう、あらかじめ書いてある。

/// 実行時に検出したCPU機能。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CpuFeatures {
    pub sse2: bool,
    pub avx2: bool,
    /// FMA3(積和演算)。AVX2と組で使う。
    pub fma: bool,
    pub avx512f: bool,
    /// AVX-512 VNNI(int8内積、量子化推論向け)。Zen 2は非搭載。
    pub avx512vnni: bool,
    /// AVX-VNNI(256bit幅のVNNI、Alder Lake以降)。Zen 2は非搭載。
    pub avxvnni: bool,
}

impl CpuFeatures {
    pub fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            return Self {
                sse2: std::is_x86_feature_detected!("sse2"),
                avx2: std::is_x86_feature_detected!("avx2"),
                fma: std::is_x86_feature_detected!("fma"),
                avx512f: std::is_x86_feature_detected!("avx512f"),
                avx512vnni: std::is_x86_feature_detected!("avx512vnni"),
                avxvnni: std::is_x86_feature_detected!("avxvnni"),
            };
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        Self::default()
    }

    /// ログ・ベンチマーク表示用の短い説明。
    pub fn describe(&self) -> String {
        let mut v = Vec::new();
        if self.avx512f {
            v.push("avx512f");
        }
        if self.avx512vnni {
            v.push("avx512vnni");
        }
        if self.avxvnni {
            v.push("avxvnni");
        }
        if self.avx2 {
            v.push("avx2");
        }
        if self.fma {
            v.push("fma3");
        }
        if self.sse2 {
            v.push("sse2");
        }
        if v.is_empty() {
            v.push("scalar");
        }
        v.join("+")
    }
}

/// 初回のみCPU機能を検出してキャッシュする。
pub fn cpu_features() -> &'static CpuFeatures {
    use std::sync::OnceLock;
    static F: OnceLock<CpuFeatures> = OnceLock::new();
    F.get_or_init(CpuFeatures::detect)
}

// ---------------------------------------------------------------------------
// f32 内積(GEMM・Attentionのホットループ)
// ---------------------------------------------------------------------------

fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    // 4本のアキュムレータへ分けることで、スカラー経路でも命令レベル並列性を稼ぐ。
    let n = a.len().min(b.len());
    let mut acc = [0.0f32; 4];
    let mut i = 0;
    while i + 4 <= n {
        acc[0] += a[i] * b[i];
        acc[1] += a[i + 1] * b[i + 1];
        acc[2] += a[i + 2] * b[i + 2];
        acc[3] += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut s = acc[0] + acc[1] + acc[2] + acc[3];
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// AVX2 + FMA3 による f32 内積。
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_f32_avx2_fma(a: &[f32], b: &[f32]) -> (f32, usize) {
        let n = a.len().min(b.len());
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0;
        // 16要素/反復(2本のアキュムレータでFMAのレイテンシを隠す)
        while i + 16 <= n {
            let av0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv0 = _mm256_loadu_ps(b.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(av0, bv0, acc0);
            let av1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let bv1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
            acc1 = _mm256_fmadd_ps(av1, bv1, acc1);
            i += 16;
        }
        while i + 8 <= n {
            let av = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv = _mm256_loadu_ps(b.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(av, bv, acc0);
            i += 8;
        }
        let sum = _mm256_add_ps(acc0, acc1);
        // 水平加算
        let hi = _mm256_extractf128_ps(sum, 1);
        let lo = _mm256_castps256_ps128(sum);
        let mut s128 = _mm_add_ps(hi, lo);
        s128 = _mm_hadd_ps(s128, s128);
        s128 = _mm_hadd_ps(s128, s128);
        (_mm_cvtss_f32(s128), i)
    }

    /// 【実機未検証】AVX-512F による f32 内積(この開発機は非搭載)。
    #[target_feature(enable = "avx512f")]
    pub unsafe fn dot_f32_avx512(a: &[f32], b: &[f32]) -> (f32, usize) {
        let n = a.len().min(b.len());
        let mut acc = _mm512_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            let av = _mm512_loadu_ps(a.as_ptr().add(i));
            let bv = _mm512_loadu_ps(b.as_ptr().add(i));
            acc = _mm512_fmadd_ps(av, bv, acc);
            i += 16;
        }
        (_mm512_reduce_add_ps(acc), i)
    }

    /// 【実機未検証】AVX-512 VNNI による int8 内積
    /// (`_mm512_dpbusd_epi32`: u8 × i8 → i32 の積和を1命令で4要素分)。
    #[target_feature(enable = "avx512vnni,avx512bw,avx512f")]
    pub unsafe fn dot_i8_avx512vnni(a: &[u8], b: &[i8]) -> (i32, usize) {
        let n = a.len().min(b.len());
        let mut acc = _mm512_setzero_si512();
        let mut i = 0;
        while i + 64 <= n {
            let av = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
            let bv = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);
            acc = _mm512_dpbusd_epi32(acc, av, bv);
            i += 64;
        }
        (_mm512_reduce_add_epi32(acc), i)
    }

    /// 【実機未検証】AVX-VNNI(256bit幅)による int8 内積。
    #[target_feature(enable = "avxvnni,avx2")]
    pub unsafe fn dot_i8_avxvnni(a: &[u8], b: &[i8]) -> (i32, usize) {
        let n = a.len().min(b.len());
        let mut acc = _mm256_setzero_si256();
        let mut i = 0;
        while i + 32 <= n {
            let av = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let bv = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            acc = _mm256_dpbusd_avx_epi32(acc, av, bv);
            i += 32;
        }
        let mut tmp = [0i32; 8];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, acc);
        (tmp.iter().sum(), i)
    }
}

/// f32 の内積。実行時CPU機能検出で AVX-512F → AVX2+FMA3 → スカラー を選ぶ。
///
/// 【注意】SIMD経路は加算順序がスカラー経路と異なり、またFMA3は中間丸めを
/// 行わないため、**結果はスカラー実装とビット単位では一致しない**
/// (浮動小数点加算は結合則を満たさないため原理的に避けられない)。
/// 誤差はいずれも同オーダーであり、テストでは相対誤差1e-5で検証している。
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let f = cpu_features();
        if f.avx512f && n >= 16 {
            let (s, done) = unsafe { x86::dot_f32_avx512(&a[..n], &b[..n]) };
            return s + dot_f32_scalar(&a[done..n], &b[done..n]);
        }
        if f.avx2 && f.fma && n >= 8 {
            let (s, done) = unsafe { x86::dot_f32_avx2_fma(&a[..n], &b[..n]) };
            return s + dot_f32_scalar(&a[done..n], &b[done..n]);
        }
    }
    dot_f32_scalar(&a[..n], &b[..n])
}

// ---------------------------------------------------------------------------
// axpy(`acc += scale * src`)と、それを使ったCPU GEMM
// ---------------------------------------------------------------------------

fn axpy_scalar(acc: &mut [f32], src: &[f32], scale: f32) {
    for (a, &s) in acc.iter_mut().zip(src.iter()) {
        *a += scale * s;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_axpy {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn axpy_avx2_fma(acc: &mut [f32], src: &[f32], scale: f32) -> usize {
        let n = acc.len().min(src.len());
        let sv = _mm256_set1_ps(scale);
        let mut i = 0;
        while i + 8 <= n {
            let a = _mm256_loadu_ps(acc.as_ptr().add(i));
            let s = _mm256_loadu_ps(src.as_ptr().add(i));
            _mm256_storeu_ps(acc.as_mut_ptr().add(i), _mm256_fmadd_ps(sv, s, a));
            i += 8;
        }
        i
    }

    /// 【実機未検証】AVX-512F版(この開発機は非搭載)。
    #[target_feature(enable = "avx512f")]
    pub unsafe fn axpy_avx512(acc: &mut [f32], src: &[f32], scale: f32) -> usize {
        let n = acc.len().min(src.len());
        let sv = _mm512_set1_ps(scale);
        let mut i = 0;
        while i + 16 <= n {
            let a = _mm512_loadu_ps(acc.as_ptr().add(i));
            let s = _mm512_loadu_ps(src.as_ptr().add(i));
            _mm512_storeu_ps(acc.as_mut_ptr().add(i), _mm512_fmadd_ps(sv, s, a));
            i += 16;
        }
        i
    }
}

/// `acc += scale * src`(実行時ディスパッチ)。
pub fn axpy(acc: &mut [f32], src: &[f32], scale: f32) {
    let n = acc.len().min(src.len());
    #[allow(unused_assignments)]
    let mut done = 0usize;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let f = cpu_features();
        done = if f.avx512f {
            unsafe { x86_axpy::axpy_avx512(&mut acc[..n], &src[..n], scale) }
        } else if f.avx2 && f.fma {
            unsafe { x86_axpy::axpy_avx2_fma(&mut acc[..n], &src[..n], scale) }
        } else {
            0
        };
    }
    axpy_scalar(&mut acc[done..n], &src[done..n], scale);
}

/// CPU上のGEMM `C = alpha*A·B(ᵀ) + beta*C`(行優先)。
///
/// - `transpose_b == false`: Bは`k x n`。**k方向のaxpy蓄積**へ組み替えて
///   計算する(素朴な`b[kk*n+col]`のストライドアクセスではSIMDロードが
///   できないため、出力行に対する連続アクセスへ変換する)。
/// - `transpose_b == true`: Bは`n x k`。行同士の内積([`dot_f32`])。
///
/// 行ごとにrayonで並列化する。
#[allow(clippy::too_many_arguments)]
pub fn sgemm_cpu(
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    transpose_b: bool,
    beta: f32,
    c: &mut [f32],
) {
    use rayon::prelude::*;
    c.par_chunks_mut(n).enumerate().for_each(|(row, c_row)| {
        let a_row = &a[row * k..row * k + k];
        if transpose_b {
            for (col, cv) in c_row.iter_mut().enumerate() {
                let acc = dot_f32(a_row, &b[col * k..col * k + k]);
                *cv = alpha * acc + beta * *cv;
            }
        } else {
            let mut acc = vec![0.0f32; n];
            for (kk, &av) in a_row.iter().enumerate() {
                if av != 0.0 {
                    axpy(&mut acc, &b[kk * n..kk * n + n], av);
                }
            }
            for (cv, &av) in c_row.iter_mut().zip(acc.iter()) {
                *cv = alpha * av + beta * *cv;
            }
        }
    });
}

fn dot_i8_scalar(a: &[u8], b: &[i8]) -> i32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x as i32 * y as i32).sum()
}

/// int8 量子化推論向けの内積(u8 活性化 × i8 重み → i32)。
///
/// AVX-512 VNNI → AVX-VNNI → スカラー。**VNNI経路はこの開発機
/// (Ryzen 9 3950X / Zen 2)が非搭載のため実機未検証**——VNNI対応CPUへ
/// 載せ替えれば`CpuFeatures::detect()`が自動的に有効化する。
/// 整数演算のため、どの経路でも結果は**完全に一致する**(丸め誤差なし)。
pub fn dot_i8(a: &[u8], b: &[i8]) -> i32 {
    let n = a.len().min(b.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let f = cpu_features();
        if f.avx512vnni && n >= 64 {
            let (s, done) = unsafe { x86::dot_i8_avx512vnni(&a[..n], &b[..n]) };
            return s + dot_i8_scalar(&a[done..n], &b[done..n]);
        }
        if f.avxvnni && n >= 32 {
            let (s, done) = unsafe { x86::dot_i8_avxvnni(&a[..n], &b[..n]) };
            return s + dot_i8_scalar(&a[done..n], &b[done..n]);
        }
    }
    dot_i8_scalar(&a[..n], &b[..n])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_f32(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as i32 as f32) / 1_000_000.0
            })
            .collect()
    }

    #[test]
    fn dot_f32_matches_scalar_reference_for_many_lengths() {
        for len in [0usize, 1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 63, 64, 65, 1000, 1023] {
            let a = pseudo_f32(len, 11 + len as u64);
            let b = pseudo_f32(len, 977 + len as u64);
            let got = dot_f32(&a, &b);
            let want = dot_f32_scalar(&a, &b);
            let tol = 1e-5 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tol,
                "len={len} got={got} want={want} (diff={})",
                (got - want).abs()
            );
        }
    }

    #[test]
    fn dot_i8_matches_scalar_reference_exactly() {
        for len in [0usize, 1, 31, 32, 33, 63, 64, 65, 500] {
            let a: Vec<u8> = (0..len).map(|i| ((i * 37 + 11) % 256) as u8).collect();
            let b: Vec<i8> = (0..len).map(|i| (((i * 53 + 7) % 256) as i32 - 128) as i8).collect();
            assert_eq!(dot_i8(&a, &b), dot_i8_scalar(&a, &b), "len={len}");
        }
    }

    /// スカラー実装との速度比較(手動ベンチマーク)。
    /// `cargo test -p opencuda-blas --release simd::tests::manual_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn manual_bench_dot_f32_simd_vs_scalar() {
        use std::time::Instant;
        println!("検出CPU機能: {}", cpu_features().describe());
        for len in [64usize, 256, 768, 4096] {
            let a = pseudo_f32(len, 3);
            let b = pseudo_f32(len, 5);
            let iters = 200_000usize;
            let t0 = Instant::now();
            let mut s = 0.0f32;
            for _ in 0..iters {
                s += std::hint::black_box(dot_f32_scalar(&a, &b));
            }
            let scalar_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let t1 = Instant::now();
            let mut s2 = 0.0f32;
            for _ in 0..iters {
                s2 += std::hint::black_box(dot_f32(&a, &b));
            }
            let simd_ms = t1.elapsed().as_secs_f64() * 1000.0;
            println!(
                "k={len:>5}: scalar {scalar_ms:>8.1}ms / simd {simd_ms:>8.1}ms / {:.2}x  (sum check {s:.3} vs {s2:.3})",
                scalar_ms / simd_ms
            );
        }
    }

    #[test]
    fn detected_features_are_reported() {
        let f = cpu_features();
        assert!(!f.describe().is_empty());
    }
}
