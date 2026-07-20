//! # opencuda-blas
//!
//! LLM推論に必要な高レベルカーネルを提供する（Phase 3）。
//!
//! 設計方針: 各ベンダーの最速ライブラリ（cuBLAS / rocBLAS / oneMKL）を
//! 自動選択し、無ければ汎用カーネル（CPU / Vulkan）にフォールバックする。
//! LLM推論は GEMM と Attention が計算時間の大半を占めるため、この少数の
//! カーネルを各バックエンドで最適化することが「実用的なフル機能」への近道。
//!
//! ## 実装状況（Phase 3 前半、このパスで実装した範囲）
//!
//! - `sgemm` の `GemmPath::CpuNaive` 経路: **実装済み**。
//!   `examples/matmul` にあった naive 三重ループカーネルを、実際に行列
//!   データ（`a`/`b`/`c` スライス）を受け取れる形にしてこのクレートへ
//!   移植した（旧シグネチャは次元のみを受け取り、実データを渡す手段が
//!   無かったため、何も計算できない不完全な形だった。今回シグネチャ自体を
//!   修正した）。`opencuda_core::GpuDevice::launch_kernel` 経由で実行する
//!   ため、CPUバックエンド（`opencuda-cpu`、rayonで各要素を並列実行）上で
//!   本物のカーネルディスパッチが走る。
//! - `scaled_dot_product_attention`: **実装済み**。素朴な（非flash）
//!   scaled dot-product attentionで、seq_len×seq_len のスコア行列を
//!   全部メモリ上に展開して計算する。QKᵀ 部分は本クレートの GEMM系
//!   カーネル（B転置版）を、softmax は行ごとにホスト側CPUで（rayonで
//!   行並列に）、P·V 部分は `sgemm` をそのまま再利用して計算する。
//! - `GemmPath::CuBlas` / `RocBlas` / `OneMkl` / `VulkanGeneric`
//!   （GPUベンダー別の実装経路）と `quantize_int4`（INT4/INT8量子化）は
//!   **このパスでは対象外、引き続きスタブのまま**。
//! - **`flash_attention`という名前の関数は実装していない**。文献上の
//!   Flash Attention はオンラインsoftmax + タイル化により
//!   seq_len×seq_len のスコア行列全体をメモリに展開しない、という
//!   メモリ効率化が本質。今回実装したのはそれとは異なる、素朴な
//!   （全展開する）attentionなので、誇大表現を避けるため
//!   `scaled_dot_product_attention`という正直な名前にした。真のFlash
//!   Attention（タイル化・オンラインsoftmax）は引き続き別増分として
//!   `flash_attention`にスタブを残す。

use opencuda_core::{CompiledKernel, GpuDevice, GpuVendor, KernelArg, LaunchConfig, ResolvedArg, Result, ThreadCtx};
use rayon::prelude::*;

/// GEMM のバックエンド選択。ベンダーごとに最速経路へ振り分ける。
pub fn select_gemm_path(device: &dyn GpuDevice) -> GemmPath {
    match &device.info().vendor {
        GpuVendor::Nvidia { .. } => GemmPath::CuBlas,
        GpuVendor::Amd { .. } => GemmPath::RocBlas,
        GpuVendor::Intel { .. } => GemmPath::OneMkl,
        GpuVendor::Cpu => GemmPath::CpuNaive,
        GpuVendor::Unknown => GemmPath::VulkanGeneric,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmPath {
    CuBlas,        // NVIDIA   (Phase 2/3, 未実装)
    RocBlas,       // AMD      (Phase 2/3, 未実装)
    OneMkl,        // Intel    (Phase 4, 未実装)
    VulkanGeneric, // 汎用     (Phase 1 後半, 未実装)
    CpuNaive,      // CPU      (実装済み、examples/matmul を移植)
}

/// デバイス上に確保したメモリを、スコープを抜けるときに必ず解放するための
/// 小さなRAIIガード。`Arc<dyn GpuDevice>` を要求する `DeviceBuffer` とは異なり
/// `&dyn GpuDevice` のみで完結するため、`sgemm`/attention のシグネチャを
/// `Arc` 化せずに済む。
struct ScopedAlloc<'a> {
    device: &'a dyn GpuDevice,
    ptr: opencuda_core::DevicePtr,
}

impl<'a> ScopedAlloc<'a> {
    fn new(device: &'a dyn GpuDevice, bytes: usize) -> Result<Self> {
        let ptr = device.alloc(bytes)?;
        Ok(Self { device, ptr })
    }

    fn ptr(&self) -> opencuda_core::DevicePtr {
        self.ptr
    }
}

impl<'a> Drop for ScopedAlloc<'a> {
    fn drop(&mut self) {
        if let Err(e) = self.device.free(self.ptr) {
            tracing::warn!("opencuda-blas: ScopedAlloc drop free failed: {e}");
        }
    }
}

fn f32_to_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32スライスを読み取り専用のu8スライスとして見るだけ。
    // examples/matmul, aruaru-llm/scoring.rs と同じ最小キャストパターン。
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn f32_from_bytes_mut(v: &mut [f32]) -> &mut [u8] {
    // SAFETY: 同上、可変版。
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}

/// `GemmPath::CpuNaive` 用の内部カーネル起動。`transpose_b` が true のとき
/// `b` は `n x k`（行優先）として渡され、`b[col*k+kk]` でアクセスする
/// （QKᵀ の計算に使う）。false のときは通常の `k x n` として
/// `b[kk*n+col]` でアクセスする（通常の GEMM / P·V に使う）。
#[allow(clippy::too_many_arguments)]
fn launch_naive_gemm(
    device: &dyn GpuDevice,
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    transpose_b: bool,
    beta: f32,
    c: &mut [f32],
) -> Result<()> {
    if a.len() != m * k {
        anyhow::bail!("sgemm: a.len()={} != m*k={}", a.len(), m * k);
    }
    if b.len() != k * n {
        anyhow::bail!("sgemm: b.len()={} != k*n={}", b.len(), k * n);
    }
    if c.len() != m * n {
        anyhow::bail!("sgemm: c.len()={} != m*n={}", c.len(), m * n);
    }

    let bytes_a = std::mem::size_of_val(a);
    let bytes_b = std::mem::size_of_val(b);
    let bytes_c = std::mem::size_of_val(c);

    let da = ScopedAlloc::new(device, bytes_a)?;
    let db = ScopedAlloc::new(device, bytes_b)?;
    let dc = ScopedAlloc::new(device, bytes_c)?;

    device.memcpy_h2d(da.ptr(), f32_to_bytes(a))?;
    device.memcpy_h2d(db.ptr(), f32_to_bytes(b))?;
    // beta*C の項があるため、既存の c の内容もデバイス側へ転送しておく。
    device.memcpy_h2d(dc.ptr(), f32_to_bytes(c))?;

    let kernel = CompiledKernel::native("sgemm_naive", move |ctx: ThreadCtx, args: &[ResolvedArg]| {
        let idx = ctx.global_id_x() as usize;
        let m = args[3].as_usize().unwrap();
        let k = args[4].as_usize().unwrap();
        let n = args[5].as_usize().unwrap();
        let alpha = args[6].as_f32().unwrap();
        let beta = args[7].as_f32().unwrap();

        if idx >= m * n {
            return;
        }
        let row = idx / n;
        let col = idx % n;
        let (a_ptr, _) = args[0].as_ptr().unwrap();
        let (b_ptr, _) = args[1].as_ptr().unwrap();
        let (c_ptr, _) = args[2].as_ptr().unwrap();

        let mut acc = 0.0f32;
        unsafe {
            let a = a_ptr as *const f32;
            let b = b_ptr as *const f32;
            let c = c_ptr as *mut f32;
            for kk in 0..k {
                let b_val = if transpose_b {
                    b.add(col * k + kk).read()
                } else {
                    b.add(kk * n + col).read()
                };
                acc += a.add(row * k + kk).read() * b_val;
            }
            let old_c = c.add(idx).read();
            c.add(idx).write(alpha * acc + beta * old_c);
        }
    });

    let cfg = LaunchConfig::linear((m * n) as u32, 256);
    device.launch_kernel(
        &kernel,
        &cfg,
        &[
            KernelArg::Ptr(da.ptr()),
            KernelArg::Ptr(db.ptr()),
            KernelArg::Ptr(dc.ptr()),
            KernelArg::Usize(m),
            KernelArg::Usize(k),
            KernelArg::Usize(n),
            KernelArg::F32(alpha),
            KernelArg::F32(beta),
        ],
    )?;
    device.synchronize()?;

    device.memcpy_d2h(f32_from_bytes_mut(c), dc.ptr())?;
    Ok(())
}

/// 単精度 GEMM: `C = alpha * A·B + beta * C`。
///
/// `a` は `m x k`、`b` は `k x n`、`c` は `m x n`（すべて行優先）。
/// `c` は入力（`beta*C` 項に使う既存値）と出力を兼ねる。
///
/// 旧シグネチャ（`sgemm(device, m, k, n)`）は次元だけを受け取り、実際の
/// 行列データを渡す手段が無い不完全なものだった（何も計算できない）ため、
/// このパスで実データを受け取る形に修正した。ワークスペース内・
/// リポジトリ横断で `opencuda_blas::sgemm` を呼び出す既存コードは
/// 見つからなかった（`aruaru-llm` は `opencuda-core`/`opencuda-cpu` のみに
/// 依存し、`opencuda-blas` 自体には依存していない）ため、破壊的変更の
/// 影響範囲は無い。
#[allow(clippy::too_many_arguments)]
pub fn sgemm(
    device: &dyn GpuDevice,
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) -> Result<()> {
    let path = select_gemm_path(device);
    tracing::debug!("sgemm path = {path:?}");
    match path {
        GemmPath::CpuNaive => launch_naive_gemm(device, m, k, n, alpha, a, b, false, beta, c),
        other => anyhow::bail!("sgemm: {other:?} backend not yet implemented (Phase 3)"),
    }
}

/// 素朴な（非Flash）scaled dot-product attention。
///
/// `q`/`k`/`v` はいずれも `seq_len x head_dim`（行優先、単一ヘッド分）。
/// 計算内容:
/// 1. `scores = Q·Kᵀ / sqrt(head_dim)` （`seq_len x seq_len`）
/// 2. `probs = softmax(scores)`（行ごと、数値安定のため各行の最大値を引く）
/// 3. `output = probs·V` （`seq_len x head_dim`）
///
/// **これは文献上の Flash Attention ではない**: `scores`/`probs` の
/// `seq_len x seq_len` 行列をまるごとメモリに展開しており、オンライン
/// softmaxもタイル化も行わない。それらのメモリ効率化こそが Flash
/// Attentionの本質であるため、誇張を避けてこの正直な名前にしている
/// （`flash_attention` という別関数を、真のタイル化実装向けのスタブとして
/// 残してある）。
///
/// QKᵀ の計算は本クレートの GEMM系カーネル（`transpose_b=true`）を、
/// `probs·V` の計算は [`sgemm`] をそのまま再利用する。softmax は行同士に
/// 依存が無いため rayon でホスト側CPU並列に計算する。
pub fn scaled_dot_product_attention(
    device: &dyn GpuDevice,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    if q.len() != seq_len * head_dim {
        anyhow::bail!("attention: q.len()={} != seq_len*head_dim={}", q.len(), seq_len * head_dim);
    }
    if k.len() != seq_len * head_dim {
        anyhow::bail!("attention: k.len()={} != seq_len*head_dim={}", k.len(), seq_len * head_dim);
    }
    if v.len() != seq_len * head_dim {
        anyhow::bail!("attention: v.len()={} != seq_len*head_dim={}", v.len(), seq_len * head_dim);
    }

    // 1. scores = Q・Kᵀ / sqrt(head_dim)
    //    launch_naive_gemm の alpha でスケーリングを同時に適用する。
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; seq_len * seq_len];
    launch_naive_gemm(device, seq_len, head_dim, seq_len, scale, q, k, true, 0.0, &mut scores)?;

    // 2. 行ごとのsoftmax（数値安定のため各行の最大値を引く）。行同士は独立
    //    なので rayon で並列化する。
    let mut probs = vec![0.0f32; seq_len * seq_len];
    probs
        .par_chunks_mut(seq_len)
        .zip(scores.par_chunks(seq_len))
        .for_each(|(out_row, in_row)| {
            let max = in_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for (o, &s) in out_row.iter_mut().zip(in_row.iter()) {
                let e = (s - max).exp();
                *o = e;
                sum += e;
            }
            if sum > 0.0 {
                for o in out_row.iter_mut() {
                    *o /= sum;
                }
            }
        });

    // 3. output = probs・V （通常の GEMM、sgemm をそのまま再利用）
    let mut output = vec![0.0f32; seq_len * head_dim];
    sgemm(device, seq_len, seq_len, head_dim, 1.0, &probs, v, 0.0, &mut output)?;

    Ok(output)
}

/// 真の Flash Attention（オンラインsoftmax + タイル化によりスコア行列を
/// 全展開しない）は未実装（Phase 3の次増分）。
/// 素朴な非タイル化実装は [`scaled_dot_product_attention`] を参照。
pub fn flash_attention(_device: &dyn GpuDevice) -> Result<()> {
    anyhow::bail!("flash_attention: true tiled/online-softmax flash attention not yet implemented; see scaled_dot_product_attention for the naive (non-tiled) implementation that IS implemented")
}

/// INT4量子化済みテンソル（グループ単位の対称量子化）。
///
/// - `data`: 量子化値（4bit、[-8, 7]）を2値/バイトでニブルパックしたもの。
///   偶数インデックスの値が下位ニブル、奇数インデックスが上位ニブル。
///   各ニブルは「符号付き値 + 8」（0..=15）として格納する。
/// - `scales`: グループごとのスケール（`dequant = (nibble - 8) as f32 * scale`）。
/// - `group_size`: 1グループの要素数（スケールを共有する単位）。
/// - `len`: 元の要素数（奇数の場合、最終バイトの上位ニブルはパディング）。
#[derive(Debug, Clone)]
pub struct QuantizedInt4Tensor {
    pub data: Vec<u8>,
    pub scales: Vec<f32>,
    pub group_size: usize,
    pub len: usize,
}

/// INT8量子化済みテンソル（グループ単位の対称量子化、1値/バイト）。
/// `dequant = q as f32 * scale`（qは[-127, 127]）。
#[derive(Debug, Clone)]
pub struct QuantizedInt8Tensor {
    pub data: Vec<i8>,
    pub scales: Vec<f32>,
    pub group_size: usize,
    pub len: usize,
}

/// グループごとのスケールを計算する（`max_abs / q_max`）。全要素0の
/// グループはスケール0とし、量子化値も0になる（0除算を避けるため
/// 量子化側でスケール0を特別扱いする）。
fn group_scales(input: &[f32], group_size: usize, q_max: f32) -> Vec<f32> {
    input
        .par_chunks(group_size)
        .map(|chunk| {
            let max_abs = chunk.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            if max_abs == 0.0 {
                0.0
            } else {
                max_abs / q_max
            }
        })
        .collect()
}

/// 要素ごとの対称量子化をデバイスカーネルとして起動する共通部。
/// 出力は1値/バイトの符号付き量子化値（i8をu8バッファとして扱う）。
/// INT4のニブルパッキングはバイト共有による書き込み競合を避けるため
/// ホスト側で行う（カーネルは常に1値/バイトで書く）。
fn launch_quantize_kernel(
    device: &dyn GpuDevice,
    input: &[f32],
    scales: &[f32],
    group_size: usize,
    q_min: f32,
    q_max: f32,
) -> Result<Vec<i8>> {
    let len = input.len();
    let bytes_in = std::mem::size_of_val(input);
    let bytes_scales = std::mem::size_of_val(scales);

    let din = ScopedAlloc::new(device, bytes_in)?;
    let dscales = ScopedAlloc::new(device, bytes_scales)?;
    let dout = ScopedAlloc::new(device, len)?;

    device.memcpy_h2d(din.ptr(), f32_to_bytes(input))?;
    device.memcpy_h2d(dscales.ptr(), f32_to_bytes(scales))?;

    let kernel = CompiledKernel::native("quantize_symmetric", move |ctx: ThreadCtx, args: &[ResolvedArg]| {
        let idx = ctx.global_id_x() as usize;
        let len = args[3].as_usize().unwrap();
        if idx >= len {
            return;
        }
        let group_size = args[4].as_usize().unwrap();
        let q_min = args[5].as_f32().unwrap();
        let q_max = args[6].as_f32().unwrap();
        let (in_ptr, _) = args[0].as_ptr().unwrap();
        let (scales_ptr, _) = args[1].as_ptr().unwrap();
        let (out_ptr, _) = args[2].as_ptr().unwrap();
        unsafe {
            let x = (in_ptr as *const f32).add(idx).read();
            let scale = (scales_ptr as *const f32).add(idx / group_size).read();
            let q = if scale == 0.0 { 0.0 } else { (x / scale).round().clamp(q_min, q_max) };
            (out_ptr as *mut i8).add(idx).write(q as i8);
        }
    });

    let cfg = LaunchConfig::linear(len as u32, 256);
    device.launch_kernel(
        &kernel,
        &cfg,
        &[
            KernelArg::Ptr(din.ptr()),
            KernelArg::Ptr(dscales.ptr()),
            KernelArg::Ptr(dout.ptr()),
            KernelArg::Usize(len),
            KernelArg::Usize(group_size),
            KernelArg::F32(q_min),
            KernelArg::F32(q_max),
        ],
    )?;
    device.synchronize()?;

    let mut out = vec![0i8; len];
    // SAFETY: i8スライスをu8スライスとして受けるだけ（f32_to_bytesと同じ最小パターン）。
    let out_bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, len) };
    device.memcpy_d2h(out_bytes, dout.ptr())?;
    Ok(out)
}

fn validate_quantize_args(input: &[f32], group_size: usize) -> Result<()> {
    if input.is_empty() {
        anyhow::bail!("quantize: input must not be empty");
    }
    if group_size == 0 {
        anyhow::bail!("quantize: group_size must be > 0");
    }
    Ok(())
}

/// INT4量子化（グループ単位の対称量子化、Phase 3）。
///
/// llama.cpp系のQ4量子化と同じ発想の「グループごとにスケールを持つ
/// 対称量子化」: 各グループの`max_abs / 7`をスケールとし、
/// `round(x / scale)`を[-7, 7]へクランプして4bit（+8オフセットの
/// ニブル）に格納する。要素ごとの量子化は`GpuDevice::launch_kernel`経由の
/// 実カーネルディスパッチで行い（CPUバックエンドではrayon並列）、
/// ニブルパッキング（2値/バイト）はバイト共有の書き込み競合を避けるため
/// ホスト側で行う。
///
/// 旧シグネチャ（`quantize_int4(device) -> Result<()>`）は入力を受け取る
/// 手段が無いスタブだった。`sgemm`のシグネチャ修正と同じ経緯で、実データを
/// 受け取る形へ変更した（ワークスペース内外に旧シグネチャの呼び出し元は
/// 存在しない）。
pub fn quantize_int4(device: &dyn GpuDevice, input: &[f32], group_size: usize) -> Result<QuantizedInt4Tensor> {
    validate_quantize_args(input, group_size)?;
    // 対称レンジ[-7, 7]を使う（-8を許すとmax_abs側の符号によって精度が
    // 非対称になるため、Q4系の慣例に合わせて±7で対称にする）。
    let scales = group_scales(input, group_size, 7.0);
    let q = launch_quantize_kernel(device, input, &scales, group_size, -7.0, 7.0)?;

    // ニブルパック: 偶数idx→下位、奇数idx→上位。格納値は q + 8（0..=15）。
    let mut data = vec![0u8; input.len().div_ceil(2)];
    for (i, &v) in q.iter().enumerate() {
        let nibble = (v + 8) as u8 & 0x0F;
        if i % 2 == 0 {
            data[i / 2] |= nibble;
        } else {
            data[i / 2] |= nibble << 4;
        }
    }
    Ok(QuantizedInt4Tensor { data, scales, group_size, len: input.len() })
}

/// [`quantize_int4`]の逆変換（ホスト側）。
pub fn dequantize_int4(t: &QuantizedInt4Tensor) -> Vec<f32> {
    (0..t.len)
        .map(|i| {
            let byte = t.data[i / 2];
            let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            let q = nibble as i32 - 8;
            q as f32 * t.scales[i / t.group_size]
        })
        .collect()
}

/// INT8量子化（グループ単位の対称量子化、Phase 3）。
/// スケールは`max_abs / 127`、量子化値は[-127, 127]（1値/バイト）。
pub fn quantize_int8(device: &dyn GpuDevice, input: &[f32], group_size: usize) -> Result<QuantizedInt8Tensor> {
    validate_quantize_args(input, group_size)?;
    let scales = group_scales(input, group_size, 127.0);
    let data = launch_quantize_kernel(device, input, &scales, group_size, -127.0, 127.0)?;
    Ok(QuantizedInt8Tensor { data, scales, group_size, len: input.len() })
}

/// [`quantize_int8`]の逆変換（ホスト側）。
pub fn dequantize_int8(t: &QuantizedInt8Tensor) -> Vec<f32> {
    (0..t.len).map(|i| t.data[i] as f32 * t.scales[i / t.group_size]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencuda_cpu::CpuDevice;
    use std::sync::Arc;

    fn cpu_device() -> Arc<CpuDevice> {
        CpuDevice::new(0)
    }

    #[test]
    fn sgemm_2x2_identity_alpha_one_beta_zero() {
        // A = [[1,2],[3,4]], B = [[1,0],[0,1]] (identity) => A·B = A
        let device = cpu_device();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![0.0; 4];
        sgemm(device.as_ref(), 2, 2, 2, 1.0, &a, &b, 0.0, &mut c).unwrap();
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn sgemm_2x2_with_alpha_and_beta_scaling() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // A·B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        // C_initial = [[1,1],[1,1]]
        // alpha=2, beta=3 => C = 2*[[19,22],[43,50]] + 3*[[1,1],[1,1]]
        //                      = [[38,44],[86,100]] + [[3,3],[3,3]] = [[41,47],[89,103]]
        let device = cpu_device();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![1.0, 1.0, 1.0, 1.0];
        sgemm(device.as_ref(), 2, 2, 2, 2.0, &a, &b, 3.0, &mut c).unwrap();
        assert_eq!(c, vec![41.0, 47.0, 89.0, 103.0]);
    }

    #[test]
    fn sgemm_3x3_matches_hand_computed_product() {
        // A = I(3), B = [[1,2,3],[4,5,6],[7,8,9]] => A·B = B (alpha=1, beta=0)
        let device = cpu_device();
        let a = vec![
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0,
        ];
        let b = vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0,
        ];
        let mut c = vec![0.0; 9];
        sgemm(device.as_ref(), 3, 3, 3, 1.0, &a, &b, 0.0, &mut c).unwrap();
        assert_eq!(c, b);
    }

    #[test]
    fn sgemm_rejects_mismatched_dimensions() {
        let device = cpu_device();
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![0.0; 4];
        // a has only 2 elements but m*k=4 is expected.
        assert!(sgemm(device.as_ref(), 2, 2, 2, 1.0, &a, &b, 0.0, &mut c).is_err());
    }

    #[test]
    fn attention_seq_len_one_returns_v_unchanged() {
        // seq_len=1のとき、softmaxの分母には1項しかないので確率は必ず1.0、
        // よってoutput = probs・V = 1.0 * V = V。
        let device = cpu_device();
        let head_dim = 4;
        let q = vec![0.5, -1.0, 2.0, 0.25];
        let k = vec![1.0, 1.0, 1.0, 1.0];
        let v = vec![10.0, 20.0, 30.0, 40.0];
        let out = scaled_dot_product_attention(device.as_ref(), &q, &k, &v, 1, head_dim).unwrap();
        assert_eq!(out, v);
    }

    #[test]
    fn attention_identical_keys_produce_uniform_average_of_v() {
        // 全ての行のKが同一なら、Qの値に関わらずQ・Kᵀの各要素は行内で同一
        // スコアになるため、softmaxは各行で一様分布(1/seq_len)になる。
        // よってoutputの各行は V の行の単純平均になる。
        let device = cpu_device();
        let seq_len = 3;
        let head_dim = 2;
        let q = vec![1.0, 2.0, -3.0, 0.5, 4.0, -1.0]; // 値は何でもよい(Kが同一なので無関係)
        let k = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // 3行とも同じK
        let v = vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0,
        ];
        // Vの列平均: col0 = (1+3+5)/3 = 3.0, col1 = (2+4+6)/3 = 4.0
        let expected_row = vec![3.0, 4.0];
        let out = scaled_dot_product_attention(device.as_ref(), &q, &k, &v, seq_len, head_dim).unwrap();
        for row in 0..seq_len {
            let got_row = &out[row * head_dim..(row + 1) * head_dim];
            for (g, e) in got_row.iter().zip(expected_row.iter()) {
                assert!((g - e).abs() < 1e-4, "row {row}: got {got_row:?}, expected {expected_row:?}");
            }
        }
    }

    #[test]
    fn attention_rejects_mismatched_dimensions() {
        let device = cpu_device();
        let q = vec![1.0, 2.0];
        let k = vec![1.0, 1.0, 1.0, 1.0];
        let v = vec![1.0, 1.0, 1.0, 1.0];
        assert!(scaled_dot_product_attention(device.as_ref(), &q, &k, &v, 2, 2).is_err());
    }

    #[test]
    fn quantize_int4_roundtrip_error_is_bounded_by_half_scale() {
        // 対称量子化の理論誤差上限は scale/2。グループごとにそれを検証する。
        let device = cpu_device();
        let input: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.37 - 11.0) * 0.5).collect();
        let group_size = 16;
        let t = quantize_int4(device.as_ref(), &input, group_size).unwrap();
        let restored = dequantize_int4(&t);
        assert_eq!(restored.len(), input.len());
        for (i, (&x, &r)) in input.iter().zip(restored.iter()).enumerate() {
            let scale = t.scales[i / group_size];
            assert!(
                (x - r).abs() <= scale * 0.5 + 1e-6,
                "idx {i}: x={x}, restored={r}, scale={scale}"
            );
        }
    }

    #[test]
    fn quantize_int4_packs_two_values_per_byte_with_odd_len_padding() {
        let device = cpu_device();
        let input = vec![7.0, -7.0, 0.0, 3.5, 1.0]; // 奇数個（5要素→3バイト）
        let t = quantize_int4(device.as_ref(), &input, input.len()).unwrap();
        assert_eq!(t.data.len(), 3);
        assert_eq!(t.len, 5);
        // scale = 7.0/7 = 1.0。q = [7, -7, 0, 4(3.5切り上げ丸め), 1]
        // 格納ニブル(+8): [15, 1, 8, 12, 9]
        assert_eq!(t.data[0], (1 << 4) | 15); // 下位=15(q=7), 上位=1(q=-7)
        assert_eq!(t.data[1], (12 << 4) | 8); // 下位=8(q=0), 上位=12(q=4)
        assert_eq!(t.data[2], 9); // 下位=9(q=1), 上位=パディング0
        let restored = dequantize_int4(&t);
        assert_eq!(restored, vec![7.0, -7.0, 0.0, 4.0, 1.0]);
    }

    #[test]
    fn quantize_int4_all_zero_group_stays_zero() {
        let device = cpu_device();
        let input = vec![0.0f32; 8];
        let t = quantize_int4(device.as_ref(), &input, 4).unwrap();
        assert_eq!(t.scales, vec![0.0, 0.0]);
        assert_eq!(dequantize_int4(&t), input);
    }

    #[test]
    fn quantize_int4_respects_group_boundaries() {
        // グループ0は大きな値、グループ1は小さな値。グループ1のスケールが
        // グループ0に汚染されない(=小さな値の分解能が保たれる)ことを検証。
        let device = cpu_device();
        let input = vec![700.0, -350.0, 0.007, -0.0035];
        let t = quantize_int4(device.as_ref(), &input, 2).unwrap();
        assert!((t.scales[0] - 100.0).abs() < 1e-4);
        assert!((t.scales[1] - 0.001).abs() < 1e-7);
        let restored = dequantize_int4(&t);
        assert!((restored[2] - 0.007).abs() < 0.001 * 0.5 + 1e-7);
    }

    #[test]
    fn quantize_int8_roundtrip_error_is_bounded_by_half_scale() {
        let device = cpu_device();
        let input: Vec<f32> = (0..100).map(|i| ((i as f32) - 50.0) * 1.3).collect();
        let group_size = 25;
        let t = quantize_int8(device.as_ref(), &input, group_size).unwrap();
        let restored = dequantize_int8(&t);
        for (i, (&x, &r)) in input.iter().zip(restored.iter()).enumerate() {
            let scale = t.scales[i / group_size];
            assert!((x - r).abs() <= scale * 0.5 + 1e-6, "idx {i}: x={x}, restored={r}");
        }
    }

    #[test]
    fn quantize_int8_is_more_precise_than_int4_on_same_input() {
        let device = cpu_device();
        let input: Vec<f32> = (0..32).map(|i| (i as f32 * 0.911).sin() * 10.0).collect();
        let t4 = quantize_int4(device.as_ref(), &input, 32).unwrap();
        let t8 = quantize_int8(device.as_ref(), &input, 32).unwrap();
        let err4: f32 = input.iter().zip(dequantize_int4(&t4)).map(|(x, r)| (x - r).abs()).sum();
        let err8: f32 = input.iter().zip(dequantize_int8(&t8)).map(|(x, r)| (x - r).abs()).sum();
        assert!(err8 < err4, "int8 total err {err8} should be < int4 total err {err4}");
    }

    #[test]
    fn quantize_rejects_empty_input_and_zero_group_size() {
        let device = cpu_device();
        assert!(quantize_int4(device.as_ref(), &[], 4).is_err());
        assert!(quantize_int4(device.as_ref(), &[1.0], 0).is_err());
        assert!(quantize_int8(device.as_ref(), &[], 4).is_err());
        assert!(quantize_int8(device.as_ref(), &[1.0], 0).is_err());
    }
}
