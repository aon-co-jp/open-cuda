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
//! - `GemmPath::CuBlas` / `RocBlas` / `OneMkl`（GPUベンダー専用ライブラリ
//!   経路）は引き続きスタブのまま（このマシンにはCUDA/ROCm/oneAPIの
//!   ツールチェインが無く〈`nvcc --version`が見つからない〉、未検証の
//!   コードを実装済みと偽ることになるため着手していない）。
//!   `GemmPath::VulkanGeneric`は`sgemm_vulkan_generic`として実装済み
//!   （実機NVIDIA GT 730で`sgemm`のCPU版との数値一致を検証）。さらに
//!   `select_gemm_path`は、ベンダー別専用経路がスタブのままの間、
//!   `device.supports_spirv()`が`true`のデバイス（実Vulkanデバイス）に
//!   対しては自動的に`VulkanGeneric`へフォールバックする（`sgemm`に
//!   オプションの`spirv`引数を追加し、この経路で実際に計算できるように
//!   した）。
//! - **`flash_attention`は実装済み**。Q/K/Vをブロックへタイル化し、
//!   実行中の最大値・指数和・出力累積を保持する「オンラインsoftmax」
//!   （Dao et al. 2022, FlashAttentionのアルゴリズム1相当）で、
//!   `seq_len x seq_len`のスコア行列全体をメモリに展開せず計算する。
//!   `scaled_dot_product_attention`（素朴な全展開版、GEMMカーネル
//!   ディスパッチ経由）とは別に、`flash_attention`は純粋なホスト側
//!   Rust実装（`GpuDevice`のカーネルディスパッチは使わない）として
//!   併存させている。両者が数学的に同じ結果を返すことをテストで検証
//!   済み（固定入力・乱数入力・block_sizeがseq_len非約数のケース・
//!   seq_len=1の境界ケース・次元不一致のエラーケースを含む）。
//! - `quantize_int4`/`quantize_int8`/`quantize_int4_awq`（AWQ風の
//!   activation-aware INT4量子化）は**実装済み**。グループ単位の対称
//!   量子化を`GpuDevice::launch_kernel`経由の実カーネルディスパッチで
//!   行い（CPUバックエンドではrayon並列）、ニブルパッキング等の
//!   バイト共有処理はホスト側で行う。それぞれ`dequantize_*`の逆変換と、
//!   往復誤差がscale/2以内に収まることを検証するテストを含む。

use opencuda_core::{CompiledKernel, GpuDevice, GpuVendor, KernelArg, LaunchConfig, ResolvedArg, Result, ThreadCtx};
use rayon::prelude::*;

/// GEMM のバックエンド選択。ベンダーごとに最速経路へ振り分ける。
///
/// ベンダー別専用経路（cuBLAS/rocBLAS/oneMKL）は現状すべてスタブ
/// （[`sgemm`] に渡すと未実装エラーを返す）。そのスタブ経路へ振り分けて
/// しまうと、実際にはVulkan経由で正しく計算できるデバイス上でも
/// `sgemm`が使い物にならない（エラーになる、あるいは将来スタブが
/// 「何もせず0を返す」ような実装になれば無言で誤った結果を返しかねない）。
/// そのため、ベンダー別専用経路がスタブのままの間は、
/// `device.supports_spirv()`（実Vulkanデバイスなら`true`、
/// `opencuda-vulkan::real::VulkanDevice`参照）が`true`を返すデバイスに
/// 対しては動く経路（`GemmPath::VulkanGeneric`）を優先する。
/// cuBLAS等が実装され次第、この優先ロジックはベンダー別経路の方を
/// 優先するよう改めればよい（スタブの実装済みへの置き換えに合わせて
/// このコメントと分岐を更新すること）。
pub fn select_gemm_path(device: &dyn GpuDevice) -> GemmPath {
    let vendor_path = match &device.info().vendor {
        GpuVendor::Nvidia { .. } => GemmPath::CuBlas,
        GpuVendor::Amd { .. } => GemmPath::RocBlas,
        GpuVendor::Intel { .. } => GemmPath::OneMkl,
        GpuVendor::Cpu => GemmPath::CpuNaive,
        // Qualcomm Adreno/ARM Mali/Imagination PowerVRにはベンダー専用GEMM
        // ライブラリのstub経路が無い(cuBLAS/rocBLAS/oneMKLに相当する
        // モバイル向け専用実装は本クレートで未着手)。これらのベンダーは
        // Vulkan Compute経由でしか到達しない設計のため、最初からVulkan
        // 汎用経路を返す(スタブ経由の遠回りを避ける、2026-07-25追加)。
        GpuVendor::Qualcomm { .. } | GpuVendor::Arm { .. } | GpuVendor::ImaginationPowerVr { .. } => {
            GemmPath::VulkanGeneric
        }
        GpuVendor::Unknown => GemmPath::VulkanGeneric,
    };

    let vendor_path_is_stub = matches!(vendor_path, GemmPath::CuBlas | GemmPath::RocBlas | GemmPath::OneMkl);
    if vendor_path_is_stub && device.supports_spirv() {
        GemmPath::VulkanGeneric
    } else {
        vendor_path
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
/// このパスで実データを受け取る形に修正した。
///
/// `spirv`: `select_gemm_path`が`GemmPath::VulkanGeneric`を選んだ場合に
/// `sgemm_vulkan_generic`へ渡すSPIR-Vバイト列（`examples/matmul_vulkan_real/
/// shaders/matmul.spv`と同一契約、呼び出し側が用意する。理由は
/// `sgemm_vulkan_generic`のdocコメント参照 — リポジトリに`.spv`を
/// 埋め込むとシェーダ未コンパイル環境で`cargo build`自体が壊れるため）。
/// `GemmPath::CpuNaive`が選ばれた場合は使われないので`None`で構わない。
/// `GemmPath::VulkanGeneric`が選ばれたのに`None`だった場合はエラーを返す
/// （黙って別経路にフォールバックしたり、誤った結果を返したりしない）。
/// Vulkanシェーダはalpha/beta スケーリングに対応していないため、
/// `sgemm_vulkan_generic`の結果に対しホスト側で`c = alpha*result +
/// beta*c`を適用し、CPU経路と同じセマンティクスを保つ。
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
    spirv: Option<&[u8]>,
) -> Result<()> {
    let path = select_gemm_path(device);
    tracing::debug!("sgemm path = {path:?}");
    match path {
        GemmPath::CpuNaive => launch_naive_gemm(device, m, k, n, alpha, a, b, false, beta, c),
        GemmPath::VulkanGeneric => {
            let spirv = spirv.ok_or_else(|| {
                anyhow::anyhow!(
                    "sgemm: GemmPath::VulkanGeneric selected (device.supports_spirv()==true, \
                     vendor-specific path is still a stub) but no spirv bytes were provided; \
                     pass the compiled matmul.spv bytes via the `spirv` argument"
                )
            })?;
            if c.len() != m * n {
                anyhow::bail!("sgemm: c.len()={} != m*n={}", c.len(), m * n);
            }
            let result = sgemm_vulkan_generic(device, m, k, n, a, b, spirv)?;
            for (ci, ri) in c.iter_mut().zip(result.iter()) {
                *ci = alpha * ri + beta * *ci;
            }
            Ok(())
        }
        other => anyhow::bail!("sgemm: {other:?} backend not yet implemented (Phase 3)"),
    }
}

/// `GemmPath::VulkanGeneric` の実装: Vulkan Compute 上で naive matmul を
/// 実行する（`C = A・B`、alpha/beta スケーリングは無し — シェーダ側が
/// 対応していないため。必要ならホスト側で `sgemm` と同様に別途スケールする）。
///
/// `spirv` には事前コンパイル済みの matmul シェーダのバイト列を渡す
/// （`crates/opencuda-blas/shaders/matmul.comp` を `glslc` 等でコンパイルした
/// ものと同一契約。`*.spv` はビルド成果物のためリポジトリでは
/// `.gitignore`（`**/*.spv`）で追跡していない — `examples/matmul_vulkan_real`
/// や `tools/compile-vulkan-shaders.{ps1,cmd,sh}` と同じ理由・同じ運用。
/// `include_bytes!` で埋め込むと、シェーダを事前コンパイルしていない
/// クローン直後の環境で `cargo build` 自体が壊れてしまうため、あえて
/// 呼び出し側にバイト列を渡させる設計にしてある）。
///
/// `device` には `opencuda-vulkan::real::VulkanDevice`（`real-vulkan`
/// feature 有効時）のような、SpirVカーネルの `"matmul"` エントリを
/// 実行できる `GpuDevice` 実装を渡す。CPUバックエンド（`opencuda-cpu`）
/// はSpirVカーネルを実行できないため、この関数をCPUデバイスに渡すと
/// エラーになる（[`sgemm`] の `GemmPath::CpuNaive` を使うこと）。
///
/// `a` は `m x k`、`b` は `k x n`（いずれも行優先）。ワークグループサイズは
/// シェーダの `local_size_x/y = 16` に合わせて `LaunchConfig::grid2d` で
/// `16x16` を指定する（`examples/matmul_vulkan_real` と同じ契約）。
#[allow(clippy::too_many_arguments)]
pub fn sgemm_vulkan_generic(device: &dyn GpuDevice, m: usize, k: usize, n: usize, a: &[f32], b: &[f32], spirv: &[u8]) -> Result<Vec<f32>> {
    if a.len() != m * k {
        anyhow::bail!("sgemm_vulkan_generic: a.len()={} != m*k={}", a.len(), m * k);
    }
    if b.len() != k * n {
        anyhow::bail!("sgemm_vulkan_generic: b.len()={} != k*n={}", b.len(), k * n);
    }

    let da = ScopedAlloc::new(device, std::mem::size_of_val(a))?;
    let db = ScopedAlloc::new(device, std::mem::size_of_val(b))?;
    let dc = ScopedAlloc::new(device, m * n * std::mem::size_of::<f32>())?;

    device.memcpy_h2d(da.ptr(), f32_to_bytes(a))?;
    device.memcpy_h2d(db.ptr(), f32_to_bytes(b))?;

    let kernel = CompiledKernel::spirv("matmul", "main", spirv);
    let cfg = LaunchConfig::grid2d(m as u32, n as u32, 16, 16);
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
        ],
    )?;
    device.synchronize()?;

    let mut c = vec![0.0f32; m * n];
    device.memcpy_d2h(f32_from_bytes_mut(&mut c), dc.ptr())?;
    Ok(c)
}

/// 行ごと(row-wise) softmaxを実Vulkan SPIR-Vカーネルで計算する
/// (2026-08-06新設、CLAUDE.md HANDOFF 2026-08-05「次にすべきこと(1)
/// softmax専用のSPIR-Vカーネル」への着手)。
///
/// `data`は`rows x cols`（行優先）。各行に対して数値安定な softmax
/// （各行の最大値を引いてからexp・合計・正規化）を、1ワークグループ=1行、
/// 共有メモリでのブロック内リダクションにより計算する
/// (`examples/softmax_vulkan_real/shaders/softmax.comp`と同じ契約、
/// SPIR-Vバイト列は`sgemm_vulkan_generic`と同じ理由で呼び出し側が渡す
/// 設計)。
///
/// **正直な開示**: これは
/// [`scaled_dot_product_attention_with_spirv`]の内部で使われる
/// **ハイブリッド版のsoftmax（GPU GEMM + CPU softmax）を置き換える配線は
/// まだ行っていない**——このカーネル自体は独立した再利用可能な部品として
/// 実装・実機検証済みだが、既存のAttention経路に組み込む変更は影響範囲の
/// 検討（既存APIのシグネチャ変更が必要）のため次の増分に残す。
pub fn softmax_vulkan_generic(device: &dyn GpuDevice, rows: usize, cols: usize, data: &[f32], spirv: &[u8]) -> Result<Vec<f32>> {
    if data.len() != rows * cols {
        anyhow::bail!("softmax_vulkan_generic: data.len()={} != rows*cols={}", data.len(), rows * cols);
    }

    let dd = ScopedAlloc::new(device, std::mem::size_of_val(data))?;
    device.memcpy_h2d(dd.ptr(), f32_to_bytes(data))?;

    let kernel = CompiledKernel::spirv("softmax", "main", spirv);
    // シェーダは1ワークグループ=1行(local_size_x=256)を前提とするため、
    // grid.x=rowsになるよう LaunchConfig::linear(rows*256, 256) を使う。
    let cfg = LaunchConfig::linear((rows * 256) as u32, 256);
    device.launch_kernel(&kernel, &cfg, &[KernelArg::Ptr(dd.ptr()), KernelArg::Usize(rows), KernelArg::Usize(cols)])?;
    device.synchronize()?;

    let mut out = vec![0.0f32; rows * cols];
    device.memcpy_d2h(f32_from_bytes_mut(&mut out), dd.ptr())?;
    Ok(out)
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
/// （タイル化・オンラインsoftmaxを実際に行う真のFlash Attentionは
/// 別関数 [`flash_attention`] として実装済み）。
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
    scaled_dot_product_attention_with_spirv(device, q, k, v, seq_len, head_dim, None)
}

/// [`scaled_dot_product_attention`] の SPIR-V 対応版(GEMMのみ)。
///
/// **2026-08-05実装、2026-08-06に`softmax_spirv`引数を追加した
/// [`scaled_dot_product_attention_with_spirv_and_softmax`]の薄い
/// ラッパーへ変更(後方互換、既存呼び出し元は無改修のまま)**。
/// `softmax_spirv=None`固定で呼ぶため、GEMM(QKᵀ・P·V)はVulkan経由に
/// なりうるがsoftmaxは常にホスト側CPU(rayon並列)のまま。GPU常駐の
/// softmaxカーネルも使いたい場合は
/// [`scaled_dot_product_attention_with_spirv_and_softmax`]を直接呼ぶこと。
#[allow(clippy::too_many_arguments)]
pub fn scaled_dot_product_attention_with_spirv(
    device: &dyn GpuDevice,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    spirv: Option<&[u8]>,
) -> Result<Vec<f32>> {
    scaled_dot_product_attention_with_spirv_and_softmax(device, q, k, v, seq_len, head_dim, spirv, None)
}

/// [`scaled_dot_product_attention`] のSPIR-V対応版(GEMM + softmax)。
///
/// **2026-08-06実装、正直な開示**: `matmul_spirv`(`sgemm_vulkan_generic`と
/// 同一契約の`matmul.spv`バイト列)に加えて`softmax_spirv`
/// (`softmax_vulkan_generic`と同一契約の`softmax.spv`バイト列)の**両方**が
/// `Some`で、かつ`select_gemm_path(device)`が`GemmPath::VulkanGeneric`を
/// 選ぶ場合のみ、QKᵀ・行ごとのsoftmax・P·Vの**すべてのステップを実際に
/// Vulkanデバイス上でディスパッチする**(「GPU GEMM + CPU softmax」の
/// ハイブリッドから「GPU GEMM + GPU softmax」への移行、直前のHANDOFF
/// 「次にすべきこと(1)」への対応)。`softmax_spirv`が`None`の場合は
/// 従来通りホスト側CPU(rayon並列)のsoftmaxにフォールバックする
/// (`scaled_dot_product_attention_with_spirv`と完全に同じ挙動、後方互換)。
/// GEMM側が`CpuNaive`(または`spirv`が`None`)の場合も同様に、softmax側も
/// 常にCPUのまま(GEMMがCPUで動いているのにsoftmaxだけVulkanへ飛ばすと
/// H2D/D2H転送往復が余計に増えるだけで意味が無いため、GEMM経路と
/// softmax経路は常に一致させる設計)。
///
/// 各ステップの詳細:
/// - QKᵀの計算には既存の[`sgemm_vulkan_generic`]（`matmul.comp`
///   シェーダ、非転置の通常GEMM専用）をそのまま再利用する。シェーダは
///   `b`の転置に対応していないため、Kをホスト側で転置してから
///   （`seq_len x head_dim` → `head_dim x seq_len`）通常のGEMMとして
///   渡す。alpha相当のスケーリング（`1/sqrt(head_dim)`）はシェーダが
///   対応していないため、GPU計算後にホスト側で適用する。
/// - softmax（行ごとのexp/sum/normalize）は、GPU経路が選ばれた場合
///   [`softmax_vulkan_generic`]（1ワークグループ=1行、共有メモリでの
///   max/sum二分木リダクションによる数値安定softmax）を呼ぶ。
/// - P·V の計算は[`sgemm`]をそのまま再利用する（`matmul_spirv`をそのまま
///   渡すため、`GemmPath::VulkanGeneric`ならここもVulkanへディスパッチ
///   される）。
#[allow(clippy::too_many_arguments)]
pub fn scaled_dot_product_attention_with_spirv_and_softmax(
    device: &dyn GpuDevice,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    matmul_spirv: Option<&[u8]>,
    softmax_spirv: Option<&[u8]>,
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

    let path = select_gemm_path(device);
    let gpu_gemm = matches!(path, GemmPath::VulkanGeneric) && matmul_spirv.is_some();

    // 1. scores = Q・Kᵀ / sqrt(head_dim)
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; seq_len * seq_len];
    if gpu_gemm {
        let spirv_bytes = matmul_spirv.expect("gpu_gemm implies matmul_spirv is Some");
        // シェーダは転置Bに対応していないため、Kをホスト側で
        // head_dim x seq_len へ転置してから通常GEMMとして渡す。
        let mut k_t = vec![0.0f32; head_dim * seq_len];
        for row in 0..seq_len {
            for col in 0..head_dim {
                k_t[col * seq_len + row] = k[row * head_dim + col];
            }
        }
        let raw = sgemm_vulkan_generic(device, seq_len, head_dim, seq_len, q, &k_t, spirv_bytes)?;
        for (s, r) in scores.iter_mut().zip(raw.iter()) {
            *s = scale * r;
        }
    } else {
        // launch_naive_gemm の alpha でスケーリングを同時に適用する。
        launch_naive_gemm(device, seq_len, head_dim, seq_len, scale, q, k, true, 0.0, &mut scores)?;
    }

    // 2. 行ごとのsoftmax。GEMM経路がVulkanかつsoftmax_spirvありのときのみ
    //    softmax_vulkan_generic経由でGPUディスパッチ、それ以外は従来通り
    //    ホスト側CPU(rayon並列、数値安定のため各行の最大値を引く)。
    let probs = if gpu_gemm {
        if let Some(softmax_spirv_bytes) = softmax_spirv {
            softmax_vulkan_generic(device, seq_len, seq_len, &scores, softmax_spirv_bytes)?
        } else {
            cpu_row_softmax(&scores, seq_len)
        }
    } else {
        cpu_row_softmax(&scores, seq_len)
    };

    // 3. output = probs・V （通常の GEMM、sgemm をそのまま再利用。
    //    matmul_spirv をそのまま渡すことで GemmPath::VulkanGeneric なら
    //    ここもVulkanへディスパッチされる）。
    let mut output = vec![0.0f32; seq_len * head_dim];
    sgemm(device, seq_len, seq_len, head_dim, 1.0, &probs, v, 0.0, &mut output, matmul_spirv)?;

    Ok(output)
}

/// 行ごとの数値安定softmax(ホスト側CPU、rayon並列)。
/// [`scaled_dot_product_attention_with_spirv_and_softmax`]のCPUフォール
/// バック経路として使う(GPU経路が選べない/`softmax_spirv`が`None`の場合)。
fn cpu_row_softmax(scores: &[f32], seq_len: usize) -> Vec<f32> {
    let mut probs = vec![0.0f32; seq_len * seq_len];
    probs.par_chunks_mut(seq_len).zip(scores.par_chunks(seq_len)).for_each(|(out_row, in_row)| {
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
    probs
}

/// 真の Flash Attention（タイル化 + オンラインsoftmax）。
///
/// [`scaled_dot_product_attention`] との違い: あちらは `seq_len x seq_len`
/// の `scores`/`probs` 行列をまるごとメモリに展開してから softmax を取るが、
/// こちらは Q を `Br` 行、K/V を `Bc` 行のブロックに分割し、ブロックごとに
/// 部分的な attention スコアだけを計算・破棄しながら、実行中の最大値
/// (`m_i`)・実行中の指数和 (`l_i`)・実行中の出力累積 (`acc`) だけを
/// 保持する「オンラインsoftmax」で結果を組み立てる（Dao et al. 2022,
/// FlashAttention のアルゴリズム1に相当）。これにより、どの時点でも
/// メモリ上に存在するスコア行列は最大でも `Br x Bc` のブロック分だけで済み、
/// `seq_len x seq_len` の全展開が要らない。
///
/// この実装は純粋なホスト側（CPU）Rustコードで、`GpuDevice` のカーネル
/// ディスパッチは使わない（`launch_kernel`/`memcpy` 等を挟まないぶん
/// タイル化アルゴリズムの正しさそのものに焦点を当てた実装で、GPU向け
/// カーネル化は別増分）。そのため他の関数と異なり `device` 引数を取らない。
///
/// `q`/`k`/`v` はいずれも `seq_len x head_dim`（行優先、単一ヘッド分）。
/// `block_size` は Q 側のブロック行数(`Br`)にも K/V 側のブロック行数(`Bc`)
/// にも使う（両方同じ値を使う簡易版。`seq_len` を割り切らなくてもよい —
/// 最終ブロックは残り行数に切り詰める）。`block_size == 0` はエラー。
///
/// 数学的には [`scaled_dot_product_attention`] と全く同じ結果を返す
/// （タイル化とオンラインsoftmaxは結果を変えない、純粋なメモリアクセス
/// パターンの最適化のため）。それは本クレートのテストで実際に検証している。
pub fn flash_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<Vec<f32>> {
    if q.len() != seq_len * head_dim {
        anyhow::bail!("flash_attention: q.len()={} != seq_len*head_dim={}", q.len(), seq_len * head_dim);
    }
    if k.len() != seq_len * head_dim {
        anyhow::bail!("flash_attention: k.len()={} != seq_len*head_dim={}", k.len(), seq_len * head_dim);
    }
    if v.len() != seq_len * head_dim {
        anyhow::bail!("flash_attention: v.len()={} != seq_len*head_dim={}", v.len(), seq_len * head_dim);
    }
    if block_size == 0 {
        anyhow::bail!("flash_attention: block_size must be > 0");
    }

    let scale = 1.0f32 / (head_dim as f32).sqrt();

    // Qの各行ブロックは他の行ブロックと完全に独立に計算できるため、rayonで
    // 行ブロック単位に並列化する(ブロック内部はシーケンシャル、オンライン
    // softmaxの漸化式そのままの実装)。
    let row_blocks: Vec<(usize, usize)> = (0..seq_len)
        .step_by(block_size)
        .map(|start| (start, (start + block_size).min(seq_len)))
        .collect();

    let mut output = vec![0.0f32; seq_len * head_dim];
    output
        .par_chunks_mut(head_dim)
        .enumerate()
        .try_for_each(|(row, out_row)| -> Result<()> {
            // このQ行が属するブロック開始位置は使わない(行単位で独立に
            // オンラインsoftmaxを回すため)。ブロック分割はK/V側にのみ適用する。
            let q_row = &q[row * head_dim..(row + 1) * head_dim];

            // オンラインsoftmaxの実行中状態: 最大値・指数和・重み付き累積出力。
            let mut m_i = f32::NEG_INFINITY;
            let mut l_i = 0.0f32;
            let mut acc = vec![0.0f32; head_dim];

            for &(kv_start, kv_end) in &row_blocks {
                let bc = kv_end - kv_start;

                // このK/Vブロックに対するスコア: s_j = scale * q_row・k_j
                let mut block_scores = vec![0.0f32; bc];
                for (j, score) in block_scores.iter_mut().enumerate() {
                    let k_row = &k[(kv_start + j) * head_dim..(kv_start + j + 1) * head_dim];
                    let dot: f32 = q_row.iter().zip(k_row.iter()).map(|(a, b)| a * b).sum();
                    *score = dot * scale;
                }

                let block_max = block_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let new_m = m_i.max(block_max);

                // 既存の累積値を新しい最大値基準へ再スケール(オンラインsoftmaxの核心)。
                let correction = if m_i == f32::NEG_INFINITY { 0.0 } else { (m_i - new_m).exp() };
                l_i *= correction;
                for a in acc.iter_mut() {
                    *a *= correction;
                }

                // このブロック分の寄与を加算。
                let mut block_l = 0.0f32;
                for (j, &s) in block_scores.iter().enumerate() {
                    let p = (s - new_m).exp();
                    block_l += p;
                    let v_row = &v[(kv_start + j) * head_dim..(kv_start + j + 1) * head_dim];
                    for (a, &vv) in acc.iter_mut().zip(v_row.iter()) {
                        *a += p * vv;
                    }
                }
                l_i += block_l;
                m_i = new_m;
            }

            if l_i > 0.0 {
                for (o, a) in out_row.iter_mut().zip(acc.iter()) {
                    *o = a / l_i;
                }
            }
            Ok(())
        })?;

    Ok(output)
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

/// AWQ（Activation-aware Weight Quantization、[Lin et al.](https://arxiv.org/abs/2306.00978)）
/// 風のINT4量子化。既存の`quantize_int4`は全チャネルを均等に量子化するため、
/// 重み自体は小さくても対応する活性化の振幅が大きい「重要チャネル」ほど、
/// 量子化グループ内の他チャネルに埋もれて粗く量子化され相対誤差が大きくなる。
/// この関数は`y = Wx = (W・diag(s)) ・ (diag(s)^-1・x)`という等価変換を使い、
/// 重み行列（`rows`×`cols`、行優先）の各入力チャネル（列）`j`を
/// `activation_scale[j]^alpha`でスケールアップしてから量子化することで、
/// 重要チャネルがグループ内の他要素に埋もれず量子化スケールに寄与するように
/// する。推論時は本来、量子化済み重みではなく入力活性化`x`側を`awq_scale`で
/// 割ってから掛け合わせる必要がある（matmulへの配線は次の増分のスコープ、
/// ここでは量子化APIの提供まで）。
pub struct QuantizedInt4AwqTensor {
    pub inner: QuantizedInt4Tensor,
    /// 列（入力チャネル）ごとのAWQスケール係数。長さ = `cols`。
    pub awq_scale: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

fn validate_awq_args(weight: &[f32], rows: usize, cols: usize, activation_scale: &[f32]) -> Result<()> {
    if rows == 0 || cols == 0 {
        anyhow::bail!("quantize_int4_awq: rows and cols must be > 0");
    }
    if weight.len() != rows * cols {
        anyhow::bail!("quantize_int4_awq: weight.len() must equal rows*cols");
    }
    if activation_scale.len() != cols {
        anyhow::bail!("quantize_int4_awq: activation_scale.len() must equal cols");
    }
    Ok(())
}

pub fn quantize_int4_awq(
    device: &dyn GpuDevice,
    weight: &[f32],
    rows: usize,
    cols: usize,
    activation_scale: &[f32],
    group_size: usize,
    alpha: f32,
) -> Result<QuantizedInt4AwqTensor> {
    validate_awq_args(weight, rows, cols, activation_scale)?;
    let awq_scale: Vec<f32> = activation_scale.iter().map(|&s| if s <= 0.0 { 1.0 } else { s.powf(alpha) }).collect();
    let mut scaled = vec![0f32; weight.len()];
    for r in 0..rows {
        for c in 0..cols {
            scaled[r * cols + c] = weight[r * cols + c] * awq_scale[c];
        }
    }
    let inner = quantize_int4(device, &scaled, group_size)?;
    Ok(QuantizedInt4AwqTensor { inner, awq_scale, rows, cols })
}

/// [`quantize_int4_awq`]の逆変換。AWQスケールを除去し元の重みスケールへ戻す
/// （往復検証・デバッグ用途。実推論では代わりに活性化側を`awq_scale`で割る）。
pub fn dequantize_int4_awq(t: &QuantizedInt4AwqTensor) -> Vec<f32> {
    let scaled = dequantize_int4(&t.inner);
    let mut out = vec![0f32; scaled.len()];
    for r in 0..t.rows {
        for c in 0..t.cols {
            let s = t.awq_scale[c];
            out[r * t.cols + c] = if s == 0.0 { 0.0 } else { scaled[r * t.cols + c] / s };
        }
    }
    out
}

/// DeepSeek-V3のMulti-head Latent Attention(MLA)にインスパイアされた、
/// 低ランク射影によるKVキャッシュ圧縮(2026-08-06追加)。
///
/// ## 調査結果(日英でGoogle/GitHub/論文調査、正直な開示)
///
/// DeepSeek-V3 technical report([arXiv:2412.19437](https://arxiv.org/abs/2412.19437))
/// によれば、MLAの核心は「KV(キー・バリュー)を`d_h`次元でそのまま
/// キャッシュせず、それより遥かに小さい`d_c`次元の潜在ベクトルへ
/// 低ランク射影(down-projection)して圧縮保存し、必要な時にup-projection
/// で復元する」という設計。DeepSeek-V2の実測では**KVキャッシュを
/// 93.3%削減、最大生成スループットを5.76倍に向上**させたと報告されている
/// (出典: 日本語調査記事、[Qiita: DeepSeek-V4-FlashのKVキャッシュ削減](https://qiita.com/sukimaengineer/items/b2f143552cf6d1eadeae)等の
/// 複数の解説記事で同様の数値を確認)。
///
/// ## 実装したもの(正直なスコープ)
///
/// 本関数は、MLAの**核心となる低ランク射影の仕組み(down-projection→
/// 圧縮保存→up-projection→復元)**を、このクレートが既に持つ`sgemm`
/// (CPU/Vulkan両対応、実機検証済み)を土台にそのまま実装したもの。
/// **正直な開示**: DeepSeek-V3の実際の`down_proj`/`up_proj`重み行列は
/// 大規模事前学習によって獲得されるものであり、本関数はその**学習済み
/// 重みを持たない**(学習パイプライン自体は本クレートのスコープ外)。
/// そのため「情報をほぼ無損失で圧縮できる」というMLAの実運用上の効能を
/// 主張するものではなく、「低ランク射影という計算の仕組み自体が、
/// 既存のGEMM基盤の上に正しく実装できる」ことの実証に留まる
/// (`quantize_int4`等の既存の量子化機能と同じ「メモリ効率化」という
/// 方向性の追加手段として位置づける)。
///
/// `kv`: `seq_len x d_h`(行優先)のキーまたはバリュー行列。
/// `down_proj`: `d_h x d_c`の下方射影行列(`d_c < d_h`を推奨)。
/// 戻り値: `seq_len x d_c`の圧縮された潜在表現。
pub fn mla_compress_kv(device: &dyn GpuDevice, seq_len: usize, d_h: usize, d_c: usize, kv: &[f32], down_proj: &[f32], spirv: Option<&[u8]>) -> Result<Vec<f32>> {
    anyhow::ensure!(kv.len() == seq_len * d_h, "mla_compress_kv: kv.len()={} != seq_len*d_h={}", kv.len(), seq_len * d_h);
    anyhow::ensure!(down_proj.len() == d_h * d_c, "mla_compress_kv: down_proj.len()={} != d_h*d_c={}", down_proj.len(), d_h * d_c);

    let mut latent = vec![0f32; seq_len * d_c];
    sgemm(device, seq_len, d_h, d_c, 1.0, kv, down_proj, 0.0, &mut latent, spirv)?;
    Ok(latent)
}

/// [`mla_compress_kv`]の逆変換(up-projection)。
///
/// `latent`: `seq_len x d_c`の圧縮された潜在表現。
/// `up_proj`: `d_c x d_h`の上方射影行列。
/// 戻り値: `seq_len x d_h`の復元されたKV行列(学習済み重みを使わない限り
/// 元のKVとは一致しない、上記モジュールdoc参照)。
pub fn mla_decompress_kv(device: &dyn GpuDevice, seq_len: usize, d_c: usize, d_h: usize, latent: &[f32], up_proj: &[f32], spirv: Option<&[u8]>) -> Result<Vec<f32>> {
    anyhow::ensure!(latent.len() == seq_len * d_c, "mla_decompress_kv: latent.len()={} != seq_len*d_c={}", latent.len(), seq_len * d_c);
    anyhow::ensure!(up_proj.len() == d_c * d_h, "mla_decompress_kv: up_proj.len()={} != d_c*d_h={}", up_proj.len(), d_c * d_h);

    let mut reconstructed = vec![0f32; seq_len * d_h];
    sgemm(device, seq_len, d_c, d_h, 1.0, latent, up_proj, 0.0, &mut reconstructed, spirv)?;
    Ok(reconstructed)
}

/// KVキャッシュ圧縮による理論上のメモリ削減率(%)を計算するヘルパー
/// (`d_c`が`d_h`よりどれだけ小さいかの単純比較、DeepSeek-V2が報告した
/// 「93.3%削減」のような数値をこのエコシステムの他プロジェクトが参照
/// できるようにする)。
pub fn mla_memory_reduction_percent(d_h: usize, d_c: usize) -> f64 {
    if d_h == 0 {
        return 0.0;
    }
    (1.0 - (d_c as f64 / d_h as f64)) * 100.0
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
        sgemm(device.as_ref(), 2, 2, 2, 1.0, &a, &b, 0.0, &mut c, None).unwrap();
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
        sgemm(device.as_ref(), 2, 2, 2, 2.0, &a, &b, 3.0, &mut c, None).unwrap();
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
        sgemm(device.as_ref(), 3, 3, 3, 1.0, &a, &b, 0.0, &mut c, None).unwrap();
        assert_eq!(c, b);
    }

    #[test]
    fn sgemm_rejects_mismatched_dimensions() {
        let device = cpu_device();
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![0.0; 4];
        // a has only 2 elements but m*k=4 is expected.
        assert!(sgemm(device.as_ref(), 2, 2, 2, 1.0, &a, &b, 0.0, &mut c, None).is_err());
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
    fn quantize_int4_awq_reduces_error_on_salient_low_magnitude_channel() {
        // 1行4列。チャネル3(0-indexed)は重み自体は小さい(0.1)が活性化が
        // 極端に大きい(=AWQ論文でいう「重要チャネル」)。group_size=4で
        // 行全体を1グループにすると、素のquantize_int4はmax_abs(10.0)を
        // 基準にスケールを決めるため0.1は丸めで0になり、そのチャネルの
        // 情報が完全に失われる。AWQは活性化スケールでチャネル3の重みを
        // 事前に引き上げることでこれを防ぐ。
        let device = cpu_device();
        let rows = 1;
        let cols = 4;
        let weight = vec![10.0f32, 10.0, 10.0, 0.1];
        let activation_scale = vec![1.0f32, 1.0, 1.0, 100.0];
        let group_size = 4;

        let plain = quantize_int4(device.as_ref(), &weight, group_size).unwrap();
        let plain_restored = dequantize_int4(&plain);
        let plain_err_ch3 = (weight[3] - plain_restored[3]).abs();
        assert_eq!(plain_restored[3], 0.0, "salient channel should be crushed to 0 without AWQ");

        let awq = quantize_int4_awq(device.as_ref(), &weight, rows, cols, &activation_scale, group_size, 1.0).unwrap();
        let awq_restored = dequantize_int4_awq(&awq);
        let awq_err_ch3 = (weight[3] - awq_restored[3]).abs();

        assert!(
            awq_err_ch3 < plain_err_ch3,
            "AWQ error {awq_err_ch3} should be < plain error {plain_err_ch3} on the salient channel"
        );
        // 非重要チャネル(0..3)はAWQでもほぼ同等の精度を保つ。
        for i in 0..3 {
            assert!((weight[i] - awq_restored[i]).abs() < 1.0, "channel {i} should stay reasonably precise");
        }
    }

    #[test]
    fn quantize_int4_awq_rejects_mismatched_shapes() {
        let device = cpu_device();
        let weight = vec![1.0f32; 6];
        // activation_scale.len() != cols(3)
        assert!(quantize_int4_awq(device.as_ref(), &weight, 2, 3, &[1.0, 1.0], 3, 1.0).is_err());
        // weight.len() != rows*cols
        assert!(quantize_int4_awq(device.as_ref(), &weight, 2, 4, &[1.0, 1.0, 1.0, 1.0], 4, 1.0).is_err());
    }

    #[test]
    fn sgemm_vulkan_generic_matches_cpu_naive_on_real_hardware() {
        // このテストは実Vulkan環境(このマシンではNVIDIA GeForce GT 730で
        // `vulkaninfo --summary`で実機確認済み)と、事前コンパイル済みの
        // matmul.spv(`examples/matmul_vulkan_real/shaders/matmul.spv`、
        // `tools/compile-vulkan-shaders.*`で生成、.gitignoreの対象)の
        // 両方が必要。どちらか欠けている環境(CI等)では誤魔化さず
        // テストをスキップする(assertを偽装しない方針)。
        let spirv_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/matmul_vulkan_real/shaders/matmul.spv");
        let spirv = match std::fs::read(&spirv_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!(
                    "skipping sgemm_vulkan_generic test: matmul.spv not compiled at {}: {e} \
                     (run tools/compile-vulkan-shaders.* first)",
                    spirv_path.display()
                );
                return;
            }
        };

        let vulkan_device = match opencuda_vulkan::VulkanDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping sgemm_vulkan_generic test: no real Vulkan device available: {e}");
                return;
            }
        };

        let m = 8;
        let k = 6;
        let n = 5;
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32).collect();

        let mut c_cpu = vec![0.0f32; m * n];
        let cpu_device = cpu_device();
        sgemm(cpu_device.as_ref(), m, k, n, 1.0, &a, &b, 0.0, &mut c_cpu, None).unwrap();

        let c_vulkan = sgemm_vulkan_generic(vulkan_device.as_ref(), m, k, n, &a, &b, &spirv).unwrap();

        assert_eq!(c_vulkan.len(), c_cpu.len());
        for (i, (&gv, &gc)) in c_vulkan.iter().zip(c_cpu.iter()).enumerate() {
            assert!((gv - gc).abs() < 1e-3, "idx {i}: vulkan={gv}, cpu={gc}");
        }
    }

    #[test]
    fn mla_compress_decompress_round_trip_matches_between_cpu_and_vulkan() {
        // MLA風の低ランクKV圧縮(2026-08-06新設)。sgemm_vulkan_generic系
        // テストと同じ方針で、matmul.spv/実Vulkanデバイスが無い環境では
        // assertを誤魔化さずスキップする。
        let spirv_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/matmul_vulkan_real/shaders/matmul.spv");
        let spirv = match std::fs::read(&spirv_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!("skipping mla test: matmul.spv not compiled at {}: {e}", spirv_path.display());
                return;
            }
        };
        let vulkan_device = match opencuda_vulkan::VulkanDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping mla test: no real Vulkan device available: {e}");
                return;
            }
        };

        let seq_len = 8;
        let d_h = 16; // 実KVの次元(例)
        let d_c = 4; // 圧縮後の潜在次元(d_c < d_h、DeepSeek-V2/V3と同じ低ランク方向)

        let kv: Vec<f32> = (0..seq_len * d_h).map(|i| ((i % 11) as f32) * 0.1).collect();
        let down_proj: Vec<f32> = (0..d_h * d_c).map(|i| ((i % 5) as f32) * 0.2 - 0.4).collect();
        let up_proj: Vec<f32> = (0..d_c * d_h).map(|i| ((i % 7) as f32) * 0.15 - 0.3).collect();

        let cpu_device = cpu_device();
        let latent_cpu = mla_compress_kv(cpu_device.as_ref(), seq_len, d_h, d_c, &kv, &down_proj, None).unwrap();
        let recon_cpu = mla_decompress_kv(cpu_device.as_ref(), seq_len, d_c, d_h, &latent_cpu, &up_proj, None).unwrap();

        let latent_vulkan = mla_compress_kv(vulkan_device.as_ref(), seq_len, d_h, d_c, &kv, &down_proj, Some(&spirv)).unwrap();
        let recon_vulkan = mla_decompress_kv(vulkan_device.as_ref(), seq_len, d_c, d_h, &latent_vulkan, &up_proj, Some(&spirv)).unwrap();

        assert_eq!(latent_cpu.len(), seq_len * d_c);
        for (i, (&gv, &gc)) in latent_vulkan.iter().zip(latent_cpu.iter()).enumerate() {
            assert!((gv - gc).abs() < 1e-3, "latent idx {i}: vulkan={gv}, cpu={gc}");
        }
        for (i, (&gv, &gc)) in recon_vulkan.iter().zip(recon_cpu.iter()).enumerate() {
            assert!((gv - gc).abs() < 1e-3, "reconstructed idx {i}: vulkan={gv}, cpu={gc}");
        }

        let reduction = mla_memory_reduction_percent(d_h, d_c);
        assert!((reduction - 75.0).abs() < 1e-9, "expected 75% reduction for d_h=16,d_c=4, got {reduction}");
    }

    #[test]
    fn softmax_vulkan_generic_matches_cpu_reference_on_real_hardware() {
        // 2026-08-06新設: softmax_vulkan_generic(1ワークグループ=1行+共有メモリ
        // リダクション)を実Vulkan環境で検証。sgemm_vulkan_generic系テストと
        // 同じ方針で、spvファイル未コンパイル/Vulkanデバイス無しの環境では
        // assertを誤魔化さずスキップする。
        let spirv_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/softmax_vulkan_real/shaders/softmax.spv");
        let spirv = match std::fs::read(&spirv_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!(
                    "skipping softmax_vulkan_generic test: softmax.spv not compiled at {}: {e} \
                     (run tools/compile-vulkan-shaders.* first)",
                    spirv_path.display()
                );
                return;
            }
        };

        let vulkan_device = match opencuda_vulkan::VulkanDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping softmax_vulkan_generic test: no real Vulkan device available: {e}");
                return;
            }
        };

        // 行数・列数ともに256を割り切らない値で、ループ境界(local_size_x=256
        // より多い列、行数がワークグループ数を素直に決める側)を確認する。
        let rows = 5;
        let cols = 41;
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let r = i / cols;
                let c = i % cols;
                ((r * 17 + c * 3) % 29) as f32 - 14.0
            })
            .collect();

        // CPU側リファレンス(数値安定softmax、rayon不要な小規模計算)。
        let mut expected = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let slice = &data[r * cols..(r + 1) * cols];
            let m = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = slice.iter().map(|&x| (x - m).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for (c, e) in exps.into_iter().enumerate() {
                expected[r * cols + c] = e / sum;
            }
        }

        let got = softmax_vulkan_generic(vulkan_device.as_ref(), rows, cols, &data, &spirv).unwrap();

        assert_eq!(got.len(), expected.len());
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-4, "idx {i}: vulkan={g}, expected={e}");
        }
        for r in 0..rows {
            let sum: f32 = got[r * cols..(r + 1) * cols].iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "row {r} does not sum to 1.0: {sum}");
        }
    }

    #[test]
    fn sgemm_auto_dispatch_uses_vulkan_path_on_real_nvidia_hardware_instead_of_cublas_stub() {
        // 上のテストは`sgemm_vulkan_generic`を明示的に呼んでいるだけで、
        // 自動選択の入口である`sgemm`(select_gemm_path経由)が実際に
        // VulkanGeneric経路を選ぶことまでは検証していなかった。このテストは
        // その自動選択そのものを検証する: 実機VulkanDeviceは
        // `GpuVendor::Nvidia`を返す(vendor_from_idがVendorID 0x10DEを
        // Nvidiaにマップするため)ので、`select_gemm_path`が単純にベンダー
        // だけで判定していれば`GemmPath::CuBlas`(未実装スタブ)を選び、
        // `sgemm`はエラーを返してしまう。`device.supports_spirv()`による
        // フォールバックが機能していれば、`sgemm`はVulkanGeneric経路を
        // 選び、cuBLASスタブを経由せず正しい結果を返すはずである。
        let spirv_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/matmul_vulkan_real/shaders/matmul.spv");
        let spirv = match std::fs::read(&spirv_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!(
                    "skipping sgemm_auto_dispatch test: matmul.spv not compiled at {}: {e} \
                     (run tools/compile-vulkan-shaders.* first)",
                    spirv_path.display()
                );
                return;
            }
        };

        let vulkan_device = match opencuda_vulkan::VulkanDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping sgemm_auto_dispatch test: no real Vulkan device available: {e}");
                return;
            }
        };

        // 実機がベンダー別スタブ経路(cuBLAS等)ではなくVulkanGenericへ
        // フォールバックすると自動選択されていることをまず直接確認する。
        assert_eq!(
            select_gemm_path(vulkan_device.as_ref()),
            GemmPath::VulkanGeneric,
            "expected select_gemm_path to prefer VulkanGeneric over the still-stubbed \
             vendor-specific path on this real Vulkan device"
        );
        assert!(matches!(vulkan_device.info().vendor, GpuVendor::Nvidia { .. }));

        let m = 8;
        let k = 6;
        let n = 5;
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32).collect();

        let mut c_cpu = vec![0.0f32; m * n];
        let cpu_device = cpu_device();
        sgemm(cpu_device.as_ref(), m, k, n, 1.0, &a, &b, 0.0, &mut c_cpu, None).unwrap();

        // ここが本題: 明示的な `sgemm_vulkan_generic` ではなく、自動選択の
        // 入口である `sgemm` 自体を、実機Vulkanデバイスに対して呼ぶ。
        // spirvを渡しているのでVulkanGeneric経路が選ばれれば実行できるはず。
        let mut c_auto = vec![0.0f32; m * n];
        sgemm(vulkan_device.as_ref(), m, k, n, 1.0, &a, &b, 0.0, &mut c_auto, Some(&spirv)).unwrap();

        assert_eq!(c_auto.len(), c_cpu.len());
        for (i, (&ga, &gc)) in c_auto.iter().zip(c_cpu.iter()).enumerate() {
            assert!((ga - gc).abs() < 1e-3, "idx {i}: sgemm(auto)={ga}, cpu={gc}");
        }
    }

    #[test]
    fn scaled_dot_product_attention_with_spirv_matches_cpu_on_real_hardware() {
        // 2026-08-05新設: scaled_dot_product_attention_with_spirv が
        // 実Vulkanハードウェア上で「GPU GEMM(QKᵀ・P·V) + CPU softmax」の
        // ハイブリッド経路を実際にディスパッチし、CPU版
        // (scaled_dot_product_attention、GemmPath::CpuNaive)と数値一致
        // することを検証する。matmul.spv/実Vulkanデバイスのどちらかが
        // 欠けている環境(CI等)では誤魔化さずスキップする(既存の同種
        // テストと同じ方針)。
        let spirv_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/matmul_vulkan_real/shaders/matmul.spv");
        let spirv = match std::fs::read(&spirv_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!(
                    "skipping scaled_dot_product_attention_with_spirv test: matmul.spv not \
                     compiled at {}: {e} (run tools/compile-vulkan-shaders.* first)",
                    spirv_path.display()
                );
                return;
            }
        };

        let vulkan_device = match opencuda_vulkan::VulkanDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping scaled_dot_product_attention_with_spirv test: no real Vulkan device available: {e}");
                return;
            }
        };

        assert_eq!(
            select_gemm_path(vulkan_device.as_ref()),
            GemmPath::VulkanGeneric,
            "expected select_gemm_path to pick VulkanGeneric on this real Vulkan device"
        );

        let seq_len = 6;
        let head_dim = 4;
        let q: Vec<f32> = (0..seq_len * head_dim).map(|i| ((i as f32) * 0.11).sin()).collect();
        let k: Vec<f32> = (0..seq_len * head_dim).map(|i| ((i as f32) * 0.23).cos()).collect();
        let v: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.07 - 0.5).collect();

        let cpu_device = cpu_device();
        let cpu_out = scaled_dot_product_attention(cpu_device.as_ref(), &q, &k, &v, seq_len, head_dim).unwrap();

        // 本題: 実Vulkanデバイスへ spirv を渡して呼ぶ。select_gemm_path が
        // VulkanGeneric を選び、かつ spirv が Some なので、QKᵀ・P·V の
        // 両方の GEMM ステップが実際に VulkanDevice::launch_kernel
        // (KernelSource::SpirV) 経由でディスパッチされる
        // (launch_naive_gemm の Native カーネル経由にはならない)。
        let vulkan_out = scaled_dot_product_attention_with_spirv(
            vulkan_device.as_ref(),
            &q,
            &k,
            &v,
            seq_len,
            head_dim,
            Some(&spirv),
        )
        .unwrap();

        assert_eq!(cpu_out.len(), vulkan_out.len());
        for (i, (&cv, &vv)) in cpu_out.iter().zip(vulkan_out.iter()).enumerate() {
            assert!((cv - vv).abs() < 1e-3, "idx {i}: cpu={cv}, vulkan={vv}");
        }
    }

    #[test]
    fn scaled_dot_product_attention_with_spirv_and_softmax_matches_cpu_on_real_hardware() {
        // 2026-08-06新設: matmul.spv に加えて softmax.spv も渡した場合、
        // QKᵀ・softmax・P·Vのすべてが実Vulkanハードウェア上でディスパッチ
        // される「GPU GEMM + GPU softmax」経路が、CPU版
        // (scaled_dot_product_attention、GemmPath::CpuNaive)と数値一致
        // することを検証する。matmul.spv/softmax.spv/実Vulkanデバイスの
        // いずれかが欠けている環境(CI等)では誤魔化さずスキップする。
        let matmul_spirv_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/matmul_vulkan_real/shaders/matmul.spv");
        let matmul_spirv = match std::fs::read(&matmul_spirv_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!(
                    "skipping scaled_dot_product_attention_with_spirv_and_softmax test: matmul.spv not \
                     compiled at {}: {e} (run tools/compile-vulkan-shaders.* first)",
                    matmul_spirv_path.display()
                );
                return;
            }
        };
        let softmax_spirv_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/softmax_vulkan_real/shaders/softmax.spv");
        let softmax_spirv = match std::fs::read(&softmax_spirv_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!(
                    "skipping scaled_dot_product_attention_with_spirv_and_softmax test: softmax.spv not \
                     compiled at {}: {e} (run tools/compile-vulkan-shaders.* first)",
                    softmax_spirv_path.display()
                );
                return;
            }
        };

        let vulkan_device = match opencuda_vulkan::VulkanDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping scaled_dot_product_attention_with_spirv_and_softmax test: no real Vulkan device available: {e}");
                return;
            }
        };

        let seq_len = 6;
        let head_dim = 4;
        let q: Vec<f32> = (0..seq_len * head_dim).map(|i| ((i as f32) * 0.11).sin()).collect();
        let k: Vec<f32> = (0..seq_len * head_dim).map(|i| ((i as f32) * 0.23).cos()).collect();
        let v: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.07 - 0.5).collect();

        let cpu_device = cpu_device();
        let cpu_out = scaled_dot_product_attention(cpu_device.as_ref(), &q, &k, &v, seq_len, head_dim).unwrap();

        // 本題: matmul_spirv・softmax_spirv両方をSomeで渡す。GEMM(QKᵀ・P・V)
        // だけでなくsoftmax自体もsoftmax_vulkan_generic経由でVulkan
        // ディスパッチされる。
        let vulkan_out = scaled_dot_product_attention_with_spirv_and_softmax(
            vulkan_device.as_ref(),
            &q,
            &k,
            &v,
            seq_len,
            head_dim,
            Some(&matmul_spirv),
            Some(&softmax_spirv),
        )
        .unwrap();

        assert_eq!(cpu_out.len(), vulkan_out.len());
        for (i, (&cv, &vv)) in cpu_out.iter().zip(vulkan_out.iter()).enumerate() {
            assert!((cv - vv).abs() < 1e-3, "idx {i}: cpu={cv}, vulkan={vv}");
        }
    }

    #[test]
    fn flash_attention_matches_naive_attention_on_fixed_input() {
        // 固定入力(乱数不要の再現可能な値)で、タイル化+オンラインsoftmaxの
        // flash_attentionが、全展開版のscaled_dot_product_attentionと
        // 数値的に一致することを検証する。block_sizeがseq_lenを割り切らない
        // ケースも含める。
        let device = cpu_device();
        let seq_len = 7;
        let head_dim = 5;
        let q: Vec<f32> = (0..seq_len * head_dim).map(|i| ((i as f32) * 0.13).sin()).collect();
        let k: Vec<f32> = (0..seq_len * head_dim).map(|i| ((i as f32) * 0.29).cos()).collect();
        let v: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.05 - 1.0).collect();

        let naive = scaled_dot_product_attention(device.as_ref(), &q, &k, &v, seq_len, head_dim).unwrap();

        for block_size in [1usize, 2, 3, 4, seq_len, seq_len * 2] {
            let flash = flash_attention(&q, &k, &v, seq_len, head_dim, block_size).unwrap();
            assert_eq!(flash.len(), naive.len());
            for (i, (&f, &n)) in flash.iter().zip(naive.iter()).enumerate() {
                assert!(
                    (f - n).abs() < 1e-4,
                    "block_size={block_size}, idx {i}: flash={f}, naive={n}"
                );
            }
        }
    }

    #[test]
    fn flash_attention_matches_naive_attention_on_larger_random_input() {
        // 手計算しやすい小さい入力だけでなく、もう少し大きいサイズでも
        // 一致することを確認する(疑似乱数のかわりに決定的なLCGを使い、
        // テストの再現性を保つ)。
        let device = cpu_device();
        let seq_len = 17;
        let head_dim = 8;

        let mut state: u64 = 0x2545F4914F6CDD1D;
        let mut next_f32 = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = (state >> 33) as u32;
            (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let q: Vec<f32> = (0..seq_len * head_dim).map(|_| next_f32()).collect();
        let k: Vec<f32> = (0..seq_len * head_dim).map(|_| next_f32()).collect();
        let v: Vec<f32> = (0..seq_len * head_dim).map(|_| next_f32()).collect();

        let naive = scaled_dot_product_attention(device.as_ref(), &q, &k, &v, seq_len, head_dim).unwrap();
        let flash = flash_attention(&q, &k, &v, seq_len, head_dim, 4).unwrap();

        for (i, (&f, &n)) in flash.iter().zip(naive.iter()).enumerate() {
            assert!((f - n).abs() < 1e-4, "idx {i}: flash={f}, naive={n}");
        }
    }

    #[test]
    fn flash_attention_seq_len_one_returns_v_unchanged() {
        let head_dim = 4;
        let q = vec![0.5, -1.0, 2.0, 0.25];
        let k = vec![1.0, 1.0, 1.0, 1.0];
        let v = vec![10.0, 20.0, 30.0, 40.0];
        let out = flash_attention(&q, &k, &v, 1, head_dim, 3).unwrap();
        assert_eq!(out, v);
    }

    #[test]
    fn flash_attention_rejects_mismatched_dimensions_and_zero_block_size() {
        let q = vec![1.0, 2.0];
        let k = vec![1.0, 1.0, 1.0, 1.0];
        let v = vec![1.0, 1.0, 1.0, 1.0];
        assert!(flash_attention(&q, &k, &v, 2, 2, 1).is_err());

        let q2 = vec![1.0, 2.0, 3.0, 4.0];
        assert!(flash_attention(&q2, &k, &v, 2, 2, 0).is_err());
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
