//! # open-cuda-llm
//!
//! 自己回帰デコーダ(GPT系アーキテクチャ)のforward pass実装。
//! `open-raid-z`の2026-07-21マーケティング調査ロードマップで言う
//! 「Python製AIライブラリのRust移植 1〜6位」のうち、**1位のvLLM相当**の
//! MVP(最小実用実装)にあたる。`open-cuda-bert`(2位Transformers相当、
//! エンコーダ専用)・`opencuda-blas`(3位NumPy相当、GEMM/Attention)が
//! 既に存在していたので、本クレートはそれらの上に「トークンを1つずつ
//! 生成していく」自己回帰パス(KVキャッシュ付き貪欲デコード)を追加する。
//!
//! ## 正直な開示(スコープの限界)
//!
//! - **これは本家vLLMの核心的な最適化(PagedAttention、連続バッチング
//!   〈continuous batching〉、複数リクエストの同時処理)を一切実装して
//!   いない**。単一シーケンスを1件ずつ、KVキャッシュを使って逐次デコード
//!   するだけの素朴な実装(いわば「vLLMが最適化する前のベースライン」)。
//! - **2026-07-25追記: 実在の学習済み重み(GPT-2 124M、`openai-community/gpt2`)
//!   を読み込む`GptModel::load`を追加した**(`open-cuda-bert::BertModel::load`
//!   と同じ設計、safetensorsを直接パース)。デフォルトのコンストラクタは
//!   引き続き決定的な疑似乱数(`SplitMix64`)による`load_random`(既存の
//!   最小構成テスト・KVキャッシュ数値一致テストはこちらを使い続ける、
//!   後方互換)。実重みを使うにはGPT-2自身のBPE語彙に対応した
//!   `GptTokenizer`(下記)が必要——`ByteTokenizer`(バイト値=トークンID)を
//!   実重みと組み合わせても、GPT-2のBPE語彙とは無関係なIDを渡すことに
//!   なるため意味のある出力は得られない。
//! - **トークナイザ**: 従来通り既定は`ByteTokenizer`(UTF-8バイト単位、
//!   外部ファイル不要)。実重みでの検証用に`tokenizers`クレートによる
//!   本格的なBPE/SentencePiece対応`GptTokenizer`(GPT-2の`tokenizer.json`を
//!   読み込む)も追加した。
//! - Attentionは`open-cuda-bert`と同じく`opencuda-blas::scaled_dot_product_attention`
//!   (非タイル化の素朴な実装)をそのまま使う。KVキャッシュ付きの1トークン
//!   ずつの生成では、クエリ行を`n`回複製して`n x n`のattentionを計算し
//!   先頭行だけを使うという簡易的な方法で「新規トークンのクエリ×
//!   過去全体のキー/バリュー」を計算している(数学的には正しいが、
//!   本来必要な計算量よりO(n)倍無駄が多い——専用のcausal-attention
//!   カーネルを`opencuda-blas`に追加するのが次の最適化)。

use std::path::Path;

use anyhow::{Context, Result};
use opencuda_core::GpuDevice;
use serde::Deserialize;

/// デコーダの設定(GPT系アーキテクチャの最小構成)。
#[derive(Debug, Clone)]
pub struct GptConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,
    pub layer_norm_eps: f32,
}

impl GptConfig {
    /// テスト・デモ用の極小構成(実運用サイズではない)。
    pub fn tiny(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            intermediate_size: 64,
            max_seq_len: 256,
            layer_norm_eps: 1e-5,
        }
    }
}

/// `config.json`(Hugging Face GPT-2形式)の生パース用構造体。
/// `GptConfig`と1対1ではない(フィールド名が異なる、`n_ctx`と
/// `n_positions`のどちらか一方しか無いモデルもある)ため、変換関数
/// `GPT2Config::into_gpt_config`を介する。
#[derive(Debug, Deserialize)]
struct GPT2Config {
    vocab_size: usize,
    n_embd: usize,
    n_layer: usize,
    n_head: usize,
    #[serde(default)]
    n_ctx: Option<usize>,
    #[serde(default)]
    n_positions: Option<usize>,
    #[serde(default = "default_gpt2_eps")]
    layer_norm_epsilon: f32,
}

fn default_gpt2_eps() -> f32 {
    1e-5
}

impl GPT2Config {
    fn into_gpt_config(self) -> Result<GptConfig> {
        let max_seq_len = self
            .n_positions
            .or(self.n_ctx)
            .context("open-cuda-llm: config.json must have n_positions or n_ctx")?;
        anyhow::ensure!(self.n_embd % self.n_head == 0, "open-cuda-llm: n_embd {} not divisible by n_head {}", self.n_embd, self.n_head);
        Ok(GptConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.n_embd,
            num_layers: self.n_layer,
            num_heads: self.n_head,
            intermediate_size: 4 * self.n_embd,
            max_seq_len,
            layer_norm_eps: self.layer_norm_epsilon,
        })
    }
}

fn tensor_f32(tensors: &safetensors::SafeTensors, name: &str) -> Result<Vec<f32>> {
    let view = tensors.tensor(name).with_context(|| format!("open-cuda-llm: missing tensor '{name}'"))?;
    anyhow::ensure!(
        view.dtype() == safetensors::Dtype::F32,
        "open-cuda-llm: tensor '{name}' has unexpected dtype {:?} (expected F32)",
        view.dtype()
    );
    let bytes = view.data();
    anyhow::ensure!(bytes.len() % 4 == 0, "open-cuda-llm: tensor '{name}' byte length not a multiple of 4");
    let mut out = vec![0.0f32; bytes.len() / 4];
    for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *dst = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(out)
}

/// `[out_dim, in_dim]`(行優先)を`[in_dim, out_dim]`へ転置する
/// (`open-cuda-bert`の同名ヘルパーと同じ用途。GPT-2のトークン埋め込み
/// `wte.weight`は`[vocab_size, hidden]`で保存されており、重み共有
/// 〈weight tying〉される`lm_head`としては`[hidden, vocab_size]`
/// レイアウトが必要なため転置する)。
fn transpose(src: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; src.len()];
    for o in 0..out_dim {
        for i in 0..in_dim {
            out[i * out_dim + o] = src[o * in_dim + i];
        }
    }
    out
}

/// GPT-2の`Conv1D`層(`nn.Linear`と違い`[in_dim, out_dim]`のまま保存、
/// 転置不要)から`Linear`を組み立てる。
fn load_conv1d(tensors: &safetensors::SafeTensors, prefix: &str, in_dim: usize, out_dim: usize) -> Result<Linear> {
    let weight = tensor_f32(tensors, &format!("{prefix}.weight"))?;
    anyhow::ensure!(
        weight.len() == in_dim * out_dim,
        "open-cuda-llm: '{prefix}.weight' has {} elements, expected {}x{}",
        weight.len(),
        in_dim,
        out_dim
    );
    let bias = tensor_f32(tensors, &format!("{prefix}.bias"))?;
    Ok(Linear { weight_t: weight, bias, in_dim, out_dim, spirv_matmul: None, dxil_offload: None })
}

// **2026-08-04変更**: GPT-2はQ/K/Vを1本の`c_attn`(`[hidden, 3*hidden]`)へ
// 融合して保存している。以前はこれを列方向に3分割し、Q/K/Vそれぞれ独立
// した`Linear`(3回のGEMM呼び出し)として扱っていたが、safetensors側の
// レイアウトが既に`Linear::forward`が要求する`[in_dim, out_dim]`
// (`in_dim=hidden, out_dim=3*hidden`)そのものであるため、分割は不要
// ——単に`load_conv1d`で1本の融合`Linear`として読み込めばよい。これに
// より推論側のディスパッチ回数がQ/K/Vぶん1/3になる(`load_conv1d`を
// そのまま再利用、専用関数〈旧`load_fused_qkv`〉は廃止)。分割後に3つの
// 独立したGEMMで計算していた場合と、1回のGEMMで計算してから列方向に
// 出力を切り出す場合とで数値結果は完全に一致する(同じ内積・同じ累積
// 順序、下記HANDOFF 2026-08-04参照)。

fn load_layer_norm(tensors: &safetensors::SafeTensors, prefix: &str, eps: f32) -> Result<LayerNorm> {
    let weight = tensor_f32(tensors, &format!("{prefix}.weight"))?;
    let bias = tensor_f32(tensors, &format!("{prefix}.bias"))?;
    Ok(LayerNorm { weight, bias, eps })
}

/// 決定的な疑似乱数生成器(SplitMix64)。学習済み重みが無い現段階で、
/// 「毎回同じ初期化になる」ことを保証するためだけに使う(暗号用途ではない)。
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// `[-scale, scale]`の範囲の一様乱数(小さめの初期化スケール、
    /// Xavier初期化の簡易近似)。
    fn next_f32(&mut self, scale: f32) -> f32 {
        let bits = (self.next_u64() >> 40) as u32; // 上位24bit程度を使う
        let unit = (bits as f32) / (1u32 << 24) as f32; // [0, 1)
        (unit * 2.0 - 1.0) * scale
    }
}

fn random_vec(rng: &mut SplitMix64, len: usize, scale: f32) -> Vec<f32> {
    (0..len).map(|_| rng.next_f32(scale)).collect()
}

struct Linear {
    weight_t: Vec<f32>, // in_dim x out_dim (行優先)
    bias: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
    /// コンパイル済み`matmul.spv`(`opencuda_blas::sgemm`が
    /// `GemmPath::VulkanGeneric`を選んだ際に必要、2026-08-05配線)。
    /// `GptModel::set_matmul_spirv`経由で全`Linear`インスタンスに同じ
    /// `Arc`を共有させる。未設定(`None`)ならCPU実行(`GemmPath::
    /// CpuNaive`)のまま——既存の挙動を一切変えない後方互換なデフォルト。
    spirv_matmul: Option<std::sync::Arc<Vec<u8>>>,
    /// **2026-08-23新設**: D3D12 Compute(DXIL)への密GEMMオフロード。
    ///
    /// `spirv_matmul`とは設計が異なり、**呼び出し側が`forward`へ渡す
    /// `device`とは別のデバイス**(`opencuda-directx::real::DirectXDevice`)
    /// をこの`Linear`自身が保持する。理由: `DirectXDevice`は
    /// `KernelSource::Dxil`しか実行できず、`launch_naive_gemm`の
    /// Rustクロージャカーネル(Attention・LayerNorm等が使う)を実行
    /// できない。したがってモデル全体をDirectXデバイス上で走らせる
    /// ことはできず、「密GEMMだけGPUへ、それ以外はCPU(open-cpu SIMD)で」
    /// というハイブリッド構成にしてある。
    dxil_offload: Option<DxilMatmulOffload>,
}

/// [`GptModel::set_matmul_dxil_offload`]で配線される、密GEMM専用の
/// D3D12 Computeオフロード先(2026-08-23新設)。
#[derive(Clone)]
pub struct DxilMatmulOffload {
    device: std::sync::Arc<dyn GpuDevice>,
    dxil: std::sync::Arc<Vec<u8>>,
    /// この`Linear`の重み行列(`weight_t`, in_dim×out_dim)をデバイスへ
    /// 常駐させたポインタ。重みは推論中不変なので、毎回H2D転送するのは
    /// 純粋な無駄(`lm_head`なら768×50257×4 ≒ 154MBを1トークンごとに
    /// 転送してしまう)。実測でこの常駐化により6〜10倍速くなった
    /// (`opencuda-blas/tests/sgemm_directx_bench.rs`参照)。
    ///
    /// 所有権は[`ResidentWeights`]が持ち、`GptModel`のdrop時にまとめて
    /// 解放される。
    b_ptr: opencuda_core::DevicePtr,
}

/// DXILオフロードでデバイス常駐させた重みの解放を担うRAIIハンドル
/// (2026-08-23新設)。`GptModel`が保持し、dropされた時点で全ポインタを
/// `device.free()`する。
pub struct ResidentWeights {
    device: std::sync::Arc<dyn GpuDevice>,
    ptrs: Vec<opencuda_core::DevicePtr>,
}

impl Drop for ResidentWeights {
    fn drop(&mut self) {
        for p in self.ptrs.drain(..) {
            if let Err(e) = self.device.free(p) {
                tracing::warn!("open-cuda-llm: failed to free resident DXIL weight buffer: {e}");
            }
        }
    }
}

impl std::fmt::Debug for DxilMatmulOffload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DxilMatmulOffload").field("device", &self.device.info().name).field("dxil_len", &self.dxil.len()).finish()
    }
}

impl Linear {
    fn random(rng: &mut SplitMix64, in_dim: usize, out_dim: usize) -> Self {
        let scale = 1.0 / (in_dim as f32).sqrt();
        Self { weight_t: random_vec(rng, in_dim * out_dim, scale), bias: vec![0.0; out_dim], in_dim, out_dim, spirv_matmul: None, dxil_offload: None }
    }

    fn forward(&self, device: &dyn GpuDevice, x: &[f32], seq_len: usize) -> Result<Vec<f32>> {
        debug_assert_eq!(x.len(), seq_len * self.in_dim);
        let mut out = vec![0.0f32; seq_len * self.out_dim];
        // DXILオフロードが配線済みなら、密GEMMだけをD3D12 Computeで実行する
        // (失敗した場合は黙って誤結果を返さず、そのままエラーを返す)。
        if let Some(off) = &self.dxil_offload {
            let result = opencuda_blas::sgemm_directx_resident_b(&*off.device, seq_len, self.in_dim, self.out_dim, x, off.b_ptr, &off.dxil)?;
            out.copy_from_slice(&result);
            for row in 0..seq_len {
                for c in 0..self.out_dim {
                    out[row * self.out_dim + c] += self.bias[c];
                }
            }
            return Ok(out);
        }
        let spirv = self.spirv_matmul.as_deref().map(|v| v.as_slice());
        opencuda_blas::sgemm(device, seq_len, self.in_dim, self.out_dim, 1.0, x, &self.weight_t, 0.0, &mut out, spirv)?;
        for row in 0..seq_len {
            for c in 0..self.out_dim {
                out[row * self.out_dim + c] += self.bias[c];
            }
        }
        Ok(out)
    }
}

struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    eps: f32,
}

impl LayerNorm {
    fn identity(dim: usize, eps: f32) -> Self {
        Self { weight: vec![1.0; dim], bias: vec![0.0; dim], eps }
    }

    fn forward(&self, x: &mut [f32], seq_len: usize, dim: usize) {
        for row in 0..seq_len {
            let slice = &mut x[row * dim..(row + 1) * dim];
            let mean: f32 = slice.iter().sum::<f32>() / dim as f32;
            let var: f32 = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
            let inv_std = 1.0 / (var + self.eps).sqrt();
            for (i, v) in slice.iter_mut().enumerate() {
                *v = (*v - mean) * inv_std * self.weight[i] + self.bias[i];
            }
        }
    }
}

/// `gelu_new`(GPT-2のHugging Face実装が使うtanh近似GELU、
/// `config.json`の`activation_function: "gelu_new"`に対応)。
/// `0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))`。
fn gelu_inplace(x: &mut [f32]) {
    const SQRT_2_OVER_PI: f64 = 0.7978845608028654;
    for v in x.iter_mut() {
        let xf = *v as f64;
        let inner = SQRT_2_OVER_PI * (xf + 0.044715 * xf.powi(3));
        *v = (0.5 * xf * (1.0 + inner.tanh())) as f32;
    }
}

/// **アーキテクチャ注記(2026-07-25、safetensorsローダー追加時に変更)**:
/// 当初は(BERT/GPT-1系と同じ)post-LN——Attention/FFNの後に残差加算+LN——
/// だったが、実在のGPT-2の学習済み重みをそのまま読み込んで意味のある
/// 出力を得るには、GPT-2が採用しているpre-LN(Attention/FFNの「前」に
/// 正規化を適用し、残差加算は正規化前の`hidden`に対して行う)へ構造を
/// 合わせる必要があったため変更した。ランダム初期化パス(`load_random`)
/// はpre-LN/post-LNどちらでも数学的な整合性(KVキャッシュ増分計算と
/// フルスクラッチ再計算の一致)に影響しないため、既存テストへの
/// 悪影響は無い。
struct DecoderLayer {
    ln_1: LayerNorm,
    /// Q/K/Vを1本に融合した`Linear`(`out_dim = 3*hidden`、列0..hidden=Q・
    /// hidden..2*hidden=K・2*hidden..3*hidden=V)。2026-08-04に3本の独立
    /// した`Linear`から統合(上記`load_conv1d`呼び出し箇所のコメント参照)。
    qkv: Linear,
    attn_out: Linear,
    ln_2: LayerNorm,
    intermediate: Linear,
    output: Linear,
    /// **2026-08-07新設**: [`GptModel::enable_mla_kv_compression`]配線先。
    /// ヘッドごとのMLA低ランク射影(`opencuda_blas::mla_compress_kv`/
    /// `mla_decompress_kv`土台)、`None`(既定)なら従来通りKVキャッシュを
    /// フル精度のまま保持する(後方互換、既存テストへの影響ゼロ)。
    mla: Option<Vec<MlaHeadProjection>>,
}

impl DecoderLayer {
    fn random(rng: &mut SplitMix64, hidden: usize, intermediate: usize, eps: f32) -> Self {
        Self {
            ln_1: LayerNorm::identity(hidden, eps),
            qkv: Linear::random(rng, hidden, 3 * hidden),
            attn_out: Linear::random(rng, hidden, hidden),
            ln_2: LayerNorm::identity(hidden, eps),
            intermediate: Linear::random(rng, hidden, intermediate),
            output: Linear::random(rng, intermediate, hidden),
            mla: None,
        }
    }

    /// 1トークン分(`hidden`は`hidden_size`長の単一行)を処理し、
    /// このレイヤーの`cache`へ今回のk/vを追加した上で出力を返す
    /// (causalマスクは「まだキャッシュに存在しない未来のトークンは
    /// そもそも追加されていない」ことで自然に実現される、明示的な
    /// マスク行列は不要)。pre-LN(GPT-2方式、上記構造体docコメント参照)。
    #[allow(clippy::too_many_arguments)]
    fn forward_step(
        &self,
        device: &dyn GpuDevice,
        hidden: &[f32],
        cache: &mut [KvCacheHead],
        hidden_size: usize,
        num_heads: usize,
        softmax_spirv: Option<&[u8]>,
        flash_spirv: Option<(&[u8], usize)>,
    ) -> Result<Vec<f32>> {
        let head_dim = hidden_size / num_heads;

        let mut normed = hidden.to_vec();
        self.ln_1.forward(&mut normed, 1, hidden_size);

        // 2026-08-04: Q/K/Vを3回の別々のGEMMではなく、融合`c_attn`への
        // 1回のGEMMで計算し、出力を列方向に3分割する(下記`forward_prefill`
        // と全く同じ分割規約)。
        let qkv = self.qkv.forward(device, &normed, 1)?;
        let q = &qkv[0..hidden_size];
        let k = &qkv[hidden_size..2 * hidden_size];
        let v = &qkv[2 * hidden_size..3 * hidden_size];

        let mut context = vec![0.0f32; hidden_size];
        for (h, cache_head) in cache.iter_mut().enumerate().take(num_heads) {
            let col_start = h * head_dim;
            let q_h = &q[col_start..col_start + head_dim];
            let k_h = &k[col_start..col_start + head_dim];
            let v_h = &v[col_start..col_start + head_dim];

            let spirv = self.qkv.spirv_matmul.as_deref().map(|v| v.as_slice());
            let proj = self.mla.as_ref().map(|v| &v[h]);
            cache_head.push(device, k_h, v_h, proj, spirv)?;
            let n = cache_head.n;
            let (k_all, v_all) = cache_head.current_kv(device, head_dim, proj, spirv)?;

            // qを n 回複製して n x n の attention を計算し、先頭行(全行
            // 同一)だけを使う(モジュールdocコメント参照、素朴だが正しい)。
            let mut q_full = vec![0.0f32; n * head_dim];
            for row in q_full.chunks_exact_mut(head_dim) {
                row.copy_from_slice(q_h);
            }
            // **2026-08-07新設**: `flash_spirv`が配線済みなら、QKᵀ・softmax・
            // P·Vを1回のディスパッチで完結する`flash_attention_with_spirv`
            // (open-cuda側2026-08-07 HANDOFF「次にすべきこと(1)」対応)を使う。
            // `q_full`/`k_all`/`v_all`はいずれもすでに`n*head_dim`長(先頭行
            // 複製方式、モジュールdocコメント参照)で`flash_attention_with_
            // spirv`の`seq_len=n`契約とそのまま一致するため、追加の変換は
            // 不要。未配線(`None`、既定)の場合は従来通り
            // `scaled_dot_product_attention_with_spirv_and_softmax`
            // (GEMM+softmaxを別々にディスパッチする経路)にフォールバックする
            // (後方互換、既存呼び出し元・テストへの影響なし)。
            let out = if let Some((flash_bytes, block_size)) = flash_spirv {
                opencuda_blas::flash_attention_with_spirv(device, &q_full, &k_all, &v_all, n, head_dim, block_size, flash_bytes)?
            } else {
                opencuda_blas::scaled_dot_product_attention_with_spirv_and_softmax(
                    device,
                    &q_full,
                    &k_all,
                    &v_all,
                    n,
                    head_dim,
                    spirv,
                    softmax_spirv,
                )?
            };
            context[col_start..col_start + head_dim].copy_from_slice(&out[0..head_dim]);
        }

        let attn_dense = self.attn_out.forward(device, &context, 1)?;
        let mut hidden2 = hidden.to_vec();
        for (a, b) in hidden2.iter_mut().zip(attn_dense.iter()) {
            *a += b; // residual(正規化前のhiddenへ加算、pre-LN方式)
        }

        let mut normed2 = hidden2.clone();
        self.ln_2.forward(&mut normed2, 1, hidden_size);

        let mut intermediate = self.intermediate.forward(device, &normed2, 1)?;
        gelu_inplace(&mut intermediate);

        let ffn_out = self.output.forward(device, &intermediate, 1)?;
        let mut hidden3 = hidden2.clone();
        for (a, b) in hidden3.iter_mut().zip(ffn_out.iter()) {
            *a += b; // residual
        }

        Ok(hidden3)
    }

    /// **2026-08-04新設(プリフィル/デコード分離、`aruaru-llm`側
    /// 2026-07-26 HANDOFFで指摘された次の設計変更(a))**: プロンプト全体
    /// (`seq_len`トークン)を1回のバッチ処理として通す。`forward_step`が
    /// 1トークンずつ`seq_len=1`でLinear/GEMMを呼ぶのに対し、本メソッドは
    /// Q/K/V融合GEMM・`attn_out`・`intermediate`・`output`の4つのLinearを
    /// いずれも`seq_len=プロンプト長`の**本当のGEMM(m>1)**として1回ずつ
    /// 呼ぶ(レイヤーあたりのディスパッチ回数が`4*seq_len`から`4`へ削減)。
    /// Attention自体は引き続き位置ごとの因果性(causality)を守るため、
    /// 各行(トークン位置)を昇順に処理し、その位置までのキャッシュのみを
    /// 参照する(`forward_step`をprompt長ぶん呼んだ場合と全く同じ順序で
    /// KVキャッシュを構築・参照する)。
    ///
    /// **数値的な同値性の根拠**: LayerNorm・Linear(GEMM)・GELU・残差加算は
    /// いずれも「各行(トークン位置)ごとに独立」した計算であり、バッチ
    /// (`seq_len`行まとめて1回のGEMM)で計算しても、行ごとに`seq_len`回
    /// `forward_step`を呼んだ場合と同じ内積・同じ累積順序になる
    /// (`sgemm`のCPU素朴実装は出力の各要素を独立に`sum_k`で計算するため、
    /// `m`が1でもNでも各行の計算結果は変わらない)。Attentionもキャッシュへの
    /// push順序を`forward_step`の逐次呼び出しと同じ昇順に保つことで、
    /// 同一の入力から同一の出力が得られる。よって本メソッドは
    /// 「挙動を変えない最適化」であり、`forward_step`をprompt長ぶん
    /// ループした場合とビット完全に一致する(open-cuda-llm側テスト
    /// `prefill_batch_matches_sequential_forward_step`で検証)。
    #[allow(clippy::too_many_arguments)]
    fn forward_prefill(
        &self,
        device: &dyn GpuDevice,
        hidden_batch: &[f32],
        seq_len: usize,
        caches: &mut [KvCacheHead],
        hidden_size: usize,
        num_heads: usize,
        softmax_spirv: Option<&[u8]>,
        flash_spirv: Option<(&[u8], usize)>,
    ) -> Result<Vec<f32>> {
        let head_dim = hidden_size / num_heads;
        debug_assert_eq!(hidden_batch.len(), seq_len * hidden_size);

        let mut normed = hidden_batch.to_vec();
        self.ln_1.forward(&mut normed, seq_len, hidden_size);

        // 1回のバッチGEMM(m=seq_len)でQ/K/Vをまとめて計算する。
        let qkv = self.qkv.forward(device, &normed, seq_len)?;

        let mut context = vec![0.0f32; seq_len * hidden_size];
        for row in 0..seq_len {
            let qkv_row = &qkv[row * 3 * hidden_size..(row + 1) * 3 * hidden_size];
            let q_row = &qkv_row[0..hidden_size];
            let k_row = &qkv_row[hidden_size..2 * hidden_size];
            let v_row = &qkv_row[2 * hidden_size..3 * hidden_size];

            for (h, cache_head) in caches.iter_mut().enumerate().take(num_heads) {
                let col_start = h * head_dim;
                let q_h = &q_row[col_start..col_start + head_dim];
                let k_h = &k_row[col_start..col_start + head_dim];
                let v_h = &v_row[col_start..col_start + head_dim];

                let spirv = self.qkv.spirv_matmul.as_deref().map(|v| v.as_slice());
                let proj = self.mla.as_ref().map(|v| &v[h]);
                cache_head.push(device, k_h, v_h, proj, spirv)?;
                let n = cache_head.n;
                let (k_all, v_all) = cache_head.current_kv(device, head_dim, proj, spirv)?;

                let mut q_full = vec![0.0f32; n * head_dim];
                for q_full_row in q_full.chunks_exact_mut(head_dim) {
                    q_full_row.copy_from_slice(q_h);
                }
                // 上記`forward_step`と同じ理由(2026-08-07新設)でflash_spirv
                // 優先の分岐にする。
                let out = if let Some((flash_bytes, block_size)) = flash_spirv {
                    opencuda_blas::flash_attention_with_spirv(device, &q_full, &k_all, &v_all, n, head_dim, block_size, flash_bytes)?
                } else {
                    opencuda_blas::scaled_dot_product_attention_with_spirv_and_softmax(
                        device,
                        &q_full,
                        &k_all,
                        &v_all,
                        n,
                        head_dim,
                        spirv,
                        softmax_spirv,
                    )?
                };
                context[row * hidden_size + col_start..row * hidden_size + col_start + head_dim].copy_from_slice(&out[0..head_dim]);
            }
        }

        // 1回のバッチGEMM(m=seq_len)。
        let attn_dense = self.attn_out.forward(device, &context, seq_len)?;
        let mut hidden2 = hidden_batch.to_vec();
        for (a, b) in hidden2.iter_mut().zip(attn_dense.iter()) {
            *a += b;
        }

        let mut normed2 = hidden2.clone();
        self.ln_2.forward(&mut normed2, seq_len, hidden_size);

        // 1回のバッチGEMM(m=seq_len)。
        let mut intermediate = self.intermediate.forward(device, &normed2, seq_len)?;
        gelu_inplace(&mut intermediate);

        // 1回のバッチGEMM(m=seq_len)。
        let ffn_out = self.output.forward(device, &intermediate, seq_len)?;
        let mut hidden3 = hidden2.clone();
        for (a, b) in hidden3.iter_mut().zip(ffn_out.iter()) {
            *a += b;
        }

        Ok(hidden3)
    }
}

/// **2026-08-07新設**: [`opencuda_blas::mla_compress_kv`]/`mla_decompress_kv`を
/// ヘッド単位で使うための射影行列一式(DeepSeek-V3のMLA構想と同じ
/// down-projection/up-projectionのペア)。[`GptModel::enable_mla_kv_compression`]
/// が乱数初期化する——実運用の学習済み重みではない点は`opencuda-blas`側の
/// 既存の開示と同じ(このクレートでの「配線」は、計算経路が正しく
/// KVキャッシュ経路まで繋がることの実証が目的)。
/// **2026-08-08新設**: [`GptModel::enable_mla_kv_compression_calibrated`]の
/// 核心部分。`rows_flat`(`num_rows x dim`の実活性化行列、行優先)の非中心
/// 二次モーメント行列`XᵀX`(`dim x dim`、対称半正定値)を`nalgebra`の対称
/// 固有値分解で解き、固有値降順で上位`d_c`個の固有ベクトルを`down_proj`
/// (`dim x d_c`)の列として並べる。直交基底なので`up_proj`(`d_c x dim`)は
/// 単純に転置(このモジュールdocコメントの制約(a)参照)。
fn pca_top_directions(rows_flat: &[f32], num_rows: usize, dim: usize, d_c: usize) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(rows_flat.len(), num_rows * dim);
    let x = nalgebra::DMatrix::from_row_slice(num_rows, dim, rows_flat);
    let cov = x.transpose() * &x; // dim x dim
    let eig = nalgebra::SymmetricEigen::new(cov);

    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[b].partial_cmp(&eig.eigenvalues[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut down_proj = vec![0f32; dim * d_c];
    for (col, &src_col) in order.iter().take(d_c).enumerate() {
        for row in 0..dim {
            down_proj[row * d_c + col] = eig.eigenvectors[(row, src_col)];
        }
    }
    let mut up_proj = vec![0f32; d_c * dim];
    for r in 0..d_c {
        for c in 0..dim {
            up_proj[r * dim + c] = down_proj[c * d_c + r];
        }
    }
    (down_proj, up_proj)
}

struct MlaHeadProjection {
    /// `head_dim x d_c`。
    down_proj: Vec<f32>,
    /// `d_c x head_dim`。
    up_proj: Vec<f32>,
    d_c: usize,
}

/// ヘッド単位のKVキャッシュ(`DecoderLayer::forward_step`内で使用)。
///
/// **2026-08-07変更**: [`MlaHeadProjection`]が渡された場合、フル精度の
/// `k`/`v`ではなく、低ランク射影された潜在表現(`k_latent`/`v_latent`、
/// `d_c`次元)のみを保持する(実際のメモリ削減の配線——
/// `opencuda-blas::mla_compress_kv`/`mla_decompress_kv`が単体の部品として
/// 存在するだけだった状態を、KVキャッシュの実経路まで繋いだ)。
/// `None`(既定)の場合は従来通りフル精度のまま`k`/`v`に積む
/// (後方互換、既存の数値一致テストへの影響ゼロ)。
struct KvCacheHead {
    k: Vec<f32>,
    v: Vec<f32>,
    k_latent: Vec<f32>,
    v_latent: Vec<f32>,
    n: usize,
}

impl KvCacheHead {
    fn empty() -> Self {
        Self { k: Vec::new(), v: Vec::new(), k_latent: Vec::new(), v_latent: Vec::new(), n: 0 }
    }

    /// 1トークンぶん(`head_dim`長)のk/vをキャッシュへ追加する。
    /// `proj`が`Some`なら、フル精度のまま保持せず`mla_compress_kv`で
    /// `d_c`次元へ圧縮してから保存する(`spirv`は`sgemm`のVulkan経路用、
    /// 既存の`Linear::forward`呼び出しと同じ規約)。
    fn push(&mut self, device: &dyn GpuDevice, k_row: &[f32], v_row: &[f32], proj: Option<&MlaHeadProjection>, spirv: Option<&[u8]>) -> Result<()> {
        match proj {
            Some(p) => {
                let head_dim = k_row.len();
                let k_lat = opencuda_blas::mla_compress_kv(device, 1, head_dim, p.d_c, k_row, &p.down_proj, spirv)?;
                let v_lat = opencuda_blas::mla_compress_kv(device, 1, head_dim, p.d_c, v_row, &p.down_proj, spirv)?;
                self.k_latent.extend_from_slice(&k_lat);
                self.v_latent.extend_from_slice(&v_lat);
            }
            None => {
                self.k.extend_from_slice(k_row);
                self.v.extend_from_slice(v_row);
            }
        }
        self.n += 1;
        Ok(())
    }

    /// これまでキャッシュした`n x head_dim`のk/vを返す。`proj`が`Some`の
    /// 場合は潜在表現から`mla_decompress_kv`で復元する(学習済み重みを
    /// 使わない限り元のk/vとは一致しない点は`opencuda-blas`側の開示通り、
    /// このクレートはあくまで「復元してAttentionへ渡す」経路の配線を担う)。
    fn current_kv(&self, device: &dyn GpuDevice, head_dim: usize, proj: Option<&MlaHeadProjection>, spirv: Option<&[u8]>) -> Result<(Vec<f32>, Vec<f32>)> {
        match proj {
            Some(p) => {
                let k = opencuda_blas::mla_decompress_kv(device, self.n, p.d_c, head_dim, &self.k_latent, &p.up_proj, spirv)?;
                let v = opencuda_blas::mla_decompress_kv(device, self.n, p.d_c, head_dim, &self.v_latent, &p.up_proj, spirv)?;
                Ok((k, v))
            }
            None => Ok((self.k.clone(), self.v.clone())),
        }
    }

    /// **2026-08-17新設(投機的デコード`GptModel::generate_speculative`
    /// 向け)**: キャッシュ長を`new_n`(`<= self.n`)へ巻き戻す。ドラフト
    /// モデルが提案した複数トークンをターゲットモデルで一括検証した後、
    /// 実際に採用された分だけを残して残りを捨てるために使う。
    /// **正直な開示・制約**: MLA低ランク圧縮(`k_latent`/`v_latent`が
    /// 非空)のキャッシュには未対応——`generate_speculative`側で
    /// MLA有効モデルを`ensure!`で拒否しているため、この関数が
    /// 実際にMLA圧縮キャッシュへ呼ばれることは無い前提。
    fn truncate(&mut self, new_n: usize, head_dim: usize) {
        debug_assert!(new_n <= self.n, "open-cuda-llm: KvCacheHead::truncate: new_n={new_n} must be <= current n={}", self.n);
        debug_assert!(self.k_latent.is_empty() && self.v_latent.is_empty(), "open-cuda-llm: KvCacheHead::truncate: MLA-compressed caches are not supported");
        self.k.truncate(new_n * head_dim);
        self.v.truncate(new_n * head_dim);
        self.n = new_n;
    }
}

/// GPT系デコーダ本体。`load_random`(現状唯一のコンストラクタ、学習済み
/// 重みローダーは未実装)で生成し、`generate`で貪欲デコードする。
pub struct GptModel {
    config: GptConfig,
    word_embeddings: Vec<f32>,
    position_embeddings: Vec<f32>,
    layers: Vec<DecoderLayer>,
    final_ln: LayerNorm,
    lm_head: Linear,
    /// コンパイル済み`softmax.spv`(`opencuda_blas::softmax_vulkan_generic`が
    /// 期待するのと同じシェーダバイト列)。2026-08-06新設、
    /// [`set_softmax_spirv`](Self::set_softmax_spirv)経由で配線される。
    /// `None`のまま(既定)なら、`spirv_matmul`が配線済みでAttentionの
    /// GEMM自体はVulkan経由になっていても、softmaxステップは従来通り
    /// ホスト側CPU(rayon並列)のまま(`scaled_dot_product_attention_
    /// with_spirv_and_softmax`の後方互換フォールバック規約通り)。
    softmax_spirv: Option<std::sync::Arc<Vec<u8>>>,
    /// コンパイル済み`flash_attention.spv`+ディスパッチ時の`block_size`
    /// (`opencuda_blas::flash_attention_with_spirv`が期待するのと同じ
    /// シェーダバイト列)。2026-08-07新設、
    /// [`set_flash_attention_spirv`](Self::set_flash_attention_spirv)経由で
    /// 配線される。`Some`の場合、Attention計算は`softmax_spirv`経由の
    /// 「GEMM+softmaxを別々にディスパッチ」する経路より優先され、QKᵀ・
    /// softmax・P·Vを1回のディスパッチで完結するfused flash attention
    /// カーネルを使う(レイヤーあたりのAttentionディスパッチ回数が3回から
    /// 1回へ削減される)。`None`のまま(既定)なら従来通り
    /// `softmax_spirv`配線の有無に応じた経路のまま(後方互換)。
    flash_attn_spirv: Option<(std::sync::Arc<Vec<u8>>, usize)>,
    /// DXILオフロード(2026-08-23新設)でデバイス常駐させた重みの所有者。
    /// がdropされるとここから全バッファが解放される。
    dxil_resident_weights: Option<ResidentWeights>,
}

impl GptModel {
    pub fn config(&self) -> &GptConfig {
        &self.config
    }

    /// 決定的な疑似乱数(`seed`)で初期化されたモデルを構築する
    /// (学習済み重みローダーは次の増分、モジュールdocコメント参照)。
    pub fn load_random(config: GptConfig, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let hidden = config.hidden_size;
        let word_embeddings = random_vec(&mut rng, config.vocab_size * hidden, 0.02);
        let position_embeddings = random_vec(&mut rng, config.max_seq_len * hidden, 0.02);
        let layers = (0..config.num_layers)
            .map(|_| DecoderLayer::random(&mut rng, hidden, config.intermediate_size, config.layer_norm_eps))
            .collect();
        let final_ln = LayerNorm::identity(hidden, config.layer_norm_eps);
        let lm_head = Linear::random(&mut rng, hidden, config.vocab_size);
        Self { config, word_embeddings, position_embeddings, layers, final_ln, lm_head, softmax_spirv: None, flash_attn_spirv: None, dxil_resident_weights: None }
    }

    /// `dir`配下の`config.json`・`model.safetensors`(Hugging Face GPT-2形式、
    /// 例: `openai-community/gpt2`)を読み込む。`open-cuda-bert::BertModel::load`
    /// と同じ設計(config.json→safetensorsの順で読み、レイヤーごとに
    /// テンソル名`h.{i}.*`を辿る)。
    ///
    /// アーキテクチャ上の注意点(`BertModel::load`との違い):
    /// - GPT-2は`Conv1D`層(`[in_dim, out_dim]`のまま保存、転置不要)を使う。
    ///   `nn.Linear`(`[out_dim, in_dim]`、転置が必要)のBERTとは逆。
    /// - Q/K/Vは`c_attn`という1本の融合`Conv1D`(`[hidden, 3*hidden]`)に
    ///   まとまっているため、列方向に3分割する(`load_fused_qkv`)。
    /// - `lm_head`はトークン埋め込み`wte.weight`と重み共有(weight tying)
    ///   されており、safetensors内に別テンソルとして存在しない
    ///   ——`wte.weight`(`[vocab, hidden]`)を転置して使う。
    pub fn load(dir: &Path) -> Result<Self> {
        let config_json = std::fs::read_to_string(dir.join("config.json"))
            .with_context(|| format!("open-cuda-llm: failed to read config.json in {dir:?}"))?;
        let raw_config: GPT2Config = serde_json::from_str(&config_json).context("open-cuda-llm: failed to parse config.json")?;
        let config = raw_config.into_gpt_config()?;

        let weights_bytes = std::fs::read(dir.join("model.safetensors"))
            .with_context(|| format!("open-cuda-llm: failed to read model.safetensors in {dir:?}"))?;
        let tensors = safetensors::SafeTensors::deserialize(&weights_bytes).context("open-cuda-llm: failed to parse model.safetensors")?;

        let hidden = config.hidden_size;

        // 2026-07-27追記(実E2E検証で発見した実バグの修正): Hugging Face上の
        // GPT-2互換モデルは、変換元スクリプトによってテンソル名に
        // `transformer.`プレフィックスが付く場合(例: distilgpt2の
        // `distilbert/distilgpt2`)と付かない場合(例: openai-communityの
        // `gpt2`本体)が実際に混在する——同じGPT-2アーキテクチャでも
        // 保存規約が統一されていない。プレフィックスの有無を`wte.weight`
        // の存在で自動判定し、以降の全テンソル名にこの`key_prefix`を
        // 前置することで両方を吸収する(モデルごとの個別分岐は増やさない)。
        let key_prefix = if tensors.tensor("wte.weight").is_ok() {
            ""
        } else if tensors.tensor("transformer.wte.weight").is_ok() {
            "transformer."
        } else {
            ""
        };

        let word_embeddings = tensor_f32(&tensors, &format!("{key_prefix}wte.weight"))?;
        anyhow::ensure!(
            word_embeddings.len() == config.vocab_size * hidden,
            "open-cuda-llm: '{key_prefix}wte.weight' has {} elements, expected {}x{}",
            word_embeddings.len(),
            config.vocab_size,
            hidden
        );
        let position_embeddings = tensor_f32(&tensors, &format!("{key_prefix}wpe.weight"))?;
        anyhow::ensure!(
            position_embeddings.len() == config.max_seq_len * hidden,
            "open-cuda-llm: '{key_prefix}wpe.weight' has {} elements, expected {}x{}",
            position_embeddings.len(),
            config.max_seq_len,
            hidden
        );

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("{key_prefix}h.{i}");
            let qkv = load_conv1d(&tensors, &format!("{p}.attn.c_attn"), hidden, 3 * hidden)?;
            layers.push(DecoderLayer {
                ln_1: load_layer_norm(&tensors, &format!("{p}.ln_1"), config.layer_norm_eps)?,
                qkv,
                attn_out: load_conv1d(&tensors, &format!("{p}.attn.c_proj"), hidden, hidden)?,
                ln_2: load_layer_norm(&tensors, &format!("{p}.ln_2"), config.layer_norm_eps)?,
                intermediate: load_conv1d(&tensors, &format!("{p}.mlp.c_fc"), hidden, config.intermediate_size)?,
                output: load_conv1d(&tensors, &format!("{p}.mlp.c_proj"), config.intermediate_size, hidden)?,
                mla: None,
            });
        }

        let final_ln = load_layer_norm(&tensors, &format!("{key_prefix}ln_f"), config.layer_norm_eps)?;
        let lm_head = Linear {
            weight_t: transpose(&word_embeddings, config.vocab_size, hidden),
            bias: vec![0.0; config.vocab_size], // GPT-2のlm_headはbias無し(weight tying)
            in_dim: hidden,
            out_dim: config.vocab_size,
            spirv_matmul: None,
            dxil_offload: None,
        };

        Ok(Self { config, word_embeddings, position_embeddings, layers, final_ln, lm_head, softmax_spirv: None, flash_attn_spirv: None, dxil_resident_weights: None })
    }

    /// コンパイル済み`matmul.spv`(`opencuda_blas::sgemm_vulkan_generic`が
    /// 期待するのと同じシェーダバイト列)をこのモデル内の全`Linear`
    /// (各レイヤーのQKV融合/attn_out/intermediate/output + `lm_head`)へ
    /// 配線する(2026-08-05新設)。呼び出し元(例: `aruaru-llm`の
    /// `--features real-vulkan`)が`opencuda-vulkan::real::VulkanDevice`を
    /// 使う場合はこれを呼んでから`generate`すること——呼ばなければ
    /// `sgemm`は`GemmPath::CpuNaive`(既存の既定挙動)のまま動く。
    ///
    /// **2026-08-05当時の開示(現在は解消済み)**: 当初はこの関数だけでは
    /// Attention計算(`opencuda_blas::scaled_dot_product_attention`が
    /// 内部で使う`launch_naive_gemm`経由のRustクロージャカーネル)側の
    /// 別のギャップ(`VulkanDevice::launch_kernel`が`KernelSource::SpirV`
    /// 以外を受け付けない)によりAttention自体がVulkanデバイス上で失敗
    /// していたが、`opencuda_blas::scaled_dot_product_attention_with_spirv`
    /// (2026-08-05、`open-cuda`側`CLAUDE.md`参照)の追加によりGEMM
    /// (QKᵀ・P·V)はVulkanディスパッチ可能になった。softmaxステップを
    /// GPU常駐にするには、別途[`set_softmax_spirv`](Self::set_softmax_spirv)
    /// を呼ぶこと(2026-08-06追加)。
    pub fn set_matmul_spirv(&mut self, spirv: Vec<u8>) {
        let spirv = std::sync::Arc::new(spirv);
        for layer in &mut self.layers {
            layer.qkv.spirv_matmul = Some(spirv.clone());
            layer.attn_out.spirv_matmul = Some(spirv.clone());
            layer.intermediate.spirv_matmul = Some(spirv.clone());
            layer.output.spirv_matmul = Some(spirv.clone());
        }
        self.lm_head.spirv_matmul = Some(spirv);
    }

    /// **2026-08-23新設**: 密GEMM(各レイヤーのQKV融合/attn_out/
    /// intermediate/output + `lm_head`)を、渡された**DXIL実行可能な
    /// デバイス**(`opencuda-directx::real::DirectXDevice`)へオフロード
    /// するよう配線する。`dxil`には`opencuda-directx/shaders/matmul.dxil`
    /// と同一契約(引数6個 a/b/c/m/k/n、`numthreads(8,8,1)`)の
    /// 事前コンパイル済みバイト列を渡すこと。
    ///
    /// **正直な開示**(誇張しないための明記):
    /// - オフロードされるのは上記の密GEMMのみ。Attention
    ///   (QKᵀ・softmax・P·V)・LayerNorm・GELU・埋め込み参照は
    ///   引き続き`generate`へ渡した`device`(通常は`CpuDevice`、
    ///   `opencuda-blas`経由で`open-cpu`のSIMDディスパッチが効く)で
    ///   実行される。DirectXDevice は`KernelSource::Dxil`しか実行できず、
    ///   これらが使うRustクロージャカーネルを実行できないため。
    /// - `matmul.hlsl`はタイリング等の最適化をしていないnaive実装であり、
    ///   小さい行列ではPCIe転送・ディスパッチのオーバーヘッドがCPUを
    ///   上回る可能性がある。速度が上がるかどうかは実測すること。
    /// - `set_matmul_spirv`と併用した場合、`forward`ではDXILオフロードが
    ///   優先される(両方を配線する構成は想定していない)。
    /// - 全`Linear`の重みをデバイスVRAMへ常駐させる(GPT-2 124Mでおよそ
    ///   0.5GB)。VRAMが足りない等でアップロードに失敗した場合は、
    ///   途中まで確保したバッファを解放した上で`Err`を返し、モデルは
    ///   **一切変更しない**(部分配線という中途半端な状態を作らない)。
    ///
    /// **実測(2026-08-23、開発機 NVIDIA GeForce GT 730 + AVX2 CPU)**:
    /// この経路はCPU(AVX2)より**遅かった**(GPT-2形状のGEMMで3〜30倍
    /// 遅い、`opencuda-blas/tests/sgemm_directx_bench.rs`参照)。
    /// naiveなmatmulシェーダーと非力なGPUの組み合わせが原因。より強力な
    /// 統合GPU/弱いCPUの組み合わせでは有利になり得るが、**この開発機では
    /// 高速化は確認できていない**。必ず実測してから有効化すること。
    pub fn set_matmul_dxil_offload(&mut self, device: std::sync::Arc<dyn GpuDevice>, dxil: Vec<u8>) -> Result<()> {
        let dxil = std::sync::Arc::new(dxil);
        let mut resident = ResidentWeights { device: device.clone(), ptrs: Vec::new() };

        // まず全重みをアップロードし、対応する`DevicePtr`を集める
        // (ここで失敗しても`resident`のdropが確保済み分を解放する)。
        let mut ptrs: Vec<opencuda_core::DevicePtr> = Vec::new();
        {
            let mut upload = |w: &[f32]| -> Result<()> {
                let p = opencuda_blas::upload_resident_matrix(&*device, w)?;
                resident.ptrs.push(p);
                ptrs.push(p);
                Ok(())
            };
            for layer in &self.layers {
                upload(&layer.qkv.weight_t)?;
                upload(&layer.attn_out.weight_t)?;
                upload(&layer.intermediate.weight_t)?;
                upload(&layer.output.weight_t)?;
            }
            upload(&self.lm_head.weight_t)?;
        }

        // すべて成功してから配線する。
        let mut it = ptrs.into_iter();
        let mk = |ptr: opencuda_core::DevicePtr| DxilMatmulOffload { device: device.clone(), dxil: dxil.clone(), b_ptr: ptr };
        for layer in &mut self.layers {
            layer.qkv.dxil_offload = Some(mk(it.next().expect("qkv ptr")));
            layer.attn_out.dxil_offload = Some(mk(it.next().expect("attn_out ptr")));
            layer.intermediate.dxil_offload = Some(mk(it.next().expect("intermediate ptr")));
            layer.output.dxil_offload = Some(mk(it.next().expect("output ptr")));
        }
        self.lm_head.dxil_offload = Some(mk(it.next().expect("lm_head ptr")));

        self.dxil_resident_weights = Some(resident);
        Ok(())
    }

    /// コンパイル済み`softmax.spv`(`opencuda_blas::softmax_vulkan_generic`
    /// が期待するのと同じシェーダバイト列)を配線する(2026-08-06新設)。
    /// [`set_matmul_spirv`](Self::set_matmul_spirv)と併用することで、
    /// Attention計算のQKᵀ・softmax・P·Vのすべてが実Vulkanデバイス上で
    /// ディスパッチされる(「GPU GEMM + CPU softmax」のハイブリッドから
    /// 「GPU GEMM + GPU softmax」への移行)。`set_matmul_spirv`を呼ばずに
    /// これだけ呼んでも、GEMM側が`GemmPath::VulkanGeneric`を選ばない
    /// 限りsoftmaxもCPUのままとなる(`opencuda_blas::scaled_dot_
    /// product_attention_with_spirv_and_softmax`の設計、GEMM経路と
    /// softmax経路を常に一致させる方針)。
    pub fn set_softmax_spirv(&mut self, spirv: Vec<u8>) {
        self.softmax_spirv = Some(std::sync::Arc::new(spirv));
    }

    /// **2026-08-07新設**: コンパイル済み`flash_attention.spv`
    /// (`opencuda_blas::flash_attention_with_spirv`が期待するのと同じ
    /// シェーダバイト列)を配線し、Attention計算をQKᵀ・softmax・P·Vが
    /// 1回のディスパッチで完結するfused flash attentionカーネルへ切り替える
    /// (open-cuda側2026-08-07 HANDOFF「次にすべきこと(1)」——
    /// `scaled_dot_product_attention_with_spirv_and_softmax`〈GEMM/softmax
    /// を別々にディスパッチ、レイヤーあたり3回〉から`flash_attention_with_
    /// spirv`〈1回〉への切り替え——への対応)。`set_matmul_spirv`/
    /// `set_softmax_spirv`と併用する必要はない(このモデルの内部Attention
    /// 経路では、`flash_attn_spirv`が`Some`の場合はそちらを優先し、
    /// `softmax_spirv`は無視する設計、他方Linear層のGEMM〈QKV/attn_out/
    /// intermediate/output〉は引き続き`set_matmul_spirv`で別途配線する
    /// 必要がある)。
    ///
    /// `block_size`は`opencuda_blas::flash_attention_with_spirv`の
    /// タイルサイズ引数(`0`はエラー、`head_dim`・`block_size`とも256を
    /// 超えるとシェーダの固定長ローカル配列の制約によりエラーになる、
    /// 詳細は`opencuda-blas`側のdocコメント参照)。
    pub fn set_flash_attention_spirv(&mut self, spirv: Vec<u8>, block_size: usize) {
        self.flash_attn_spirv = Some((std::sync::Arc::new(spirv), block_size));
    }

    /// **2026-08-07新設**: DeepSeek-V3のMLA(Multi-Head Latent Attention)に
    /// インスパイアされた低ランクKVキャッシュ圧縮(`opencuda_blas::
    /// mla_compress_kv`/`mla_decompress_kv`、2026-08-06追加)を、このモデルの
    /// 実際のKVキャッシュ経路(`forward_step`/`forward_prefill`が使う
    /// `KvCacheHead`)へ配線する(前回HANDOFF「次にすべきこと(1)」への対応、
    /// それまでは`opencuda-blas`単体の部品のまま呼び出し元が未接続だった)。
    ///
    /// `d_c`: ヘッドあたりの圧縮後次元(`head_dim`より小さい値を指定、
    /// `opencuda_blas::mla_memory_reduction_percent(head_dim, d_c)`で削減率を
    /// 事前に確認できる)。`seed`: 射影行列(`down_proj`/`up_proj`)の
    /// 決定的な乱数初期化シード。
    ///
    /// **正直な開示**: `opencuda-blas`側のdocコメント通り、射影行列は
    /// 学習済み重みではなく乱数初期化のため、圧縮・復元後のk/vは元の
    /// 値と一致しない(情報の一部が失われる)。この関数が担保するのは
    /// 「低ランク圧縮の計算経路が実際のKVキャッシュ・Attention計算まで
    /// 正しく配線され、`generate`がエンドツーエンドで動作する」ことで
    /// あり、生成品質の維持を主張するものではない——既に学習済みの
    /// モデル(`load`で読み込んだ実重み)にこれを適用すると出力の質が
    /// 劣化することが予想されるため、実運用では推奨しない
    /// (将来、学習済みのMLA射影重みを読み込めるようになった場合に
    /// 置き換えることを想定した土台)。
    ///
    /// `d_c >= head_dim`の場合はエラーを返す(圧縮になっていないため)。
    pub fn enable_mla_kv_compression(&mut self, d_c: usize, seed: u64) -> Result<()> {
        let head_dim = self.config.hidden_size / self.config.num_heads;
        anyhow::ensure!(d_c > 0 && d_c < head_dim, "open-cuda-llm: enable_mla_kv_compression: d_c={d_c} must satisfy 0 < d_c < head_dim={head_dim}");

        let mut rng = SplitMix64::new(seed);
        for layer in &mut self.layers {
            let projections = (0..self.config.num_heads)
                .map(|_| MlaHeadProjection {
                    down_proj: random_vec(&mut rng, head_dim * d_c, 0.02),
                    up_proj: random_vec(&mut rng, d_c * head_dim, 0.02),
                    d_c,
                })
                .collect();
            layer.mla = Some(projections);
        }
        Ok(())
    }

    /// **2026-08-08新設**: [`Self::enable_mla_kv_compression`]の乱数射影が
    /// 実測(このマシンのGT730、実GPT-2 124M重み)で生成品質を明確に劣化させた
    /// (反復・破綻した出力)ことを受けての対応。乱数射影の代わりに、
    /// **実際のサンプル文でプリフィルを走らせて集めた本物のK/V活性化統計に
    /// PCA(主成分分析、`kv`の非中心二次モーメント行列`kvᵀkv`の固有値分解)を
    /// 適用し、分散が最大の上位`d_c`個の方向を`down_proj`の基底として使う**。
    /// Johnson–Lindenstrauss型の乱数射影は次元`d_c`が大きい(理論上は
    /// 対象次元の対数のオーダー)場合にのみ距離をよく保存する保証があり、
    /// 本タスクのように`d_c=16`(`head_dim=64`からの75%圧縮)のような小さい
    /// 値では実データの分散構造を全く反映できない――これが乱数射影で品質が
    /// 崩壊した数学的な理由。PCAは逆に「実データの分散が集中する方向」を
    /// 直接見つけるため、この失敗モードに正面から対応する標準的な次元削減
    /// 手法(数十年来の線形代数、`nalgebra`の対称行列固有値分解を使用)。
    ///
    /// **設計上の制約(正直な開示)**: (a) 直交基底によるPCAのため
    /// `up_proj`は`down_proj`の転置に固定される(既存の`MlaHeadProjection`
    /// が`down_proj`/`up_proj`を独立フィールドとして持つ設計とは異なる
    /// 制約が入るが、直交射影の理論上は転置が最適な再構成行列であるため
    /// 問題にならない)。(b) 本実装はバイアス項(平均オフセット)を持たない
    /// ため、列平均を引く「中心化PCA」ではなく非中心(uncentered)PCAを使う
    /// ――`mla_compress_kv`/`mla_decompress_kv`が単純な行列積のみで平均加算
    /// を行わない既存契約に合わせた選択(中心化した場合、復元時に平均を
    /// 足し戻す処理が必要になり既存APIと非互換になるため)。(c)
    /// K活性化とV活性化は同じ`down_proj`/`up_proj`を共有する既存設計
    /// (`KvCacheHead::push`/`current_kv`参照)に合わせ、両方の活性化行列を
    /// 縦に連結してから単一のPCA基底を求める(K専用・V専用の別基底には
    /// していない)。(d) 較正に使う`sample_prompts`(すでにトークンID化済み、
    /// トークナイザはこのクレートの関知するところではないため呼び出し側で
    /// エンコード済みのものを渡す)がモデルの実運用時の入力分布を代表して
    /// いない場合、汎化しない可能性がある――較正プロンプトと全く異なる
    /// 話題・文体の入力では効果が薄れる、またはより悪化することもありうる
    /// (小サンプルでの過学習の一種)。呼び出し側は較正プロンプトと異なる
    /// held-outプロンプトで品質を確認すべき。
    ///
    /// 各ヘッドの較正データ行数が`d_c`未満の場合はエラーを返す
    /// (意味のあるPCA基底を作れないため、黙って劣化した基底を使わない)。
    pub fn enable_mla_kv_compression_calibrated(&mut self, d_c: usize, device: &dyn GpuDevice, sample_prompts: &[Vec<u32>]) -> Result<()> {
        let head_dim = self.config.hidden_size / self.config.num_heads;
        anyhow::ensure!(d_c > 0 && d_c < head_dim, "open-cuda-llm: enable_mla_kv_compression_calibrated: d_c={d_c} must satisfy 0 < d_c < head_dim={head_dim}");
        anyhow::ensure!(!sample_prompts.is_empty(), "open-cuda-llm: enable_mla_kv_compression_calibrated: sample_prompts must not be empty");
        anyhow::ensure!(
            self.layers.iter().all(|l| l.mla.is_none()),
            "open-cuda-llm: enable_mla_kv_compression_calibrated: some layers already have MLA compression enabled \
             (calibration must run against the uncompressed model, otherwise it would collect already-lossy latents \
             instead of real full-precision activations)"
        );

        let num_layers = self.config.num_layers;
        let num_heads = self.config.num_heads;
        let mut per_layer_head_rows: Vec<Vec<Vec<f32>>> = (0..num_layers).map(|_| vec![Vec::new(); num_heads]).collect();

        for prompt in sample_prompts {
            anyhow::ensure!(!prompt.is_empty(), "open-cuda-llm: enable_mla_kv_compression_calibrated: calibration prompt must not be empty");
            let mut caches = self.new_caches();
            self.forward_prefill_all_layers(device, prompt, 0, &mut caches)?;
            for (layer_idx, layer_caches) in caches.into_iter().enumerate() {
                for (head_idx, cache_head) in layer_caches.into_iter().enumerate() {
                    let bucket = &mut per_layer_head_rows[layer_idx][head_idx];
                    // proj=Noneでのプリフィルなので cache_head.k/.v はフル精度
                    // (KvCacheHead::pushのdocコメント参照)。K・V両方を縦に連結し、
                    // 同一のdown_proj/up_projで両方を扱う既存設計に合わせる。
                    bucket.extend_from_slice(&cache_head.k);
                    bucket.extend_from_slice(&cache_head.v);
                }
            }
        }

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let mut projections = Vec::with_capacity(num_heads);
            for (head_idx, rows_flat) in per_layer_head_rows[layer_idx].iter().enumerate() {
                let num_rows = rows_flat.len() / head_dim;
                anyhow::ensure!(
                    num_rows >= d_c,
                    "open-cuda-llm: enable_mla_kv_compression_calibrated: only {num_rows} calibration rows collected for \
                     layer {layer_idx} head {head_idx}, need >= d_c={d_c} for a meaningful PCA basis (pass longer/more sample prompts)"
                );
                let (down_proj, up_proj) = pca_top_directions(rows_flat, num_rows, head_dim, d_c);
                projections.push(MlaHeadProjection { down_proj, up_proj, d_c });
            }
            layer.mla = Some(projections);
        }
        Ok(())
    }

    /// 新規のKVキャッシュ集合(レイヤー数 x ヘッド数)を作る。
    fn new_caches(&self) -> Vec<Vec<KvCacheHead>> {
        (0..self.config.num_layers).map(|_| (0..self.config.num_heads).map(|_| KvCacheHead::empty()).collect()).collect()
    }

    /// 1トークンぶん進め、そのトークンの次を予測するロジット(語彙数長)を返す。
    fn forward_step(&self, device: &dyn GpuDevice, token_id: u32, pos: usize, caches: &mut [Vec<KvCacheHead>]) -> Result<Vec<f32>> {
        anyhow::ensure!(pos < self.config.max_seq_len, "open-cuda-llm: position {pos} exceeds max_seq_len {}", self.config.max_seq_len);
        let hidden_size = self.config.hidden_size;
        let tok = token_id as usize;
        anyhow::ensure!(tok < self.config.vocab_size, "open-cuda-llm: token id {tok} out of vocab range");

        let word_row = &self.word_embeddings[tok * hidden_size..(tok + 1) * hidden_size];
        let pos_row = &self.position_embeddings[pos * hidden_size..(pos + 1) * hidden_size];
        let mut hidden: Vec<f32> = word_row.iter().zip(pos_row.iter()).map(|(w, p)| w + p).collect();

        let softmax_spirv = self.softmax_spirv.as_deref().map(|v| v.as_slice());
        let flash_spirv = self.flash_attn_spirv.as_ref().map(|(bytes, bs)| (bytes.as_slice(), *bs));
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            hidden = layer.forward_step(device, &hidden, cache, hidden_size, self.config.num_heads, softmax_spirv, flash_spirv)?;
        }

        self.final_ln.forward(&mut hidden, 1, hidden_size);
        self.lm_head.forward(device, &hidden, 1)
    }

    /// `forward_prefill_all_layers`/`forward_prefill_all_layers_per_position`
    /// 共通のembedding+レイヤー通過部分。`start_pos`は`token_ids[0]`が
    /// 置かれる絶対位置(既存キャッシュに続けて追記する場合に使う、
    /// 2026-08-17新設——投機的デコードの検証バッチが、プロンプト直後
    /// ではなく既存キャッシュの続きから始まる位置埋め込みを正しく
    /// 計算するために必要になった)。プロンプトの初回呼び出しは
    /// `start_pos=0`を渡せば従来と完全に同じ挙動になる(後方互換)。
    fn forward_prefill_hidden(&self, device: &dyn GpuDevice, token_ids: &[u32], start_pos: usize, caches: &mut [Vec<KvCacheHead>]) -> Result<Vec<f32>> {
        let hidden_size = self.config.hidden_size;
        let seq_len = token_ids.len();

        let mut hidden_batch = vec![0.0f32; seq_len * hidden_size];
        for (row, &tok) in token_ids.iter().enumerate() {
            let pos = start_pos + row;
            anyhow::ensure!(pos < self.config.max_seq_len, "open-cuda-llm: position {pos} exceeds max_seq_len {}", self.config.max_seq_len);
            let tok = tok as usize;
            anyhow::ensure!(tok < self.config.vocab_size, "open-cuda-llm: token id {tok} out of vocab range");
            let word_row = &self.word_embeddings[tok * hidden_size..(tok + 1) * hidden_size];
            let pos_row = &self.position_embeddings[pos * hidden_size..(pos + 1) * hidden_size];
            let dst = &mut hidden_batch[row * hidden_size..(row + 1) * hidden_size];
            for (d, (w, p)) in dst.iter_mut().zip(word_row.iter().zip(pos_row.iter())) {
                *d = w + p;
            }
        }

        let softmax_spirv = self.softmax_spirv.as_deref().map(|v| v.as_slice());
        let flash_spirv = self.flash_attn_spirv.as_ref().map(|(bytes, bs)| (bytes.as_slice(), *bs));
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            hidden_batch = layer.forward_prefill(device, &hidden_batch, seq_len, cache, hidden_size, self.config.num_heads, softmax_spirv, flash_spirv)?;
        }
        Ok(hidden_batch)
    }

    /// **2026-08-04新設**: プロンプト全体(`token_ids`)を`DecoderLayer::
    /// forward_prefill`でバッチ処理し、最終位置のロジットを返す
    /// (プリフィル/デコード分離、上記`forward_prefill`のdocコメント参照)。
    /// **2026-08-17変更**: `start_pos`引数を追加(`forward_prefill_hidden`
    /// 参照)——既存呼び出し元(プロンプトの初回prefill)は`start_pos=0`を
    /// 渡すことで挙動は完全に無変更。
    fn forward_prefill_all_layers(&self, device: &dyn GpuDevice, token_ids: &[u32], start_pos: usize, caches: &mut [Vec<KvCacheHead>]) -> Result<Vec<f32>> {
        let hidden_size = self.config.hidden_size;
        let seq_len = token_ids.len();
        let hidden_batch = self.forward_prefill_hidden(device, token_ids, start_pos, caches)?;

        // 最終位置のみLayerNorm+lm_headを適用すれば十分(`generate`が
        // 必要とするのは次トークン予測用のロジットのみのため、他の行を
        // 正規化する計算は省く)。
        let last_row_start = (seq_len - 1) * hidden_size;
        let mut last_hidden = hidden_batch[last_row_start..last_row_start + hidden_size].to_vec();
        self.final_ln.forward(&mut last_hidden, 1, hidden_size);
        self.lm_head.forward(device, &last_hidden, 1)
    }

    /// **2026-08-17新設(投機的デコード`generate_speculative`向け)**:
    /// `forward_prefill_all_layers`と同じバッチprefillを行うが、最終位置
    /// だけでなく**全位置**のロジットを返す(`seq_len * vocab_size`の
    /// フラット配列、行`i`が位置`start_pos+i`の次トークン予測に対応)。
    /// 投機的デコードの検証ステップでは、ドラフトモデルが提案した複数
    /// トークンそれぞれについてターゲットモデルの貪欲選択を知る必要が
    /// あるため、最終位置だけでは足りない。
    fn forward_prefill_all_layers_per_position(&self, device: &dyn GpuDevice, token_ids: &[u32], start_pos: usize, caches: &mut [Vec<KvCacheHead>]) -> Result<Vec<f32>> {
        let hidden_size = self.config.hidden_size;
        let seq_len = token_ids.len();
        let mut hidden_batch = self.forward_prefill_hidden(device, token_ids, start_pos, caches)?;
        self.final_ln.forward(&mut hidden_batch, seq_len, hidden_size);
        self.lm_head.forward(device, &hidden_batch, seq_len)
    }

    /// 貪欲デコード(argmax、サンプリング温度無し)で`max_new_tokens`個
    /// トークンを生成する。`prompt_ids`自体は出力に含めない
    /// (呼び出し側で連結すること)。繰り返しペナルティ無し
    /// (`repetition_penalty=1.0`)で`generate_with_repetition_penalty`を
    /// 呼ぶ薄いラッパー——既存呼び出し元の挙動は完全に無変更(後方互換)。
    ///
    /// **2026-08-04変更(プリフィル/デコード分離)**: プロンプトの初回
    /// forwardは`forward_prefill_all_layers`(バッチGEMM)で処理し、
    /// 生成された各トークンの逐次デコードは従来通り`forward_step`
    /// (`seq_len=1`)のままとする(`aruaru-llm`側CLAUDE.md 2026-07-26
    /// HANDOFFで指摘された設計変更(a))。
    pub fn generate(&self, device: &std::sync::Arc<dyn GpuDevice>, prompt_ids: &[u32], max_new_tokens: usize) -> Result<Vec<u32>> {
        self.generate_with_repetition_penalty(device, prompt_ids, max_new_tokens, 1.0)
    }

    /// `generate`と同じ貪欲デコードだが、既に登場したトークン(プロンプト+
    /// これまでに生成済みのトークン、両方を対象)のlogitへ繰り返しペナルティを
    /// 適用する(CTRL論文〈Keskar et al. 2019〉のrepetition penalty方式:
    /// logitが正なら`/penalty`、負なら`*penalty`——`penalty>1.0`で
    /// そのトークンが再び選ばれにくくなる)。
    ///
    /// `aruaru-llm`側ユーザー報告「しつこく繰り返すバグ」(対話ファイン
    /// チューニング無しの素のGPT-2貪欲デコードが同一文字列の無限ループに
    /// 陥る、既知のGPT-2系の劣化モード)への根本対応。`repetition_penalty
    /// =1.0`を渡すと`generate`と完全に同一の挙動になる(早期returnで
    /// 一切のペナルティ処理を行わない、既存テストとの数値一致を保証)。
    pub fn generate_with_repetition_penalty(
        &self,
        device: &std::sync::Arc<dyn GpuDevice>,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        repetition_penalty: f32,
    ) -> Result<Vec<u32>> {
        anyhow::ensure!(!prompt_ids.is_empty(), "open-cuda-llm: prompt_ids must not be empty");
        let device_ref = device.as_ref();
        let mut caches = self.new_caches();

        let mut logits = self.forward_prefill_all_layers(device_ref, prompt_ids, 0, &mut caches)?;
        let pos = prompt_ids.len();

        let mut seen: std::collections::HashSet<u32> = prompt_ids.iter().copied().collect();
        apply_repetition_penalty(&mut logits, &seen, repetition_penalty);

        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut next = argmax(&logits);
        for pos_now in (pos..).take(max_new_tokens) {
            generated.push(next);
            seen.insert(next);
            if pos_now >= self.config.max_seq_len {
                break;
            }
            logits = self.forward_step(device_ref, next, pos_now, &mut caches)?;
            apply_repetition_penalty(&mut logits, &seen, repetition_penalty);
            next = argmax(&logits);
        }
        Ok(generated)
    }

    /// **2026-08-17新設**: 全レイヤー・全ヘッドのKVキャッシュを`new_n`
    /// トークン分へ巻き戻す(`KvCacheHead::truncate`参照)。
    fn truncate_caches(&self, caches: &mut [Vec<KvCacheHead>], new_n: usize) {
        let head_dim = self.config.hidden_size / self.config.num_heads;
        for layer_caches in caches.iter_mut() {
            for head_cache in layer_caches.iter_mut() {
                head_cache.truncate(new_n, head_dim);
            }
        }
    }

    /// **2026-08-17新設**: DeepSeekの「DSpark」(ロスレス投機的デコード、
    /// 2026-06-27公開・MITライセンス)・および学術的には
    /// Leviathan et al. 2023 "Fast Inference from Transformers via
    /// Speculative Decoding"に遡る手法を、貪欲デコード(このクレートが
    /// 唯一対応するデコード方式)向けに実装したもの。ユーザー承認
    /// (週次リサーチルーティンでのDSpark/llama.cpp Multi-Token
    /// Prediction調査結果への2026-08-17 YES回答)を受けて実装した。
    ///
    /// ## 目的(`aruaru-llm`側の既知のボトルネックへの対応)
    ///
    /// `aruaru-llm`のCLAUDE.mdは、GT730のような低性能GPUでは「1トークン
    /// デコードのGEMMが極めて軽く、Vulkanディスパッチの固定オーバー
    /// ヘッドが支配的になる」ことを複数回実測してきた。本関数は、軽量な
    /// `draft`モデル(例: `distilgpt2`)に複数トークンを先に提案させ、
    /// 本命の`self`(ターゲット、例: `gpt2-medium`)モデルは**1回の
    /// バッチprefillで複数トークンをまとめて検証**することで、ターゲット
    /// モデル側のディスパッチ回数を「採用トークン数」ではなく「ラウンド数」
    /// のオーダーへ削減する——ディスパッチ固定オーバーヘッドが支配的な
    /// 環境ほど効果が大きいはずだが、この初回実装では実機ベンチマークは
    /// 未実施(下記「正直な開示」参照、`aruaru-llm`側配線後に計測予定)。
    ///
    /// ## ロスレス性の根拠(貪欲デコード限定、誇張しない範囲での説明)
    ///
    /// ラウンドごとに、ドラフトモデルが提案した`x_0..x_{k-1}`のうち、
    /// ターゲットモデルが同じ位置で貪欲に選んだであろうトークンと一致
    /// する先頭`m`個(`0 <= m <= k`)をそのまま採用し、最初に食い違った
    /// 位置(または全て一致した場合はその直後の位置)ではターゲット自身の
    /// 貪欲選択を採用する。帰納的に、この手続きで生成される系列は
    /// `self.generate()`(ターゲット単体の貪欲デコード)と1トークン単位で
    /// ビット完全に一致する——ドラフトモデルの品質は出力の正しさには
    /// 一切影響せず、採用率(高速化率)にのみ影響する。この性質自体は
    /// テスト`generate_speculative_matches_plain_greedy_decode`で
    /// 数値的に検証している。
    ///
    /// ## 正直な開示・制約
    ///
    /// - サンプリング(温度・top-k/top-p)は未対応(このクレート全体が
    ///   貪欲デコードのみ対応のため、既存の`generate`と同じ制約)。
    /// - `self`(ターゲット)・`draft`は同じ語彙(トークナイザ・
    ///   `vocab_size`)を共有する前提——実運用では同じGPT-2ファミリー内の
    ///   異なるサイズ(例: ターゲット`gpt2-medium`+ドラフト`distilgpt2`)を
    ///   想定する。異なる場合はエラーを返す。
    /// - MLA低ランクKVキャッシュ圧縮(`enable_mla_kv_compression*`)が
    ///   有効なモデルは未対応(`KvCacheHead::truncate`が非圧縮キャッシュ
    ///   のみ対応のため)——該当する場合はエラーを返す。
    /// - 繰り返しペナルティ(`generate_with_repetition_penalty`)は未統合。
    /// - **速度面の実測結果(2026-08-17、CPU実行、誇張しない)**:
    ///   実重み(ターゲット`gpt2`124M・ドラフト`distilgpt2`82M、
    ///   `draft_k=4`・`max_new_tokens=16`)で計測したところ、採用率
    ///   80%(12/15)と高かったにもかかわらず、**素の`generate()`より
    ///   実際には遅かった**(plain=4.63秒 vs speculative=7.65秒、テスト
    ///   `real_gpt2_speculative_decoding_matches_plain_greedy_and_reports_
    ///   acceptance`の`--nocapture`実測)。CPU素朴GEMM実装では
    ///   ディスパッチ固定オーバーヘッドという「削減すべきコスト」自体が
    ///   ほぼ存在しないため、(a)ドラフトモデルの計算コスト、(b)検証
    ///   ラウンドごとにドラフト側のKVキャッシュを毎回truncate+再構築する
    ///   本実装のコスト、が純増分になってしまい逆効果だった。**本命の
    ///   対象は`aruaru-llm`が繰り返し記録してきたVulkanディスパッチ
    ///   オーバーヘッド支配的な環境(`--features real-vulkan`)であり、
    ///   その環境での速度検証は未実施**(次の増分、下記「次にすべき
    ///   こと」参照)——CPU実行での本結果だけを見て「投機的デコードは
    ///   有効」と主張することはしない。
    #[allow(clippy::too_many_arguments)]
    pub fn generate_speculative(
        &self,
        device: &std::sync::Arc<dyn GpuDevice>,
        draft: &GptModel,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        draft_k: usize,
    ) -> Result<(Vec<u32>, SpeculativeStats)> {
        anyhow::ensure!(!prompt_ids.is_empty(), "open-cuda-llm: generate_speculative: prompt_ids must not be empty");
        anyhow::ensure!(draft_k >= 1, "open-cuda-llm: generate_speculative: draft_k must be >= 1");
        anyhow::ensure!(
            self.config.vocab_size == draft.config.vocab_size,
            "open-cuda-llm: generate_speculative: target vocab_size={} != draft vocab_size={}",
            self.config.vocab_size,
            draft.config.vocab_size
        );
        anyhow::ensure!(
            self.layers.iter().all(|l| l.mla.is_none()) && draft.layers.iter().all(|l| l.mla.is_none()),
            "open-cuda-llm: generate_speculative: MLA-compressed models are not supported yet (KvCacheHead::truncate only handles full-precision caches)"
        );
        if max_new_tokens == 0 {
            return Ok((Vec::new(), SpeculativeStats::default()));
        }

        let device_ref = device.as_ref();
        let vocab = self.config.vocab_size;

        let mut target_caches = self.new_caches();
        let mut draft_caches = draft.new_caches();

        // プロンプトをそれぞれ独立にプリフィル(既存`generate`と同じ、
        // 挙動を変えない範囲での起点)。
        let target_prompt_logits = self.forward_prefill_all_layers(device_ref, prompt_ids, 0, &mut target_caches)?;
        let draft_prompt_logits = draft.forward_prefill_all_layers(device_ref, prompt_ids, 0, &mut draft_caches)?;

        let mut pos = prompt_ids.len();
        // t_0: プロンプト直後にターゲットモデルが貪欲に選ぶトークン
        // (ラウンド0のドラフト提案x_0と比較する基準)。
        let mut pending_target_pick = argmax(&target_prompt_logits);
        // ドラフトモデルの最初の提案x_0。
        let mut pending_draft_pick = argmax(&draft_prompt_logits);

        let mut output = Vec::with_capacity(max_new_tokens);
        let mut stats = SpeculativeStats::default();

        while output.len() < max_new_tokens {
            let remaining = max_new_tokens - output.len();
            let k = draft_k.min(remaining).min(self.config.max_seq_len.saturating_sub(pos)).max(1);

            // 1) ドラフトモデルでx_0..x_{k-1}を自己回帰的に提案する
            //    (ドラフト自身のKVキャッシュへ逐次push、この成長分は
            //    検証後に使い切り、下記4番で正確に作り直す)。
            let mut draft_tokens = Vec::with_capacity(k);
            let mut next_draft_input = pending_draft_pick;
            for i in 0..k {
                draft_tokens.push(next_draft_input);
                if i + 1 < k {
                    let logits = draft.forward_step(device_ref, next_draft_input, pos + i, &mut draft_caches)?;
                    next_draft_input = argmax(&logits);
                }
            }

            // 2) ターゲットモデルで一括検証(投機的デコードの高速化の核心
            //    ——k個のトークンをk回ではなく1回のバッチprefillで処理)。
            let verify_logits = self.forward_prefill_all_layers_per_position(device_ref, &draft_tokens, pos, &mut target_caches)?;

            // target_predictions[i] = ターゲットが位置pos+iで貪欲に選ぶ
            // トークン(i=0はプロンプト/前ラウンド由来のpending_target_pick、
            // i=1..kはverify_logitsの各行から)。
            let mut target_predictions = Vec::with_capacity(k + 1);
            target_predictions.push(pending_target_pick);
            for j in 0..k {
                target_predictions.push(argmax(&verify_logits[j * vocab..(j + 1) * vocab]));
            }

            stats.proposed += k;
            let mut m = 0usize;
            while m < k && draft_tokens[m] == target_predictions[m] {
                m += 1;
            }
            stats.accepted += m;
            let correction = target_predictions[m];

            // 3) 採用分をoutputへ反映(max_new_tokensの上限を超えない範囲)。
            for &tok in &draft_tokens[0..m] {
                if output.len() >= max_new_tokens {
                    break;
                }
                output.push(tok);
            }
            if output.len() < max_new_tokens {
                output.push(correction);
            }

            // 4) 次ラウンドに向けてターゲット・ドラフト双方のKVキャッシュを
            //    「実際に採用された系列(採用分+補正トークン)」だけに
            //    揃え直す(3番のmax_new_tokens上限による打ち切りとは独立に、
            //    次ラウンドを回さないループ終了時はこの後処理ごと省略しても
            //    正しさに影響しないため、ループ条件で自然に止まる)。
            self.truncate_caches(&mut target_caches, pos + m);
            let next_target_logits = self.forward_step(device_ref, correction, pos + m, &mut target_caches)?;
            pending_target_pick = argmax(&next_target_logits);

            draft.truncate_caches(&mut draft_caches, pos);
            let mut dpos = pos;
            let mut last_draft_logits = None;
            for &tok in draft_tokens[0..m].iter().chain(std::iter::once(&correction)) {
                last_draft_logits = Some(draft.forward_step(device_ref, tok, dpos, &mut draft_caches)?);
                dpos += 1;
            }
            pending_draft_pick = argmax(&last_draft_logits.expect("committed list always has at least the correction token"));

            pos += m + 1;
        }

        output.truncate(max_new_tokens);
        Ok((output, stats))
    }
}

/// [`GptModel::generate_speculative`]の1呼び出し全体を通じた、ドラフト
/// モデルの提案採用状況(高速化率の目安)。
#[derive(Debug, Clone, Copy, Default)]
pub struct SpeculativeStats {
    /// ドラフトモデルが提案した(検証対象になった)トークンの総数。
    pub proposed: usize,
    /// そのうちターゲットモデルの貪欲選択と一致し実際に採用された数。
    pub accepted: usize,
}

impl SpeculativeStats {
    /// 採用率(0.0〜1.0)。`proposed==0`なら`0.0`。
    pub fn acceptance_rate(&self) -> f32 {
        if self.proposed == 0 {
            0.0
        } else {
            self.accepted as f32 / self.proposed as f32
        }
    }
}

/// `logits`のうち`seen`に含まれるトークンIDへ繰り返しペナルティを適用する。
/// `penalty == 1.0`(既定・ペナルティ無し)の場合は`logits`を一切変更せず
/// 即座に返る——`generate`(既存呼び出し元)の数値的な挙動を完全に保つ。
fn apply_repetition_penalty(logits: &mut [f32], seen: &std::collections::HashSet<u32>, penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    for &tok in seen {
        if let Some(logit) = logits.get_mut(tok as usize) {
            *logit = if *logit > 0.0 { *logit / penalty } else { *logit * penalty };
        }
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

/// UTF-8バイト単位の素朴なトークナイザ(語彙: 0..256がバイト値そのもの、
/// 256以降を特殊トークン用に予約)。本格的なBPE/SentencePieceの代わりに、
/// 追加のモデルファイルを一切必要としない自己完結な構成にするための
/// 設計判断(モジュールdocコメント参照)。
pub struct ByteTokenizer;

impl ByteTokenizer {
    pub const VOCAB_SIZE: usize = 256 + 4;
    pub const BOS: u32 = 256;
    pub const EOS: u32 = 257;

    pub fn encode(text: &str) -> Vec<u32> {
        text.bytes().map(|b| b as u32).collect()
    }

    /// 特殊トークン(256以上)は無視して復号する(不正なUTF-8境界は
    /// `String::from_utf8_lossy`で置換文字化)。
    pub fn decode(ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().filter(|&&id| id < 256).map(|&id| id as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// GPT-2の`tokenizer.json`(Hugging Face fast tokenizer形式、バイトレベルBPE)を
/// ロードする薄いラッパー(`open-cuda-bert::BertTokenizer`と同じ設計)。
/// `GptModel::load`で読み込んだ実重みは、GPT-2自身のBPE語彙で学習された
/// ものなので、意味のある出力を得るには`ByteTokenizer`(語彙0..256=生バイト)
/// ではなく本トークナイザを使う必要がある——**正直な開示**: `ByteTokenizer`の
/// トークンIDはGPT-2のBPE語彙IDとは無関係(たまたま小さい整数という以外
/// 対応関係が無い)ため、実重み+`ByteTokenizer`の組み合わせでは意味のある
/// 生成は期待できない。実重みを試す場合は必ず本`GptTokenizer`を使うこと。
pub struct GptTokenizer {
    inner: tokenizers::Tokenizer,
}

impl GptTokenizer {
    pub fn load(dir: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("open-cuda-llm: failed to load tokenizer.json: {e}"))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(text, false).map_err(|e| anyhow::anyhow!("open-cuda-llm: tokenizer encode failed: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner.decode(ids, true).map_err(|e| anyhow::anyhow!("open-cuda-llm: tokenizer decode failed: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencuda_cpu::CpuDevice;
    use std::sync::Arc;

    fn device() -> Arc<dyn GpuDevice> {
        CpuDevice::new(0)
    }

    #[test]
    fn generates_requested_number_of_tokens_without_panicking() {
        let config = GptConfig::tiny(ByteTokenizer::VOCAB_SIZE);
        let model = GptModel::load_random(config, 42);
        let device = device();
        let prompt = ByteTokenizer::encode("hi");
        let generated = model.generate(&device, &prompt, 8).unwrap();
        assert_eq!(generated.len(), 8);
        // decodeがpanicしないこと(語彙範囲外を混ぜても壊れない)も確認。
        let _ = ByteTokenizer::decode(&generated);
    }

    /// **2026-08-17新設**: `generate_speculative`の核心的な正しさの検証
    /// ——ドラフトモデルにターゲットと同一のモデル(同一シード)を使うと、
    /// ドラフトの提案は常にターゲットの貪欲選択と一致する(採用率100%)
    /// はずであり、出力は`generate()`(ターゲット単体の貪欲デコード)と
    /// 完全に一致するはず。
    #[test]
    fn generate_speculative_with_identical_draft_matches_plain_greedy_decode_with_full_acceptance() {
        let config = GptConfig::tiny(ByteTokenizer::VOCAB_SIZE);
        let target = GptModel::load_random(config.clone(), 7);
        let draft = GptModel::load_random(config, 7); // 同一シード=同一重み
        let device = device();
        let prompt = ByteTokenizer::encode("speculative decoding should be lossless");

        let plain = target.generate(&device, &prompt, 12).unwrap();
        let (speculative, stats) = target.generate_speculative(&device, &draft, &prompt, 12, 4).unwrap();

        assert_eq!(speculative, plain);
        assert_eq!(stats.proposed, stats.accepted, "identical draft/target should have 100% acceptance");
    }

    /// **2026-08-17新設**: ドラフトとターゲットが異なるモデル(異なる
    /// シード=異なる重み)の場合、提案は一部しか一致しないはずだが、
    /// それでも出力は`generate()`とビット完全に一致し続けなければ
    /// ならない(ロスレス性の本質的な検証、DSpark/Leviathan et al.の
    /// 手法が保証する性質)。
    #[test]
    fn generate_speculative_with_different_draft_still_matches_plain_greedy_decode() {
        let config = GptConfig::tiny(ByteTokenizer::VOCAB_SIZE);
        let target = GptModel::load_random(config.clone(), 100);
        let draft = GptModel::load_random(config, 999); // 異なるシード=異なる重み
        let device = device();
        let prompt = ByteTokenizer::encode("a different draft model still has to be lossless");

        let plain = target.generate(&device, &prompt, 16).unwrap();
        let (speculative, stats) = target.generate_speculative(&device, &draft, &prompt, 16, 3).unwrap();

        assert_eq!(speculative, plain, "output must be byte-identical to plain greedy decode regardless of draft quality");
        assert!(stats.proposed > 0, "expected at least one speculative round to have proposed tokens");
    }

    /// 語彙サイズが異なるモデル同士は拒否されることを確認。
    #[test]
    fn generate_speculative_rejects_mismatched_vocab_size() {
        let device = device();
        let target = GptModel::load_random(GptConfig::tiny(300), 1);
        let draft = GptModel::load_random(GptConfig::tiny(200), 1);
        let prompt = vec![1u32, 2, 3];
        assert!(target.generate_speculative(&device, &draft, &prompt, 4, 2).is_err());
    }

    /// MLA圧縮を有効化したモデルはまだ未対応であることを確認
    /// (`KvCacheHead::truncate`が非圧縮キャッシュのみ対応のため)。
    #[test]
    fn generate_speculative_rejects_mla_compressed_target_or_draft() {
        let device = device();
        let config = GptConfig::tiny(ByteTokenizer::VOCAB_SIZE);
        let mut target = GptModel::load_random(config.clone(), 1);
        target.enable_mla_kv_compression(2, 1).unwrap();
        let draft = GptModel::load_random(config, 1);
        let prompt = ByteTokenizer::encode("hi");
        assert!(target.generate_speculative(&device, &draft, &prompt, 4, 2).is_err());
    }

    /// **2026-08-07新設**: `enable_mla_kv_compression`をKVキャッシュ経路
    /// (`forward_step`/`forward_prefill`の両方、プリフィル+逐次デコード)へ
    /// 実際に配線したことのE2E検証。前回HANDOFF「次にすべきこと(1)」への
    /// 対応(それまでは`opencuda-blas::mla_compress_kv`/`mla_decompress_kv`が
    /// 単体の部品として実装されているだけで、呼び出し元が未接続だった)。
    #[test]
    fn mla_kv_compression_enabled_model_generates_without_panicking() {
        let config = GptConfig::tiny(ByteTokenizer::VOCAB_SIZE); // hidden=32, num_heads=4 => head_dim=8
        let mut model = GptModel::load_random(config, 123);
        model.enable_mla_kv_compression(2, 999).unwrap(); // head_dim=8 -> d_c=2 (75%削減)
        let device = device();
        let prompt = ByteTokenizer::encode("mla compression path");
        let generated = model.generate(&device, &prompt, 6).unwrap();
        assert_eq!(generated.len(), 6);
    }

    /// `d_c >= head_dim`は圧縮になっていないため拒否されることを確認。
    #[test]
    fn mla_kv_compression_rejects_non_reducing_d_c() {
        let config = GptConfig::tiny(ByteTokenizer::VOCAB_SIZE); // head_dim=8
        let mut model = GptModel::load_random(config, 1);
        assert!(model.enable_mla_kv_compression(8, 1).is_err());
        assert!(model.enable_mla_kv_compression(0, 1).is_err());
    }

    /// KVキャッシュ圧縮を有効にしたモデルと無効なモデルとで、同一プロンプト
    /// に対する生成結果が(射影が乱数のため情報が失われ)実際に異なりうる
    /// ことを確認する回帰テスト——「配線したが実は何も変わっていない
    /// (常に同じ経路に落ちる)」という見逃しを防ぐ。乱数の組み合わせ次第で
    /// 稀に一致する可能性はゼロではないため、複数シードで試して少なくとも
    /// 1つは異なることを確認する(フレーキーさを避ける設計)。
    #[test]
    fn mla_kv_compression_actually_changes_generation_versus_uncompressed() {
        let prompt = ByteTokenizer::encode("does compression change output");
        let device = device();
        let mut any_different = false;
        for seed in [1u64, 2, 3, 4, 5] {
            let config = GptConfig::tiny(ByteTokenizer::VOCAB_SIZE);
            let baseline = GptModel::load_random(config.clone(), seed);
            let out_baseline = baseline.generate(&device, &prompt, 6).unwrap();

            let mut compressed = GptModel::load_random(config, seed);
            compressed.enable_mla_kv_compression(2, seed).unwrap();
            let out_compressed = compressed.generate(&device, &prompt, 6).unwrap();

            if out_baseline != out_compressed {
                any_different = true;
                break;
            }
        }
        assert!(any_different, "expected mla-compressed KV cache path to actually be exercised (output should differ from uncompressed baseline for at least one seed)");
    }

    #[test]
    fn same_seed_and_prompt_produce_identical_output_deterministically() {
        let config = GptConfig::tiny(ByteTokenizer::VOCAB_SIZE);
        let device = device();
        let prompt = ByteTokenizer::encode("determinism check");

        let model_a = GptModel::load_random(config.clone(), 7);
        let out_a = model_a.generate(&device, &prompt, 5).unwrap();

        let model_b = GptModel::load_random(config, 7);
        let out_b = model_b.generate(&device, &prompt, 5).unwrap();

        assert_eq!(out_a, out_b, "same seed should yield byte-identical generation (random weights, but deterministic RNG)");
    }

    #[test]
    fn different_seeds_produce_different_weights_and_usually_different_output() {
        let config = GptConfig::tiny(ByteTokenizer::VOCAB_SIZE);
        let device = device();
        let prompt = ByteTokenizer::encode("seed sensitivity");

        let model_a = GptModel::load_random(config.clone(), 1);
        let out_a = model_a.generate(&device, &prompt, 5).unwrap();

        let model_b = GptModel::load_random(config, 2);
        let out_b = model_b.generate(&device, &prompt, 5).unwrap();

        assert_ne!(out_a, out_b, "different seeds should (with overwhelming probability) yield different random weights and thus different output");
    }

    /// KVキャッシュを使った逐次デコードの数値的な正しさの検証:
    /// 「1トークンずつキャッシュを積みながら計算したロジット」が、
    /// 「毎回シーケンス全体を(キャッシュ無しで)フルスクラッチ計算した
    /// ロジット」と一致することを確認する(causalマスクを明示的な
    /// マスク行列ではなく「キャッシュに未来のトークンを入れない」ことで
    /// 実現している設計の正しさを裏付ける、opencuda-blasの既存テスト
    /// (Flash Attention数値一致)と同じ考え方)。
    #[test]
    fn incremental_kv_cache_decoding_matches_full_recompute_at_each_position() {
        let config = GptConfig::tiny(ByteTokenizer::VOCAB_SIZE);
        let model = GptModel::load_random(config.clone(), 99);
        let device = device();
        let tokens = ByteTokenizer::encode("abcde");

        // 増分キャッシュ版: forward_stepを直接呼び、各位置のロジットを記録。
        let mut caches = model.new_caches();
        let mut incremental_logits = Vec::new();
        for (pos, &tok) in tokens.iter().enumerate() {
            let logits = model.forward_step(device.as_ref(), tok, pos, &mut caches).unwrap();
            incremental_logits.push(logits);
        }

        // フルスクラッチ版: 毎回新しいキャッシュで先頭から`pos`個を再計算。
        for (pos, _) in tokens.iter().enumerate() {
            let mut caches = model.new_caches();
            let mut logits = Vec::new();
            for (p, &tok) in tokens[..=pos].iter().enumerate() {
                logits = model.forward_step(device.as_ref(), tok, p, &mut caches).unwrap();
            }
            for (a, b) in logits.iter().zip(incremental_logits[pos].iter()) {
                assert!((a - b).abs() < 1e-4, "position {pos}: incremental={a} vs full-recompute={b}");
            }
        }
    }

    fn gpt2_model_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/gpt2")
    }

    /// `distilgpt2`(2026-08-17、`generate_speculative`のドラフトモデルとして
    /// 使う実重み検証向けに新設)。
    fn distilgpt2_model_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/distilgpt2")
    }

    /// 合成(ランダムだが正しい形状の)safetensorsファイルを作り、
    /// `GptModel::load`がテンソル名・形状の契約通りに読み込めることを検証する。
    /// 実GPT-2重み(500MB超)が無い環境でもローダーのロジック自体を
    /// 検証できるようにするための単体テスト(モジュールdocコメント/
    /// タスク指示の「ネットワーク不到達時は単体テストに留めてよい」に対応)。
    /// 合成safetensorsフィクスチャを作って`GptModel::load`する共通ヘルパー。
    /// `key_prefix`に`""`を渡せば`openai-community/gpt2`本体と同じ無印の
    /// テンソル名規約、`"transformer."`を渡せば`distilbert/distilgpt2`等
    /// 一部モデルで使われる`transformer.`プレフィックス付き規約を再現する
    /// (2026-07-27、実E2Eダウンロード検証でdistilgpt2のロードが
    /// `missing tensor 'wte.weight'`で失敗する実バグを発見したことへの
    /// 回帰テスト——`GptModel::load`が両方の規約を吸収できることを
    /// 検証する)。
    fn build_and_load_synthetic_model(dir_suffix: &str, key_prefix: &str) -> GptModel {
        use safetensors::tensor::{Dtype, TensorView};
        use std::collections::HashMap;

        let vocab = 37usize;
        let hidden = 8usize;
        let n_head = 2usize;
        let n_layer = 1usize;
        let n_positions = 16usize;
        let inner = 4 * hidden;

        // 決定的な適当な値で埋めた各テンソルを用意する(数値の意味は無い、
        // 形状・テンソル名がGPT-2の契約と一致するかだけを検証する)。
        let mut rng = SplitMix64::new(1234);
        let mut buffers: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        let mut push = |name: String, shape: Vec<usize>, rng: &mut SplitMix64| {
            let len: usize = shape.iter().product();
            let data = random_vec(rng, len, 0.1);
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            buffers.push((name, shape, bytes));
        };
        push(format!("{key_prefix}wte.weight"), vec![vocab, hidden], &mut rng);
        push(format!("{key_prefix}wpe.weight"), vec![n_positions, hidden], &mut rng);
        for i in 0..n_layer {
            let p = format!("{key_prefix}h.{i}");
            push(format!("{p}.ln_1.weight"), vec![hidden], &mut rng);
            push(format!("{p}.ln_1.bias"), vec![hidden], &mut rng);
            push(format!("{p}.attn.c_attn.weight"), vec![hidden, 3 * hidden], &mut rng);
            push(format!("{p}.attn.c_attn.bias"), vec![3 * hidden], &mut rng);
            push(format!("{p}.attn.c_proj.weight"), vec![hidden, hidden], &mut rng);
            push(format!("{p}.attn.c_proj.bias"), vec![hidden], &mut rng);
            push(format!("{p}.ln_2.weight"), vec![hidden], &mut rng);
            push(format!("{p}.ln_2.bias"), vec![hidden], &mut rng);
            push(format!("{p}.mlp.c_fc.weight"), vec![hidden, inner], &mut rng);
            push(format!("{p}.mlp.c_fc.bias"), vec![inner], &mut rng);
            push(format!("{p}.mlp.c_proj.weight"), vec![inner, hidden], &mut rng);
            push(format!("{p}.mlp.c_proj.bias"), vec![hidden], &mut rng);
        }
        push(format!("{key_prefix}ln_f.weight"), vec![hidden], &mut rng);
        push(format!("{key_prefix}ln_f.bias"), vec![hidden], &mut rng);

        let mut views: HashMap<String, TensorView> = HashMap::new();
        for (name, shape, bytes) in &buffers {
            views.insert(name.clone(), TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap());
        }
        let serialized = safetensors::serialize(&views, &None).unwrap();

        let dir = std::env::temp_dir().join(format!("open-cuda-llm-synthetic-gpt2-{dir_suffix}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), serialized).unwrap();
        std::fs::write(
            dir.join("config.json"),
            format!(
                r#"{{"vocab_size":{vocab},"n_embd":{hidden},"n_layer":{n_layer},"n_head":{n_head},"n_positions":{n_positions},"layer_norm_epsilon":1e-5}}"#
            ),
        )
        .unwrap();

        let model = GptModel::load(&dir).unwrap();
        assert_eq!(model.config().vocab_size, vocab);
        assert_eq!(model.config().hidden_size, hidden);
        assert_eq!(model.config().num_layers, n_layer);
        assert_eq!(model.layers.len(), n_layer);

        // ロードしたモデルで実際に生成が(パニックせず)動くことも確認する
        // (形状が正しいだけでなく、forward_step一式が最後まで通ることの検証)。
        let device = device();
        let generated = model.generate(&device, &[1, 2, 3], 4).unwrap();
        assert_eq!(generated.len(), 4);

        let _ = std::fs::remove_dir_all(&dir);
        model
    }

    #[test]
    fn load_parses_gpt2_shaped_safetensors_and_config_without_panicking() {
        build_and_load_synthetic_model("no-prefix", "");
    }

    /// 2026-07-27追記の回帰テスト: `transformer.`プレフィックス付きの
    /// テンソル名規約(distilgpt2等が実際に使う規約)でも`GptModel::load`
    /// が正しくロードできることを確認する(実際にHugging Faceから
    /// distilgpt2をダウンロードして`GET /v1/models/select`したところ
    /// `missing tensor 'wte.weight'`で失敗した実バグの回帰テスト)。
    #[test]
    fn load_parses_transformer_prefixed_safetensors_like_distilgpt2() {
        build_and_load_synthetic_model("transformer-prefix", "transformer.");
    }

    /// 実GPT-2(124M、`openai-community/gpt2`のsafetensors)がこのマシンに
    /// ダウンロード済みの場合のみ実行する検証。**正直な開示**: 完全に流暢な
    /// 文章生成を検証しているのではなく、(a) 実重みが実際にロードできる
    /// こと、(b) ランダム初期化(`load_random`)と実重み(`load`)とで、
    /// 同一プロンプトに対する貪欲デコードの出力トークン列が異なる
    /// (=重みが実際に使われている、配線ミスで常に同じ出力になっていない)
    /// ことを確認するに留める。GPT-2自身のBPEトークナイザ(`GptTokenizer`)
    /// を使う。
    #[test]
    fn real_gpt2_weights_load_and_produce_output_distinct_from_random_init() {
        let dir = gpt2_model_dir();
        if !dir.join("model.safetensors").exists() {
            eprintln!("skipping: real GPT-2 weights not present at {dir:?} (see CLAUDE.md HANDOFF for download instructions)");
            return;
        }

        let model = GptModel::load(&dir).unwrap();
        assert_eq!(model.config().vocab_size, 50257);
        assert_eq!(model.config().hidden_size, 768);
        assert_eq!(model.config().num_layers, 12);
        assert_eq!(model.config().num_heads, 12);

        let tokenizer = GptTokenizer::load(&dir).unwrap();
        let prompt_ids = tokenizer.encode("The quick brown fox").unwrap();
        assert!(!prompt_ids.is_empty());

        let device = device();
        let real_out = model.generate(&device, &prompt_ids, 12).unwrap();
        let real_text = tokenizer.decode(&real_out).unwrap();
        eprintln!("real GPT-2 weights greedy continuation: {real_text:?} (token ids: {real_out:?})");

        // 同じプロンプト・同じトークナイザ語彙空間で、ランダム初期化モデルと
        // 生成結果が異なることを確認する(重みが実際に効いていることの
        // 最低限の裏付け——出力が「流暢」であることまでは主張しない)。
        let random_config = GptConfig {
            vocab_size: model.config().vocab_size,
            hidden_size: model.config().hidden_size,
            num_layers: model.config().num_layers,
            num_heads: model.config().num_heads,
            intermediate_size: model.config().intermediate_size,
            max_seq_len: model.config().max_seq_len,
            layer_norm_eps: model.config().layer_norm_eps,
        };
        let random_model = GptModel::load_random(random_config, 42);
        let random_out = random_model.generate(&device, &prompt_ids, 12).unwrap();
        let random_text = tokenizer.decode(&random_out).unwrap();
        eprintln!("random-init weights greedy continuation: {random_text:?} (token ids: {random_out:?})");

        assert_ne!(real_out, random_out, "real GPT-2 weights should produce different greedy output than random init for the same prompt");
    }

    /// **2026-08-17新設**: `generate_speculative`を実重み(ターゲット
    /// `gpt2`124M・ドラフト`distilgpt2`82M、同じGPT-2ファミリー・同じ
    /// GPT-2 BPE語彙)で検証する。型チェック・合成フィクスチャだけで
    /// 完了と報告しない方針の実践——ロスレス性(`generate()`とのビット
    /// 完全一致)・採用率・大まかな所要時間差を実際に計測して記録する。
    #[test]
    fn real_gpt2_speculative_decoding_matches_plain_greedy_and_reports_acceptance() {
        let target_dir = gpt2_model_dir();
        let draft_dir = distilgpt2_model_dir();
        if !target_dir.join("model.safetensors").exists() || !draft_dir.join("model.safetensors").exists() {
            eprintln!(
                "skipping: real gpt2/distilgpt2 weights not present at {target_dir:?} / {draft_dir:?} \
                 (see CLAUDE.md HANDOFF for download instructions)"
            );
            return;
        }

        let target = GptModel::load(&target_dir).unwrap();
        let draft = GptModel::load(&draft_dir).unwrap();
        assert_eq!(target.config().vocab_size, draft.config().vocab_size, "gpt2 and distilgpt2 must share the same GPT-2 BPE vocabulary");

        let tokenizer = GptTokenizer::load(&target_dir).unwrap();
        let prompt_ids = tokenizer.encode("The quick brown fox").unwrap();
        let device = device();
        let max_new_tokens = 16;

        let t0 = std::time::Instant::now();
        let plain = target.generate(&device, &prompt_ids, max_new_tokens).unwrap();
        let plain_elapsed = t0.elapsed();

        let t1 = std::time::Instant::now();
        let (speculative, stats) = target.generate_speculative(&device, &draft, &prompt_ids, max_new_tokens, 4).unwrap();
        let speculative_elapsed = t1.elapsed();

        assert_eq!(speculative, plain, "speculative decoding output must be byte-identical to plain greedy decode on real weights");

        eprintln!(
            "generate_speculative (real gpt2 target + distilgpt2 draft, draft_k=4, max_new_tokens={max_new_tokens}): \
             plain={plain_elapsed:?}, speculative={speculative_elapsed:?}, \
             acceptance={}/{} ({:.1}%)",
            stats.accepted,
            stats.proposed,
            stats.acceptance_rate() * 100.0
        );
    }

    /// **2026-08-10新設**: `aruaru-llm`側ユーザー報告「しつこく繰り返す
    /// バグ」(対話ファインチューニング無しの素のGPT-2貪欲デコードが
    /// "Student: Hello"等の同一文字列を無限ループする)に対する
    /// `generate_with_repetition_penalty`の実効性を実GPT-2 124M重みで
    /// 検証する。open-english側と同じプロンプト構造(会話プロンプト+
    /// "Student: <発話>\nTrainer:")で再現し、ペナルティ無し版が実際に
    /// "Student:"を繰り返すこと、ペナルティ適用版がその繰り返しを避ける
    /// (生成列内の"Student:"相当のトークン列の出現回数が減る)ことを確認。
    #[test]
    fn repetition_penalty_reduces_degenerate_loop_on_real_gpt2_weights() {
        let dir = gpt2_model_dir();
        if !dir.join("model.safetensors").exists() {
            eprintln!("skipping: real GPT-2 weights not present at {dir:?} (see CLAUDE.md HANDOFF for download instructions)");
            return;
        }

        let model = GptModel::load(&dir).unwrap();
        let tokenizer = GptTokenizer::load(&dir).unwrap();
        let prompt = "You are a friendly English conversation trainer at a maid cafe.\nStudent: Hello\nTrainer:";
        let prompt_ids = tokenizer.encode(prompt).unwrap();
        assert!(!prompt_ids.is_empty());

        let device = device();
        let no_penalty = model.generate_with_repetition_penalty(&device, &prompt_ids, 40, 1.0).unwrap();
        let with_penalty = model.generate_with_repetition_penalty(&device, &prompt_ids, 40, 1.3).unwrap();

        let no_penalty_text = tokenizer.decode(&no_penalty).unwrap();
        let with_penalty_text = tokenizer.decode(&with_penalty).unwrap();
        eprintln!("no repetition penalty : {no_penalty_text:?}");
        eprintln!("repetition_penalty=1.3: {with_penalty_text:?}");

        let count_student = |s: &str| s.matches("Student:").count();
        let no_penalty_repeats = count_student(&no_penalty_text);
        let with_penalty_repeats = count_student(&with_penalty_text);

        // ペナルティ無し版は実際に劣化ループへ陥ること(この既知の失敗
        // モードを再現できていること自体の裏取り)を確認したうえで、
        // ペナルティ版がその繰り返し回数を実際に減らすことを検証する。
        assert!(no_penalty_repeats >= 2, "expected the unpenalized baseline to actually exhibit the known repetition loop, got: {no_penalty_text:?}");
        assert!(
            with_penalty_repeats < no_penalty_repeats,
            "repetition_penalty=1.3 should reduce 'Student:' recurrences versus no penalty (no_penalty={no_penalty_repeats}, with_penalty={with_penalty_repeats})"
        );

        // penalty=1.0の場合は`generate`(既存API)と完全に同一の出力になる
        // ことも確認する(後方互換性の実証)。
        let via_generate = model.generate(&device, &prompt_ids, 40).unwrap();
        assert_eq!(via_generate, no_penalty, "generate() must be byte-identical to generate_with_repetition_penalty(..., 1.0)");
    }

    /// **2026-08-08新設**: [`GptModel::enable_mla_kv_compression_calibrated`]
    /// (PCA較正版MLA)の実証テスト。実GPT-2 124M重みで、(a)非圧縮、
    /// (b)乱数射影MLA(`enable_mla_kv_compression`、既知の劣化)、
    /// (c)PCA較正版MLA(`enable_mla_kv_compression_calibrated`)の3経路を
    /// 同一プロンプト・同一`d_c`で比較し、実際に生成された文字列を
    /// `eprintln!`で残す(数値上の改善だけでなく、読み手が実際に文章を
    /// 見て判断できるようにする、タスク指示「fabricate結果しない」への
    /// 対応)。較正プロンプトとは別のheld-outプロンプトでも実行し、
    /// 較正データへの過学習でないかを正直に確認する。
    #[test]
    fn calibrated_pca_mla_kv_compression_on_real_gpt2_weights() {
        let dir = gpt2_model_dir();
        if !dir.join("model.safetensors").exists() {
            eprintln!("skipping calibrated PCA MLA test: real GPT-2 weights not present at {dir:?}");
            return;
        }

        let tokenizer = GptTokenizer::load(&dir).unwrap();
        let device: std::sync::Arc<dyn GpuDevice> = CpuDevice::new(0);

        // head_dim = 768/12 = 64 のうち d_c=16 (75%圧縮)、タスク指示の
        // 数値と一致させる。
        let d_c = 16;

        let calibration_prompts_text = [
            "The weather today is quite pleasant and sunny.",
            "In economics, supply and demand determine prices in a market.",
            "She walked into the kitchen and started making breakfast.",
            "The history of ancient Rome spans over a thousand years.",
            "Computers process information using binary logic circuits.",
            "The mountain trail was steep but offered a beautiful view.",
            "Scientists discovered a new species of frog in the rainforest.",
            "He picked up his guitar and began to play a soft melody.",
        ];
        let calibration_prompts: Vec<Vec<u32>> = calibration_prompts_text.iter().map(|t| tokenizer.encode(t).unwrap()).collect();

        let test_prompt_calibration_style = "The quick brown fox";
        let test_prompt_holdout = "def compute_gradient(weights, learning_rate):";

        for (label, prompt_text) in [("calibration-style prompt", test_prompt_calibration_style), ("held-out prompt", test_prompt_holdout)] {
            let prompt_ids = tokenizer.encode(prompt_text).unwrap();

            // (a) 非圧縮ベースライン
            let baseline_model = GptModel::load(&dir).unwrap();
            let baseline_out = baseline_model.generate(&device, &prompt_ids, 16).unwrap();
            let baseline_text = tokenizer.decode(&baseline_out).unwrap();

            // (b) 乱数射影MLA(既知の劣化)
            let mut random_mla_model = GptModel::load(&dir).unwrap();
            random_mla_model.enable_mla_kv_compression(d_c, 999).unwrap();
            let random_mla_out = random_mla_model.generate(&device, &prompt_ids, 16).unwrap();
            let random_mla_text = tokenizer.decode(&random_mla_out).unwrap();

            // (c) PCA較正版MLA
            let mut pca_mla_model = GptModel::load(&dir).unwrap();
            pca_mla_model.enable_mla_kv_compression_calibrated(d_c, device.as_ref(), &calibration_prompts).unwrap();
            let pca_mla_out = pca_mla_model.generate(&device, &prompt_ids, 16).unwrap();
            let pca_mla_text = tokenizer.decode(&pca_mla_out).unwrap();

            eprintln!("=== {label} ({prompt_text:?}) ===");
            eprintln!("  uncompressed         : {baseline_text:?}");
            eprintln!("  random-projection MLA: {random_mla_text:?}");
            eprintln!("  PCA-calibrated MLA    : {pca_mla_text:?}");

            // 正直な最低限のassert: PCA較正版は乱数射影版と異なる(=実際に
            // 別の基底を使っている)ことのみ機械的に確認する。「PCA版の方が
            // 流暢である」ことは自動テストでは判定できない主観的な質なので、
            // 上記eprintln!の実文字列を人間が読んで判断する設計。
            assert_ne!(
                random_mla_out, pca_mla_out,
                "PCA-calibrated and random-projection MLA should use genuinely different projection bases and thus usually diverge in output"
            );
        }
    }

    /// **2026-08-04新設**: プリフィル/デコード分離(`forward_prefill_all_layers`)
    /// +QKV融合GEMM(`DecoderLayer::qkv`)というディスパッチ削減最適化が、
    /// 「挙動を変えない最適化」であることの回帰テスト。プロンプトを
    /// バッチ処理する`GptModel::generate`(内部で`forward_prefill_all_layers`
    /// を使う)の出力が、1トークンずつ`forward_step`をループする素朴な
    /// リファレンス実装と完全に一致する(ビット完全)ことを確認する。
    #[test]
    fn prefill_batch_generate_matches_token_by_token_forward_step_reference() {
        let config = GptConfig::tiny(ByteTokenizer::VOCAB_SIZE);
        let model = GptModel::load_random(config, 2026);
        let device = device();
        let prompt = ByteTokenizer::encode("prefill batching regression check");

        // 最適化されたパス(forward_prefill_all_layers経由)。
        let optimized = model.generate(&device, &prompt, 10).unwrap();

        // リファレンス実装: forward_stepを1トークンずつ呼ぶだけの素朴な
        // ループ(最適化前の`generate`のロジックをそのままここへ複製)。
        let mut caches = model.new_caches();
        let mut pos = 0usize;
        let mut logits = Vec::new();
        for &tok in &prompt {
            logits = model.forward_step(device.as_ref(), tok, pos, &mut caches).unwrap();
            pos += 1;
        }
        let mut reference = Vec::with_capacity(10);
        let mut next = argmax(&logits);
        for _ in 0..10 {
            reference.push(next);
            if pos >= model.config().max_seq_len {
                break;
            }
            logits = model.forward_step(device.as_ref(), next, pos, &mut caches).unwrap();
            pos += 1;
            next = argmax(&logits);
        }

        assert_eq!(optimized, reference, "prefill-batched generate() must match token-by-token forward_step reference exactly (behavior-preserving optimization)");
    }

    /// 上記と同じ趣旨だが、単一トークンのプロンプト(`seq_len=1`)という
    /// 境界ケースでも一致することを確認する(バッチ処理コードパスが
    /// `seq_len=1`のときも正しく`forward_step`ループの特殊ケースとして
    /// 振る舞うことの検証)。
    #[test]
    fn prefill_batch_generate_matches_reference_for_single_token_prompt() {
        let config = GptConfig::tiny(ByteTokenizer::VOCAB_SIZE);
        let model = GptModel::load_random(config, 4242);
        let device = device();
        let prompt = ByteTokenizer::encode("x");
        assert_eq!(prompt.len(), 1);

        let optimized = model.generate(&device, &prompt, 6).unwrap();

        let mut caches = model.new_caches();
        let mut pos = 0usize;
        let mut logits = Vec::new();
        for &tok in &prompt {
            logits = model.forward_step(device.as_ref(), tok, pos, &mut caches).unwrap();
            pos += 1;
        }
        let mut reference = Vec::with_capacity(6);
        let mut next = argmax(&logits);
        for _ in 0..6 {
            reference.push(next);
            if pos >= model.config().max_seq_len {
                break;
            }
            logits = model.forward_step(device.as_ref(), next, pos, &mut caches).unwrap();
            pos += 1;
            next = argmax(&logits);
        }

        assert_eq!(optimized, reference);
    }

    /// 2026-08-05配線の実機検証: `GptModel::set_matmul_spirv`経由で
    /// `Linear::forward`が実際に`GemmPath::VulkanGeneric`を使うように
    /// なり、CPU実行(`GemmPath::CpuNaive`)と数値一致する出力を返すことを
    /// 実Vulkanハードウェア上で確認する。実Vulkan環境(このマシンでは
    /// NVIDIA GeForce GT 730)と事前コンパイル済み`matmul.spv`
    /// (`examples/matmul_vulkan_real/shaders/matmul.spv`、
    /// `tools/compile-vulkan-shaders.*`で生成)の両方が必要——どちらか
    /// 欠けている環境(CI等)では誤魔化さずスキップする
    /// (`opencuda-blas`の同種テストと同じ方針)。
    ///
    /// **意図的に`Linear::forward`単体を叩き、`GptModel::generate`は
    /// 呼ばない(正直な開示・スコープの境界)**: 最初はモデル全体の
    /// `generate()`をCPU/Vulkan両方で実行し出力一致を見る設計で書いたが、
    /// 実機で実行したところ`VulkanDevice::launch_kernel`が
    /// `kernel source not supported by this backend: Native`で
    /// **panicすることを実際に確認した**——`scaled_dot_product_attention`
    /// が内部で使う`launch_naive_gemm`はSPIR-Vではなく
    /// `KernelSource::Native`(Rustクロージャ)を要求するため、
    /// `VulkanDevice`(SPIR-Vカーネルしか受理しない実装)へ渡すと
    /// Attention計算そのものが即座に失敗する。これは今回のGEMM配線
    /// 修正とは別の、独立したギャップ(SPIR-V版のattentionカーネルが
    /// 存在しない)であり、本テストのスコープでは解消しない
    /// (`CLAUDE.md`HANDOFFに記録)。そのため本テストは、実際に
    /// Vulkan経路を通る範囲——`Linear::forward`(GEMM)——だけを直接
    /// 検証範囲とし、Attentionを経由する`generate()`は使わない。
    #[test]
    fn set_matmul_spirv_makes_linear_forward_use_vulkan_and_matches_cpu_output() {
        let spirv_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/matmul_vulkan_real/shaders/matmul.spv");
        let spirv = match std::fs::read(&spirv_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!(
                    "skipping set_matmul_spirv test: matmul.spv not compiled at {}: {e} \
                     (run tools/compile-vulkan-shaders.* first)",
                    spirv_path.display()
                );
                return;
            }
        };

        let vulkan_device = match opencuda_vulkan::VulkanDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping set_matmul_spirv test: no real Vulkan device available: {e}");
                return;
            }
        };
        let vulkan_device: std::sync::Arc<dyn GpuDevice> = vulkan_device;
        let cpu_device: std::sync::Arc<dyn GpuDevice> = opencuda_cpu::CpuDevice::new(0);

        let mut rng = SplitMix64::new(42);
        let mut linear = Linear::random(&mut rng, 16, 24);
        let x: Vec<f32> = (0..3 * 16).map(|i| (i % 5) as f32 * 0.1).collect();

        let cpu_out = linear.forward(cpu_device.as_ref(), &x, 3).unwrap();

        linear.spirv_matmul = Some(std::sync::Arc::new(spirv));
        let vulkan_out = linear.forward(vulkan_device.as_ref(), &x, 3).unwrap();

        assert_eq!(cpu_out.len(), vulkan_out.len());
        for (i, (&cv, &vv)) in cpu_out.iter().zip(vulkan_out.iter()).enumerate() {
            assert!((cv - vv).abs() < 1e-3, "idx {i}: cpu={cv}, vulkan={vv}");
        }
    }

    /// 2026-08-05(続き)追加: 上のテストが書かれた時点では
    /// `scaled_dot_product_attention`がVulkanデバイス上で
    /// `KernelSource::Native`を要求し即座にpanicしたため、`generate()`
    /// 全体をVulkanで動かす検証は意図的に外していた
    /// (`opencuda-blas`側に`scaled_dot_product_attention_with_spirv`
    /// を新設し、`DecoderLayer::forward_step`/`forward_prefill`の
    /// 呼び出し箇所を切り替えたことで解消——このテストがその
    /// エンドツーエンド検証)。`GptModel::set_matmul_spirv`を呼んだ
    /// モデルに対して`generate()`を実Vulkanハードウェア(NVIDIA GT 730)
    /// 上で実行し、CPU実行(spirv未設定のモデル)と生成トークン列が
    /// 完全一致することを確認する。
    #[test]
    fn generate_end_to_end_matches_cpu_on_real_vulkan_hardware_after_set_matmul_spirv() {
        let spirv_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/matmul_vulkan_real/shaders/matmul.spv");
        let spirv = match std::fs::read(&spirv_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!(
                    "skipping generate_end_to_end test: matmul.spv not compiled at {}: {e} \
                     (run tools/compile-vulkan-shaders.* first)",
                    spirv_path.display()
                );
                return;
            }
        };

        let vulkan_device = match opencuda_vulkan::VulkanDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping generate_end_to_end test: no real Vulkan device available: {e}");
                return;
            }
        };
        let vulkan_device: std::sync::Arc<dyn GpuDevice> = vulkan_device;
        let cpu_device: std::sync::Arc<dyn GpuDevice> = opencuda_cpu::CpuDevice::new(0);

        let config = GptConfig::tiny(ByteTokenizer::VOCAB_SIZE);
        let prompt = ByteTokenizer::encode("hi");

        let cpu_model = GptModel::load_random(config.clone(), 42);
        let cpu_out = cpu_model.generate(&cpu_device, &prompt, 6).unwrap();

        let mut vulkan_model = GptModel::load_random(config, 42);
        vulkan_model.set_matmul_spirv(spirv);
        // 本題: これまで`VulkanDevice::launch_kernel`が
        // `kernel source not supported by this backend: Native`で
        // panicしていた経路。scaled_dot_product_attention_with_spirvへの
        // 切り替えにより、ここでpanicせず最後まで完走することを確認する。
        let vulkan_out = vulkan_model.generate(&vulkan_device, &prompt, 6).unwrap();

        assert_eq!(cpu_out, vulkan_out, "CPU and Vulkan generation should produce byte-identical token sequences for the same seed/prompt");
    }

    /// **2026-08-23新設**: `set_matmul_dxil_offload`(密GEMMのD3D12
    /// Computeオフロード)を配線したモデルの`generate()`が、実D3D12
    /// ハードウェア上で最後まで完走し、純CPU実行と生成トークン列が
    /// 完全一致することを確認する(上のVulkan版に対応するDirectX版)。
    ///
    /// D3D12デバイスを作れない環境(非Windows・ドライバ無し)では
    /// 失敗ではなくスキップする(既存のVulkan実機テストと同じ方針)。
    #[test]
    #[cfg(windows)]
    fn generate_end_to_end_matches_cpu_on_real_d3d12_after_set_matmul_dxil_offload() {
        const MATMUL_DXIL: &[u8] = include_bytes!("../../opencuda-directx/shaders/matmul.dxil");

        let dx_device = match opencuda_directx::real::DirectXDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping generate_end_to_end (dxil) test: no real D3D12 device available: {e}");
                return;
            }
        };
        let dx_device: std::sync::Arc<dyn GpuDevice> = dx_device;
        let cpu_device: std::sync::Arc<dyn GpuDevice> = opencuda_cpu::CpuDevice::new(0);

        let config = GptConfig::tiny(ByteTokenizer::VOCAB_SIZE);
        let prompt = ByteTokenizer::encode("hi");

        let cpu_model = GptModel::load_random(config.clone(), 42);
        let cpu_out = cpu_model.generate(&cpu_device, &prompt, 6).unwrap();

        let mut dx_model = GptModel::load_random(config, 42);
        dx_model.set_matmul_dxil_offload(dx_device, MATMUL_DXIL.to_vec()).expect("wire dxil offload");
        // 密GEMMだけがD3D12へ行き、Attention/LayerNorm/GELUはCPUデバイス
        // 上で走るハイブリッド構成。`generate`へ渡すのはCPUデバイス。
        let dx_out = dx_model.generate(&cpu_device, &prompt, 6).unwrap();

        assert_eq!(
            cpu_out, dx_out,
            "CPU-only and DXIL-offloaded generation should produce identical token sequences for the same seed/prompt"
        );
    }

    /// **2026-08-06新設**: `set_matmul_spirv`に加えて`set_softmax_spirv`も
    /// 呼んだ場合(「GPU GEMM + GPU softmax」経路)でも、`generate()`が
    /// 実Vulkanハードウェア上で最後まで完走し、CPU実行と生成トークン列が
    /// 完全一致することを確認する(直上テストの「GPU GEMM + CPU softmax」
    /// 版に対応するGPU常駐softmax版)。
    #[test]
    fn generate_end_to_end_matches_cpu_on_real_vulkan_hardware_after_set_matmul_and_softmax_spirv() {
        let matmul_spirv_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/matmul_vulkan_real/shaders/matmul.spv");
        let matmul_spirv = match std::fs::read(&matmul_spirv_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!(
                    "skipping generate_end_to_end (matmul+softmax) test: matmul.spv not compiled at {}: {e} \
                     (run tools/compile-vulkan-shaders.* first)",
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
                    "skipping generate_end_to_end (matmul+softmax) test: softmax.spv not compiled at {}: {e} \
                     (run tools/compile-vulkan-shaders.* first)",
                    softmax_spirv_path.display()
                );
                return;
            }
        };

        let vulkan_device = match opencuda_vulkan::VulkanDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping generate_end_to_end (matmul+softmax) test: no real Vulkan device available: {e}");
                return;
            }
        };
        let vulkan_device: std::sync::Arc<dyn GpuDevice> = vulkan_device;
        let cpu_device: std::sync::Arc<dyn GpuDevice> = opencuda_cpu::CpuDevice::new(0);

        let config = GptConfig::tiny(ByteTokenizer::VOCAB_SIZE);
        let prompt = ByteTokenizer::encode("hi");

        let cpu_model = GptModel::load_random(config.clone(), 42);
        let cpu_out = cpu_model.generate(&cpu_device, &prompt, 6).unwrap();

        let mut vulkan_model = GptModel::load_random(config, 42);
        vulkan_model.set_matmul_spirv(matmul_spirv);
        vulkan_model.set_softmax_spirv(softmax_spirv);
        let vulkan_out = vulkan_model.generate(&vulkan_device, &prompt, 6).unwrap();

        assert_eq!(
            cpu_out, vulkan_out,
            "CPU and Vulkan(GPU GEMM + GPU softmax) generation should produce byte-identical token sequences for the same seed/prompt"
        );
    }

    /// **2026-08-07新設**: `set_flash_attention_spirv`(本HANDOFF増分、
    /// open-cuda側2026-08-07 HANDOFF「次にすべきこと(1)」——素朴な
    /// `scaled_dot_product_attention_with_spirv_and_softmax`から
    /// `flash_attention_with_spirv`〈1ディスパッチで完結するfused
    /// カーネル〉への切り替え——への対応)を配線した場合でも、`generate()`
    /// が実Vulkanハードウェア上で最後まで完走し、CPU実行と生成トークン列が
    /// 完全一致することを確認する。上記2テスト(GEMM+CPU softmax /
    /// GEMM+GPU softmax)と並ぶ第3の経路。
    #[test]
    fn generate_end_to_end_matches_cpu_on_real_vulkan_hardware_after_set_matmul_and_flash_attention_spirv() {
        let matmul_spirv_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/matmul_vulkan_real/shaders/matmul.spv");
        let matmul_spirv = match std::fs::read(&matmul_spirv_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!(
                    "skipping generate_end_to_end (matmul+flash_attention) test: matmul.spv not compiled at {}: {e} \
                     (run tools/compile-vulkan-shaders.* first)",
                    matmul_spirv_path.display()
                );
                return;
            }
        };
        let flash_spirv_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/flash_attention_vulkan_real/shaders/flash_attention.spv");
        let flash_spirv = match std::fs::read(&flash_spirv_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!(
                    "skipping generate_end_to_end (matmul+flash_attention) test: flash_attention.spv not compiled at {}: {e} \
                     (run tools/compile-vulkan-shaders.* first)",
                    flash_spirv_path.display()
                );
                return;
            }
        };

        let vulkan_device = match opencuda_vulkan::VulkanDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping generate_end_to_end (matmul+flash_attention) test: no real Vulkan device available: {e}");
                return;
            }
        };
        let vulkan_device: std::sync::Arc<dyn GpuDevice> = vulkan_device;
        let cpu_device: std::sync::Arc<dyn GpuDevice> = opencuda_cpu::CpuDevice::new(0);

        let config = GptConfig::tiny(ByteTokenizer::VOCAB_SIZE);
        let prompt = ByteTokenizer::encode("hi");

        let cpu_model = GptModel::load_random(config.clone(), 42);
        let cpu_out = cpu_model.generate(&cpu_device, &prompt, 6).unwrap();

        let mut vulkan_model = GptModel::load_random(config, 42);
        vulkan_model.set_matmul_spirv(matmul_spirv);
        // block_size=4: GptConfig::tiny()のhead_dim=32/4=8、キャッシュ長も
        // 今回のプロンプト長(2〜8トークン程度)で256を大きく下回るため、
        // シェーダのMAX_DIM=256制約には掛からない。
        vulkan_model.set_flash_attention_spirv(flash_spirv, 4);
        let vulkan_out = vulkan_model.generate(&vulkan_device, &prompt, 6).unwrap();

        assert_eq!(
            cpu_out, vulkan_out,
            "CPU and Vulkan(GPU GEMM + fused flash attention) generation should produce byte-identical token sequences for the same seed/prompt"
        );
    }

    /// **2026-08-07新設**: `enable_mla_kv_compression`で配線したKVキャッシュ
    /// 圧縮経路が、実Vulkanハードウェア(NVIDIA GT 730)上でも
    /// `mla_compress_kv`/`mla_decompress_kv`のVulkan経路(`spirv`引数付き)を
    /// 通してエラー無く最後まで完走することを確認する。`set_matmul_spirv`
    /// (GEMM)経由でMLAの圧縮・復元用GEMMにも同じ`matmul.spv`が渡るため、
    /// これが実Vulkanデバイス上でのMLA配線の実機検証にあたる(前回HANDOFF
    /// 「次にすべきこと(1)」の最終検証)。CPU版とのトークン列一致は
    /// 主張しない(射影行列が乱数のため、CPU/Vulkanどちらの経路でも
    /// 情報の一部が失われる非可逆変換であり、両者が一致する保証は無い
    /// ——実際に確認したいのは「実機で最後まで動くこと」)。
    #[test]
    fn mla_kv_compression_completes_on_real_vulkan_hardware() {
        let spirv_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/matmul_vulkan_real/shaders/matmul.spv");
        let spirv = match std::fs::read(&spirv_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!("skipping mla_kv_compression_completes_on_real_vulkan_hardware: matmul.spv not compiled at {}: {e}", spirv_path.display());
                return;
            }
        };

        let vulkan_device = match opencuda_vulkan::VulkanDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping mla_kv_compression_completes_on_real_vulkan_hardware: no real Vulkan device available: {e}");
                return;
            }
        };
        let vulkan_device: std::sync::Arc<dyn GpuDevice> = vulkan_device;

        let config = GptConfig::tiny(ByteTokenizer::VOCAB_SIZE); // hidden=32, num_heads=4 => head_dim=8
        let mut model = GptModel::load_random(config, 7);
        model.set_matmul_spirv(spirv);
        model.enable_mla_kv_compression(2, 42).unwrap(); // 75%削減

        let prompt = ByteTokenizer::encode("mla on real vulkan hardware");
        let generated = model.generate(&vulkan_device, &prompt, 6).unwrap();
        assert_eq!(generated.len(), 6, "generate() should complete end-to-end through the compressed KV cache path on real Vulkan hardware");
    }
}

#[cfg(test)]
mod bench_manual {
    use super::*;
    use opencuda_cpu::CpuDevice;
    use std::sync::Arc;
    use std::time::Instant;

    /// 手動実行専用(既定のcargo testでは実行しない、`--ignored`指定時のみ)。
    /// 実GPT-2 124M重みでプロンプト長ごとのgenerate()所要時間を計測し、
    /// プリフィルバッチ化の効果を実測するためのベンチマーク。
    #[test]
    #[ignore]
    fn manual_bench_real_gpt2_generate_timing() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/gpt2");
        if !dir.join("model.safetensors").exists() {
            eprintln!("skipping: no real GPT-2 weights");
            return;
        }
        let model = GptModel::load(&dir).unwrap();
        let tokenizer = GptTokenizer::load(&dir).unwrap();
        let device: Arc<dyn GpuDevice> = CpuDevice::new(0);
        let prompt = tokenizer.encode("The quick brown fox jumps over the lazy dog and continues running through the forest").unwrap();
        eprintln!("prompt_len={}", prompt.len());
        let start = Instant::now();
        let out = model.generate(&device, &prompt, 20).unwrap();
        let elapsed = start.elapsed();
        eprintln!("generate(20 new tokens) elapsed={elapsed:?}, out_len={}", out.len());
    }
}
