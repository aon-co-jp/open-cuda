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
    Ok(Linear { weight_t: weight, bias, in_dim, out_dim })
}

/// GPT-2はQ/K/Vを1本の`c_attn`(`[hidden, 3*hidden]`)へ融合して保存する。
/// `Linear::forward`はヘッドごとに独立した`Linear`を要求するため、
/// 出力次元(列)方向に3分割してQ/K/Vそれぞれの`Linear`を作る。
fn load_fused_qkv(tensors: &safetensors::SafeTensors, prefix: &str, hidden: usize) -> Result<(Linear, Linear, Linear)> {
    let weight = tensor_f32(tensors, &format!("{prefix}.weight"))?;
    anyhow::ensure!(
        weight.len() == hidden * 3 * hidden,
        "open-cuda-llm: '{prefix}.weight' has {} elements, expected {}x{}",
        weight.len(),
        hidden,
        3 * hidden
    );
    let bias = tensor_f32(tensors, &format!("{prefix}.bias"))?;
    anyhow::ensure!(bias.len() == 3 * hidden, "open-cuda-llm: '{prefix}.bias' has {} elements, expected {}", bias.len(), 3 * hidden);

    let split = |idx: usize| -> (Vec<f32>, Vec<f32>) {
        let col_start = idx * hidden;
        let mut w = vec![0.0f32; hidden * hidden];
        for (dst_row, src_row) in w.chunks_exact_mut(hidden).zip(weight.chunks_exact(3 * hidden)) {
            dst_row.copy_from_slice(&src_row[col_start..col_start + hidden]);
        }
        let b = bias[col_start..col_start + hidden].to_vec();
        (w, b)
    };

    let (qw, qb) = split(0);
    let (kw, kb) = split(1);
    let (vw, vb) = split(2);
    Ok((
        Linear { weight_t: qw, bias: qb, in_dim: hidden, out_dim: hidden },
        Linear { weight_t: kw, bias: kb, in_dim: hidden, out_dim: hidden },
        Linear { weight_t: vw, bias: vb, in_dim: hidden, out_dim: hidden },
    ))
}

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
}

impl Linear {
    fn random(rng: &mut SplitMix64, in_dim: usize, out_dim: usize) -> Self {
        let scale = 1.0 / (in_dim as f32).sqrt();
        Self { weight_t: random_vec(rng, in_dim * out_dim, scale), bias: vec![0.0; out_dim], in_dim, out_dim }
    }

    fn forward(&self, device: &dyn GpuDevice, x: &[f32], seq_len: usize) -> Result<Vec<f32>> {
        debug_assert_eq!(x.len(), seq_len * self.in_dim);
        let mut out = vec![0.0f32; seq_len * self.out_dim];
        opencuda_blas::sgemm(device, seq_len, self.in_dim, self.out_dim, 1.0, x, &self.weight_t, 0.0, &mut out, None)?;
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
    query: Linear,
    key: Linear,
    value: Linear,
    attn_out: Linear,
    ln_2: LayerNorm,
    intermediate: Linear,
    output: Linear,
}

impl DecoderLayer {
    fn random(rng: &mut SplitMix64, hidden: usize, intermediate: usize, eps: f32) -> Self {
        Self {
            ln_1: LayerNorm::identity(hidden, eps),
            query: Linear::random(rng, hidden, hidden),
            key: Linear::random(rng, hidden, hidden),
            value: Linear::random(rng, hidden, hidden),
            attn_out: Linear::random(rng, hidden, hidden),
            ln_2: LayerNorm::identity(hidden, eps),
            intermediate: Linear::random(rng, hidden, intermediate),
            output: Linear::random(rng, intermediate, hidden),
        }
    }

    /// 1トークン分(`hidden`は`hidden_size`長の単一行)を処理し、
    /// このレイヤーの`cache`へ今回のk/vを追加した上で出力を返す
    /// (causalマスクは「まだキャッシュに存在しない未来のトークンは
    /// そもそも追加されていない」ことで自然に実現される、明示的な
    /// マスク行列は不要)。pre-LN(GPT-2方式、上記構造体docコメント参照)。
    fn forward_step(&self, device: &dyn GpuDevice, hidden: &[f32], cache: &mut [KvCacheHead], hidden_size: usize, num_heads: usize) -> Result<Vec<f32>> {
        let head_dim = hidden_size / num_heads;

        let mut normed = hidden.to_vec();
        self.ln_1.forward(&mut normed, 1, hidden_size);

        let q = self.query.forward(device, &normed, 1)?;
        let k = self.key.forward(device, &normed, 1)?;
        let v = self.value.forward(device, &normed, 1)?;

        let mut context = vec![0.0f32; hidden_size];
        for (h, cache_head) in cache.iter_mut().enumerate().take(num_heads) {
            let col_start = h * head_dim;
            let q_h = &q[col_start..col_start + head_dim];
            let k_h = &k[col_start..col_start + head_dim];
            let v_h = &v[col_start..col_start + head_dim];

            cache_head.push(k_h, v_h);
            let n = cache_head.n;

            // qを n 回複製して n x n の attention を計算し、先頭行(全行
            // 同一)だけを使う(モジュールdocコメント参照、素朴だが正しい)。
            let mut q_full = vec![0.0f32; n * head_dim];
            for row in q_full.chunks_exact_mut(head_dim) {
                row.copy_from_slice(q_h);
            }
            let out = opencuda_blas::scaled_dot_product_attention(device, &q_full, &cache_head.k, &cache_head.v, n, head_dim)?;
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
}

/// ヘッド単位のKVキャッシュ(`DecoderLayer::forward_step`内で使用)。
struct KvCacheHead {
    k: Vec<f32>,
    v: Vec<f32>,
    n: usize,
}

impl KvCacheHead {
    fn empty() -> Self {
        Self { k: Vec::new(), v: Vec::new(), n: 0 }
    }

    fn push(&mut self, k_row: &[f32], v_row: &[f32]) {
        self.k.extend_from_slice(k_row);
        self.v.extend_from_slice(v_row);
        self.n += 1;
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
        Self { config, word_embeddings, position_embeddings, layers, final_ln, lm_head }
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
            let (query, key, value) = load_fused_qkv(&tensors, &format!("{p}.attn.c_attn"), hidden)?;
            layers.push(DecoderLayer {
                ln_1: load_layer_norm(&tensors, &format!("{p}.ln_1"), config.layer_norm_eps)?,
                query,
                key,
                value,
                attn_out: load_conv1d(&tensors, &format!("{p}.attn.c_proj"), hidden, hidden)?,
                ln_2: load_layer_norm(&tensors, &format!("{p}.ln_2"), config.layer_norm_eps)?,
                intermediate: load_conv1d(&tensors, &format!("{p}.mlp.c_fc"), hidden, config.intermediate_size)?,
                output: load_conv1d(&tensors, &format!("{p}.mlp.c_proj"), config.intermediate_size, hidden)?,
            });
        }

        let final_ln = load_layer_norm(&tensors, &format!("{key_prefix}ln_f"), config.layer_norm_eps)?;
        let lm_head = Linear {
            weight_t: transpose(&word_embeddings, config.vocab_size, hidden),
            bias: vec![0.0; config.vocab_size], // GPT-2のlm_headはbias無し(weight tying)
            in_dim: hidden,
            out_dim: config.vocab_size,
        };

        Ok(Self { config, word_embeddings, position_embeddings, layers, final_ln, lm_head })
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

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            hidden = layer.forward_step(device, &hidden, cache, hidden_size, self.config.num_heads)?;
        }

        self.final_ln.forward(&mut hidden, 1, hidden_size);
        self.lm_head.forward(device, &hidden, 1)
    }

    /// 貪欲デコード(argmax、サンプリング温度無し)で`max_new_tokens`個
    /// トークンを生成する。`prompt_ids`自体は出力に含めない
    /// (呼び出し側で連結すること)。
    pub fn generate(&self, device: &std::sync::Arc<dyn GpuDevice>, prompt_ids: &[u32], max_new_tokens: usize) -> Result<Vec<u32>> {
        anyhow::ensure!(!prompt_ids.is_empty(), "open-cuda-llm: prompt_ids must not be empty");
        let device_ref = device.as_ref();
        let mut caches = self.new_caches();

        let mut pos = 0usize;
        let mut logits = Vec::new();
        for &tok in prompt_ids {
            logits = self.forward_step(device_ref, tok, pos, &mut caches)?;
            pos += 1;
        }

        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut next = argmax(&logits);
        for _ in 0..max_new_tokens {
            generated.push(next);
            if pos >= self.config.max_seq_len {
                break;
            }
            logits = self.forward_step(device_ref, next, pos, &mut caches)?;
            pos += 1;
            next = argmax(&logits);
        }
        Ok(generated)
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
}
