//! # opencuda-bert
//!
//! BERT系エンコーダ(multilingual-e5-small等)のforward pass実装。
//! `opencuda-blas`の実カーネル(`sgemm`によるLinear層、
//! `scaled_dot_product_attention`によるマルチヘッドAttention)の上に
//! 積み上げた、文埋め込み(sentence embedding)専用のライブラリ。
//!
//! ## 位置づけ
//!
//! `aruaru-llm`(前回まで)は固定語彙へのbag-of-wordsドット積で意図分類を
//! 行っていた。本クレートは、学習済み埋め込みモデル(multilingual-e5-small、
//! MITライセンス、日本語含む100言語対応)を実際にロード・実行し、
//! 文の意味を反映した384次元ベクトルを返す。これにより`aruaru-llm`側は
//! bag-of-wordsからコサイン類似度ベースの意図分類へ移行できる。
//!
//! ## 正直な開示
//!
//! - **これはエンコーダ専用**(文の意味理解・分類・検索向け)であり、文章を
//!   生成する自己回帰デコーダ(いわゆる「LLM」の対話生成能力)は実装して
//!   いない。前回のロードマップで言う「1位vLLM相当」はまだ次の段階。
//! - Attentionは`opencuda-blas::scaled_dot_product_attention`(素朴な非
//!   タイル化実装)をそのまま利用しており、真のFlash Attentionではない。
//! - 推論経路はすべてCPUバックエンド(`opencuda_cpu::CpuDevice`)を想定。
//!   GPU専用パス(Vulkan/CUDA/ROCm)の`sgemm`はまだ未実装のため、本クレートも
//!   その制約を継承する。

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use opencuda_core::GpuDevice;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct BertConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    #[serde(default = "default_eps")]
    pub layer_norm_eps: f32,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
}

fn default_eps() -> f32 {
    1e-12
}

/// `y = x @ W^T + b` 相当のLinear層。`weight_t`はロード時に転置済み
/// (`in_dim x out_dim`、行優先)——safetensors側の`[out_dim, in_dim]`を
/// そのまま`opencuda_blas::sgemm`(`A(m×k)・B(k×n)`)へ渡すため。
struct Linear {
    weight_t: Vec<f32>,
    bias: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl Linear {
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
    /// `x`: `seq_len x dim`(行優先)を行ごとに正規化する(ホスト側、
    /// 行同士に依存が無いため並列化の余地はあるが hidden_size=384 程度の
    /// 小規模なため単純なループで十分)。
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

/// 厳密なGELU(`hidden_act: "gelu"`、tanh近似ではなくerfベース、
/// 元のBERT論文・実装と一致させる)。
fn gelu_inplace(x: &mut [f32]) {
    const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;
    for v in x.iter_mut() {
        let xf = *v as f64;
        *v = (0.5 * xf * (1.0 + libm::erf(xf * INV_SQRT2))) as f32;
    }
}

struct EncoderLayer {
    query: Linear,
    key: Linear,
    value: Linear,
    attn_out: Linear,
    attn_ln: LayerNorm,
    intermediate: Linear,
    output: Linear,
    output_ln: LayerNorm,
}

impl EncoderLayer {
    fn forward(&self, device: &dyn GpuDevice, hidden: &[f32], seq_len: usize, hidden_size: usize, num_heads: usize) -> Result<Vec<f32>> {
        let head_dim = hidden_size / num_heads;

        let q = self.query.forward(device, hidden, seq_len)?;
        let k = self.key.forward(device, hidden, seq_len)?;
        let v = self.value.forward(device, hidden, seq_len)?;

        // マルチヘッド分割: 各ヘッドの列区間を抜き出し、opencuda-blasの
        // scaled_dot_product_attention(単一ヘッド用)をヘッドごとに呼ぶ。
        let mut context = vec![0.0f32; seq_len * hidden_size];
        for h in 0..num_heads {
            let col_start = h * head_dim;
            let extract_head = |src: &[f32]| -> Vec<f32> {
                let mut buf = vec![0.0f32; seq_len * head_dim];
                for row in 0..seq_len {
                    buf[row * head_dim..(row + 1) * head_dim]
                        .copy_from_slice(&src[row * hidden_size + col_start..row * hidden_size + col_start + head_dim]);
                }
                buf
            };
            let q_h = extract_head(&q);
            let k_h = extract_head(&k);
            let v_h = extract_head(&v);
            let out_h = opencuda_blas::scaled_dot_product_attention(device, &q_h, &k_h, &v_h, seq_len, head_dim)?;
            for row in 0..seq_len {
                context[row * hidden_size + col_start..row * hidden_size + col_start + head_dim]
                    .copy_from_slice(&out_h[row * head_dim..(row + 1) * head_dim]);
            }
        }

        let mut attn_dense = self.attn_out.forward(device, &context, seq_len)?;
        for i in 0..attn_dense.len() {
            attn_dense[i] += hidden[i]; // residual
        }
        self.attn_ln.forward(&mut attn_dense, seq_len, hidden_size);

        let mut intermediate = self.intermediate.forward(device, &attn_dense, seq_len)?;
        gelu_inplace(&mut intermediate);

        let mut ffn_out = self.output.forward(device, &intermediate, seq_len)?;
        for i in 0..ffn_out.len() {
            ffn_out[i] += attn_dense[i]; // residual
        }
        self.output_ln.forward(&mut ffn_out, seq_len, hidden_size);

        Ok(ffn_out)
    }
}

pub struct BertModel {
    config: BertConfig,
    word_embeddings: Vec<f32>,
    position_embeddings: Vec<f32>,
    token_type_embeddings: Vec<f32>,
    emb_ln: LayerNorm,
    layers: Vec<EncoderLayer>,
}

fn tensor_f32(tensors: &safetensors::SafeTensors, name: &str) -> Result<Vec<f32>> {
    let view = tensors.tensor(name).with_context(|| format!("opencuda-bert: missing tensor '{name}'"))?;
    anyhow::ensure!(
        view.dtype() == safetensors::Dtype::F32,
        "opencuda-bert: tensor '{name}' has unexpected dtype {:?} (expected F32)",
        view.dtype()
    );
    let bytes = view.data();
    anyhow::ensure!(bytes.len() % 4 == 0, "opencuda-bert: tensor '{name}' byte length not a multiple of 4");
    let mut out = vec![0.0f32; bytes.len() / 4];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(out)
}

/// `[out_dim, in_dim]`(行優先、safetensorsのLinear重みの標準形)を
/// `[in_dim, out_dim]`へ転置する。384×384〜384×1536程度の小規模行列を
/// モデルロード時に1回だけ転置するため、素朴な二重ループで十分。
fn transpose(src: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; src.len()];
    for o in 0..out_dim {
        for i in 0..in_dim {
            out[i * out_dim + o] = src[o * in_dim + i];
        }
    }
    out
}

fn load_linear(tensors: &safetensors::SafeTensors, prefix: &str, in_dim: usize, out_dim: usize) -> Result<Linear> {
    let weight = tensor_f32(tensors, &format!("{prefix}.weight"))?;
    anyhow::ensure!(
        weight.len() == out_dim * in_dim,
        "opencuda-bert: '{prefix}.weight' has {} elements, expected {}x{}",
        weight.len(),
        out_dim,
        in_dim
    );
    let bias = tensor_f32(tensors, &format!("{prefix}.bias"))?;
    Ok(Linear { weight_t: transpose(&weight, out_dim, in_dim), bias, in_dim, out_dim })
}

fn load_layer_norm(tensors: &safetensors::SafeTensors, prefix: &str, eps: f32) -> Result<LayerNorm> {
    let weight = tensor_f32(tensors, &format!("{prefix}.weight"))?;
    let bias = tensor_f32(tensors, &format!("{prefix}.bias"))?;
    Ok(LayerNorm { weight, bias, eps })
}

impl BertModel {
    /// 埋め込みベクトルの次元数(=隠れ層サイズ)。呼び出し側が事前に
    /// ベクトルのバッファサイズを知りたい場合などに使う。
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// `dir`配下の`config.json`・`model.safetensors`を読み込む。
    pub fn load(dir: &Path) -> Result<Self> {
        let config_json = std::fs::read_to_string(dir.join("config.json"))
            .with_context(|| format!("opencuda-bert: failed to read config.json in {dir:?}"))?;
        let config: BertConfig = serde_json::from_str(&config_json).context("opencuda-bert: failed to parse config.json")?;

        let weights_bytes = std::fs::read(dir.join("model.safetensors"))
            .with_context(|| format!("opencuda-bert: failed to read model.safetensors in {dir:?}"))?;
        let tensors = safetensors::SafeTensors::deserialize(&weights_bytes).context("opencuda-bert: failed to parse model.safetensors")?;

        let hidden = config.hidden_size;
        let word_embeddings = tensor_f32(&tensors, "embeddings.word_embeddings.weight")?;
        let position_embeddings = tensor_f32(&tensors, "embeddings.position_embeddings.weight")?;
        let token_type_embeddings = tensor_f32(&tensors, "embeddings.token_type_embeddings.weight")?;
        let emb_ln = load_layer_norm(&tensors, "embeddings.LayerNorm", config.layer_norm_eps)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("encoder.layer.{i}");
            layers.push(EncoderLayer {
                query: load_linear(&tensors, &format!("{p}.attention.self.query"), hidden, hidden)?,
                key: load_linear(&tensors, &format!("{p}.attention.self.key"), hidden, hidden)?,
                value: load_linear(&tensors, &format!("{p}.attention.self.value"), hidden, hidden)?,
                attn_out: load_linear(&tensors, &format!("{p}.attention.output.dense"), hidden, hidden)?,
                attn_ln: load_layer_norm(&tensors, &format!("{p}.attention.output.LayerNorm"), config.layer_norm_eps)?,
                intermediate: load_linear(&tensors, &format!("{p}.intermediate.dense"), hidden, config.intermediate_size)?,
                output: load_linear(&tensors, &format!("{p}.output.dense"), config.intermediate_size, hidden)?,
                output_ln: load_layer_norm(&tensors, &format!("{p}.output.LayerNorm"), config.layer_norm_eps)?,
            });
        }

        Ok(Self { config, word_embeddings, position_embeddings, token_type_embeddings, emb_ln, layers })
    }

    /// トークンID列(単一シーケンス、パディング無し)から文埋め込みを計算する。
    /// 平均プーリング(全トークンの単純平均、E5系モデルの標準的な手法)+
    /// L2正規化を行う。`token_type_ids`は単一セグメント想定で常に0。
    pub fn embed_tokens(&self, device: &Arc<dyn GpuDevice>, token_ids: &[u32]) -> Result<Vec<f32>> {
        let seq_len = token_ids.len();
        anyhow::ensure!(seq_len > 0, "opencuda-bert: token_ids must not be empty");
        anyhow::ensure!(
            seq_len <= self.config.max_position_embeddings,
            "opencuda-bert: seq_len {seq_len} exceeds max_position_embeddings {}",
            self.config.max_position_embeddings
        );

        let hidden_size = self.config.hidden_size;
        let mut hidden = vec![0.0f32; seq_len * hidden_size];
        for (row, &tok) in token_ids.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(tok < self.config.vocab_size, "opencuda-bert: token id {tok} out of vocab range");
            for c in 0..hidden_size {
                hidden[row * hidden_size + c] = self.word_embeddings[tok * hidden_size + c]
                    + self.position_embeddings[row * hidden_size + c]
                    + self.token_type_embeddings[c]; // token_type=0の行
            }
        }
        self.emb_ln.forward(&mut hidden, seq_len, hidden_size);

        let device_ref = device.as_ref();
        for layer in &self.layers {
            hidden = layer.forward(device_ref, &hidden, seq_len, hidden_size, self.config.num_attention_heads)?;
        }

        // 平均プーリング(パディング無し前提のため全トークンを対象)。
        let mut pooled = vec![0.0f32; hidden_size];
        for row in 0..seq_len {
            for c in 0..hidden_size {
                pooled[c] += hidden[row * hidden_size + c];
            }
        }
        for v in pooled.iter_mut() {
            *v /= seq_len as f32;
        }

        // L2正規化。
        let norm: f32 = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in pooled.iter_mut() {
                *v /= norm;
            }
        }
        Ok(pooled)
    }
}

/// 2つのL2正規化済みベクトルのコサイン類似度(=内積)を計算する。
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// `tokenizer.json`(Hugging Face fast tokenizer形式)をロードする薄いラッパー。
pub struct BertTokenizer {
    inner: tokenizers::Tokenizer,
}

impl BertTokenizer {
    pub fn load(dir: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("opencuda-bert: failed to load tokenizer.json: {e}"))?;
        Ok(Self { inner })
    }

    /// `add_special_tokens=true`でエンコードし、トークンID列を返す。
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("opencuda-bert: tokenizer encode failed: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }
}

/// テキストを直接埋め込みベクトルへ変換する便利関数
/// (`BertTokenizer::encode` → [`BertModel::embed_tokens`])。
pub fn embed_text(model: &BertModel, tokenizer: &BertTokenizer, device: &Arc<dyn GpuDevice>, text: &str) -> Result<Vec<f32>> {
    let ids = tokenizer.encode(text)?;
    model.embed_tokens(device, &ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencuda_cpu::CpuDevice;
    use std::path::PathBuf;

    fn model_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../aruaru-llm/models/multilingual-e5-small")
    }

    #[test]
    fn embeds_text_as_unit_length_vector_of_hidden_size() {
        let dir = model_dir();
        if !dir.join("model.safetensors").exists() {
            eprintln!("skipping: model not present at {dir:?}");
            return;
        }
        let model = BertModel::load(&dir).unwrap();
        let tokenizer = BertTokenizer::load(&dir).unwrap();
        let device: Arc<dyn GpuDevice> = CpuDevice::new(0);
        let emb = embed_text(&model, &tokenizer, &device, "query: 申請の手続きについて教えてください").unwrap();
        assert_eq!(emb.len(), model.config.hidden_size);
        let norm: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "norm should be ~1.0 after L2 normalize, got {norm}");
    }

    #[test]
    fn similar_sentences_score_higher_than_unrelated_ones() {
        let dir = model_dir();
        if !dir.join("model.safetensors").exists() {
            eprintln!("skipping: model not present at {dir:?}");
            return;
        }
        let model = BertModel::load(&dir).unwrap();
        let tokenizer = BertTokenizer::load(&dir).unwrap();
        let device: Arc<dyn GpuDevice> = CpuDevice::new(0);

        let query = embed_text(&model, &tokenizer, &device, "query: マイナンバーカードの申請をしたい").unwrap();
        let gov = embed_text(&model, &tokenizer, &device, "passage: 行政手続き・マイナンバーに関するご案内").unwrap();
        let unrelated = embed_text(&model, &tokenizer, &device, "passage: 今日の天気は晴れです").unwrap();

        let sim_gov = cosine_similarity(&query, &gov);
        let sim_unrelated = cosine_similarity(&query, &unrelated);
        assert!(sim_gov > sim_unrelated, "expected gov-related similarity ({sim_gov}) > unrelated ({sim_unrelated})");
    }
}
