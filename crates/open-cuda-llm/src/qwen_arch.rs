//! Qwen2/Qwen2.5系アーキテクチャ(RoPE・Grouped Query Attention・RMSNorm・
//! SwiGLU MLP)のforward pass実装(2026-09-11新設)。
//!
//! ## 経緯・正直な開示
//!
//! `GptModel`(このクレートの既存の中核)は GPT-2 アーキテクチャ専用
//! (`Conv1D`の重み配置・LayerNorm・GELU)で、Llama/Qwen 系
//! (RoPE・Grouped Query Attention・RMSNorm・SwiGLU)の重みは原理的に
//! ロードできない——このモジュールはその制約を解消するために、
//! **既存の `GptModel` 経路には一切手を触れず**、並行する新しい
//! アーキテクチャ経路として追加する(`Linear`/`LayerNorm` とは別に
//! `RmsNorm` を新設、Attention 自体は既存の
//! `opencuda_blas::scaled_dot_product_attention` を再利用)。
//!
//! ## スコープの限界(誇張しない)
//!
//! - 対応するのは **Qwen2/Qwen2.5 系の dense(非MoE)アーキテクチャ**の
//!   みで、2026年9月に話題になった Qwen3.5 系・DeepSeek-V4.1-Flash
//!   (Mixture-of-Experts、Causal Encoder-Decoder、FP4 KVキャッシュ圧縮
//!   等の新設計)には**まだ対応していない**。これらは全く別の計算
//!   グラフ(ルーティング・専門家選択・エンコーダ/デコーダ分割)が
//!   必要で、今回の変更の範囲を大きく超える。まずは同じ「RoPE+GQA+
//!   RMSNorm+SwiGLU」系統の基盤を用意し、実在する小型モデル
//!   (Qwen2.5-0.5B 等)で end-to-end に動くことを検証する第一段階。
//! - この開発機(GT730、VRAM 2GB)では大型モデルはそもそも実行不可能
//!   なため、実重みでの検証は小型モデル限定になる。
//! - PagedAttention・連続バッチング等、`GptModel`同様に本家vLLM相当の
//!   最適化は無い(単一シーケンス・KVキャッシュ付き逐次デコードのみ)。
//! - `tokenizer.json` は既存の[`crate::GptTokenizer`]（`tokenizers`
//!   クレート、HF公式実装）をそのまま流用できる(Qwenも同じ
//!   `tokenizer.json` 形式のため新規実装は不要)。

use std::path::Path;

use anyhow::{ensure, Context, Result};
use opencuda_core::GpuDevice;

use super::{argmax, apply_repetition_penalty, pca_top_directions, random_vec, tensor_f32, transpose, KvCacheHead, Linear, MlaHeadProjection, SplitMix64};

/// Qwen2/Qwen2.5 dense アーキテクチャの設定。`config.json` の該当
/// フィールドにそのまま対応する。
#[derive(Debug, Clone, serde::Deserialize)]
pub struct QwenConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    #[serde(rename = "num_hidden_layers")]
    pub num_layers: usize,
    #[serde(rename = "num_attention_heads")]
    pub num_heads: usize,
    /// Grouped Query Attention: クエリヘッド数より少ない(または同数=
    /// 通常のMulti-Head Attention)キー/バリューヘッド数。
    #[serde(rename = "num_key_value_heads")]
    pub num_kv_heads: usize,
    pub intermediate_size: usize,
    #[serde(default = "default_max_seq_len")]
    #[serde(rename = "max_position_embeddings")]
    pub max_seq_len: usize,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_max_seq_len() -> usize {
    32768
}
fn default_rms_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    1_000_000.0
}

impl QwenConfig {
    /// テスト・デモ用の極小構成(実運用サイズではない)。
    pub fn tiny(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2, // GQA: 4クエリヘッドを2KVヘッドで共有(ヘッドあたり2本)
            intermediate_size: 64,
            max_seq_len: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            tie_word_embeddings: true,
        }
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

struct RmsNorm {
    weight: Vec<f32>,
    eps: f32,
}

impl RmsNorm {
    fn identity(dim: usize, eps: f32) -> Self {
        Self { weight: vec![1.0; dim], eps }
    }

    /// 1行(`dim`長)を正規化する(`forward_step`が1トークンずつ処理する
    /// ため、既存`LayerNorm::forward`のような複数行版は不要)。
    fn forward_row(&self, x: &mut [f32]) {
        let dim = x.len();
        let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv_rms = 1.0 / (ms + self.eps).sqrt();
        for (v, w) in x.iter_mut().zip(&self.weight) {
            *v = *v * inv_rms * w;
        }
    }
}

/// 1 行のロジットを対数確率(log-softmax)にする。
fn log_softmax_row(row: &[f32]) -> Vec<f32> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = row.iter().map(|v| (v - max).exp()).sum();
    let lse = max + sum.ln();
    row.iter().map(|v| v - lse).collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// RoPE(Rotary Position Embedding)の回転角(`cos`/`sin`、各`head_dim/2`長)を
/// 位置`pos`について事前計算する。HF Llama/Qwen実装と同じ
/// "rotate_half"方式(前半次元と後半次元のペアを回転させる)。
fn rope_cos_sin(head_dim: usize, theta: f32, pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; half];
    let mut sin = vec![0.0f32; half];
    for i in 0..half {
        let freq = 1.0f32 / theta.powf((2 * i) as f32 / head_dim as f32);
        let angle = pos as f32 * freq;
        cos[i] = angle.cos();
        sin[i] = angle.sin();
    }
    (cos, sin)
}

/// 1ヘッドぶん(`head_dim`長)のq/kベクトルへ"rotate_half" RoPEを適用する。
fn apply_rope(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = cos.len();
    debug_assert_eq!(x.len(), half * 2);
    for i in 0..half {
        let x1 = x[i];
        let x2 = x[i + half];
        x[i] = x1 * cos[i] - x2 * sin[i];
        x[i + half] = x2 * cos[i] + x1 * sin[i];
    }
}

struct QwenLayer {
    input_layernorm: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    post_attention_layernorm: RmsNorm,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    /// **2026-09-11新設**: [`QwenModel::enable_mla_kv_compression`]配線先。
    /// `GptModel`の同名機構(`DecoderLayer::mla`)の移植だが、**KVヘッド
    /// 単位**(GQAでは`num_kv_heads` < `num_heads`)で持つ点が異なる——
    /// GPT-2にはGQAが無く`num_heads`個持っていたのに対し、こちらは
    /// 実際にKVキャッシュを持つヘッド数ぶんだけで足りる(かつそれが
    /// 正しい: 複数のクエリヘッドが同じKVヘッドのキャッシュを共有する
    /// ため、圧縮もKVヘッド単位で行うのが筋が通る)。`None`(既定)なら
    /// 従来通りフル精度でKVキャッシュを保持する(後方互換)。
    mla: Option<Vec<MlaHeadProjection>>,
}

impl QwenLayer {
    fn random(rng: &mut SplitMix64, cfg: &QwenConfig) -> Self {
        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.num_kv_heads * head_dim;
        Self {
            input_layernorm: RmsNorm::identity(hidden, cfg.rms_norm_eps),
            q_proj: Linear::random(rng, hidden, hidden),
            k_proj: Linear::random(rng, hidden, kv_dim),
            v_proj: Linear::random(rng, hidden, kv_dim),
            o_proj: Linear::random(rng, hidden, hidden),
            post_attention_layernorm: RmsNorm::identity(hidden, cfg.rms_norm_eps),
            gate_proj: Linear::random(rng, hidden, cfg.intermediate_size),
            up_proj: Linear::random(rng, hidden, cfg.intermediate_size),
            down_proj: Linear::random(rng, cfg.intermediate_size, hidden),
            mla: None,
        }
    }
}

/// `QwenLayer`ごとのKVキャッシュ(KVヘッド数ぶん、GQAではクエリヘッド数より
/// 少ない)。
struct LayerCache {
    kv: Vec<KvCacheHead>,
}

impl LayerCache {
    fn new(num_kv_heads: usize) -> Self {
        Self { kv: (0..num_kv_heads).map(|_| KvCacheHead::empty()).collect() }
    }
}

pub struct QwenModel {
    config: QwenConfig,
    embed_tokens: Vec<f32>, // [vocab_size, hidden_size] 行優先(embeddingルックアップ用、転置不要)
    layers: Vec<QwenLayer>,
    norm: RmsNorm,
    /// `tie_word_embeddings=true`なら`None`(`embed_tokens`を転置して使う)。
    lm_head: Option<Linear>,
}

impl QwenModel {
    pub fn load_random(config: QwenConfig, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let embed_tokens = random_vec(&mut rng, config.vocab_size * config.hidden_size, 0.02);
        let layers = (0..config.num_layers).map(|_| QwenLayer::random(&mut rng, &config)).collect();
        let norm = RmsNorm::identity(config.hidden_size, config.rms_norm_eps);
        let lm_head = if config.tie_word_embeddings { None } else { Some(Linear::random(&mut rng, config.hidden_size, config.vocab_size)) };
        Self { config, embed_tokens, layers, norm, lm_head }
    }

    /// 実在の学習済み重み(Qwen2/Qwen2.5、`config.json` + 単一ファイルの
    /// `model.safetensors`)を読み込む。**正直な開示**: 分割済み
    /// (`model-00001-of-00002.safetensors`等)のsharded checkpointには
    /// 未対応(`GptModel::load`と同じ制約——単一ファイルのモデルのみ)。
    pub fn load(dir: &Path) -> Result<Self> {
        let config_bytes = std::fs::read(dir.join("config.json")).with_context(|| format!("open-cuda-llm: failed to read {}/config.json", dir.display()))?;
        let config: QwenConfig = serde_json::from_slice(&config_bytes).context("open-cuda-llm: failed to parse Qwen config.json")?;
        ensure!(
            config.num_heads > 0 && config.num_kv_heads > 0 && config.num_heads % config.num_kv_heads == 0,
            "open-cuda-llm: num_attention_heads ({}) must be a positive multiple of num_key_value_heads ({})",
            config.num_heads,
            config.num_kv_heads
        );
        ensure!(config.hidden_size % config.num_heads == 0, "open-cuda-llm: hidden_size ({}) must be divisible by num_attention_heads ({})", config.hidden_size, config.num_heads);

        let weights_path = dir.join("model.safetensors");
        let data = std::fs::read(&weights_path).with_context(|| format!("open-cuda-llm: failed to read {}", weights_path.display()))?;
        let tensors = safetensors::SafeTensors::deserialize(&data).context("open-cuda-llm: failed to parse model.safetensors")?;

        let hidden = config.hidden_size;
        let head_dim = config.head_dim();
        let kv_dim = config.num_kv_heads * head_dim;

        let embed_tokens = tensor_f32(&tensors, "model.embed_tokens.weight")?;
        ensure!(embed_tokens.len() == config.vocab_size * hidden, "open-cuda-llm: model.embed_tokens.weight has {} elements, expected {}x{}", embed_tokens.len(), config.vocab_size, hidden);

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("model.layers.{i}");
            layers.push(QwenLayer {
                input_layernorm: RmsNorm { weight: tensor_f32(&tensors, &format!("{p}.input_layernorm.weight"))?, eps: config.rms_norm_eps },
                q_proj: load_linear(&tensors, &format!("{p}.self_attn.q_proj"), hidden, hidden)?,
                k_proj: load_linear(&tensors, &format!("{p}.self_attn.k_proj"), hidden, kv_dim)?,
                v_proj: load_linear(&tensors, &format!("{p}.self_attn.v_proj"), hidden, kv_dim)?,
                o_proj: load_linear(&tensors, &format!("{p}.self_attn.o_proj"), hidden, hidden)?,
                post_attention_layernorm: RmsNorm { weight: tensor_f32(&tensors, &format!("{p}.post_attention_layernorm.weight"))?, eps: config.rms_norm_eps },
                gate_proj: load_linear(&tensors, &format!("{p}.mlp.gate_proj"), hidden, config.intermediate_size)?,
                up_proj: load_linear(&tensors, &format!("{p}.mlp.up_proj"), hidden, config.intermediate_size)?,
                down_proj: load_linear(&tensors, &format!("{p}.mlp.down_proj"), config.intermediate_size, hidden)?,
                mla: None,
            });
        }

        let norm = RmsNorm { weight: tensor_f32(&tensors, "model.norm.weight")?, eps: config.rms_norm_eps };

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            let raw = tensor_f32(&tensors, "lm_head.weight")?; // [vocab_size, hidden] 行優先(標準nn.Linear)
            ensure!(raw.len() == config.vocab_size * hidden, "open-cuda-llm: lm_head.weight has {} elements, expected {}x{}", raw.len(), config.vocab_size, hidden);
            let weight_t = transpose(&raw, config.vocab_size, hidden); // -> [hidden, vocab_size]
            Some(Linear { weight_t, bias: vec![0.0; config.vocab_size], in_dim: hidden, out_dim: config.vocab_size, spirv_matmul: None, dxil_offload: None, fp8_weight: None })
        };

        Ok(Self { config, embed_tokens, layers, norm, lm_head })
    }

    /// KVキャッシュをヘッドあたり`head_dim`次元から`d_c`次元(`d_c < head_dim`)
    /// へ低ランク圧縮する(`GptModel::enable_mla_kv_compression`の移植、
    /// `QwenLayer::mla`のモジュールdoc参照——GQAではKVヘッド単位で行う)。
    ///
    /// **正直な開示(`GptModel`側の既存知見をそのまま引き継ぐ)**: ここでの
    /// 射影は乱数(`SplitMix64`)によるものであり、実際のK/V活性化統計に
    /// 基づくPCA較正版(`GptModel::enable_mla_kv_compression_calibrated`)
    /// ではない。`GptModel`側では実重みで乱数射影が生成品質を明確に
    /// 劣化させることが実測されている(モジュールdoc該当箇所参照)——
    /// `QwenModel`でも同様の劣化が起きる可能性が高く、圧縮率とのトレード
    /// オフを事前に検証してから使うこと。PCA較正版の`QwenModel`移植は
    /// 今回のスコープ外(次の増分)。
    pub fn enable_mla_kv_compression(&mut self, d_c: usize, seed: u64) -> Result<()> {
        let head_dim = self.config.head_dim();
        ensure!(d_c > 0 && d_c < head_dim, "open-cuda-llm: QwenModel::enable_mla_kv_compression: d_c={d_c} must satisfy 0 < d_c < head_dim={head_dim}");
        let mut rng = SplitMix64::new(seed);
        for layer in &mut self.layers {
            let projections = (0..self.config.num_kv_heads)
                .map(|_| MlaHeadProjection { down_proj: random_vec(&mut rng, head_dim * d_c, 0.02), up_proj: random_vec(&mut rng, d_c * head_dim, 0.02), d_c })
                .collect();
            layer.mla = Some(projections);
        }
        Ok(())
    }

    /// [`enable_mla_kv_compression`]のPCA較正版(`GptModel::
    /// enable_mla_kv_compression_calibrated`の移植、2026-09-13)。
    ///
    /// 前回のHANDOFFで「今回のスコープ外(次の増分)」と明記していた
    /// PCA較正版を、`GptModel`側で確立済みの設計(`open-cuda-llm/src/
    /// lib.rs`の`pca_top_directions`、非中心〈uncentered〉PCA、
    /// K/V活性化を縦連結して単一基底を求める、直交基底のため
    /// `up_proj=down_proj`の転置)をそのまま`QwenModel`(GQA、KVヘッド
    /// 単位で射影)へ適用する。
    ///
    /// `GptModel`版は`forward_prefill_all_layers`(バッチ処理版)で
    /// 較正データを集めるが、`QwenModel`にはまだその一括版が無いため、
    /// `forward_step`をプロンプトのトークン数ぶん逐次呼び出して
    /// キャッシュを埋める(結果は`GptModel`版と数学的に同じ——単に
    /// 「1トークンずつ」か「まとめて」かの実行効率の違いのみ、
    /// 較正に使う活性化統計そのものは変わらない)。
    pub fn enable_mla_kv_compression_calibrated(&mut self, d_c: usize, device: &dyn GpuDevice, sample_prompts: &[Vec<u32>]) -> Result<()> {
        let head_dim = self.config.head_dim();
        ensure!(d_c > 0 && d_c < head_dim, "open-cuda-llm: QwenModel::enable_mla_kv_compression_calibrated: d_c={d_c} must satisfy 0 < d_c < head_dim={head_dim}");
        ensure!(!sample_prompts.is_empty(), "open-cuda-llm: QwenModel::enable_mla_kv_compression_calibrated: sample_prompts must not be empty");
        ensure!(
            self.layers.iter().all(|l| l.mla.is_none()),
            "open-cuda-llm: QwenModel::enable_mla_kv_compression_calibrated: some layers already have MLA compression \
             enabled (calibration must run against the uncompressed model, otherwise it would collect already-lossy \
             latents instead of real full-precision activations)"
        );

        let num_layers = self.config.num_layers;
        let num_kv_heads = self.config.num_kv_heads;
        let mut per_layer_head_rows: Vec<Vec<Vec<f32>>> = (0..num_layers).map(|_| vec![Vec::new(); num_kv_heads]).collect();

        for prompt in sample_prompts {
            ensure!(!prompt.is_empty(), "open-cuda-llm: QwenModel::enable_mla_kv_compression_calibrated: calibration prompt must not be empty");
            let mut caches = self.new_caches();
            for &token_id in prompt {
                self.forward_step(device, token_id, &mut caches)?;
            }
            for (layer_idx, layer_caches) in caches.into_iter().enumerate() {
                for (head_idx, cache_head) in layer_caches.kv.into_iter().enumerate() {
                    let bucket = &mut per_layer_head_rows[layer_idx][head_idx];
                    // proj=NoneでのforwardなのでKvCacheHead.k/.vはフル精度
                    // (KvCacheHead::pushのdocコメント参照、GptModel版と同じ)。
                    bucket.extend_from_slice(&cache_head.k);
                    bucket.extend_from_slice(&cache_head.v);
                }
            }
        }

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let mut projections = Vec::with_capacity(num_kv_heads);
            for (head_idx, rows_flat) in per_layer_head_rows[layer_idx].iter().enumerate() {
                let num_rows = rows_flat.len() / head_dim;
                ensure!(
                    num_rows >= d_c,
                    "open-cuda-llm: QwenModel::enable_mla_kv_compression_calibrated: only {num_rows} calibration rows \
                     collected for layer {layer_idx} kv_head {head_idx}, need >= d_c={d_c} for a meaningful PCA basis \
                     (pass longer/more sample prompts)"
                );
                let (down_proj, up_proj) = pca_top_directions(rows_flat, num_rows, head_dim, d_c);
                projections.push(MlaHeadProjection { down_proj, up_proj, d_c });
            }
            layer.mla = Some(projections);
        }
        Ok(())
    }

    fn new_caches(&self) -> Vec<LayerCache> {
        (0..self.config.num_layers).map(|_| LayerCache::new(self.config.num_kv_heads)).collect()
    }

    /// 1トークンぶんを処理し、次トークン予測のlogits(`vocab_size`長)を返す。
    /// `caches`は呼び出し側が保持し続ける(逐次デコードのKVキャッシュ、
    /// `GptModel::forward_step`と同じ設計)。
    fn forward_step(&self, device: &dyn GpuDevice, token_id: u32, caches: &mut [LayerCache]) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let heads_per_kv = cfg.num_heads / cfg.num_kv_heads;
        let pos = caches[0].kv[0].n;
        let (cos, sin) = rope_cos_sin(head_dim, cfg.rope_theta, pos);

        let tok = token_id as usize;
        ensure!(tok < cfg.vocab_size, "open-cuda-llm: token id {tok} out of vocab range {}", cfg.vocab_size);
        let mut hidden_state = self.embed_tokens[tok * hidden..(tok + 1) * hidden].to_vec();

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            // ---- Attention サブ層(pre-norm) ----
            let mut normed = hidden_state.clone();
            layer.input_layernorm.forward_row(&mut normed);

            let q = layer.q_proj.forward(device, &normed, 1)?;
            let k = layer.k_proj.forward(device, &normed, 1)?;
            let v = layer.v_proj.forward(device, &normed, 1)?;

            // KVヘッドごとにRoPEを適用してからキャッシュへpush。
            let mut k_heads_rot = Vec::with_capacity(cfg.num_kv_heads);
            for kvh in 0..cfg.num_kv_heads {
                let mut k_h = k[kvh * head_dim..(kvh + 1) * head_dim].to_vec();
                apply_rope(&mut k_h, &cos, &sin);
                let v_h = &v[kvh * head_dim..(kvh + 1) * head_dim];
                let proj = layer.mla.as_ref().map(|v| &v[kvh]);
                cache.kv[kvh].push(device, &k_h, v_h, proj, None)?;
                k_heads_rot.push(k_h);
            }

            let mut context = vec![0.0f32; hidden];
            for qh in 0..cfg.num_heads {
                let kvh = qh / heads_per_kv;
                let mut q_h = q[qh * head_dim..(qh + 1) * head_dim].to_vec();
                apply_rope(&mut q_h, &cos, &sin);

                let proj = layer.mla.as_ref().map(|v| &v[kvh]);
                let (k_all, v_all) = cache.kv[kvh].current_kv(device, head_dim, proj, None)?;
                let n = cache.kv[kvh].n;
                // `scaled_dot_product_attention`はseq_len行のqを要求するため、
                // 単一のクエリ行をn回複製して先頭行だけを使う
                // (`GptModel::forward_step`と同じ手法、コメント参照)。
                let mut q_full = vec![0.0f32; n * head_dim];
                for row in q_full.chunks_exact_mut(head_dim) {
                    row.copy_from_slice(&q_h);
                }
                let out = opencuda_blas::scaled_dot_product_attention(device, &q_full, &k_all, &v_all, n, head_dim)?;
                context[qh * head_dim..(qh + 1) * head_dim].copy_from_slice(&out[0..head_dim]);
            }

            let attn_out = layer.o_proj.forward(device, &context, 1)?;
            for (h, a) in hidden_state.iter_mut().zip(&attn_out) {
                *h += a;
            }

            // ---- MLP(SwiGLU)サブ層(pre-norm) ----
            let mut normed2 = hidden_state.clone();
            layer.post_attention_layernorm.forward_row(&mut normed2);
            let gate = layer.gate_proj.forward(device, &normed2, 1)?;
            let up = layer.up_proj.forward(device, &normed2, 1)?;
            let mut mlp_hidden = vec![0.0f32; gate.len()];
            for i in 0..gate.len() {
                mlp_hidden[i] = silu(gate[i]) * up[i];
            }
            let mlp_out = layer.down_proj.forward(device, &mlp_hidden, 1)?;
            for (h, m) in hidden_state.iter_mut().zip(&mlp_out) {
                *h += m;
            }
        }

        self.norm.forward_row(&mut hidden_state);

        let logits = match &self.lm_head {
            Some(head) => head.forward(device, &hidden_state, 1)?,
            None => {
                // weight tying: embed_tokens([vocab,hidden]) を転置せず、
                // 素朴な内積で vocab_size ぶんのlogitsを計算する
                // (毎トークン transpose するより軽い)。
                let mut logits = vec![0.0f32; cfg.vocab_size];
                for (v, row) in logits.iter_mut().zip(self.embed_tokens.chunks_exact(hidden)) {
                    *v = row.iter().zip(&hidden_state).map(|(a, b)| a * b).sum();
                }
                logits
            }
        };
        Ok(logits)
    }

    // ---- 文脈つき候補の採点(OCR の穴埋めなど、短い穴を前後の文脈で埋める用途) ----

    /// `token_ids` を一括(バッチ)で処理する。KV キャッシュには `caches` が持つ長さの続きとして追加され、
    /// **全位置**の次トークン予測のロジット(`token_ids.len() * vocab_size` のフラット配列)を返す。
    /// 1 トークンずつ処理する `forward_step` と数値的に同じ結果になる(試験で確認)が、重み行列の読み出しを
    /// 全トークンで共有できるため、プロンプトが長いほど速い。MLA 圧縮が有効なモデルは未対応。
    fn forward_prefill_per_position(&self, device: &dyn GpuDevice, token_ids: &[u32], caches: &mut [LayerCache]) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let heads_per_kv = cfg.num_heads / cfg.num_kv_heads;
        let seq = token_ids.len();
        ensure!(seq > 0, "open-cuda-llm: token_ids must not be empty");
        ensure!(self.layers.iter().all(|l| l.mla.is_none()), "open-cuda-llm: batched prefill does not support MLA-compressed KV caches");
        let pos0 = caches[0].kv[0].n;
        ensure!(pos0 + seq <= cfg.max_seq_len, "open-cuda-llm: sequence too long ({} > {})", pos0 + seq, cfg.max_seq_len);

        let mut x = vec![0.0f32; seq * hidden];
        for (i, &t) in token_ids.iter().enumerate() {
            let tok = t as usize;
            ensure!(tok < cfg.vocab_size, "open-cuda-llm: token id {tok} out of vocab range {}", cfg.vocab_size);
            x[i * hidden..(i + 1) * hidden].copy_from_slice(&self.embed_tokens[tok * hidden..(tok + 1) * hidden]);
        }
        let rope: Vec<(Vec<f32>, Vec<f32>)> = (0..seq).map(|i| rope_cos_sin(head_dim, cfg.rope_theta, pos0 + i)).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            let mut normed = x.clone();
            for row in normed.chunks_exact_mut(hidden) {
                layer.input_layernorm.forward_row(row);
            }
            let q = layer.q_proj.forward(device, &normed, seq)?;
            let k = layer.k_proj.forward(device, &normed, seq)?;
            let v = layer.v_proj.forward(device, &normed, seq)?;
            let kv_dim = cfg.num_kv_heads * head_dim;
            for i in 0..seq {
                let (cos, sin) = &rope[i];
                for kvh in 0..cfg.num_kv_heads {
                    let mut k_h = k[i * kv_dim + kvh * head_dim..i * kv_dim + (kvh + 1) * head_dim].to_vec();
                    apply_rope(&mut k_h, cos, sin);
                    cache.kv[kvh].push(device, &k_h, &v[i * kv_dim + kvh * head_dim..i * kv_dim + (kvh + 1) * head_dim], None, None)?;
                }
            }
            // 因果的アテンション: 位置 i のクエリは、キャッシュ先頭から pos0+i までを見る。
            let mut context = vec![0.0f32; seq * hidden];
            for qh in 0..cfg.num_heads {
                let kvh = qh / heads_per_kv;
                let kc = &cache.kv[kvh].k;
                let vc = &cache.kv[kvh].v;
                for i in 0..seq {
                    let mut q_h = q[i * hidden + qh * head_dim..i * hidden + (qh + 1) * head_dim].to_vec();
                    apply_rope(&mut q_h, &rope[i].0, &rope[i].1);
                    let n = pos0 + i + 1;
                    let mut scores = vec![0.0f32; n];
                    let mut max = f32::NEG_INFINITY;
                    for (j, sc) in scores.iter_mut().enumerate() {
                        let kr = &kc[j * head_dim..(j + 1) * head_dim];
                        *sc = q_h.iter().zip(kr).map(|(a, b)| a * b).sum::<f32>() * scale;
                        max = max.max(*sc);
                    }
                    let mut sum = 0.0f32;
                    for sc in scores.iter_mut() {
                        *sc = (*sc - max).exp();
                        sum += *sc;
                    }
                    let out = &mut context[i * hidden + qh * head_dim..i * hidden + (qh + 1) * head_dim];
                    for (j, sc) in scores.iter().enumerate() {
                        let w = sc / sum;
                        for (o, vv) in out.iter_mut().zip(&vc[j * head_dim..(j + 1) * head_dim]) {
                            *o += w * vv;
                        }
                    }
                }
            }
            let attn_out = layer.o_proj.forward(device, &context, seq)?;
            for (h, a) in x.iter_mut().zip(&attn_out) {
                *h += a;
            }
            let mut normed2 = x.clone();
            for row in normed2.chunks_exact_mut(hidden) {
                layer.post_attention_layernorm.forward_row(row);
            }
            let gate = layer.gate_proj.forward(device, &normed2, seq)?;
            let up = layer.up_proj.forward(device, &normed2, seq)?;
            let mlp_hidden: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
            let mlp_out = layer.down_proj.forward(device, &mlp_hidden, seq)?;
            for (h, m) in x.iter_mut().zip(&mlp_out) {
                *h += m;
            }
        }
        for row in x.chunks_exact_mut(hidden) {
            self.norm.forward_row(row);
        }
        match &self.lm_head {
            Some(head) => head.forward(device, &x, seq),
            None => Ok(self.tied_logits(&x, seq)),
        }
    }

    /// 重み共有(tie_word_embeddings)の語彙ロジット。語彙が大きい(15 万語)ので、スレッドで分担する。
    fn tied_logits(&self, x: &[f32], seq: usize) -> Vec<f32> {
        let hidden = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let mut logits = vec![0.0f32; seq * vocab];
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(16);
        let chunk = vocab.div_ceil(threads);
        // 各スレッドが語彙の一部(列の範囲)を、全位置について計算する。
        let parts: Vec<(usize, Vec<f32>)> = std::thread::scope(|sc| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let (lo, hi) = ((t * chunk).min(vocab), ((t + 1) * chunk).min(vocab));
                    sc.spawn(move || {
                        let mut part = vec![0.0f32; seq * (hi - lo)];
                        for (vi, row) in self.embed_tokens[lo * hidden..hi * hidden].chunks_exact(hidden).enumerate() {
                            for i in 0..seq {
                                part[i * (hi - lo) + vi] = row.iter().zip(&x[i * hidden..(i + 1) * hidden]).map(|(a, b)| a * b).sum();
                            }
                        }
                        (lo, part)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap_or_default()).collect()
        });
        for (lo, part) in parts {
            if part.is_empty() {
                continue;
            }
            let w = part.len() / seq;
            for i in 0..seq {
                logits[i * vocab + lo..i * vocab + lo + w].copy_from_slice(&part[i * w..(i + 1) * w]);
            }
        }
        logits
    }

    fn truncate_caches(&self, caches: &mut [LayerCache], new_n: usize) {
        let head_dim = self.config.head_dim();
        for layer in caches.iter_mut() {
            for head in layer.kv.iter_mut() {
                head.truncate(new_n, head_dim);
            }
        }
    }

    /// 文脈 `prefix_ids` のあとに各 `continuations[i]` が続く確率の**対数(自然対数の和)**を返す。
    /// OCR でかすれた箇所の候補(候補の文字列+そのすぐ後ろの文脈)を、前の文脈つきで採点する用途。
    /// 前の文脈の KV キャッシュは 1 回だけ作り、候補ごとに巻き戻して再利用する。
    /// `prefix_ids` は空にできない(先頭トークンに文脈が要る。呼び出し側で先頭語を付ける)。
    pub fn score_continuations(&self, device: &std::sync::Arc<dyn GpuDevice>, prefix_ids: &[u32], continuations: &[Vec<u32>]) -> Result<Vec<f32>> {
        ensure!(!prefix_ids.is_empty(), "open-cuda-llm: prefix must not be empty");
        let device_ref = device.as_ref();
        let vocab = self.config.vocab_size;
        let mut caches = self.new_caches();
        let prefix_logits = self.forward_prefill_per_position(device_ref, prefix_ids, &mut caches)?;
        let last_prefix = log_softmax_row(&prefix_logits[(prefix_ids.len() - 1) * vocab..prefix_ids.len() * vocab]);
        let mut out = Vec::with_capacity(continuations.len());
        for cont in continuations {
            if cont.is_empty() {
                out.push(0.0);
                continue;
            }
            self.truncate_caches(&mut caches, prefix_ids.len());
            let logits = self.forward_prefill_per_position(device_ref, cont, &mut caches)?;
            let mut lp = last_prefix[cont[0] as usize];
            for i in 1..cont.len() {
                let row = log_softmax_row(&logits[(i - 1) * vocab..i * vocab]);
                lp += row[cont[i] as usize];
            }
            out.push(lp);
        }
        Ok(out)
    }

    /// 文章の各トークンの対数確率(直前までの文脈つき)を返す。先頭トークンには文脈が無いので `0.0`。
    /// 「この文字の並びは不自然(確率が極端に低い)」という OCR の誤読の検出に使う。1 回のバッチ prefill で済む。
    pub fn token_logprobs(&self, device: &std::sync::Arc<dyn GpuDevice>, ids: &[u32]) -> Result<Vec<f32>> {
        ensure!(!ids.is_empty(), "open-cuda-llm: ids must not be empty");
        let vocab = self.config.vocab_size;
        let mut caches = self.new_caches();
        let logits = self.forward_prefill_per_position(device.as_ref(), ids, &mut caches)?;
        let mut out = Vec::with_capacity(ids.len());
        out.push(0.0);
        for i in 1..ids.len() {
            let row = log_softmax_row(&logits[(i - 1) * vocab..i * vocab]);
            out.push(row[ids[i] as usize]);
        }
        Ok(out)
    }

    /// 文脈 `prefix_ids` の次に来る確率が高いトークンを上位 `k` 個(トークン ID, 対数確率)で返す。
    /// かすれて何も読めない箇所の候補をモデル自身に挙げさせる用途。
    pub fn top_next_tokens(&self, device: &std::sync::Arc<dyn GpuDevice>, prefix_ids: &[u32], k: usize) -> Result<Vec<(u32, f32)>> {
        ensure!(!prefix_ids.is_empty(), "open-cuda-llm: prefix must not be empty");
        let vocab = self.config.vocab_size;
        let mut caches = self.new_caches();
        let logits = self.forward_prefill_per_position(device.as_ref(), prefix_ids, &mut caches)?;
        let row = log_softmax_row(&logits[(prefix_ids.len() - 1) * vocab..prefix_ids.len() * vocab]);
        let mut idx: Vec<usize> = (0..vocab).collect();
        idx.sort_unstable_by(|&a, &b| row[b].total_cmp(&row[a]));
        Ok(idx.into_iter().take(k).map(|i| (i as u32, row[i])).collect())
    }

    /// 貪欲デコード(繰り返しペナルティ付き)で`max_new_tokens`個生成する。
    /// `GptModel::generate_with_repetition_penalty`と同じ呼び出し規約。
    pub fn generate(&self, device: &std::sync::Arc<dyn GpuDevice>, prompt_ids: &[u32], max_new_tokens: usize) -> Result<Vec<u32>> {
        self.generate_with_repetition_penalty(device, prompt_ids, max_new_tokens, 1.0)
    }

    pub fn generate_with_repetition_penalty(&self, device: &std::sync::Arc<dyn GpuDevice>, prompt_ids: &[u32], max_new_tokens: usize, penalty: f32) -> Result<Vec<u32>> {
        ensure!(!prompt_ids.is_empty(), "open-cuda-llm: prompt must not be empty");
        let device_ref = device.as_ref();
        let mut caches = self.new_caches();
        let mut logits = Vec::new();
        for &id in prompt_ids {
            logits = self.forward_step(device_ref, id, &mut caches)?;
        }
        let mut seen: std::collections::HashSet<u32> = prompt_ids.iter().copied().collect();
        let mut generated = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            if penalty != 1.0 {
                apply_repetition_penalty(&mut logits, &seen, penalty);
            }
            let next = argmax(&logits);
            generated.push(next);
            seen.insert(next);
            if generated.len() >= max_new_tokens {
                break;
            }
            logits = self.forward_step(device_ref, next, &mut caches)?;
        }
        Ok(generated)
    }
}

/// 標準的な`nn.Linear`(`[out_dim, in_dim]`行優先で保存、GPT-2の
/// `Conv1D`とは転置関係が逆)から`Linear`を組み立てる。biasテンソルが
/// 無ければ(Qwen2の`o_proj`/MLP系はbias無し)ゼロベクトルにする。
fn load_linear(tensors: &safetensors::SafeTensors, prefix: &str, in_dim: usize, out_dim: usize) -> Result<Linear> {
    let raw = tensor_f32(tensors, &format!("{prefix}.weight"))?;
    ensure!(raw.len() == out_dim * in_dim, "open-cuda-llm: '{prefix}.weight' has {} elements, expected {}x{}", raw.len(), out_dim, in_dim);
    let weight_t = transpose(&raw, out_dim, in_dim);
    let bias = tensor_f32(tensors, &format!("{prefix}.bias")).unwrap_or_else(|_| vec![0.0; out_dim]);
    Ok(Linear { weight_t, bias, in_dim, out_dim, spirv_matmul: None, dxil_offload: None, fp8_weight: None })
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencuda_cpu::CpuDevice;
    use std::sync::Arc;

    fn device() -> Arc<dyn GpuDevice> {
        CpuDevice::new(0)
    }

    /// バッチ prefill が、1 トークンずつの forward_step と同じロジットになること(候補採点の土台)。
    #[test]
    fn batched_prefill_matches_stepwise_logits_and_scoring_is_consistent() {
        let model = QwenModel::load_random(QwenConfig::tiny(48), 11);
        let device = device();
        let ids = [3u32, 9, 1, 7, 20, 5];
        let mut c1 = model.new_caches();
        let batched = model.forward_prefill_per_position(device.as_ref(), &ids, &mut c1).unwrap();
        let mut c2 = model.new_caches();
        for (i, &t) in ids.iter().enumerate() {
            let step = model.forward_step(device.as_ref(), t, &mut c2).unwrap();
            for (a, b) in step.iter().zip(&batched[i * 48..(i + 1) * 48]) {
                assert!((a - b).abs() < 1e-3, "pos {i}: {a} vs {b}");
            }
        }
        // 採点: 手計算(forward_step の対数確率の和)と一致。
        let prefix = [3u32, 9];
        let conts = vec![vec![1u32, 7], vec![20u32], vec![]];
        let scores = model.score_continuations(&device, &prefix, &conts).unwrap();
        let mut caches = model.new_caches();
        let mut logits = Vec::new();
        for &t in &prefix {
            logits = model.forward_step(device.as_ref(), t, &mut caches).unwrap();
        }
        let mut expect = 0.0;
        for &t in &conts[0] {
            expect += log_softmax_row(&logits)[t as usize];
            logits = model.forward_step(device.as_ref(), t, &mut caches).unwrap();
        }
        assert!((scores[0] - expect).abs() < 1e-3, "{} vs {expect}", scores[0]);
        assert!(scores[1] < 0.0 && scores[2] == 0.0);
        let top = model.top_next_tokens(&device, &prefix, 3).unwrap();
        assert_eq!(top.len(), 3);
        assert!(top[0].1 >= top[1].1 && top[1].1 >= top[2].1);
    }

    #[test]
    fn token_logprobs_match_scoring() {
        let model = QwenModel::load_random(QwenConfig::tiny(48), 5);
        let device = device();
        let ids = [3u32, 9, 1, 7];
        let lp = model.token_logprobs(&device, &ids).unwrap();
        assert_eq!(lp.len(), 4);
        assert_eq!(lp[0], 0.0);
        let scored = model.score_continuations(&device, &ids[..1], &[ids[1..].to_vec()]).unwrap();
        let sum: f32 = lp[1..].iter().sum();
        assert!((scored[0] - sum).abs() < 1e-3, "{} vs {sum}", scored[0]);
    }

    #[test]
    fn generates_requested_number_of_tokens_without_panicking() {
        let config = QwenConfig::tiny(64);
        let model = QwenModel::load_random(config, 42);
        let device = device();
        let prompt = vec![1u32, 2, 3];
        let generated = model.generate(&device, &prompt, 8).unwrap();
        assert_eq!(generated.len(), 8);
        for &t in &generated {
            assert!((t as usize) < 64, "generated token {t} out of vocab range");
        }
    }

    /// GQA配線の検証: num_kv_heads(2) < num_heads(4) でも
    /// パニックせず、ヘッド数不整合(assert)にも引っかからないこと。
    #[test]
    fn grouped_query_attention_with_fewer_kv_heads_than_query_heads_works() {
        let config = QwenConfig::tiny(32);
        assert!(config.num_kv_heads < config.num_heads);
        let model = QwenModel::load_random(config, 7);
        let device = device();
        let generated = model.generate(&device, &[0, 1], 5).unwrap();
        assert_eq!(generated.len(), 5);
    }

    /// 通常のMulti-Head Attention相当(num_kv_heads == num_heads)でも
    /// 動くこと(GQAはMHAの一般化なので、この境界条件も壊れていないか)。
    #[test]
    fn num_kv_heads_equal_to_num_heads_behaves_as_plain_multi_head_attention() {
        let mut config = QwenConfig::tiny(32);
        config.num_kv_heads = config.num_heads;
        let model = QwenModel::load_random(config, 3);
        let device = device();
        let generated = model.generate(&device, &[5], 4).unwrap();
        assert_eq!(generated.len(), 4);
    }

    /// 同一シードは同一出力(決定的な貪欲デコード)であることの回帰テスト。
    #[test]
    fn generation_is_deterministic_for_a_fixed_seed() {
        let device = device();
        let a = QwenModel::load_random(QwenConfig::tiny(50), 99).generate(&device, &[1, 2], 6).unwrap();
        let b = QwenModel::load_random(QwenConfig::tiny(50), 99).generate(&device, &[1, 2], 6).unwrap();
        assert_eq!(a, b);
    }

    /// 出力が全て同一トークンに潰れていない(退化していない)ことの
    /// ヘルスチェック——RoPE/RMSNorm配線のどこかが恒等的に効かなくなる
    /// 実装ミス(例:全て0になる)の検出用。
    #[test]
    fn rope_and_rms_norm_produce_non_degenerate_hidden_states() {
        let device = device();
        let model = QwenModel::load_random(QwenConfig::tiny(80), 5);
        let out1 = model.generate(&device, &[10], 6).unwrap();
        let out2 = model.generate(&device, &[20], 6).unwrap();
        assert_ne!(out1, out2, "different prompts should not collapse to identical output");
    }

    /// MLA圧縮(GQA向け、KVヘッド単位)を有効化しても最後まで完走すること
    /// (`GptModel`側の同名テストと同じ趣旨)。
    #[test]
    fn mla_kv_compression_enabled_qwen_model_generates_without_panicking() {
        let config = QwenConfig::tiny(64); // hidden=32, num_heads=4 => head_dim=8
        let mut model = QwenModel::load_random(config, 123);
        model.enable_mla_kv_compression(2, 999).unwrap(); // head_dim=8 -> d_c=2 (75%削減)
        let device = device();
        let generated = model.generate(&device, &[1, 2, 3], 6).unwrap();
        assert_eq!(generated.len(), 6);
    }

    /// `d_c >= head_dim`は圧縮になっていないため拒否されることを確認
    /// (`GptModel::enable_mla_kv_compression`と同じ不変条件)。
    #[test]
    fn mla_kv_compression_rejects_non_reducing_d_c() {
        let config = QwenConfig::tiny(32); // head_dim=8
        let mut model = QwenModel::load_random(config, 1);
        assert!(model.enable_mla_kv_compression(8, 1).is_err());
        assert!(model.enable_mla_kv_compression(0, 1).is_err());
    }

    #[test]
    fn rejects_num_heads_not_a_multiple_of_num_kv_heads_at_load_time() {
        // load()は実ファイルが要るため、ここではconfig制約そのもの
        // (num_heads % num_kv_heads == 0)をload_random側の前提として
        // 明示するにとどめる — 実ファイル経路の異常系はload()内のensure!で
        // 保証される(結合テストはopen-cuda-llm単体では困難なため、
        // 呼び出し元のaruaru-llm側でカタログ経由のロードをE2E検証する)。
        let cfg = QwenConfig::tiny(16);
        assert_eq!(cfg.num_heads % cfg.num_kv_heads, 0);
    }

    /// [`QwenModel::enable_mla_kv_compression_calibrated`]の基本動作
    /// (2026-09-13追加、`GptModel`版の移植)。合成の(=事前学習していない)
    /// ランダム重みモデルでの検証のため、圧縮後の生成品質そのものは
    /// 評価できない(意味のある品質評価には実学習済み重みが要る——
    /// `GptModel`版の`calibrated_pca_mla_kv_compression_on_real_gpt2_weights`
    /// と同じ制約)。ここで確認するのは「較正が成功し、パニックせず
    /// 生成まで完走する」という配線レベルの正しさ。
    #[test]
    fn calibrated_pca_mla_kv_compression_completes_and_generates_without_panicking() {
        let config = QwenConfig::tiny(64); // hidden=32, num_heads=4, num_kv_heads=2 => head_dim=8
        let mut model = QwenModel::load_random(config, 55);
        let device = device();
        // 較正用プロンプト複数(各層・各KVヘッドともd_c=2以上の行数を
        // 確保するため、十分な長さ・本数にする)。
        let sample_prompts: Vec<Vec<u32>> = vec![vec![1, 2, 3, 4, 5, 6], vec![7, 8, 9, 10], vec![11, 12, 13, 14, 15]];
        model.enable_mla_kv_compression_calibrated(2, device.as_ref(), &sample_prompts).unwrap();

        let generated = model.generate(&device, &[1, 2, 3], 6).unwrap();
        assert_eq!(generated.len(), 6);
    }

    /// 較正には`d_c >= head_dim`(圧縮になっていない)、空のプロンプト列、
    /// 既に圧縮済みのモデルへの再較正、をそれぞれ拒否すること
    /// (`GptModel`版と同じ不変条件)。
    #[test]
    fn calibrated_pca_mla_kv_compression_rejects_invalid_inputs() {
        let config = QwenConfig::tiny(32); // head_dim=8
        let device = device();

        let mut model = QwenModel::load_random(config.clone(), 1);
        assert!(model.enable_mla_kv_compression_calibrated(8, device.as_ref(), &[vec![1, 2, 3]]).is_err());
        assert!(model.enable_mla_kv_compression_calibrated(0, device.as_ref(), &[vec![1, 2, 3]]).is_err());

        let mut model2 = QwenModel::load_random(config.clone(), 2);
        assert!(model2.enable_mla_kv_compression_calibrated(2, device.as_ref(), &[]).is_err());

        // 較正データが少なすぎる(d_c未満の行数しか集まらない)場合は
        // 意味のあるPCA基底を作れないため拒否する。
        let mut model3 = QwenModel::load_random(config.clone(), 3);
        assert!(model3.enable_mla_kv_compression_calibrated(4, device.as_ref(), &[vec![1]]).is_err());

        // 既に(ランダム射影版で)圧縮済みのモデルへ較正版を重ねて
        // 適用するのは拒否する(フル精度の活性化が既に失われているため)。
        let mut model4 = QwenModel::load_random(config, 4);
        model4.enable_mla_kv_compression(2, 999).unwrap();
        assert!(model4.enable_mla_kv_compression_calibrated(2, device.as_ref(), &[vec![1, 2, 3, 4]]).is_err());
    }
}
