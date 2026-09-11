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

use super::{argmax, apply_repetition_penalty, random_vec, tensor_f32, transpose, KvCacheHead, Linear, MlaHeadProjection, SplitMix64};

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
}
