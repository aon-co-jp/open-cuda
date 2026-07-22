//! # opencuda-llm
//!
//! 自己回帰デコーダ(GPT系アーキテクチャ)のforward pass実装。
//! `open-raid-z`の2026-07-21マーケティング調査ロードマップで言う
//! 「Python製AIライブラリのRust移植 1〜6位」のうち、**1位のvLLM相当**の
//! MVP(最小実用実装)にあたる。`opencuda-bert`(2位Transformers相当、
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
//! - **学習済みの重みは無い**。実在するGPT系モデルの`safetensors`を
//!   読み込むローダーは(`opencuda-bert`のBERT版と違い)まだ実装しておらず、
//!   決定的な疑似乱数(`SplitMix64`)で初期化したランダム重みのみを使う。
//!   したがって生成されるテキストは意味を持たない——本クレートが検証
//!   しているのは「言語モデルとして自然な文章を生成できるか」ではなく
//!   「トークン埋め込み→複数デコーダ層(Self-Attention+FFN)→KVキャッシュ
//!   による逐次デコード→貪欲サンプリング、という自己回帰生成パイプライン
//!   の配線が正しく動くか」である。実在の学習済み重み(GPT-2小型版等の
//!   `safetensors`)を読み込むローダーの追加は次の増分。
//! - **トークナイザはUTF-8バイト単位の素朴なもの**(`tokenizers`クレート
//!   による本格的なBPE/SentencePieceではない)。外部ファイル・追加の
//!   数百MB級モデルダウンロードを要さずに端から端まで動く最小構成を
//!   優先した設計判断(このタスクの制約「大きすぎる依存関係を避ける」
//!   に沿った選択)。
//! - Attentionは`opencuda-bert`と同じく`opencuda-blas::scaled_dot_product_attention`
//!   (非タイル化の素朴な実装)をそのまま使う。KVキャッシュ付きの1トークン
//!   ずつの生成では、クエリ行を`n`回複製して`n x n`のattentionを計算し
//!   先頭行だけを使うという簡易的な方法で「新規トークンのクエリ×
//!   過去全体のキー/バリュー」を計算している(数学的には正しいが、
//!   本来必要な計算量よりO(n)倍無駄が多い——専用のcausal-attention
//!   カーネルを`opencuda-blas`に追加するのが次の最適化)。

use anyhow::Result;
use opencuda_core::GpuDevice;

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

fn gelu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        let xf = *v as f64;
        *v = (0.5 * xf * (1.0 + libm_erf_approx(xf * std::f64::consts::FRAC_1_SQRT_2))) as f32;
    }
}

/// `libm`に依存させないための簡易erf近似(Abramowitz&Stegun 7.1.26、
/// 最大誤差約1.5e-7)。厳密なerfではないが、本クレートは重み自体が
/// ランダムで学習済みではないため、この程度の近似精度で十分。
fn libm_erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

struct DecoderLayer {
    query: Linear,
    key: Linear,
    value: Linear,
    attn_out: Linear,
    attn_ln: LayerNorm,
    intermediate: Linear,
    output: Linear,
    output_ln: LayerNorm,
}

impl DecoderLayer {
    fn random(rng: &mut SplitMix64, hidden: usize, intermediate: usize, eps: f32) -> Self {
        Self {
            query: Linear::random(rng, hidden, hidden),
            key: Linear::random(rng, hidden, hidden),
            value: Linear::random(rng, hidden, hidden),
            attn_out: Linear::random(rng, hidden, hidden),
            attn_ln: LayerNorm::identity(hidden, eps),
            intermediate: Linear::random(rng, hidden, intermediate),
            output: Linear::random(rng, intermediate, hidden),
            output_ln: LayerNorm::identity(hidden, eps),
        }
    }

    /// 1トークン分(`hidden`は`hidden_size`長の単一行)を処理し、
    /// このレイヤーの`cache`へ今回のk/vを追加した上で出力を返す
    /// (causalマスクは「まだキャッシュに存在しない未来のトークンは
    /// そもそも追加されていない」ことで自然に実現される、明示的な
    /// マスク行列は不要)。
    fn forward_step(&self, device: &dyn GpuDevice, hidden: &[f32], cache: &mut [KvCacheHead], hidden_size: usize, num_heads: usize) -> Result<Vec<f32>> {
        let head_dim = hidden_size / num_heads;

        let q = self.query.forward(device, hidden, 1)?;
        let k = self.key.forward(device, hidden, 1)?;
        let v = self.value.forward(device, hidden, 1)?;

        let mut context = vec![0.0f32; hidden_size];
        for h in 0..num_heads {
            let col_start = h * head_dim;
            let q_h = &q[col_start..col_start + head_dim];
            let k_h = &k[col_start..col_start + head_dim];
            let v_h = &v[col_start..col_start + head_dim];

            cache[h].push(k_h, v_h);
            let n = cache[h].n;

            // qを n 回複製して n x n の attention を計算し、先頭行(全行
            // 同一)だけを使う(モジュールdocコメント参照、素朴だが正しい)。
            let mut q_full = vec![0.0f32; n * head_dim];
            for row in 0..n {
                q_full[row * head_dim..(row + 1) * head_dim].copy_from_slice(q_h);
            }
            let out = opencuda_blas::scaled_dot_product_attention(device, &q_full, &cache[h].k, &cache[h].v, n, head_dim)?;
            context[col_start..col_start + head_dim].copy_from_slice(&out[0..head_dim]);
        }

        let mut attn_dense = self.attn_out.forward(device, &context, 1)?;
        for i in 0..attn_dense.len() {
            attn_dense[i] += hidden[i];
        }
        self.attn_ln.forward(&mut attn_dense, 1, hidden_size);

        let mut intermediate = self.intermediate.forward(device, &attn_dense, 1)?;
        gelu_inplace(&mut intermediate);

        let mut ffn_out = self.output.forward(device, &intermediate, 1)?;
        for i in 0..ffn_out.len() {
            ffn_out[i] += attn_dense[i];
        }
        self.output_ln.forward(&mut ffn_out, 1, hidden_size);

        Ok(ffn_out)
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

    /// 新規のKVキャッシュ集合(レイヤー数 x ヘッド数)を作る。
    fn new_caches(&self) -> Vec<Vec<KvCacheHead>> {
        (0..self.config.num_layers).map(|_| (0..self.config.num_heads).map(|_| KvCacheHead::empty()).collect()).collect()
    }

    /// 1トークンぶん進め、そのトークンの次を予測するロジット(語彙数長)を返す。
    fn forward_step(&self, device: &dyn GpuDevice, token_id: u32, pos: usize, caches: &mut [Vec<KvCacheHead>]) -> Result<Vec<f32>> {
        anyhow::ensure!(pos < self.config.max_seq_len, "opencuda-llm: position {pos} exceeds max_seq_len {}", self.config.max_seq_len);
        let hidden_size = self.config.hidden_size;
        let tok = token_id as usize;
        anyhow::ensure!(tok < self.config.vocab_size, "opencuda-llm: token id {tok} out of vocab range");

        let mut hidden = vec![0.0f32; hidden_size];
        for c in 0..hidden_size {
            hidden[c] = self.word_embeddings[tok * hidden_size + c] + self.position_embeddings[pos * hidden_size + c];
        }

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
        anyhow::ensure!(!prompt_ids.is_empty(), "opencuda-llm: prompt_ids must not be empty");
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
}
