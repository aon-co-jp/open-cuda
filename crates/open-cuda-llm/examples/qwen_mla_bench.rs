//! QwenModelのMLA KV圧縮(PCA較正版)の実測(2026-11-19)。
//! `cargo run --release -p open-cuda-llm --example qwen_mla_bench -- <モデルdir> [d_c|0]`
//! d_c=0または省略で圧縮なし。1行のJSON風テキストで結果を出す
//! (perplexity=評価文のトークン平均負対数尤度のexp、生成トークンID、時間、KV理論バイト数/トークン)。
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use open_cuda_llm::{GptTokenizer, QwenModel};
use opencuda_core::GpuDevice;
use opencuda_cpu::CpuDevice;

const CALIB: &[&str] = &[
    "The history of the Roman Empire spans many centuries, from the founding of the city to the fall of the western provinces.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
    "東京は日本の首都であり、世界でも有数の大都市として知られています。鉄道網が発達しており、多くの人が毎日利用します。",
    "Photosynthesis is the process by which green plants convert sunlight, water and carbon dioxide into glucose and oxygen.",
];
const EVAL: &str = "Machine learning models are trained on large datasets to recognise patterns. Once trained, they can make predictions on new data, but their quality depends on how well the training data represents the real world. 日本語の文章でも同じことが言えます。";
const PROMPT: &str = "Explain in two sentences why the sky is blue.";

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().context("usage: qwen_mla_bench <model_dir> [d_c]")?);
    let d_c: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let tok = GptTokenizer::load(&dir)?;
    let t0 = Instant::now();
    let mut model = QwenModel::load(&dir)?;
    let load_s = t0.elapsed().as_secs_f32();
    let device: Arc<dyn GpuDevice> = CpuDevice::new(0);
    let mut calib_s = 0.0;
    if d_c > 0 {
        let prompts: Vec<Vec<u32>> = CALIB.iter().map(|t| tok.encode(t)).collect::<Result<_>>()?;
        let t = Instant::now();
        model.enable_mla_kv_compression_calibrated(d_c, device.as_ref(), &prompts)?;
        calib_s = t.elapsed().as_secs_f32();
    }
    let ids = tok.encode(EVAL)?;
    let t = Instant::now();
    let lp = model.token_logprobs(&device, &ids)?;
    let eval_s = t.elapsed().as_secs_f32();
    let ppl = (-(lp.iter().sum::<f32>() / lp.len() as f32)).exp();
    let p = tok.encode(PROMPT)?;
    let t = Instant::now();
    let gen = model.generate(&device, &p, 24)?;
    let gen_s = t.elapsed().as_secs_f32();
    // Qwen2.5-1.5B: 28層 x 2 KVヘッド x (K,V) x 次元 x 4バイト(f32)
    let dim = if d_c > 0 { d_c } else { 128 };
    let kv_bytes = 28 * 2 * 2 * dim * 4;
    println!(
        "RESULT d_c={d_c} load_s={load_s:.1} calib_s={calib_s:.1} eval_tokens={} eval_s={eval_s:.1} ppl={ppl:.3} gen_s={gen_s:.1} kv_bytes_per_token={kv_bytes} gen_ids={:?}",
        ids.len(),
        gen
    );
    println!("TEXT {:?}", tok.decode(&gen)?);
    Ok(())
}
