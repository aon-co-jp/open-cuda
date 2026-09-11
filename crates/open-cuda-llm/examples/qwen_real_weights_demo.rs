//! 実在する学習済み重み(Qwen2.5-0.5B-Instruct等)で`QwenModel`を検証する
//! ためのCLIデモ(2026-09-11新設)。`cargo run -p open-cuda-llm --example
//! qwen_real_weights_demo -- <モデルディレクトリ> [プロンプト] [生成トークン数]`
//!
//! `<モデルディレクトリ>`には`config.json`・`model.safetensors`・
//! `tokenizer.json`が揃っている必要がある(`QwenModel::load`/
//! `GptTokenizer::load`と同じ前提、単一ファイルのcheckpointのみ対応)。

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use open_cuda_llm::{GptTokenizer, QwenModel};
use opencuda_core::GpuDevice;
use opencuda_cpu::CpuDevice;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(args.next().context("usage: qwen_real_weights_demo <model_dir> [prompt] [max_new_tokens]")?);
    let prompt_text = args.next().unwrap_or_else(|| "Hello, who are you?".to_string());
    let max_new_tokens: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(32);

    println!("loading tokenizer from {}/tokenizer.json ...", model_dir.display());
    let tokenizer = GptTokenizer::load(&model_dir).context("failed to load tokenizer.json")?;

    println!("loading Qwen weights from {} (config.json + model.safetensors) ...", model_dir.display());
    let t0 = Instant::now();
    let mut model = QwenModel::load(&model_dir).context("failed to load Qwen model")?;
    println!("model loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // 第4引数でMLA KVキャッシュ圧縮の d_c を指定できる(省略時は無効、
    // 従来どおりフル精度)。`QwenModel::enable_mla_kv_compression`参照。
    if let Some(d_c) = args.next().and_then(|s| s.parse::<usize>().ok()) {
        println!("enabling MLA KV-cache compression (d_c={d_c}, random projection — see module doc caveat)...");
        model.enable_mla_kv_compression(d_c, 42).context("enable_mla_kv_compression failed")?;
    }

    let prompt_ids = tokenizer.encode(&prompt_text)?;
    println!("prompt: {prompt_text:?} -> {} tokens", prompt_ids.len());

    let device: Arc<dyn GpuDevice> = CpuDevice::new(0);
    println!("generating {max_new_tokens} tokens (CPU, greedy decode)...");
    let t1 = Instant::now();
    let generated = model.generate(&device, &prompt_ids, max_new_tokens)?;
    let elapsed = t1.elapsed();

    let text = tokenizer.decode(&generated)?;
    println!("---");
    println!("generated {} tokens in {:.1}s ({:.2} tok/s)", generated.len(), elapsed.as_secs_f32(), generated.len() as f32 / elapsed.as_secs_f32().max(0.001));
    println!("output: {text:?}");
    Ok(())
}
