//! 実在するDeepSeek-V2-Lite-Chat(実チェックポイント、MLA+DeepSeekMoE、
//! 分割済み・遅延ロード経路)を検証するためのCLIプローブ(2026-09-13新設)。
//! `cargo run -p open-cuda-llm --release --example deepseek_v2_lite_real_weights_probe
//!  -- <モデルディレクトリ> [プロンプト] [生成トークン数]`
//!
//! **目的**: `deepseek_arch.rs`の`ModelWeights`遅延ロード設計
//! (ヘッダのみ読み込み+オンデマンドseek/read+`ExpertSlot::Lazy`)が、
//! この開発機のメモリ制約内で実チェックポイントを実際にロード・生成
//! できるかを検証する。ロード前後・生成前後でこのプロセスの実メモリ
//! 使用量(Windows: `wmic`/`Get-Process`相当の値をRustから取得するのは
//! 面倒なため、ここでは`/proc`相当が無いWindows環境向けに単純な経過
//! 時間のみを計測し、実際のメモリ計測は呼び出し側〈PowerShellの
//! `Get-Process`〉で別途行う前提)を出力する。
//!
//! `<モデルディレクトリ>`には`config.json`・`model.safetensors.index.json`
//! +分割された`model-NNNNN-of-000004.safetensors`・`tokenizer.json`が
//! 揃っている必要がある。

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use open_cuda_llm::{DeepseekModel, GptTokenizer};
use opencuda_core::GpuDevice;
use opencuda_cpu::CpuDevice;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(args.next().context("usage: deepseek_v2_lite_real_weights_probe <model_dir> [prompt] [max_new_tokens]")?);
    let prompt_text = args.next().unwrap_or_else(|| "Hello, who are you?".to_string());
    let max_new_tokens: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);

    println!("loading tokenizer from {}/tokenizer.json ...", model_dir.display());
    let tokenizer = GptTokenizer::load(&model_dir).context("failed to load tokenizer.json")?;

    println!("loading DeepSeek weights from {} (config.json + sharded safetensors, lazy expert loading) ...", model_dir.display());
    let t0 = Instant::now();
    let model = DeepseekModel::load(&model_dir).context("failed to load DeepSeek model")?;
    println!("model loaded (header-only + lazy experts) in {:.1}s", t0.elapsed().as_secs_f32());

    let prompt_ids = tokenizer.encode(&prompt_text)?;
    println!("prompt: {prompt_text:?} -> {} tokens", prompt_ids.len());

    let device: Arc<dyn GpuDevice> = CpuDevice::new(0);
    println!("generating {max_new_tokens} tokens (CPU, greedy decode, experts loaded on first activation)...");
    let t1 = Instant::now();
    let generated = model.generate(&device, &prompt_ids, max_new_tokens)?;
    let elapsed = t1.elapsed();

    let text = tokenizer.decode(&generated)?;
    println!("---");
    println!("generated {} tokens in {:.1}s ({:.2} tok/s)", generated.len(), elapsed.as_secs_f32(), generated.len() as f32 / elapsed.as_secs_f32().max(0.001));
    println!("output: {text:?}");
    Ok(())
}
