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
const HELDOUT: &[&str] = &[
    "The committee met on Tuesday to discuss the budget, and after a long debate the members agreed to postpone the final vote until next month.",
    "In a small village by the sea, an old fisherman repaired his nets every morning while the children played on the beach nearby.",
    "To install the package, run the setup script, then open the configuration file and set the server address and the port number.",
    "昨日は雨が降っていたので、友達と一緒に家でゲームをして、夜はカレーを作って食べました。",
];
const PROMPT: &str = "Explain in two sentences why the sky is blue.";

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().context("usage: qwen_mla_bench <model_dir> [d_c]")?);
    let d_c: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let tok = GptTokenizer::load(&dir)?;
    let t0 = Instant::now();
    let fp8 = std::env::var("QWEN_FP8").is_ok();
    let device: Arc<dyn GpuDevice> = CpuDevice::new(0);
    let mut model = if let Ok(spec) = std::env::var("QWEN_PRECISION") { QwenModel::load_precision(&dir, device.as_ref(), &open_cuda_llm::PrecisionTable::parse(&spec)?)? } else if fp8 { QwenModel::load_fp8(&dir, device.as_ref(), opencuda_blas::Fp8Format::E4M3)? } else { QwenModel::load(&dir)? };
    let load_s = t0.elapsed().as_secs_f32();
    let mut calib_s = 0.0;
    if d_c > 0 {
        let prompts: Vec<Vec<u32>> = CALIB.iter().map(|t| tok.encode(t)).collect::<Result<_>>()?;
        let t = Instant::now();
        model.enable_mla_kv_compression_calibrated(d_c, device.as_ref(), &prompts)?;
        calib_s = t.elapsed().as_secs_f32();
    }
    // QWEN_SKIP=3,5 のように指定した層を飛ばす。QWEN_SWEEP=1 なら各層を1つずつ飛ばしてppl・速度を表示。
    if std::env::var("QWEN_SWEEP").is_ok() {
        let ids = tok.encode(EVAL)?;
        for i in 1..model.num_layers() {
            model.set_skipped_layers(&[i])?;
            let t = Instant::now();
            let lp = model.token_logprobs(&device, &ids)?;
            let ppl = (-(lp.iter().sum::<f32>() / lp.len() as f32)).exp();
            println!("SWEEP skip_layer={i} ppl={ppl:.3} eval_s={:.1}", t.elapsed().as_secs_f32());
        }
        return Ok(());
    }
    // QWEN_SEARCH=K: 飛ばす層をK個選ぶ。貪欲法(1つずつ最良を追加)と焼きなまし(入れ替え近傍)を比べる。
    if let Ok(k) = std::env::var("QWEN_SEARCH") {
        let k: usize = k.parse()?;
        let ids = tok.encode(EVAL)?;
        let n = model.num_layers();
        let evals = std::cell::Cell::new(0usize);
        let eval = |set: &[usize], model: &mut QwenModel| -> Result<f32> {
            evals.set(evals.get() + 1);
            model.set_skipped_layers(set)?;
            let lp = model.token_logprobs(&device, &ids)?;
            Ok((-(lp.iter().sum::<f32>() / lp.len() as f32)).exp())
        };
        let mut greedy: Vec<usize> = Vec::new();
        let mut gp = 0.0;
        for _ in 0..k {
            let mut best = (f32::INFINITY, 0usize);
            for i in 1..n {
                if greedy.contains(&i) { continue; }
                let mut c = greedy.clone();
                c.push(i);
                let p = eval(&c, &mut model)?;
                if p < best.0 { best = (p, i); }
            }
            greedy.push(best.1);
            gp = best.0;
        }
        let greedy_evals = evals.get();
        println!("SEARCH greedy k={k} set={greedy:?} ppl={gp:.3} evals={greedy_evals}");
        // 焼きなまし: 同じ評価回数の予算で、1層を入れ替える近傍を探索。
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut rnd = |m: usize| -> usize { state ^= state << 13; state ^= state >> 7; state ^= state << 17; (state % m as u64) as usize };
        let mut cur: Vec<usize> = (0..k).map(|j| 1 + (j * (n - 1)) / k).collect();
        let mut cur_p = eval(&cur, &mut model)?;
        let (mut best_set, mut best_p) = (cur.clone(), cur_p);
        let budget = greedy_evals;
        let mut t = 2.0f32;
        while evals.get() < greedy_evals + budget {
            let pos = rnd(k);
            let cand = 1 + rnd(n - 1);
            if cur.contains(&cand) { continue; }
            let mut next = cur.clone();
            next[pos] = cand;
            let p = eval(&next, &mut model)?;
            let accept = p < cur_p || ((cur_p.ln() - p.ln()) / (t * 0.1)).exp() > (rnd(1000) as f32 / 1000.0);
            if accept { cur = next; cur_p = p; }
            if cur_p < best_p { best_p = cur_p; best_set = cur.clone(); }
            t = (t * 0.97).max(0.05);
        }
        best_set.sort();
        println!("SEARCH anneal k={k} set={best_set:?} ppl={best_p:.3} evals={}", evals.get() - greedy_evals);
        return Ok(());
    }
    if let Ok(v) = std::env::var("QWEN_SKIP") {
        let idx: Vec<usize> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        model.set_skipped_layers(&idx)?;
    }
    let ids = tok.encode(EVAL)?;
    let t = Instant::now();
    let lp = model.token_logprobs(&device, &ids)?;
    let eval_s = t.elapsed().as_secs_f32();
    let ppl = (-(lp.iter().sum::<f32>() / lp.len() as f32)).exp();
    let mut held = 0.0f32;
    for t in HELDOUT {
        let hi = tok.encode(t)?;
        let hl = model.token_logprobs(&device, &hi)?;
        held += (-(hl.iter().sum::<f32>() / hl.len() as f32)).exp().ln();
    }
    let heldout_ppl = (held / HELDOUT.len() as f32).exp();
    let p = tok.encode(PROMPT)?;
    let t = Instant::now();
    let gen = model.generate(&device, &p, 24)?;
    let gen_s = t.elapsed().as_secs_f32();
    // Qwen2.5-1.5B: 28層 x 2 KVヘッド x (K,V) x 次元 x 4バイト(f32)
    let dim = if d_c > 0 { d_c } else { 128 };
    let kv_bytes = 28 * 2 * 2 * dim * 4;
    println!(
        "RESULT fp8={fp8} d_c={d_c} load_s={load_s:.1} calib_s={calib_s:.1} eval_tokens={} eval_s={eval_s:.1} ppl={ppl:.3} heldout_ppl={heldout_ppl:.3} gen_s={gen_s:.1} kv_bytes_per_token={kv_bytes} gen_ids={:?}",
        ids.len(),
        gen
    );
    println!("TEXT {:?}", tok.decode(&gen)?);
    Ok(())
}
