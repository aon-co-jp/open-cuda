//! # open-cuda-llm-wasm
//!
//! `open-cuda-llm`(GPT-2系自己回帰デコーダ、KVキャッシュ付き貪欲デコード)を
//! `wasm-bindgen`経由でJavaScriptから呼べるようにする薄いラッパー
//! (2026-08-10新設)。「オフラインのブラウザアプリとしてGPT-2推論を動かし、
//! UPDATE確認のみオンライン」という体験(サーバー常駐無し)の実証層。
//!
//! ## 正直な開示(スコープの限界、CLAUDE.md HANDOFF参照)
//!
//! - **実行パスはCPUのみ**。`opencuda-vulkan`のVulkan実行パス(`real-vulkan`
//!   feature)はブラウザでは使えない(WebGPU等への別途対応が必要、Vulkan自体は
//!   ブラウザに存在しない)ため、本クレートは常に`opencuda-cpu::CpuDevice`
//!   (rayonベースのCPU並列実行)を使う。`generate`呼び出しは事実上、
//!   ネイティブ版の`--features real-vulkan`無し・CPU実行と同じ経路を通る。
//! - **モデル重みの配信方法(fetch→IndexedDBキャッシュ等)は本クレートの
//!   スコープ外**。JS側が既に取得済みの`model.safetensors`/`config.json`/
//!   `tokenizer.json`のバイト列(または文字列)を渡す前提——ダウンロード・
//!   キャッシュ管理はJS側(呼び出し元)の責務として明確に分離した。
//! - **`rayon`(opencuda-cpuの並列実行に使用)がブラウザのWeb Worker無し
//!   環境で実際に動作するかは、`cargo check --target wasm32-unknown-unknown`
//!   でのコンパイル成功のみ確認済みで、ブラウザでの実行時動作(スレッド
//!   生成が失敗せず、単一スレッドへ自然にフォールバックするか)は
//!   未検証**——動かなかった場合は`opencuda-cpu`側にwasm32向けの
//!   シングルスレッドフォールバックパスを追加する必要がある(次回課題)。

use std::sync::Arc;

use open_cuda_llm::{GptModel, GptTokenizer};
use opencuda_cpu::CpuDevice;
use opencuda_core::GpuDevice;
use wasm_bindgen::prelude::*;

/// ブラウザのconsoleへpanicメッセージを転送する(デバッグ用、`start`で
/// 一度だけ呼ばれる)。呼ばないとwasm32のpanicは無音で失敗するため、
/// devtoolsコンソールで原因を追えるようにする目的で追加した。
#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

/// ロード済みのGPT-2モデル+トークナイザをまとめて保持するハンドル。
/// JS側はこれを`load_model`から受け取り、`generate`へそのまま渡す。
#[wasm_bindgen]
pub struct LoadedModel {
    model: GptModel,
    tokenizer: GptTokenizer,
    device: Arc<dyn GpuDevice>,
}

#[wasm_bindgen]
impl LoadedModel {
    /// `model.safetensors`のバイト列・`config.json`/`tokenizer.json`の
    /// 文字列本体から、ファイルI/O無しでモデルを構築する
    /// (`open_cuda_llm::GptModel::load_from_bytes`/
    /// `GptTokenizer::load_from_str`をそのまま使う、2026-08-10新設の
    /// wasm対応API)。失敗した場合は`JsValue`(文字列化したエラー)を返す。
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str, weights_bytes: &[u8], tokenizer_json: &str) -> Result<LoadedModel, JsValue> {
        let model = GptModel::load_from_bytes(config_json, weights_bytes).map_err(|e| JsValue::from_str(&format!("{e:#}")))?;
        let tokenizer = GptTokenizer::load_from_str(tokenizer_json).map_err(|e| JsValue::from_str(&format!("{e:#}")))?;
        // Vulkan実行パスはブラウザでは使えないため、常にCPUデバイスを使う
        // (モジュールdocコメント参照)。id=0は`CpuDevice::new`の契約上
        // 論理コアインデックスの意味を持たない(単一デバイスの識別子)。
        let device: Arc<dyn GpuDevice> = CpuDevice::new(0);
        Ok(LoadedModel { model, tokenizer, device })
    }

    /// `prompt`をトークナイズし、貪欲デコード(繰り返しペナルティ付き)で
    /// `max_new_tokens`個のトークンを生成、デコードした文字列を返す。
    /// `repetition_penalty`は`1.0`でペナルティ無効(既存の
    /// `generate_with_repetition_penalty`のセマンティクスそのまま、
    /// `open-cuda-llm`側2026-08-10 HANDOFF参照)。
    #[wasm_bindgen]
    pub fn generate(&self, prompt: &str, max_new_tokens: usize, repetition_penalty: f32) -> Result<String, JsValue> {
        let prompt_ids = self.tokenizer.encode(prompt).map_err(|e| JsValue::from_str(&format!("{e:#}")))?;
        let new_ids = self
            .model
            .generate_with_repetition_penalty(&self.device, &prompt_ids, max_new_tokens, repetition_penalty)
            .map_err(|e| JsValue::from_str(&format!("{e:#}")))?;
        // generate_with_repetition_penaltyは新規生成分のみを返す契約
        // (prompt_ids自体は含まない、open-cuda-llm本体のdocコメント参照)。
        self.tokenizer.decode(&new_ids).map_err(|e| JsValue::from_str(&format!("{e:#}")))
    }
}

/// バイト列単体からトークナイズだけを試したい場合の補助関数
/// (モデルロード無しでトークナイザの動作を確認できるよう、デバッグ用に
/// 公開)。
#[wasm_bindgen]
pub fn tokenize_preview(tokenizer_json: &str, text: &str) -> Result<Vec<u32>, JsValue> {
    let tokenizer = GptTokenizer::load_from_str(tokenizer_json).map_err(|e| JsValue::from_str(&format!("{e:#}")))?;
    tokenizer.encode(text).map_err(|e| JsValue::from_str(&format!("{e:#}")))
}
