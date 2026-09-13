//! DeepSeek-V2/V3系 Multi-head Latent Attention (MLA) アーキテクチャの
//! forward pass実装(2026-09-13新設)。
//!
//! ## 経緯・正直な開示
//!
//! これまで`lib.rs`/`qwen_arch.rs`にあった`enable_mla_kv_compression*`は
//! **「MLA風」の事後(post-hoc)retrofit**——既に学習済みの標準Attention
//! モデルのKVキャッシュを、ランダム射影またはPCAで低ランク圧縮する
//! だけのものであり、実際のDeepSeek-V2/V3が学習時から持つMLAとは別物
//! だった(このモジュールのdocコメントで前回明記した通り)。
//!
//! このモジュールは、世界中の言語(英語・日本語・中国語)でのGoogle検索・
//! GitHub調査(2026-09-13実施)で確認した実際のDeepSeek-V2/V2-Lite/V3の
//! チェックポイント構造・forward計算順序に基づき、**学習済みの低ランク
//! 射影重みをそのまま読み込んで動く**、本物のMLA構造を実装する。
//! 出典:
//! - <https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json>
//!   (実在するconfig.jsonのフィールド名・値をそのまま採用)
//! - DeepSeek-V2論文 <https://arxiv.org/pdf/2405.04434>
//! - <https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py>
//!   (forward計算の実際の順序)
//! - `qwen_arch.rs`(このクレートの既存の「実チェックポイントを読み込んで
//!   動く新アーキテクチャモジュール」のお手本、RmsNorm/RoPE/SwiGLU MLP
//!   パターンを踏襲する)
//!
//! ## MLA計算の要点(モジュールdocに要約、詳細は`forward_step`のコメント)
//!
//! - Query側: `hidden --q_a_proj--> q_lora_rank --RMSNorm--q_b_proj-->`
//!   `num_heads*(qk_nope_head_dim+qk_rope_head_dim)`という2段階低ランク
//!   射影(`q_lora_rank`が`None`の小型モデルでは`q_proj`一発)。
//! - KV側: `hidden --kv_a_proj_with_mqa--> kv_lora_rank + qk_rope_head_dim`
//!   (後半`qk_rope_head_dim`分はMQA的に**全ヘッド共有**のRoPE専用鍵)、
//!   前半`kv_lora_rank`分を`RMSNorm`してから`kv_b_proj`で
//!   `num_heads*(qk_nope_head_dim+v_head_dim)`へ展開する。
//! - **decoupled RoPE**: 各ヘッドの`q`/`k`は「RoPEを適用しないnope部分」と
//!   「RoPEを適用するrope専用部分」に分割され、RoPEはrope専用部分にしか
//!   掛からない(この点が通常のGQA/MHAのRoPEと構造的に異なる)。
//!
//! ## DeepSeekMoE対応(2026-09-13追記)
//!
//! 前回このモジュールを新設した時点では「MoEは今回のスコープ外」と
//! 明記していたが、ユーザーから「MoE対応の為の設計と実装、開発の為に
//! 世界中の言語でGoogle検索とGithub調査して」との指示を受け、実際に
//! DeepSeekMoEを実装した。世界中の言語(英語・日本語・中国語)での
//! 調査で確認した実チェックポイント構造(出典:
//! <https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/model.safetensors.index.json>、
//! <https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json>、
//! DeepSeek-V3公式推論実装 <https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py>、
//! DeepSeek-V2論文 Sec.2.2 arXiv:2405.04434):
//!
//! - `config.json`のMoEフィールド: `n_routed_experts`・`n_shared_experts`・
//!   `num_experts_per_tok`・`first_k_dense_replace`(この値未満の層番号は
//!   dense、以降はMoE)・`moe_intermediate_size`・`norm_topk_prob`・
//!   `scoring_func`・`routed_scaling_factor`。
//! - テンソル名: dense層は`mlp.gate_proj`/`mlp.up_proj`/`mlp.down_proj`
//!   (従来通り)。MoE層は`mlp.gate.weight`(ルーター、
//!   `[n_routed_experts, hidden_size]`)・
//!   `mlp.shared_experts.{gate_proj,up_proj,down_proj}`(常時適用される
//!   共有エキスパート、dense SwiGLU一本)・
//!   `mlp.experts.{0..n_routed_experts}.{gate_proj,up_proj,down_proj}`
//!   (ルーティングされるエキスパート、各々独立したdense SwiGLU)。
//! - ルーティング計算(推論専用、擬似コード): `scores = softmax(x @ gate.T)`
//!   → 上位`num_experts_per_tok`個のエキスパートを選択(元スコアを重みに
//!   使用) → `norm_topk_prob`が真なら選択後に再正規化 →
//!   `routed_scaling_factor`を乗算 → 選択エキスパートの出力を重み付き
//!   加算 → 常時計算する`shared_experts`の出力を加算。
//!
//! ## V3固有拡張(2026-09-13続き4追記): aux-loss-free補正・group-limited
//! routing・sigmoidスコアリングにも対応
//!
//! 直前(すぐ下)で「未対応」と明記した3項目について、ユーザーから
//! 「世界中の言語で設計と開発の為にGoogle検索とGithub調査して対応
//! させて」との指示を受け、DeepSeek-V3公式推論実装
//! (<https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py>
//! の`Gate`クラス、実`config.json`
//! <https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json>、
//! 論文Sec.2.1.2 arXiv:2412.19437)を調査した上で実装した。
//!
//! **非自明な発見(実装前に把握すべき要点)**: `e_score_correction_bias`は
//! 「どのエキスパートを選ぶか」(top-k選択)にのみ影響し、「選ばれた
//! エキスパートへ与える重み」は**bias加算前の生スコア**
//! (`original_scores`)がそのまま使われる——選択用と重み用でスコアを
//! 2本持つ必要がある。グループ集約方式も分岐する: biasが無ければ
//! グループ内最大値(`amax`)、biasがあれば(V3の`noaux_tc`構成)
//! グループ内**上位2個の合計**を使う。sigmoidスコアリングでは、
//! 各エキスパートのスコアが独立([0,1]で合計が1にならない)ため、
//! 選択後の正規化(`weight /= weight.sum()`)が事実上必須(公式実装では
//! `scoring_func=="sigmoid"`かどうかで無条件に分岐しており、
//! `norm_topk_prob`とは別ロジック)。
//!
//! 正確な計算順序: `scores = softmax_or_sigmoid(gate(x))` →
//! `original_scores = scores.clone()` →
//! (biasがあれば`scores += bias`、選択にのみ影響) →
//! (`n_group>1`なら、グループごとの集約スコアで上位`topk_group`グループ
//! 以外を`-inf`マスク) → 上位`num_experts_per_tok`個を選択 →
//! `weight = original_scores[selected]`(bias抜き) →
//! (`scoring_func=="sigmoid"`または`norm_topk_prob`なら`weight`を再正規化)
//! → `weight *= routed_scaling_factor`。
//!
//! **正直な開示(誇張しない、それでも残る限界)**:
//! - `e_score_correction_bias`は推論時は固定値としてそのままロードする
//!   だけ(学習時の動的更新ロジックは推論専用実装のため実装しない、
//!   調査で確認した通りDeepSeek公式推論実装にも学習ロジックは無い)。
//! - `topk_method`は`"greedy"`(バイアス無し、V2系)と`"noaux_tc"`
//!   (バイアス有り、V3系)のみ対応。それ以外の値は`load()`が拒否する。
//! - 実チェックポイント(V3本体、`n_routed_experts=256`)は671B
//!   パラメータの巨大モデルで、この開発機(GT730、VRAM 2GB)は
//!   もちろんダウンロード自体も一般的な開発機では非現実的——この実装は
//!   あくまで構造的な正しさ(テンソル形状・計算順序)を極小構成の
//!   単体テストで検証したものであり、実V3チェックポイントでの実機
//!   検証は行っていない(行うこと自体が非現実的)。
//!
//! **2026-09-13(続き5)追記・実機検証の具体的な試算(当初の見送り判断)**:
//! より小型な`deepseek-ai/DeepSeek-V2-Lite-Chat`(15.7B、`n_group=1`・
//! `topk_method="greedy"`・`scoring_func="softmax"`——この実装が対応
//! 済みの構成)について、実際に`model.safetensors.index.json`を取得して
//! 試算した結果、**当初はこの開発機での実ダウンロード・実ロード検証を
//! 見送った**。理由: 実チェックポイントは31.4GB(bf16)・4分割
//! (`model-00001-of-000004.safetensors`等)で配布されており、当時の
//! ローダー設計(各シャードの生バイト列を一括保持しつつ、全テンソルを
//! f32へ変換して永続保持する)では、ピークメモリが90GBを超える可能性が
//! 高いと判断したため(この開発機の実測総RAM32GB・空き16GB)。
//!
//! **2026-09-13(続き6)追記・遅延ロードによる再検討**: ユーザーから
//! 「システムメモリで90GB超になるなら、HDDのキャッシュを用意しても
//! ダメか」との指摘を受け、`ModelWeights`をヘッダのみ読み込み+
//! テンソルごとのオンデマンド`seek`+`read`方式へ再設計し、さらに
//! ルーティングされる個々のエキスパートを[`ExpertSlot::Lazy`]として
//! 「実際にルーターに選ばれるまでディスクを読まない」設計にした
//! (`ModelWeights`・`ExpertSlot`のdocコメント参照)。これにより
//! 常時使う部分(attention・共有エキスパート・denseの層0のMLP・埋め込み)
//! だけならf32で概算5GB程度に収まり、実チェックポイント
//! (`hidden_size=2048`・`n_routed_experts=64`・`num_experts_per_tok=6`・
//! `moe_intermediate_size=1408`)を**数トークンだけ生成する短い検証**
//! なら、追加で読み込まれるエキスパート分(1トークンあたり最大
//! `num_experts_per_tok×MoE層数`個、1個あたり約35MB)を足しても
//! 十数GB程度に収まる見込みとなり、この開発機の空きRAM(16GB)内で
//! 現実的になった——ただし生成トークン数が増えるほどルーティングが
//! 広範囲のエキスパートに触れていき、最終的には元の約60GBという上限に
//! 近づいていく点は変わらない(コンパートメント化された恒久的な解決
//! ではなく、短い検証を可能にする設計改善)。
//!
//! **2026-09-13(続き6)実機検証の結果(重要、正直に記録)**: 上記の
//! 再計算に基づき、実際に`deepseek-ai/DeepSeek-V2-Lite-Chat`(31.4GB・
//! 4分割、`scoring_func="softmax"`・`topk_method="greedy"`——この実装が
//! 対応済みの構成)をダウンロードし、この開発機で実機検証を行った。
//! **ロード自体は成功した**(`DeepseekModel::load`が19.9秒で完了、
//! ヘッダのみ読み込み+`ExpertSlot::Lazy`の設計が実際に機能することを
//! 実機で確認できた)。しかし**生成を開始した直後、プロセスの実メモリ
//! 使用量が数十秒で急激に増加**(`Get-Process`で計測、5秒間隔で約
//! 4.1GB→19.1GB)し、**システム全体の空きメモリが127.5MBまで低下**
//! (`Get-CimInstance Win32_OperatingSystem`で計測)したため、
//! 安全装置(空きメモリが2000MB未満になったら強制終了する監視スクリプト)
//! が自動的にプロセスをkillした——クラッシュには至らず安全に停止でき、
//! 直後にシステムの空きメモリは約25GBまで正常回復したことを確認した。
//!
//! **推定される原因**: 「常時使う部分は概算5GB」という設計時の見積もりは
//! 攻撃的すぎた可能性が高い——(1) 1トークンの生成では、MoE層(26層)
//! それぞれで`num_experts_per_tok=6`個のエキスパートが**独立に**選ばれる
//! ため、1トークン目だけで最大`6×26=156`個のエキスパート読み込みが
//! 発生し得る(想定通り)が、(2) それに加えて、30GBという巨大な
//! ファイル群に対して散発的な`seek`+`read`を繰り返したことで、Windowsの
//! ファイルシステムキャッシュ(ページキャッシュ)自体が数十GB規模に
//! 膨張し、`FreePhysicalMemory`(真に空いている物理ページ数)を圧迫した
//! 可能性がある——このキャッシュは本来OSが必要に応じて回収可能だが、
//! 監視スクリプトが「回収されるかどうか」を待たずに安全側に倒して
//! 即座にkillした。いずれにせよ、**この開発機ではこの構成での実際の
//! 生成完走は達成できなかった**という事実を誇張せず記録する。
//!
//! **結論**: 遅延ロード設計は「ロード」フェーズについては明確に有効
//! (ヘッダのみ読み込みで19.9秒、実データを一括読み込みしない設計が
//! 実機で機能することを確認)だが、「生成」フェーズはMoEの疎性を
//! 加味してもなお、この開発機(実測32GB RAM)には荷が重いことが実測で
//! 判明した。次回への引き継ぎ: より小さいVRAM/RAM要求のMoEチェックポイント
//! の探索、またはOSページキャッシュを無効化する読み込みフラグ
//! (`FILE_FLAG_NO_BUFFERING`等)の使用、あるいはより大きなメモリを持つ
//! 環境での再検証が必要。詳細は`open-cuda/CLAUDE.md`の同日HANDOFF追記
//! 参照。
//!
//! `ModelWeights::load_sharded`のロード経路そのものは、合成の決定的な
//! 値で埋めた極小チェックポイントを2ファイルに分割してディスクへ実際に
//! 書き出すテスト
//! (`load_parses_sharded_safetensors_checkpoint_split_across_two_files`)、
//! および実際にMoE層を含む合成チェックポイントで`ExpertSlot::Lazy`が
//! 正しく機能することを検証するテスト
//! (`load_parses_moe_checkpoint_and_lazily_loads_selected_experts`)で
//! 検証済み。
//! - **学習専用ロジックは実装しない**(推論専用実装のため無関係):
//!   auxiliary loss計算・逆伝播・expert-parallelism分散シャーディング
//!   ・capacity factor等はすべて省略(調査で確認した通り、DeepSeek公式
//!   推論実装自体にもこれらは存在しない)。
//! - **CPU実装は素朴なループ**(llama.cppの`ggml_mul_mat_id`のような
//!   バッチ化最適化は無し)——正しさ優先、`n_routed_experts`本のうち
//!   `num_experts_per_tok`本だけを計算するので無駄な計算はしていないが、
//!   メモリアクセスパターンの最適化は次の増分。
//!
//! これにより`DeepseekConfig`が正しいMoEフィールドを持つ(V2-Lite等の)
//! チェックポイントは、`first_k_dense_replace`層目以降もエンドツーエンド
//! でロードできるようになった——ただし上記の通りV3固有の拡張
//! (aux-loss-free補正・group-limited routing・sigmoidスコアリング)には
//! まだ対応していないため、V3系の完全なロードは次の増分。
//! - **absorb最適化(推論高速化)は未実装**——調査で判明した通り、
//!   `kv_b_proj`のK側/V側をQ/O側へ数学的に吸収してKVキャッシュを
//!   圧縮ベクトルのまま保持する最適化があるが、今回は正しさの検証を
//!   優先し、`kv_b_proj`で毎ステップ素直に展開してから
//!   フル精度`k`/`v`をキャッシュする(`KvCacheHead`を`proj=None`で
//!   使う、後述)。したがってメモリ削減効果は実現できていない
//!   ——absorbは正しさが確定してからの最適化増分とする。
//! - **YaRN RoPEスケーリング(`rope_scaling`)は未対応**——実際の
//!   DeepSeek-V2-Liteの`config.json`は`rope_scaling.type="yarn"`を
//!   使っているが、YaRNの周波数補間・`mscale`補正は別途実装が要る
//!   ため、今回は`qwen_arch.rs`と同じ素朴なRoPE(`rope_theta`のみ)に
//!   とどめる。長コンテキストでの精度はYaRN無しでは実際のモデルと
//!   一致しない。
//! - `scaled_dot_product_attention`(既存共有ヘルパー)は**再利用できない**
//!   ——MLAは`q`/`k`の次元(`qk_nope_head_dim+qk_rope_head_dim`)と`v`の
//!   次元(`v_head_dim`)が非対称(実チェックポイントでは192 vs 128)で、
//!   既存ヘルパーは`q`/`k`/`v`が同一`head_dim`であることを前提にした
//!   単一引数設計のため使えない。このモジュールでは
//!   QKᵀ・softmax・P·Vを素朴なCPUループで直接計算する(GPU
//!   ディスパッチ・Vulkan/DXILオフロードは今回対応しない——`Linear`側の
//!   射影計算〈`q_a_proj`等〉はGPU対応する`Linear::forward`をそのまま
//!   使うが、Attentionコア自体はCPUのみ)。
//! - この開発機(GT730、VRAM 2GB)では実DeepSeekモデル(V2-Liteでも
//!   15.7Bパラメータ)は実行不可能なため、実重みでの検証は構造的な
//!   単体テスト(ランダム重み・極小構成)に限られる——`qwen_arch.rs`が
//!   Qwen2.5-0.5Bですら実行不可能だったのと同じ制約。

use std::path::Path;

use anyhow::{ensure, Context, Result};
use opencuda_core::GpuDevice;

use super::{argmax, apply_repetition_penalty, random_vec, tensor_f32, transpose, KvCacheHead, Linear, SplitMix64};

/// DeepSeek-V2/V3系MLAアーキテクチャの設定。実在する
/// `DeepSeek-V2-Lite-Chat/config.json`のフィールド名にそのまま対応する
/// (2026-09-13のGoogle/GitHub調査で確認、フィールド値の出典は
/// モジュールdoc参照)。MoE関連フィールド(`n_routed_experts`等)は
/// このモジュールのスコープ外(モジュールdoc参照)のため意図的に含めない。
#[derive(Debug, Clone, serde::Deserialize)]
pub struct DeepseekConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    #[serde(rename = "num_hidden_layers")]
    pub num_layers: usize,
    #[serde(rename = "num_attention_heads")]
    pub num_heads: usize,
    /// Query側低ランク圧縮の次元。`null`(小型モデル、例:
    /// DeepSeek-V2-Lite)なら`q_proj`を直接使い、`q_a_proj`/`q_b_proj`
    /// 二段構成は使わない。
    #[serde(default)]
    pub q_lora_rank: Option<usize>,
    /// KV側低ランク圧縮の次元(MLAの核心、必須)。
    pub kv_lora_rank: usize,
    /// RoPEを適用しない側のQ/K次元。
    pub qk_nope_head_dim: usize,
    /// RoPEを適用する側のQ/K次元(decoupled RoPE、全ヘッド共有のK側を
    /// `kv_a_proj_with_mqa`の後半として生成する)。
    pub qk_rope_head_dim: usize,
    /// Value側の次元(`qk_nope_head_dim+qk_rope_head_dim`とは独立)。
    pub v_head_dim: usize,
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

    // ── DeepSeekMoE(2026-09-13追加、モジュールdoc参照) ──────────────
    /// この層番号未満(0始まり)はdense SwiGLU、以降はMoE。実チェックポイント
    /// のconfig.jsonに必ず存在するフィールドだが、`tiny()`等の全層dense
    /// テスト構成向けに「デフォルトは全層dense」(`usize::MAX`)にしておく。
    #[serde(default = "default_first_k_dense_replace")]
    pub first_k_dense_replace: usize,
    #[serde(default)]
    pub n_routed_experts: usize,
    #[serde(default)]
    pub n_shared_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub norm_topk_prob: bool,
    /// `"softmax"`または`"sigmoid"`(V3系)。それ以外は`load()`が拒否する。
    #[serde(default = "default_scoring_func")]
    pub scoring_func: String,
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,
    /// グループ制限ルーティング(V3系)。`1`(既定)なら実質no-op。
    #[serde(default = "default_group")]
    pub n_group: usize,
    #[serde(default = "default_group")]
    pub topk_group: usize,
    /// `"greedy"`(既定、`e_score_correction_bias`無し、V2系)または
    /// `"noaux_tc"`(V3系、`gate.e_score_correction_bias`テンソルを
    /// ロードして選択にのみ使う——モジュールdoc「V3固有拡張」参照)。
    #[serde(default = "default_topk_method")]
    pub topk_method: String,
}

fn default_max_seq_len() -> usize {
    32768
}
fn default_rms_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    10_000.0
}
fn default_first_k_dense_replace() -> usize {
    usize::MAX
}
fn default_scoring_func() -> String {
    "softmax".to_string()
}
fn default_routed_scaling_factor() -> f32 {
    1.0
}
fn default_group() -> usize {
    1
}
fn default_topk_method() -> String {
    "greedy".to_string()
}

impl DeepseekConfig {
    /// テスト用の極小構成(`q_lora_rank=None`、実DeepSeek-V2-Liteの
    /// 「query側lora無し」パスを再現)。実運用サイズではない。
    pub fn tiny(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            q_lora_rank: None,
            kv_lora_rank: 8,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 4,
            v_head_dim: 4,
            intermediate_size: 64,
            max_seq_len: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            tie_word_embeddings: true,
            first_k_dense_replace: usize::MAX, // 全層dense(MoE無し)
            n_routed_experts: 0,
            n_shared_experts: 0,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            norm_topk_prob: false,
            scoring_func: "softmax".to_string(),
            routed_scaling_factor: 1.0,
            n_group: 1,
            topk_group: 1,
            topk_method: "greedy".to_string(),
        }
    }

    /// [`tiny`]のquery側lora有り版(実DeepSeek-V3が使う`q_a_proj`/
    /// `q_b_proj`二段構成の経路もテストするため)。
    pub fn tiny_with_q_lora(vocab_size: usize) -> Self {
        Self { q_lora_rank: Some(6), ..Self::tiny(vocab_size) }
    }

    /// [`tiny`]のDeepSeekMoE有り版(実V2-Liteと同じ「先頭`first_k_dense_replace`
    /// 層はdense、以降はMoE」構成、`num_layers=2`なので層0がdense・
    /// 層1がMoEになる)。
    pub fn tiny_with_moe(vocab_size: usize) -> Self {
        Self {
            first_k_dense_replace: 1,
            n_routed_experts: 4,
            n_shared_experts: 1,
            num_experts_per_tok: 2,
            moe_intermediate_size: 8,
            norm_topk_prob: false,
            scoring_func: "softmax".to_string(),
            routed_scaling_factor: 1.0,
            ..Self::tiny(vocab_size)
        }
    }

    /// [`tiny_with_moe`]のV3拡張版: `n_group>1`のgroup-limited routing・
    /// `topk_method="noaux_tc"`(`e_score_correction_bias`使用)・
    /// `scoring_func="sigmoid"`をすべて有効にする(実DeepSeek-V3の
    /// config.jsonが実際に使う組み合わせ、モジュールdoc「V3固有拡張」
    /// 参照)。`n_routed_experts=8`を`n_group=4`グループ(各2個)に分割し、
    /// `topk_group=2`で半分のグループへ絞ってから`num_experts_per_tok=2`
    /// を選ぶ。
    pub fn tiny_with_moe_v3(vocab_size: usize) -> Self {
        Self {
            n_routed_experts: 8,
            n_shared_experts: 1,
            num_experts_per_tok: 2,
            n_group: 4,
            topk_group: 2,
            topk_method: "noaux_tc".to_string(),
            scoring_func: "sigmoid".to_string(),
            norm_topk_prob: true,
            routed_scaling_factor: 2.5,
            ..Self::tiny_with_moe(vocab_size)
        }
    }

    fn q_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    fn is_moe_layer(&self, layer_idx: usize) -> bool {
        layer_idx >= self.first_k_dense_replace
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

/// `qwen_arch::rope_cos_sin`と同じ"rotate_half"方式(このモジュール専用に
/// 複製——`qk_rope_head_dim`はQwenの`head_dim`とは無関係な独立の次元の
/// ため、共有ヘルパー化するメリットが薄く、各アーキテクチャモジュールが
/// 自己完結する既存方針〈`qwen_arch.rs`のモジュールdoc参照〉に合わせる)。
fn rope_cos_sin(rope_dim: usize, theta: f32, pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half = rope_dim / 2;
    let mut cos = vec![0.0f32; half];
    let mut sin = vec![0.0f32; half];
    for i in 0..half {
        let freq = 1.0f32 / theta.powf((2 * i) as f32 / rope_dim as f32);
        let angle = pos as f32 * freq;
        cos[i] = angle.cos();
        sin[i] = angle.sin();
    }
    (cos, sin)
}

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

struct DeepseekLayer {
    input_layernorm: RmsNorm,
    /// `q_lora_rank`が`None`の場合に使う直接射影(`hidden -> num_heads*q_head_dim`)。
    q_proj: Option<Linear>,
    /// `q_lora_rank`が`Some`の場合に使う二段射影。
    q_a_proj: Option<Linear>,
    q_a_layernorm: Option<RmsNorm>,
    q_b_proj: Option<Linear>,
    /// `hidden -> kv_lora_rank + qk_rope_head_dim`(常に存在、MLAの核心)。
    kv_a_proj_with_mqa: Linear,
    /// `dim = kv_lora_rank`。
    kv_a_layernorm: RmsNorm,
    /// `kv_lora_rank -> num_heads*(qk_nope_head_dim+v_head_dim)`。
    kv_b_proj: Linear,
    /// `num_heads*v_head_dim -> hidden`。
    o_proj: Linear,
    post_attention_layernorm: RmsNorm,
    mlp: DeepseekMlp,
}

/// dense SwiGLU MLP一本ぶん(従来のdense MLPそのもの、MoEの各
/// エキスパート・共有エキスパートにも同じ形が使い回される)。
struct DenseSwiGlu {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl DenseSwiGlu {
    fn random(rng: &mut SplitMix64, hidden: usize, intermediate: usize) -> Self {
        Self { gate_proj: Linear::random(rng, hidden, intermediate), up_proj: Linear::random(rng, hidden, intermediate), down_proj: Linear::random(rng, intermediate, hidden) }
    }

    fn forward(&self, device: &dyn GpuDevice, x: &[f32]) -> Result<Vec<f32>> {
        let gate = self.gate_proj.forward(device, x, 1)?;
        let up = self.up_proj.forward(device, x, 1)?;
        let mut mlp_hidden = vec![0.0f32; gate.len()];
        for i in 0..gate.len() {
            mlp_hidden[i] = silu(gate[i]) * up[i];
        }
        self.down_proj.forward(device, &mlp_hidden, 1)
    }
}

/// DeepSeekMoE層(モジュールdoc「DeepSeekMoE対応」参照)。`gate`が
/// トークンごとに`num_experts_per_tok`個のルーティングされるエキスパート
/// (`experts`)を選び、常時計算される`shared_experts`の出力と合算する。
struct DeepseekMoeMlp {
    /// ルーター: `hidden -> n_routed_experts`(スコア用ロジット)。
    gate: Linear,
    /// `topk_method="noaux_tc"`(V3系)のみ`Some`。`[n_routed_experts]`長、
    /// 選択(top-k)にのみ影響し重みには使わない(モジュールdoc
    /// 「V3固有拡張」参照)。
    e_score_correction_bias: Option<Vec<f32>>,
    shared_experts: DenseSwiGlu,
    experts: Vec<ExpertSlot>,
}

/// ルーティングされる個々のエキスパートの遅延ロード枠(2026-09-13続き6
/// 追加、モジュールdoc「遅延ロード」参照)。`DeepSeekMoE`は1トークン
/// あたり`num_experts_per_tok`個(実チェックポイントでは64個中6個等)
/// しか使わないため、`DeepseekModel::load`時点では**メタデータ
/// (テンソル名・shape)だけ**を保持し、実際にそのエキスパートが
/// ルーターに選ばれて初めて`ModelWeights`からディスクを読んで
/// `DenseSwiGlu`を構築・キャッシュする。`load_random`(テスト用ランダム
/// 重み、規模が小さいので遅延の恩恵が無い)では最初から`Loaded`で
/// 構築する。
enum ExpertSlot {
    Loaded(DenseSwiGlu),
    Lazy { prefix: String, hidden: usize, intermediate: usize, cell: std::sync::OnceLock<DenseSwiGlu> },
}

impl ExpertSlot {
    fn get(&self, weights: Option<&ModelWeights>) -> Result<&DenseSwiGlu> {
        match self {
            ExpertSlot::Loaded(d) => Ok(d),
            ExpertSlot::Lazy { prefix, hidden, intermediate, cell } => {
                if let Some(existing) = cell.get() {
                    return Ok(existing);
                }
                let weights = weights.context("open-cuda-llm: DeepSeekMoE lazy expert requires ModelWeights but none was retained on this model (load_random path should never construct a Lazy slot)")?;
                let loaded = load_dense_swiglu(weights, prefix, *hidden, *intermediate)?;
                Ok(cell.get_or_init(|| loaded))
            }
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax_scores(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut scores: Vec<f32> = logits.iter().map(|&v| (v - max_logit).exp()).collect();
    let sum_exp: f32 = scores.iter().sum();
    for s in &mut scores {
        *s /= sum_exp;
    }
    scores
}

enum DeepseekMlp {
    Dense(Box<DenseSwiGlu>),
    Moe(Box<DeepseekMoeMlp>),
}

impl DeepseekMlp {
    fn forward(&self, device: &dyn GpuDevice, x: &[f32], cfg: &DeepseekConfig, weights: Option<&ModelWeights>) -> Result<Vec<f32>> {
        match self {
            DeepseekMlp::Dense(dense) => dense.forward(device, x),
            DeepseekMlp::Moe(moe) => {
                // ── ルーティング計算(モジュールdoc「V3固有拡張」参照) ──
                let logits = moe.gate.forward(device, x, 1)?;
                let scores = if cfg.scoring_func == "sigmoid" { logits.iter().map(|&v| sigmoid(v)).collect::<Vec<f32>>() } else { softmax_scores(&logits) };
                // 選択にのみbiasを使い、重みには「biasを足す前」の
                // original_scoresを使う(調査で確認した非自明な仕様、
                // モジュールdoc参照)。
                let original_scores = scores.clone();
                let mut select_scores = scores;
                if let Some(bias) = &moe.e_score_correction_bias {
                    for (s, b) in select_scores.iter_mut().zip(bias) {
                        *s += b;
                    }
                }

                if cfg.n_group > 1 {
                    let group_size = select_scores.len() / cfg.n_group;
                    let mut group_scores = vec![0.0f32; cfg.n_group];
                    for (g, group_score) in group_scores.iter_mut().enumerate() {
                        let group = &select_scores[g * group_size..(g + 1) * group_size];
                        *group_score = if moe.e_score_correction_bias.is_some() {
                            // bias有り(noaux_tc): グループ内上位2個の合計。
                            let mut sorted = group.to_vec();
                            sorted.sort_unstable_by(|a, b| b.partial_cmp(a).expect("open-cuda-llm: DeepSeekMoE group score must not be NaN"));
                            sorted.iter().take(2).sum()
                        } else {
                            // bias無し: グループ内最大値。
                            group.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                        };
                    }
                    let mut group_ranked: Vec<usize> = (0..cfg.n_group).collect();
                    group_ranked.sort_unstable_by(|&a, &b| group_scores[b].partial_cmp(&group_scores[a]).expect("open-cuda-llm: DeepSeekMoE group score must not be NaN"));
                    let kept_groups = &group_ranked[..cfg.topk_group];
                    for g in 0..cfg.n_group {
                        if !kept_groups.contains(&g) {
                            for s in &mut select_scores[g * group_size..(g + 1) * group_size] {
                                *s = f32::NEG_INFINITY;
                            }
                        }
                    }
                }

                let mut ranked: Vec<usize> = (0..select_scores.len()).collect();
                ranked.sort_unstable_by(|&a, &b| select_scores[b].partial_cmp(&select_scores[a]).expect("open-cuda-llm: DeepSeekMoE router score must not be NaN"));
                let selected = &ranked[..cfg.num_experts_per_tok];

                let mut route_weights: Vec<f32> = selected.iter().map(|&i| original_scores[i]).collect();
                if cfg.scoring_func == "sigmoid" || cfg.norm_topk_prob {
                    let sum: f32 = route_weights.iter().sum();
                    for w in &mut route_weights {
                        *w /= sum;
                    }
                }
                for w in &mut route_weights {
                    *w *= cfg.routed_scaling_factor;
                }

                let mut y = vec![0.0f32; x.len()];
                for (&expert_idx, &weight) in selected.iter().zip(&route_weights) {
                    // 選ばれたエキスパートだけをこの時点で遅延ロードする
                    // (モジュールdoc「遅延ロード」参照、`ExpertSlot::get`)。
                    let expert_out = moe.experts[expert_idx].get(weights)?.forward(device, x)?;
                    for (acc, v) in y.iter_mut().zip(&expert_out) {
                        *acc += weight * v;
                    }
                }

                let shared_out = moe.shared_experts.forward(device, x)?;
                for (acc, v) in y.iter_mut().zip(&shared_out) {
                    *acc += v;
                }
                Ok(y)
            }
        }
    }
}

impl DeepseekLayer {
    fn random(rng: &mut SplitMix64, cfg: &DeepseekConfig, layer_idx: usize) -> Self {
        let hidden = cfg.hidden_size;
        let q_head_dim = cfg.q_head_dim();
        let kv_a_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim;
        let kv_b_out = cfg.num_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim);
        let attn_out_dim = cfg.num_heads * cfg.v_head_dim;

        let (q_proj, q_a_proj, q_a_layernorm, q_b_proj) = match cfg.q_lora_rank {
            None => (Some(Linear::random(rng, hidden, cfg.num_heads * q_head_dim)), None, None, None),
            Some(q_lora_rank) => {
                (None, Some(Linear::random(rng, hidden, q_lora_rank)), Some(RmsNorm::identity(q_lora_rank, cfg.rms_norm_eps)), Some(Linear::random(rng, q_lora_rank, cfg.num_heads * q_head_dim)))
            }
        };

        let kv_a_proj_with_mqa = Linear::random(rng, hidden, kv_a_dim);
        let kv_b_proj = Linear::random(rng, cfg.kv_lora_rank, kv_b_out);
        let o_proj = Linear::random(rng, attn_out_dim, hidden);

        let mlp = if cfg.is_moe_layer(layer_idx) {
            DeepseekMlp::Moe(Box::new(DeepseekMoeMlp {
                gate: Linear::random(rng, hidden, cfg.n_routed_experts),
                e_score_correction_bias: if cfg.topk_method == "noaux_tc" { Some(random_vec(rng, cfg.n_routed_experts, 0.02)) } else { None },
                shared_experts: DenseSwiGlu::random(rng, hidden, cfg.moe_intermediate_size * cfg.n_shared_experts),
                experts: (0..cfg.n_routed_experts).map(|_| ExpertSlot::Loaded(DenseSwiGlu::random(rng, hidden, cfg.moe_intermediate_size))).collect(),
            }))
        } else {
            DeepseekMlp::Dense(Box::new(DenseSwiGlu::random(rng, hidden, cfg.intermediate_size)))
        };

        Self {
            input_layernorm: RmsNorm::identity(hidden, cfg.rms_norm_eps),
            q_proj,
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            kv_a_proj_with_mqa,
            kv_a_layernorm: RmsNorm::identity(cfg.kv_lora_rank, cfg.rms_norm_eps),
            kv_b_proj,
            o_proj,
            post_attention_layernorm: RmsNorm::identity(hidden, cfg.rms_norm_eps),
            mlp,
        }
    }
}

/// ヘッドごとのKVキャッシュ(`num_heads`本、absorb最適化なしのため
/// フル精度で`k`〈`q_head_dim`長〉/`v`〈`v_head_dim`長〉を保持する——
/// モジュールdocの「absorb未実装」参照)。既存の[`KvCacheHead`]を
/// `proj=None`で流用する(この経路では`k`/`v`の長さが異なっても問題
/// ない——`KvCacheHead::push`/`current_kv`の`None`分岐は単純な
/// `Vec`への追記・複製のみで、長さの一致を仮定していない)。
struct LayerCache {
    kv: Vec<KvCacheHead>,
}

impl LayerCache {
    fn new(num_heads: usize) -> Self {
        Self { kv: (0..num_heads).map(|_| KvCacheHead::empty()).collect() }
    }
}

pub struct DeepseekModel {
    config: DeepseekConfig,
    embed_tokens: Vec<f32>,
    layers: Vec<DeepseekLayer>,
    norm: RmsNorm,
    lm_head: Option<Linear>,
    /// `load()`経由でロードされたモデルのみ`Some`(遅延ロードされる
    /// `ExpertSlot::Lazy`がディスクを読むために必要、モジュールdoc
    /// 「遅延ロード」参照)。`load_random`は全エキスパートを最初から
    /// `Loaded`で構築するため`None`のままで問題ない。
    weights: Option<ModelWeights>,
}

impl DeepseekModel {
    pub fn load_random(config: DeepseekConfig, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let embed_tokens = random_vec(&mut rng, config.vocab_size * config.hidden_size, 0.02);
        let layers = (0..config.num_layers).map(|layer_idx| DeepseekLayer::random(&mut rng, &config, layer_idx)).collect();
        let norm = RmsNorm::identity(config.hidden_size, config.rms_norm_eps);
        let lm_head = if config.tie_word_embeddings { None } else { Some(Linear::random(&mut rng, config.hidden_size, config.vocab_size)) };
        Self { config, embed_tokens, layers, norm, lm_head, weights: None }
    }

    /// 実在の学習済み重み(`config.json` + 単一ファイルの
    /// `model.safetensors`)を読み込む。**正直な開示**: モジュールdoc
    /// 参照——MoE層(`mlp.experts.*`)には対応していないため、実在する
    /// フルサイズのDeepSeek-V2/V2-Lite/V3チェックポイントは
    /// `first_k_dense_replace`以降の層で必ず失敗する(このメソッドは
    /// 「MLA構成だが全層dense FFN」の仮想的/カスタムなチェックポイント、
    /// または将来のMoE対応後に使うための土台)。
    ///
    /// **2026-09-13(続き5)追記**: 分割済み(sharded、
    /// `model-00001-of-00004.safetensors`等+`model.safetensors.index.json`)
    /// チェックポイントに対応した——実際に`deepseek-ai/DeepSeek-V2-Lite-Chat`
    /// の`model.safetensors.index.json`を取得して確認したところ、実在の
    /// チェックポイントは31.4GB・4分割で配布されており、単一ファイルの
    /// `model.safetensors`しか読めない旧実装(`QwenModel::load`と同じ
    /// 制約を踏襲していた)では構造的にロードできないことが判明したため
    /// (`ModelWeights`参照)。
    pub fn load(dir: &Path) -> Result<Self> {
        let config_bytes = std::fs::read(dir.join("config.json")).with_context(|| format!("open-cuda-llm: failed to read {}/config.json", dir.display()))?;
        let config: DeepseekConfig = serde_json::from_slice(&config_bytes).context("open-cuda-llm: failed to parse DeepSeek config.json")?;
        ensure!(config.num_heads > 0, "open-cuda-llm: num_attention_heads must be > 0");
        ensure!(config.qk_rope_head_dim % 2 == 0, "open-cuda-llm: qk_rope_head_dim ({}) must be even (rotate_half RoPE pairs dimensions)", config.qk_rope_head_dim);
        if let Some(q_lora_rank) = config.q_lora_rank {
            ensure!(q_lora_rank > 0, "open-cuda-llm: q_lora_rank, when present, must be > 0");
        }
        ensure!(config.kv_lora_rank > 0, "open-cuda-llm: kv_lora_rank must be > 0");
        if config.first_k_dense_replace < config.num_layers {
            ensure!(
                config.scoring_func == "softmax" || config.scoring_func == "sigmoid",
                "open-cuda-llm: DeepseekModel::load: scoring_func '{}' is not supported — only \"softmax\" and \"sigmoid\" are implemented",
                config.scoring_func
            );
            ensure!(
                config.topk_method == "greedy" || config.topk_method == "noaux_tc",
                "open-cuda-llm: DeepseekModel::load: topk_method '{}' is not supported — only \"greedy\" (no correction bias) and \"noaux_tc\" (V3 aux-loss-free bias) are implemented",
                config.topk_method
            );
            ensure!(config.n_routed_experts > 0, "open-cuda-llm: n_routed_experts must be > 0 when first_k_dense_replace < num_hidden_layers (some layers are MoE)");
            ensure!(config.num_experts_per_tok > 0 && config.num_experts_per_tok <= config.n_routed_experts, "open-cuda-llm: num_experts_per_tok ({}) must be in 1..=n_routed_experts ({})", config.num_experts_per_tok, config.n_routed_experts);
            ensure!(config.moe_intermediate_size > 0, "open-cuda-llm: moe_intermediate_size must be > 0 when some layers are MoE");
            ensure!(config.n_group > 0 && config.n_routed_experts % config.n_group == 0, "open-cuda-llm: n_routed_experts ({}) must be a positive multiple of n_group ({})", config.n_routed_experts, config.n_group);
            ensure!(config.topk_group > 0 && config.topk_group <= config.n_group, "open-cuda-llm: topk_group ({}) must be in 1..=n_group ({})", config.topk_group, config.n_group);
        }

        let weights = ModelWeights::load(dir)?;

        let hidden = config.hidden_size;
        let q_head_dim = config.q_head_dim();
        let kv_a_dim = config.kv_lora_rank + config.qk_rope_head_dim;
        let kv_b_out = config.num_heads * (config.qk_nope_head_dim + config.v_head_dim);
        let attn_out_dim = config.num_heads * config.v_head_dim;

        let embed_tokens = weights.tensor_f32("model.embed_tokens.weight")?;
        ensure!(embed_tokens.len() == config.vocab_size * hidden, "open-cuda-llm: model.embed_tokens.weight has {} elements, expected {}x{}", embed_tokens.len(), config.vocab_size, hidden);

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("model.layers.{i}");
            let sa = format!("{p}.self_attn");

            let (q_proj, q_a_proj, q_a_layernorm, q_b_proj) = match config.q_lora_rank {
                None => (Some(load_linear(&weights, &format!("{sa}.q_proj"), hidden, config.num_heads * q_head_dim)?), None, None, None),
                Some(q_lora_rank) => (
                    None,
                    Some(load_linear(&weights, &format!("{sa}.q_a_proj"), hidden, q_lora_rank)?),
                    Some(RmsNorm { weight: weights.tensor_f32(&format!("{sa}.q_a_layernorm.weight"))?, eps: config.rms_norm_eps }),
                    Some(load_linear(&weights, &format!("{sa}.q_b_proj"), q_lora_rank, config.num_heads * q_head_dim)?),
                ),
            };

            layers.push(DeepseekLayer {
                input_layernorm: RmsNorm { weight: weights.tensor_f32(&format!("{p}.input_layernorm.weight"))?, eps: config.rms_norm_eps },
                q_proj,
                q_a_proj,
                q_a_layernorm,
                q_b_proj,
                kv_a_proj_with_mqa: load_linear(&weights, &format!("{sa}.kv_a_proj_with_mqa"), hidden, kv_a_dim)?,
                kv_a_layernorm: RmsNorm { weight: weights.tensor_f32(&format!("{sa}.kv_a_layernorm.weight"))?, eps: config.rms_norm_eps },
                kv_b_proj: load_linear(&weights, &format!("{sa}.kv_b_proj"), config.kv_lora_rank, kv_b_out)?,
                o_proj: load_linear(&weights, &format!("{sa}.o_proj"), attn_out_dim, hidden)?,
                post_attention_layernorm: RmsNorm { weight: weights.tensor_f32(&format!("{p}.post_attention_layernorm.weight"))?, eps: config.rms_norm_eps },
                mlp: load_mlp(&weights, &p, &config, i, hidden)?,
            });
        }

        let norm = RmsNorm { weight: weights.tensor_f32("model.norm.weight")?, eps: config.rms_norm_eps };

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            let raw = weights.tensor_f32("lm_head.weight")?;
            ensure!(raw.len() == config.vocab_size * hidden, "open-cuda-llm: lm_head.weight has {} elements, expected {}x{}", raw.len(), config.vocab_size, hidden);
            let weight_t = transpose(&raw, config.vocab_size, hidden);
            Some(Linear { weight_t, bias: vec![0.0; config.vocab_size], in_dim: hidden, out_dim: config.vocab_size, spirv_matmul: None, dxil_offload: None, fp8_weight: None })
        };

        Ok(Self { config, embed_tokens, layers, norm, lm_head, weights: Some(weights) })
    }

    fn new_caches(&self) -> Vec<LayerCache> {
        (0..self.config.num_layers).map(|_| LayerCache::new(self.config.num_heads)).collect()
    }

    /// 1トークンぶんを処理し、次トークン予測のlogits(`vocab_size`長)を返す。
    /// 計算順序はDeepSeek-V2の`modeling_deepseek.py`(モジュールdoc参照)の
    /// 実装通り(query低ランク展開 → KV低ランク展開 → decoupled RoPE →
    /// nope/rope結合 → 素朴なattention)。
    fn forward_step(&self, device: &dyn GpuDevice, token_id: u32, caches: &mut [LayerCache]) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let num_heads = cfg.num_heads;
        let nope_dim = cfg.qk_nope_head_dim;
        let rope_dim = cfg.qk_rope_head_dim;
        let v_dim = cfg.v_head_dim;
        let q_head_dim = cfg.q_head_dim();
        let pos = caches[0].kv[0].n;
        let (cos, sin) = rope_cos_sin(rope_dim, cfg.rope_theta, pos);

        let tok = token_id as usize;
        ensure!(tok < cfg.vocab_size, "open-cuda-llm: token id {tok} out of vocab range {}", cfg.vocab_size);
        let mut hidden_state = self.embed_tokens[tok * hidden..(tok + 1) * hidden].to_vec();

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            // ---- Attention サブ層(pre-norm) ----
            let mut normed = hidden_state.clone();
            layer.input_layernorm.forward_row(&mut normed);

            // Query側: 二段低ランク射影(q_lora_rank=Some)か直接射影(None)か。
            let q_full = match (&layer.q_a_proj, &layer.q_a_layernorm, &layer.q_b_proj) {
                (Some(q_a), Some(q_a_ln), Some(q_b)) => {
                    let mut lat = q_a.forward(device, &normed, 1)?;
                    q_a_ln.forward_row(&mut lat);
                    q_b.forward(device, &lat, 1)?
                }
                _ => layer.q_proj.as_ref().expect("open-cuda-llm: DeepseekLayer must have either q_proj or q_a_proj/q_b_proj").forward(device, &normed, 1)?,
            };
            debug_assert_eq!(q_full.len(), num_heads * q_head_dim);

            // KV側: 常に低ランク射影(MLAの核心)。後半`rope_dim`分は
            // 全ヘッド共有のMQA的RoPE専用鍵(`k_pe`)。
            let kv_a = layer.kv_a_proj_with_mqa.forward(device, &normed, 1)?;
            debug_assert_eq!(kv_a.len(), cfg.kv_lora_rank + rope_dim);
            let mut c_kv = kv_a[..cfg.kv_lora_rank].to_vec();
            let mut k_pe_shared = kv_a[cfg.kv_lora_rank..].to_vec();
            layer.kv_a_layernorm.forward_row(&mut c_kv);
            apply_rope(&mut k_pe_shared, &cos, &sin); // 全ヘッドで共有するため1回だけ回転させる

            let kv_full = layer.kv_b_proj.forward(device, &c_kv, 1)?;
            debug_assert_eq!(kv_full.len(), num_heads * (nope_dim + v_dim));

            let mut context = vec![0.0f32; num_heads * v_dim];
            for h in 0..num_heads {
                // ---- decoupled RoPE: nope部分はそのまま、rope部分だけ回転 ----
                let q_h = &q_full[h * q_head_dim..(h + 1) * q_head_dim];
                let mut q_h_rot = vec![0.0f32; q_head_dim];
                q_h_rot[..nope_dim].copy_from_slice(&q_h[..nope_dim]);
                let mut q_pe = q_h[nope_dim..].to_vec();
                apply_rope(&mut q_pe, &cos, &sin);
                q_h_rot[nope_dim..].copy_from_slice(&q_pe);

                let kv_h = &kv_full[h * (nope_dim + v_dim)..(h + 1) * (nope_dim + v_dim)];
                let k_nope_h = &kv_h[..nope_dim];
                let v_h = &kv_h[nope_dim..];
                let mut k_h = vec![0.0f32; q_head_dim];
                k_h[..nope_dim].copy_from_slice(k_nope_h);
                k_h[nope_dim..].copy_from_slice(&k_pe_shared); // 全ヘッド共有(MQA的)のk_peをそのまま結合

                cache.kv[h].push(device, &k_h, v_h, None, None)?;

                // ---- 素朴なattention(absorb最適化なし、モジュールdoc参照) ----
                let (k_all, v_all) = cache.kv[h].current_kv(device, q_head_dim, None, None)?;
                let n = cache.kv[h].n;
                let scale = 1.0f32 / (q_head_dim as f32).sqrt();
                let mut scores = vec![0.0f32; n];
                for (t, score) in scores.iter_mut().enumerate() {
                    let k_t = &k_all[t * q_head_dim..(t + 1) * q_head_dim];
                    *score = q_h_rot.iter().zip(k_t).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for score in &mut scores {
                    *score = (*score - max_score).exp();
                    sum_exp += *score;
                }
                let out_h = &mut context[h * v_dim..(h + 1) * v_dim];
                for (t, prob) in scores.iter().enumerate() {
                    let weight = prob / sum_exp;
                    let v_t = &v_all[t * v_dim..(t + 1) * v_dim];
                    for (o, v) in out_h.iter_mut().zip(v_t) {
                        *o += weight * v;
                    }
                }
            }

            let attn_out = layer.o_proj.forward(device, &context, 1)?;
            for (h, a) in hidden_state.iter_mut().zip(&attn_out) {
                *h += a;
            }

            // ---- MLP(dense SwiGLUまたはDeepSeekMoE)サブ層(pre-norm) ----
            let mut normed2 = hidden_state.clone();
            layer.post_attention_layernorm.forward_row(&mut normed2);
            let mlp_out = layer.mlp.forward(device, &normed2, cfg, self.weights.as_ref())?;
            for (h, m) in hidden_state.iter_mut().zip(&mlp_out) {
                *h += m;
            }
        }

        self.norm.forward_row(&mut hidden_state);

        let logits = match &self.lm_head {
            Some(head) => head.forward(device, &hidden_state, 1)?,
            None => {
                let mut logits = vec![0.0f32; cfg.vocab_size];
                for (v, row) in logits.iter_mut().zip(self.embed_tokens.chunks_exact(hidden)) {
                    *v = row.iter().zip(&hidden_state).map(|(a, b)| a * b).sum();
                }
                logits
            }
        };
        Ok(logits)
    }

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

/// 単一の`model.safetensors`、または`model.safetensors.index.json`+
/// 複数の`model-NNNNN-of-MMMMM.safetensors`という分割済み(sharded)
/// チェックポイントの両方を透過的に読めるようにする抽象化
/// (2026-09-13追加、`load()`のdocコメント参照)。
///
/// **2026-09-13(続き6)設計変更(遅延ロード)**: 当初は各シャードの
/// 生バイト列を丸ごとメモリへ読み込んでいたが(全シャード合計で
/// 実チェックポイントの場合31.4GB)、ユーザーから「システムメモリで
/// 90GB超になるなら、HDDのキャッシュを用意してもダメか」との指摘を
/// 受け再設計した。OSのページキャッシュは`std::fs::read`した内容を
/// 裏で自動キャッシュするだけで、Rustプロセス自身が二重に確保する
/// ヒープメモリ(変換後のf32配列)を減らしはしない——真の問題は
/// 「全テンソルを一括でf32へ変換し永続保持する」設計そのものだった。
///
/// そこで、各safetensorsファイルの**ヘッダ(数KB程度)だけ**を起動時に
/// 読み、テンソルごとの位置(ファイルパス・バイトオフセット・dtype・
/// shape)だけを`tensor_meta`に記録する。実データは
/// [`tensor_f32`](Self::tensor_f32)が呼ばれた時点で該当バイト範囲だけ
/// `seek`+`read`し、その場でf32へ変換する(ディスクを実質的な
/// バッキングストアとして使う設計)。さらにDeepSeekMoEの疎性
/// (1トークンあたり`n_routed_experts`個中`num_experts_per_tok`個しか
/// 使わない)を活かし、ルーティングされる個々のエキスパートは
/// [`ExpertSlot::Lazy`]として「実際に選ばれるまでこの`tensor_f32`すら
/// 呼ばない」設計にした(モジュールdoc・`ExpertSlot`参照)——これにより
/// 常駐メモリは「常時使う部分(attention・共有エキスパート・denseの
/// MLP)+実際に選ばれたエキスパートの累積」だけで済み、モデル全体を
/// 一括でf32展開する必要が無くなる。
struct ModelWeights {
    /// テンソル名 → (ファイルパス, ファイル内でのデータ開始バイト位置,
    /// dtype/shape/data_offsetsを含むsafetensorsヘッダのJSONエントリ)。
    tensor_meta: std::collections::HashMap<String, (std::path::PathBuf, u64, serde_json::Value)>,
}

#[derive(serde::Deserialize)]
struct SafetensorsIndex {
    weight_map: std::collections::HashMap<String, String>,
}

/// safetensorsファイルの先頭ヘッダだけを読む(データ本体は読まない)。
/// フォーマット: 先頭8バイトがヘッダ長(リトルエンディアンu64)、続く
/// その長さぶんがJSONヘッダ、以降がテンソル実データ
/// (<https://github.com/huggingface/safetensors>の仕様通り)。戻り値は
/// (データ本体の開始バイト位置, `__metadata__`を除いたヘッダのJSON map)。
fn read_safetensors_header(path: &Path) -> Result<(u64, serde_json::Map<String, serde_json::Value>)> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).with_context(|| format!("open-cuda-llm: failed to open {}", path.display()))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf).with_context(|| format!("open-cuda-llm: failed to read safetensors header length from {}", path.display()))?;
    let header_len = u64::from_le_bytes(len_buf);
    let mut header_buf = vec![0u8; header_len as usize];
    file.read_exact(&mut header_buf).with_context(|| format!("open-cuda-llm: failed to read safetensors header from {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&header_buf).with_context(|| format!("open-cuda-llm: failed to parse safetensors header JSON in {}", path.display()))?;
    let mut map = match value {
        serde_json::Value::Object(m) => m,
        _ => anyhow::bail!("open-cuda-llm: safetensors header in {} is not a JSON object", path.display()),
    };
    map.remove("__metadata__");
    Ok((8 + header_len, map))
}

impl ModelWeights {
    fn load(dir: &Path) -> Result<Self> {
        let index_path = dir.join("model.safetensors.index.json");
        if index_path.exists() {
            Self::load_sharded(dir, &index_path)
        } else {
            let path = dir.join("model.safetensors");
            let (data_start, header) = read_safetensors_header(&path)?;
            let tensor_meta = header.into_iter().map(|(name, entry)| (name, (path.clone(), data_start, entry))).collect();
            Ok(Self { tensor_meta })
        }
    }

    /// `model.safetensors.index.json`(`weight_map`: テンソル名→ファイル名)
    /// を読み、参照される各シャードファイルの**ヘッダだけ**を一度ずつ
    /// 読む(データ本体は読まない、モジュールdoc参照)。実在する
    /// `deepseek-ai/DeepSeek-V2-Lite-Chat`で実際に確認した形式
    /// (`model-00001-of-000004.safetensors`等、4分割・合計31.4GB)。
    fn load_sharded(dir: &Path, index_path: &Path) -> Result<Self> {
        let index_bytes = std::fs::read(index_path).with_context(|| format!("open-cuda-llm: failed to read {}", index_path.display()))?;
        let index: SafetensorsIndex = serde_json::from_slice(&index_bytes).context("open-cuda-llm: failed to parse model.safetensors.index.json")?;
        ensure!(!index.weight_map.is_empty(), "open-cuda-llm: model.safetensors.index.json has an empty weight_map");

        let mut header_cache: std::collections::HashMap<String, (u64, serde_json::Map<String, serde_json::Value>)> = std::collections::HashMap::new();
        let mut tensor_meta = std::collections::HashMap::with_capacity(index.weight_map.len());
        for (tensor_name, filename) in index.weight_map {
            if !header_cache.contains_key(&filename) {
                let shard_path = dir.join(&filename);
                let header = read_safetensors_header(&shard_path)?;
                header_cache.insert(filename.clone(), header);
            }
            let (data_start, header) = &header_cache[&filename];
            let entry = header.get(&tensor_name).with_context(|| format!("open-cuda-llm: tensor '{tensor_name}' listed in model.safetensors.index.json but not found in {filename}'s own header"))?.clone();
            tensor_meta.insert(tensor_name, (dir.join(&filename), *data_start, entry));
        }
        Ok(Self { tensor_meta })
    }

    /// 該当テンソルの実データバイト範囲だけをディスクから`seek`+`read`し、
    /// その場でf32へ変換する(モジュールdoc「遅延ロード」参照——モデル
    /// 全体を一括で読み込まない設計の核心部分)。`safetensors`クレートは
    /// バッファ全体からの解析しかサポートしないため、読み取った実データ
    /// バイト列だけを内容とする「1テンソルだけのsafetensorsバッファ」を
    /// その場で組み立てて`SafeTensors::deserialize`に通す(dtype変換
    /// ロジック自体は既存の`tensor_f32`をそのまま再利用するための
    /// 実装上の工夫、二重実装を避ける)。
    /// 実データを読まず、そのテンソル名がヘッダ上に存在するかだけを
    /// 確認する(`load_mlp`の遅延エキスパート契約チェック用)。
    fn contains(&self, name: &str) -> bool {
        self.tensor_meta.contains_key(name)
    }

    fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        use std::io::{Read, Seek, SeekFrom};

        let (path, data_start, entry) = self.tensor_meta.get(name).with_context(|| format!("open-cuda-llm: tensor '{name}' not found in checkpoint"))?;
        let offsets = entry.get("data_offsets").and_then(|v| v.as_array()).with_context(|| format!("open-cuda-llm: tensor '{name}' header entry is missing 'data_offsets'"))?;
        let begin = offsets[0].as_u64().context("open-cuda-llm: data_offsets[0] is not a u64")?;
        let end = offsets[1].as_u64().context("open-cuda-llm: data_offsets[1] is not a u64")?;
        ensure!(end >= begin, "open-cuda-llm: tensor '{name}' has invalid data_offsets [{begin}, {end}]");

        let mut file = std::fs::File::open(path).with_context(|| format!("open-cuda-llm: failed to open {}", path.display()))?;
        file.seek(SeekFrom::Start(data_start + begin)).with_context(|| format!("open-cuda-llm: failed to seek in {}", path.display()))?;
        let mut raw = vec![0u8; (end - begin) as usize];
        file.read_exact(&mut raw).with_context(|| format!("open-cuda-llm: failed to read tensor '{name}' data from {}", path.display()))?;

        let mut synthetic_entry = entry.clone();
        synthetic_entry["data_offsets"] = serde_json::json!([0, raw.len()]);
        let mut synthetic_header = serde_json::Map::new();
        synthetic_header.insert(name.to_string(), synthetic_entry);
        let header_json = serde_json::to_vec(&serde_json::Value::Object(synthetic_header)).context("open-cuda-llm: failed to build synthetic safetensors header")?;

        let mut buf = Vec::with_capacity(8 + header_json.len() + raw.len());
        buf.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        buf.extend_from_slice(&header_json);
        buf.extend_from_slice(&raw);

        let tensors = safetensors::SafeTensors::deserialize(&buf).with_context(|| format!("open-cuda-llm: failed to re-parse synthetic single-tensor buffer for '{name}'"))?;
        tensor_f32(&tensors, name)
    }
}

/// 層`layer_idx`のMLP部分をロードする(`config.first_k_dense_replace`に
/// 応じてdense/MoEを分岐、モジュールdoc「DeepSeekMoE対応」参照)。
fn load_mlp(weights: &ModelWeights, layer_prefix: &str, config: &DeepseekConfig, layer_idx: usize, hidden: usize) -> Result<DeepseekMlp> {
    if config.is_moe_layer(layer_idx) {
        let gate = load_linear(weights, &format!("{layer_prefix}.mlp.gate"), hidden, config.n_routed_experts)
            .with_context(|| format!("open-cuda-llm: layer {layer_idx}: MoE router 'mlp.gate' not found (layer_idx >= first_k_dense_replace={} so this layer is expected to be MoE)", config.first_k_dense_replace))?;
        let e_score_correction_bias = if config.topk_method == "noaux_tc" {
            let bias = weights
                .tensor_f32(&format!("{layer_prefix}.mlp.gate.e_score_correction_bias"))
                .with_context(|| format!("open-cuda-llm: layer {layer_idx}: topk_method=\"noaux_tc\" but 'mlp.gate.e_score_correction_bias' not found"))?;
            ensure!(bias.len() == config.n_routed_experts, "open-cuda-llm: layer {layer_idx}: 'mlp.gate.e_score_correction_bias' has {} elements, expected {}", bias.len(), config.n_routed_experts);
            Some(bias)
        } else {
            None
        };
        let shared_experts = load_dense_swiglu(weights, &format!("{layer_prefix}.mlp.shared_experts"), hidden, config.moe_intermediate_size * config.n_shared_experts)?;
        // ── ルーティングされる個々のエキスパートは遅延ロード ──
        // (モジュールdoc「遅延ロード」・`ExpertSlot`参照)。実データは
        // 読まないが、テンソル名の存在だけは`load()`の時点で検証し、
        // 壊れたチェックポイントを生成の途中ではなくロード時点で
        // 検知できるようにする(`ExpertSlot::get`が呼ばれるまで実データを
        // 読まない設計でも、契約違反の早期発見は諦めない)。
        let mut experts = Vec::with_capacity(config.n_routed_experts);
        for e in 0..config.n_routed_experts {
            let prefix = format!("{layer_prefix}.mlp.experts.{e}");
            ensure!(
                weights.contains(&format!("{prefix}.gate_proj.weight")),
                "open-cuda-llm: layer {layer_idx}: expert {e} tensor '{prefix}.gate_proj.weight' not found in checkpoint"
            );
            experts.push(ExpertSlot::Lazy { prefix, hidden, intermediate: config.moe_intermediate_size, cell: std::sync::OnceLock::new() });
        }
        Ok(DeepseekMlp::Moe(Box::new(DeepseekMoeMlp { gate, e_score_correction_bias, shared_experts, experts })))
    } else {
        Ok(DeepseekMlp::Dense(Box::new(load_dense_swiglu(weights, &format!("{layer_prefix}.mlp"), hidden, config.intermediate_size)?)))
    }
}

fn load_dense_swiglu(weights: &ModelWeights, prefix: &str, hidden: usize, intermediate: usize) -> Result<DenseSwiGlu> {
    Ok(DenseSwiGlu {
        gate_proj: load_linear(weights, &format!("{prefix}.gate_proj"), hidden, intermediate)?,
        up_proj: load_linear(weights, &format!("{prefix}.up_proj"), hidden, intermediate)?,
        down_proj: load_linear(weights, &format!("{prefix}.down_proj"), intermediate, hidden)?,
    })
}

fn load_linear(weights: &ModelWeights, prefix: &str, in_dim: usize, out_dim: usize) -> Result<Linear> {
    let raw = weights.tensor_f32(&format!("{prefix}.weight"))?;
    ensure!(raw.len() == out_dim * in_dim, "open-cuda-llm: '{prefix}.weight' has {} elements, expected {}x{}", raw.len(), out_dim, in_dim);
    let weight_t = transpose(&raw, out_dim, in_dim);
    let bias = weights.tensor_f32(&format!("{prefix}.bias")).unwrap_or_else(|_| vec![0.0; out_dim]);
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
        let config = DeepseekConfig::tiny(64);
        let model = DeepseekModel::load_random(config, 42);
        let device = device();
        let generated = model.generate(&device, &[1, 2, 3], 8).unwrap();
        assert_eq!(generated.len(), 8);
        for &t in &generated {
            assert!((t as usize) < 64, "generated token {t} out of vocab range");
        }
    }

    /// `q_lora_rank=Some`(実DeepSeek-V3が使う二段query射影経路)でも
    /// パニックせず完走すること。
    #[test]
    fn q_lora_rank_two_stage_query_projection_path_works() {
        let config = DeepseekConfig::tiny_with_q_lora(64);
        assert!(config.q_lora_rank.is_some());
        let model = DeepseekModel::load_random(config, 7);
        let device = device();
        let generated = model.generate(&device, &[0, 1], 5).unwrap();
        assert_eq!(generated.len(), 5);
    }

    /// `q_lora_rank=None`(実DeepSeek-V2-Liteが使う直接query射影経路)。
    #[test]
    fn q_lora_rank_none_direct_query_projection_path_works() {
        let config = DeepseekConfig::tiny(64);
        assert!(config.q_lora_rank.is_none());
        let model = DeepseekModel::load_random(config, 8);
        let device = device();
        let generated = model.generate(&device, &[0, 1], 5).unwrap();
        assert_eq!(generated.len(), 5);
    }

    /// 非対称なQ/K次元(`q_head_dim`=nope+rope)とV次元(`v_head_dim`)が
    /// 異なっていても(実チェックポイントでは192 vs 128)、素朴な
    /// attention実装が正しく動作すること(共有ヘルパーを再利用できない
    /// 理由そのものを検証する回帰テスト)。
    #[test]
    fn asymmetric_qk_and_v_head_dims_do_not_panic() {
        let mut config = DeepseekConfig::tiny(32);
        config.qk_nope_head_dim = 6;
        config.qk_rope_head_dim = 2;
        config.v_head_dim = 3; // q_head_dim=8, v_head_dim=3 (非対称)
        let model = DeepseekModel::load_random(config, 9);
        let device = device();
        let generated = model.generate(&device, &[1, 2, 3], 6).unwrap();
        assert_eq!(generated.len(), 6);
    }

    #[test]
    fn generation_is_deterministic_for_a_fixed_seed() {
        let device = device();
        let a = DeepseekModel::load_random(DeepseekConfig::tiny(50), 99).generate(&device, &[1, 2], 6).unwrap();
        let b = DeepseekModel::load_random(DeepseekConfig::tiny(50), 99).generate(&device, &[1, 2], 6).unwrap();
        assert_eq!(a, b);
    }

    /// 異なるプロンプトが同一出力へ退化していないことのヘルスチェック
    /// (RoPE/RMSNorm/低ランク射影配線のどこかが恒等的に無効化される
    /// 実装ミスの検出用、`qwen_arch.rs`と同じ趣旨)。
    #[test]
    fn different_prompts_do_not_collapse_to_identical_output() {
        let device = device();
        let model = DeepseekModel::load_random(DeepseekConfig::tiny(80), 5);
        let out1 = model.generate(&device, &[10], 6).unwrap();
        let out2 = model.generate(&device, &[20], 6).unwrap();
        assert_ne!(out1, out2, "different prompts should not collapse to identical output");
    }

    /// DeepSeekMoE(2026-09-13追加): `first_k_dense_replace=1`で層0が
    /// dense・層1がMoEという実V2-Liteと同じ構成でもパニックせず完走する
    /// こと(ルーター→top-k選択→共有エキスパート合算の配線を検証)。
    #[test]
    fn deepseekmoe_layer_generates_without_panicking() {
        let config = DeepseekConfig::tiny_with_moe(64);
        assert_eq!(config.first_k_dense_replace, 1);
        assert!(config.is_moe_layer(1));
        assert!(!config.is_moe_layer(0));
        let model = DeepseekModel::load_random(config, 21);
        let device = device();
        let generated = model.generate(&device, &[1, 2, 3], 6).unwrap();
        assert_eq!(generated.len(), 6);
    }

    /// `num_experts_per_tok == n_routed_experts`(全エキスパートを常時
    /// 使う、top-k選択の境界条件)でも壊れないこと。
    #[test]
    fn deepseekmoe_with_all_experts_selected_works() {
        let mut config = DeepseekConfig::tiny_with_moe(64);
        config.num_experts_per_tok = config.n_routed_experts; // 4 == 4
        let model = DeepseekModel::load_random(config, 22);
        let device = device();
        let generated = model.generate(&device, &[1, 2], 5).unwrap();
        assert_eq!(generated.len(), 5);
    }

    /// `norm_topk_prob=true`(選択後の重み再正規化)経路もパニックしない
    /// こと。
    #[test]
    fn deepseekmoe_with_norm_topk_prob_works() {
        let mut config = DeepseekConfig::tiny_with_moe(64);
        config.norm_topk_prob = true;
        let model = DeepseekModel::load_random(config, 23);
        let device = device();
        let generated = model.generate(&device, &[3, 4], 5).unwrap();
        assert_eq!(generated.len(), 5);
    }

    /// V3拡張(2026-09-13続き4追加): aux-loss-free補正
    /// (`e_score_correction_bias`)+group-limited routing(`n_group=4`/
    /// `topk_group=2`)+sigmoidスコアリングを全て有効にした構成
    /// (`tiny_with_moe_v3`)でもパニックせず完走すること。
    #[test]
    fn deepseekmoe_v3_extensions_generate_without_panicking() {
        let config = DeepseekConfig::tiny_with_moe_v3(64);
        assert_eq!(config.topk_method, "noaux_tc");
        assert_eq!(config.scoring_func, "sigmoid");
        assert!(config.n_group > 1);
        let model = DeepseekModel::load_random(config, 31);
        let device = device();
        let generated = model.generate(&device, &[1, 2, 3], 6).unwrap();
        assert_eq!(generated.len(), 6);
    }

    /// group-limited routingが実際にグループ外のエキスパートを除外して
    /// いることの間接検証: `topk_group=1`(最も絞り込んだ設定)でも
    /// `num_experts_per_tok`個選べる(選択候補がグループ内に十分残る)
    /// ことを確認する(境界条件、`n_group=4`・グループサイズ2・
    /// `num_experts_per_tok=2`なら`topk_group=1`でグループ内2個ちょうど
    /// 選べる)。
    #[test]
    fn deepseekmoe_v3_group_limited_routing_with_minimal_topk_group_works() {
        let mut config = DeepseekConfig::tiny_with_moe_v3(64);
        config.topk_group = 1;
        let model = DeepseekModel::load_random(config, 32);
        let device = device();
        let generated = model.generate(&device, &[4, 5], 5).unwrap();
        assert_eq!(generated.len(), 5);
    }

    /// **2026-09-13(続き5)追加**: 分割済み(sharded)safetensors
    /// チェックポイント(`model.safetensors.index.json`+複数の
    /// `.safetensors`ファイル)を実際にディスクへ書き出し、
    /// `DeepseekModel::load`が正しく読み込めることを検証する。
    /// `deepseek-ai/DeepSeek-V2-Lite-Chat`が実際に4分割・31.4GBで
    /// 配布されていることを確認した上で追加した`ModelWeights::
    /// load_sharded`の直接的な回帰テスト(実際にダウンロードするには
    /// 大きすぎるため、合成の決定的な値で埋めた極小チェックポイントを
    /// 2ファイルに分割して同じ経路を検証する)。
    #[test]
    fn load_parses_sharded_safetensors_checkpoint_split_across_two_files() {
        use safetensors::tensor::{Dtype, TensorView};
        use std::collections::HashMap;

        let vocab = 10usize;
        let hidden = 8usize;
        let num_heads = 2usize;
        let kv_lora_rank = 4usize;
        let qk_nope = 2usize;
        let qk_rope = 2usize;
        let v_dim = 2usize;
        let q_head_dim = qk_nope + qk_rope;
        let intermediate = 8usize;
        let kv_a_dim = kv_lora_rank + qk_rope;
        let kv_b_out = num_heads * (qk_nope + v_dim);
        let attn_out_dim = num_heads * v_dim;

        let mut rng = SplitMix64::new(777);
        let push = |buffers: &mut Vec<(String, Vec<usize>, Vec<u8>)>, name: String, shape: Vec<usize>, rng: &mut SplitMix64| {
            let len: usize = shape.iter().product();
            let bytes: Vec<u8> = random_vec(rng, len, 0.1).iter().flat_map(|v| v.to_le_bytes()).collect();
            buffers.push((name, shape, bytes));
        };

        // シャード1: embed_tokens + attention側の前半(q_proj〜kv_a_layernorm)。
        let mut shard1: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        push(&mut shard1, "model.embed_tokens.weight".to_string(), vec![vocab, hidden], &mut rng);
        push(&mut shard1, "model.layers.0.input_layernorm.weight".to_string(), vec![hidden], &mut rng);
        push(&mut shard1, "model.layers.0.self_attn.q_proj.weight".to_string(), vec![num_heads * q_head_dim, hidden], &mut rng);
        push(&mut shard1, "model.layers.0.self_attn.kv_a_proj_with_mqa.weight".to_string(), vec![kv_a_dim, hidden], &mut rng);
        push(&mut shard1, "model.layers.0.self_attn.kv_a_layernorm.weight".to_string(), vec![kv_lora_rank], &mut rng);

        // シャード2: attention側の後半(kv_b_proj〜o_proj)+MLP+最終norm。
        let mut shard2: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        push(&mut shard2, "model.layers.0.self_attn.kv_b_proj.weight".to_string(), vec![kv_b_out, kv_lora_rank], &mut rng);
        push(&mut shard2, "model.layers.0.self_attn.o_proj.weight".to_string(), vec![hidden, attn_out_dim], &mut rng);
        push(&mut shard2, "model.layers.0.post_attention_layernorm.weight".to_string(), vec![hidden], &mut rng);
        push(&mut shard2, "model.layers.0.mlp.gate_proj.weight".to_string(), vec![intermediate, hidden], &mut rng);
        push(&mut shard2, "model.layers.0.mlp.up_proj.weight".to_string(), vec![intermediate, hidden], &mut rng);
        push(&mut shard2, "model.layers.0.mlp.down_proj.weight".to_string(), vec![hidden, intermediate], &mut rng);
        push(&mut shard2, "model.norm.weight".to_string(), vec![hidden], &mut rng);

        let dir = std::env::temp_dir().join(format!("open-cuda-llm-deepseek-sharded-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let mut weight_map: HashMap<String, String> = HashMap::new();
        for (shard_name, buffers) in [("model-00001-of-00002.safetensors", &shard1), ("model-00002-of-00002.safetensors", &shard2)] {
            let mut views: HashMap<String, TensorView> = HashMap::new();
            for (name, shape, bytes) in buffers {
                views.insert(name.clone(), TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap());
                weight_map.insert(name.clone(), shard_name.to_string());
            }
            let serialized = safetensors::serialize(&views, &None).unwrap();
            std::fs::write(dir.join(shard_name), serialized).unwrap();
        }

        let index_json = serde_json::json!({ "metadata": { "total_size": 0 }, "weight_map": weight_map });
        std::fs::write(dir.join("model.safetensors.index.json"), serde_json::to_vec(&index_json).unwrap()).unwrap();
        std::fs::write(
            dir.join("config.json"),
            format!(
                r#"{{"vocab_size":{vocab},"hidden_size":{hidden},"num_hidden_layers":1,"num_attention_heads":{num_heads},
                "kv_lora_rank":{kv_lora_rank},"qk_nope_head_dim":{qk_nope},"qk_rope_head_dim":{qk_rope},"v_head_dim":{v_dim},
                "intermediate_size":{intermediate},"tie_word_embeddings":true}}"#
            ),
        )
        .unwrap();

        let model = DeepseekModel::load(&dir).expect("DeepseekModel::load should read a sharded checkpoint split across two safetensors files");
        let device = device();
        let generated = model.generate(&device, &[1, 2, 3], 4).unwrap();
        assert_eq!(generated.len(), 4);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// **2026-09-13(続き6)追加**: `DeepseekModel::load`が実際にDeepSeekMoE
    /// 層(`first_k_dense_replace=1`で層0=dense・層1=MoE)を含む合成
    /// safetensorsから読み込み、遅延ロードされる`ExpertSlot::Lazy`が
    /// 実際にディスクから正しく読めることを検証する(ここまでの
    /// `deepseekmoe_*`系テストは全て`load_random`〈=`ExpertSlot::Loaded`〉
    /// 経由だったため、`load()`から`ExpertSlot::Lazy`が正しく機能する
    /// ことの直接的な回帰テストが無かった——ユーザー指摘を受けた
    /// メモリ最適化〈遅延ロード〉の実装そのものを検証する)。
    #[test]
    fn load_parses_moe_checkpoint_and_lazily_loads_selected_experts() {
        use safetensors::tensor::{Dtype, TensorView};
        use std::collections::HashMap;

        let vocab = 12usize;
        let hidden = 8usize;
        let num_heads = 2usize;
        let kv_lora_rank = 4usize;
        let qk_nope = 2usize;
        let qk_rope = 2usize;
        let v_dim = 2usize;
        let q_head_dim = qk_nope + qk_rope;
        let intermediate = 8usize;
        let kv_a_dim = kv_lora_rank + qk_rope;
        let kv_b_out = num_heads * (qk_nope + v_dim);
        let attn_out_dim = num_heads * v_dim;
        let n_routed_experts = 4usize;
        let n_shared_experts = 1usize;
        let num_experts_per_tok = 2usize;
        let moe_intermediate = 4usize;

        let mut rng = SplitMix64::new(4242);
        let mut buffers: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        let mut push = |name: String, shape: Vec<usize>, rng: &mut SplitMix64| {
            let len: usize = shape.iter().product();
            let bytes: Vec<u8> = random_vec(rng, len, 0.1).iter().flat_map(|v| v.to_le_bytes()).collect();
            buffers.push((name, shape, bytes));
        };

        push("model.embed_tokens.weight".to_string(), vec![vocab, hidden], &mut rng);
        for layer_idx in 0..2 {
            let p = format!("model.layers.{layer_idx}");
            push(format!("{p}.input_layernorm.weight"), vec![hidden], &mut rng);
            push(format!("{p}.self_attn.q_proj.weight"), vec![num_heads * q_head_dim, hidden], &mut rng);
            push(format!("{p}.self_attn.kv_a_proj_with_mqa.weight"), vec![kv_a_dim, hidden], &mut rng);
            push(format!("{p}.self_attn.kv_a_layernorm.weight"), vec![kv_lora_rank], &mut rng);
            push(format!("{p}.self_attn.kv_b_proj.weight"), vec![kv_b_out, kv_lora_rank], &mut rng);
            push(format!("{p}.self_attn.o_proj.weight"), vec![hidden, attn_out_dim], &mut rng);
            push(format!("{p}.post_attention_layernorm.weight"), vec![hidden], &mut rng);
            if layer_idx == 0 {
                // first_k_dense_replace=1 なので層0はdense。
                push(format!("{p}.mlp.gate_proj.weight"), vec![intermediate, hidden], &mut rng);
                push(format!("{p}.mlp.up_proj.weight"), vec![intermediate, hidden], &mut rng);
                push(format!("{p}.mlp.down_proj.weight"), vec![hidden, intermediate], &mut rng);
            } else {
                // 層1はMoE。
                push(format!("{p}.mlp.gate.weight"), vec![n_routed_experts, hidden], &mut rng);
                push(format!("{p}.mlp.shared_experts.gate_proj.weight"), vec![moe_intermediate * n_shared_experts, hidden], &mut rng);
                push(format!("{p}.mlp.shared_experts.up_proj.weight"), vec![moe_intermediate * n_shared_experts, hidden], &mut rng);
                push(format!("{p}.mlp.shared_experts.down_proj.weight"), vec![hidden, moe_intermediate * n_shared_experts], &mut rng);
                for e in 0..n_routed_experts {
                    push(format!("{p}.mlp.experts.{e}.gate_proj.weight"), vec![moe_intermediate, hidden], &mut rng);
                    push(format!("{p}.mlp.experts.{e}.up_proj.weight"), vec![moe_intermediate, hidden], &mut rng);
                    push(format!("{p}.mlp.experts.{e}.down_proj.weight"), vec![hidden, moe_intermediate], &mut rng);
                }
            }
        }
        push("model.norm.weight".to_string(), vec![hidden], &mut rng);

        let mut views: HashMap<String, TensorView> = HashMap::new();
        for (name, shape, bytes) in &buffers {
            views.insert(name.clone(), TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap());
        }
        let serialized = safetensors::serialize(&views, &None).unwrap();

        let dir = std::env::temp_dir().join(format!("open-cuda-llm-deepseek-moe-load-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), serialized).unwrap();
        std::fs::write(
            dir.join("config.json"),
            format!(
                r#"{{"vocab_size":{vocab},"hidden_size":{hidden},"num_hidden_layers":2,"num_attention_heads":{num_heads},
                "kv_lora_rank":{kv_lora_rank},"qk_nope_head_dim":{qk_nope},"qk_rope_head_dim":{qk_rope},"v_head_dim":{v_dim},
                "intermediate_size":{intermediate},"tie_word_embeddings":true,
                "first_k_dense_replace":1,"n_routed_experts":{n_routed_experts},"n_shared_experts":{n_shared_experts},
                "num_experts_per_tok":{num_experts_per_tok},"moe_intermediate_size":{moe_intermediate},
                "scoring_func":"softmax","topk_method":"greedy"}}"#
            ),
        )
        .unwrap();

        let model = DeepseekModel::load(&dir).expect("DeepseekModel::load should read a MoE checkpoint and lazily load only the experts actually selected");
        let device = device();
        let generated = model.generate(&device, &[1, 2, 3], 5).unwrap();
        assert_eq!(generated.len(), 5);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// decoupled RoPE配線の健全性: rope専用部分を意図的に無効化(cos=1,
    /// sin=0相当になる`rope_theta`を極端値にする、等)は難しいため、
    /// 代わりに「位置を変えると出力が変わる」という間接的な検証を行う
    /// (同一トークン列でも生成が1トークン進むごとに`pos`が変わり、
    /// 何らかの形でRoPEが効いていることの健全性チェック)。
    #[test]
    fn generation_advances_position_dependent_state_across_steps() {
        let device = device();
        let model = DeepseekModel::load_random(DeepseekConfig::tiny(40), 11);
        // 同じトークンを繰り返すプロンプトでも、位置依存(RoPE)が効いて
        // いれば各ステップのlogitsは単純に同一値へ潰れない
        // (退化していないことは`generate`が末尾で同一トークンの無限
        // 繰り返しに陥っていないことでも間接的に確認できる)。
        let generated = model.generate(&device, &[3, 3, 3], 10).unwrap();
        assert_eq!(generated.len(), 10);
        let all_same = generated.iter().all(|&t| t == generated[0]);
        assert!(!all_same, "position-dependent RoPE should prevent total degeneration into a single repeated token");
    }
}
