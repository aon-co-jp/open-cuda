# open-cuda

*English*: [README-English.md](README-English.md) ·
*Other languages*: [Deutsch](README-German.md) · [Italiano](README-Italian.md) ·
[Français](README-French.md) · [Русский](README-Russian.md) ·
[Українська](README-Ukrainian.md) · [עברית](README-Hebrew.md) ·
[فارسی](README-Persian.md) · [العربية](README-Arabic.md)

> 📌 **最近の更新(2026-09-13続き7・実機検証再挑戦で成功)**: 世界中の
> 言語でのGoogle検索・GitHub調査に基づき、`FILE_FLAG_RANDOM_ACCESS`
> ヒント(Windows、先読み抑制)を追加し、監視指標も
> `FreePhysicalMemory`から`Available MBytes`(OSが回収可能な
> キャッシュを含む正しい「空き」指標)へ修正した上で
> `deepseek-ai/DeepSeek-V2-Lite-Chat`を再検証したところ、**今回は
> クラッシュなく生成が完走した**(ロード18.2秒、3トークン生成96.8秒
> ≒0.03 tok/s)。生成中にプロセスメモリは一時22GB程度まで増加したが、
> Windowsのワーキングセットトリミングで自然に縮小し安全に完了した。
> 速度は実用水準に遠いが、MLA+DeepSeekMoE+decoupled RoPE+遅延ロードが
> この開発機でエンドツーエンドに動作することを実機で確認できた。
> 詳細は[CLAUDE.md](CLAUDE.md)参照。
>
> *English*: Following multilingual research, added a
> `FILE_FLAG_RANDOM_ACCESS` hint (Windows, suppresses read-ahead) and
> fixed the monitoring metric from `FreePhysicalMemory` to `Available
> MBytes` (the correct "available" figure including OS-reclaimable
> cache). Re-verifying `deepseek-ai/DeepSeek-V2-Lite-Chat` with these
> fixes, **generation completed without a crash this time** (18.2s
> load, 96.8s for 3 tokens, ~0.03 tok/s). Process memory briefly rose
> to ~22GB during generation but shrank naturally via Windows working
> set trimming. Speed is far from practical, but the full MLA +
> DeepSeekMoE + decoupled RoPE + lazy-loading pipeline was confirmed
> working end-to-end on real hardware. See [CLAUDE.md](CLAUDE.md).

> 📌 **最近の更新(2026-09-13続き6・実機検証結果)**: 実際に
> `deepseek-ai/DeepSeek-V2-Lite-Chat`(31.4GB)をダウンロードし実機検証を
> 実施。**ロードは成功**(遅延ロード設計により19.9秒で完了)したが、
> **生成開始直後にプロセスメモリが急増しシステムの空きメモリが
> 127.5MBまで低下**、安全装置が自動的にプロセスを強制終了した(kill後
> システムは正常に25GB空きまで回復、クラッシュには至らず)。「実機
> 検証した」と誇張せず、生成の完走は達成できなかった事実を記録する。
> 原因はMoEルーティングの疎性を加味しても足りないメモリ要求+巨大
> ファイル群への散発読み込みによるOSファイルキャッシュ膨張と推定。
> 詳細は[CLAUDE.md](CLAUDE.md)参照。
>
> *English*: Actually downloaded and attempted real-checkpoint
> verification of `deepseek-ai/DeepSeek-V2-Lite-Chat` (31.4GB).
> **Loading succeeded** (19.9s, thanks to the lazy-loading design), but
> **generation triggered a memory spike that dropped system free
> memory to 127.5MB within seconds**; a safety monitor force-killed the
> process (system recovered cleanly to ~25GB free afterward — no
> crash). Recording honestly that generation did not complete on this
> machine. See [CLAUDE.md](CLAUDE.md) for the full analysis.

> 📌 **最近の更新(2026-09-13続き6)**: ユーザー指摘「HDDのキャッシュを
> 用意してもダメか」を受け、`ModelWeights`をヘッダのみ読み込み+
> オンデマンド`seek`+`read`方式へ再設計し、ルーティングされる個々の
> エキスパートを実際に選ばれるまでディスクを読まない`ExpertSlot::Lazy`
> とした。これにより常時使う部分だけならf32で概算5GB程度、短い検証
> (数トークン生成)なら十数GB程度に収まる見込みとなり、直下のエントリで
> 「見送った」実機検証を実際に試みた——結果は[CLAUDE.md](CLAUDE.md)の
> 続報エントリ参照。新規テスト1本追加、クレート全体71本成功。
>
> *English*: Following the user's question ("wouldn't a disk cache
> help?"), redesigned `ModelWeights` to parse only safetensors headers
> and seek+read tensor data on demand, and made routed experts
> (`ExpertSlot::Lazy`) load from disk only when actually selected by
> the router. This brings the always-resident footprint down to ~5GB
> (f32) and a short verification run (a few generated tokens) to
> roughly 10-20GB — within this dev machine's free RAM — so the
> real-checkpoint verification scoped out in the entry below was
> actually attempted. See the follow-up entry in
> [CLAUDE.md](CLAUDE.md) for the outcome. 1 new test added, full
> crate suite (71 tests) passing.

> 📌 **最近の更新(2026-09-13続き5)**: `DeepseekModel`にDeepSeek-V3固有の
> MoE拡張(aux-loss-free補正・group-limited routing・sigmoidスコア
> リング)を実装、加えて実チェックポイントが分割済み(sharded、4分割・
> 31.4GB)配布であることが判明したため分割safetensorsローダーを新設した。
> **正直な開示**: 実機ダウンロード・ロード検証は見送った——現在の
> ローダー設計ではピークメモリが90GBを超える可能性が高く、この開発機
> (実測: 総RAM32GB・空き16GB)では実行するとクラッシュ・システム不安定化
> の恐れがあるため。分割ロード自体は合成チェックポイントでのファイルI/O
> テストで検証済み。詳細は[CLAUDE.md](CLAUDE.md)参照。
>
> *English*: Implemented DeepSeek-V3-specific MoE extensions
> (aux-loss-free bias correction, group-limited routing, sigmoid
> scoring) in `DeepseekModel`, and added a sharded-safetensors loader
> after discovering the real checkpoint ships as 4 shards / 31.4GB.
> **Honest scope**: skipped real download/load verification — the
> current loader design could peak past 90GB of memory, and this dev
> machine (measured: 32GB total RAM, 16GB free) would likely crash or
> destabilize under that load. The sharded-loading path itself is
> verified via a synthetic on-disk fixture test. See
> [CLAUDE.md](CLAUDE.md).

> 📌 **最近の更新(2026-09-13続き2)**: `open-cuda-llm::DeepseekModel`に
> DeepSeekMoE(共有エキスパート+top-kルーティング)を実装した——直下の
> エントリで「今回はスコープ外」としたMoEに、世界中の言語での追加調査に
> 基づき対応。**正直な開示**: aux-loss-free補正(V3)・group-limited
> routing(V3)・sigmoidスコアリング(V3)は未対応(`scoring_func`が
> `"softmax"`以外なら`load()`が明示的に拒否)——V2-Lite相当の構成
> (`scoring_func="softmax"`・`n_group=1`)は読める設計。新規テスト3本
> 追加、クレート全体67本成功。詳細は[CLAUDE.md](CLAUDE.md)参照。
>
> *English*: Implemented DeepSeekMoE (shared experts + top-k routing)
> in `open-cuda-llm::DeepseekModel`, following up on the entry below
> that scoped MoE out. **Honest scope**: aux-loss-free bias correction,
> group-limited routing, and sigmoid scoring (all V3-specific) are not
> supported yet — `load()` explicitly rejects any `scoring_func` other
> than `"softmax"`. Designed to load V2-Lite-shaped configs
> (`scoring_func="softmax"`, `n_group=1`). 3 new tests added, full
> crate suite (67 tests) passing. See [CLAUDE.md](CLAUDE.md).

> 📌 **最近の更新(2026-09-13続き)**: 世界中の言語(英語・日本語・中国語)
> でのGoogle検索・GitHub調査に基づき、`open-cuda-llm`に本物の
> DeepSeek-V2/V3 Multi-head Latent Attention (MLA) アーキテクチャ
> `deepseek_arch.rs`(`DeepseekConfig`/`DeepseekModel`)を新設した——
> 直下の2026-09-13エントリで明記した「MLA風」事後圧縮とは違い、実
> チェックポイントのテンソル名(`q_a_proj`/`kv_a_proj_with_mqa`/
> `kv_b_proj`等)・decoupled RoPE構造をそのまま実装している。
> **正直な開示**: MoE(DeepSeekMoE)は未対応のため、実在する公開
> チェックポイント(ほぼ全層MoE)は`first_k_dense_replace`以降で
> ロード失敗する。absorb最適化・YaRN RoPE・GPUディスパッチも未対応。
> `aruaru-llm`側にも`POST /v1/deepseek/select`(ローカルディレクトリ
> 指定)・`POST /v1/generate-deepseek`を配線した。詳細は
> [PORTING.md](PORTING.md)・[CLAUDE.md](CLAUDE.md)参照。
>
> *English*: Based on multilingual (English/Japanese/Chinese) Google
> and GitHub research, added a real DeepSeek-V2/V3 Multi-head Latent
> Attention (MLA) architecture module `deepseek_arch.rs`
> (`DeepseekConfig`/`DeepseekModel`) to `open-cuda-llm` — unlike the
> post-hoc "MLA-style" compression noted in the entry right below,
> this implements the actual checkpoint tensor names
> (`q_a_proj`/`kv_a_proj_with_mqa`/`kv_b_proj`, etc.) and decoupled
> RoPE structure. **Honest scope**: MoE (DeepSeekMoE) is not
> supported, so real published checkpoints (almost entirely MoE
> beyond the first layer) fail to load past `first_k_dense_replace`.
> The absorb optimization, YaRN RoPE, and GPU dispatch are also not
> yet implemented. Wired `POST /v1/deepseek/select` (local directory)
> and `POST /v1/generate-deepseek` into `aruaru-llm` as well. See
> [PORTING.md](PORTING.md) / [CLAUDE.md](CLAUDE.md).

> 📌 **最近の更新(2026-09-13)**: `open-cuda-llm::QwenModel`へPCA較正版
> MLA風KVキャッシュ圧縮(`enable_mla_kv_compression_calibrated`)を
> 移植した——`GptModel`版は既にあったが、`QwenModel`側(GQA対応)は
> ランダム射影版のみで、docコメント自身が「PCA較正版の移植は次の
> 増分」と明記していたものへの対応。**正直な開示**: これは実際の
> DeepSeek-V2/V3のMLA(学習時から低ランク射影を組み込んだ専用
> アーキテクチャ)ではなく、既存モデルへの事後的な低ランクKVキャッシュ
> 圧縮(「MLA風」)——本当の意味でのMLA実装(実チェックポイント対応の
> 新規アーキテクチャモジュール)は別途まとまったセッションが必要な
> 課題として記録した。詳細は[PORTING.md](PORTING.md)・
> [CLAUDE.md](CLAUDE.md)参照。
>
> *English*: Ported PCA-calibrated MLA-style KV-cache compression
> (`enable_mla_kv_compression_calibrated`) to `open-cuda-llm::
> QwenModel` — `GptModel` already had it, but `QwenModel` (GQA) only
> had the random-projection variant, with its own doc comment noting
> the calibrated port as the next increment. **Honest scope**: this is
> not real DeepSeek-V2/V3 MLA (an architecture trained from scratch
> with low-rank projections) but post-hoc compression applied to an
> already-trained standard-attention model ("MLA-style") — genuine MLA
> support (a new architecture module for real checkpoints) is recorded
> as a separate, larger future task. See [PORTING.md](PORTING.md) /
> [CLAUDE.md](CLAUDE.md).

> 📌 **最近の更新(2026-09-05)**: `hgemm`(F16)/`dgemm`(F64)へ実Vulkan
> GPUディスパッチを実装した。`hgemm`は half 2要素を1uintへパックする
> 専用シェーダ(`hgemm.comp`、`unpackHalf2x16`/`packHalf2x16`、追加の
> Vulkan拡張不要)を新設し、`hgemm_dispatch`の`GemmPath::VulkanGeneric`
> 分岐へ実配線。`dgemm`はGLSLネイティブ`double`型のシェーダ
> (`dgemm.comp`、パッキング不要)で、`GpuDevice::supports_f64_shader()`
> (新設、`vkGetPhysicalDeviceFeatures`の`shaderFloat64`を実際に問い
> 合わせた結果)が`true`の場合のみディスパッチする。**意外な発見**:
> この開発機(NVIDIA GeForce GT 730、Kepler世代)は`shaderFloat64`を
> 実際にサポートしており、CPU参照実装と1e-9以内で一致する本物のF64
> GPU計算に成功した。続けて速度実測(`hgemm_dgemm_vulkan_bench.rs`、
> GPT-2 124M実形状)を行った結果、**このマシンではCPUの方が5〜70倍
> 速い**(既存の`sgemm`/F32での実測と同じ傾向——Vulkanディスパッチの
> 固定オーバーヘッドがGPT-2規模の軽いGEMM計算より支配的)。**正直な
> 開示**: 「動く」ことは実証したが、この環境では既定経路にする根拠は
> 無い(`real-vulkan`は既定offのopt-inのまま)。F128(qgemm)はGPU
> ハードウェアが原理的に存在しないため引き続きCPU参照実装のみ
> (恒久的な制約)。実機検証: `cargo run -p hgemm_vulkan_real --release`・
> `cargo run -p dgemm_vulkan_real --release`いずれも実機で成功、
> `cargo test -p opencuda-blas --release`は新規7件を含め全green
> (回帰なし)。詳細は[CLAUDE.md](CLAUDE.md)の2026-09-05 HANDOFF
> 追記(続き・続き2・続き3)参照。
>
> 📌 **最近の更新(2026-09-03)**: ユーザー指示「open-directx/open-cuda/
> aruaru-llmで、今後32GB VRAM級のNVIDIA/AMD/Intel GPUを想定し、
> F16/F32/F64、さらにF128まで見据えて開発する」への対応。
> `opencuda-core::KernelArg`/`ResolvedArg`へF16(`half::f16`)・F64・
> F128(自前実装のdouble-double型`DoubleDouble`)を追加し、対応する
> `hgemm`/`dgemm`/`qgemm`(CPUリファレンス実装)を新設。**正直な開示**:
> F128はGPU上のネイティブハードウェア対応がNVIDIA/AMD/Intelいずれにも
> 存在しないためソフトウェアエミュレーション(倍々精度)であり、
> 型システムの一貫性のためだけに用意している。この開発機の実機は
> GT 730(Kepler、2GB)のみのため、32GB級マルチベンダー実機での検証は
> 未実施。詳細は[CLAUDE.md](CLAUDE.md)・[OmniGPU-Design.md](OmniGPU-Design.md)
> §13参照。
>
> 📌 **最近の更新(2026-09-11)**: `open-cuda-llm`に`QwenModel`
> (Qwen2/Qwen2.5系、RoPE+Grouped Query Attention+RMSNorm+SwiGLU)を
> 新設した——既存の`GptModel`(GPT-2専用)とは完全に並行する経路。
> MLA KVキャッシュ圧縮もGQA向けに移植済み。実在モデル
> (Qwen2.5-0.5B-Instruct、Apache 2.0)でダウンロード→ロード→生成まで
> 実機検証し、文法的に正しく事実として正確な文章の生成を確認した。
> **正直な開示**: Qwen3.5-4B・DeepSeek-V4.1-Flash自体(MoE・Causal
> Encoder-Decoder)は別の大規模開発で今回は未対応。詳細は
> [CLAUDE.md](CLAUDE.md)の2026-09-11 HANDOFF追記参照。
>
> *English*: Added `QwenModel` (Qwen2/Qwen2.5: RoPE, Grouped Query
> Attention, RMSNorm, SwiGLU) alongside the existing GPT-2-only
> `GptModel`, plus GQA-aware MLA KV-cache compression. Verified
> end-to-end against real weights (Qwen2.5-0.5B-Instruct) — a real
> download → load → generate produced grammatically correct, factually
> accurate output. Honest disclosure: Qwen3.5-4B and DeepSeek-V4.1-Flash
> themselves (MoE, Causal Encoder-Decoder) remain out of scope, a
> separate larger undertaking. See the 2026-09-11 HANDOFF entry in
> [CLAUDE.md](CLAUDE.md).
>
> 📌 **最近の更新(2026-09-01)**: `open-cuda-llm::GptModel`へ
> Model Folding(層冗長性検出・実際の除去・線形アダプタ置換)を実装
> した。`analyze_layer_redundancy`/`prune_redundant_layers`
> (ShortGPT/Gromov et al.方式)・`find_best_layer_block_to_remove`/
> `remove_layer_block`(連続ブロック探索)・
> `fold_block_with_linear_adapter`(SHIFT-LLM/SlimLLM着想の
> closed-form線形置換、勾配降下法は使わない)の3段階。**正直な開示**:
> ユーザーから依頼のあった「DeepSeekの折りたたみ理論」という技術は
> 日英調査の結果実在しないと判明しており、これは無関係の代替手法
> (実測結果込み)である。詳細は[CLAUDE.md](CLAUDE.md)の2026-09-01
> HANDOFF追記を参照(利用側の`aruaru-llm`から他アカウントで再開する
> 場合は必ず`aruaru-llm/CLAUDE.md`も併せて確認すること)。
>
> 📌 **最近の更新(2026-08-10)**: `open-cuda-llm::GptModel`に
> `generate_with_repetition_penalty`(CTRL方式の繰り返しペナルティ、
> penalty>1.0で既に登場したトークンのlogitを弱める)を新設した。既存の
> `generate()`は`penalty=1.0`で呼ぶ薄いラッパーへ変更(数値的に完全同一の
> 挙動、既存テスト・呼び出し元への回帰なし)。`aruaru-llm`側ユーザー報告
> 「対話ファインチューニング無しの素のGPT-2貪欲デコードが同一文字列
> (例: "Student: Hello")を無限ループする」への根本対応として、実GPT-2
> 124M重みで新規テスト
> `repetition_penalty_reduces_degenerate_loop_on_real_gpt2_weights`を追加
> し、ペナルティ無しでは実際にループへ陥ること・`penalty=1.3`で実際に
> ループが解消し文法的に自然な文へ変わることを確認済み。`aruaru-llm`側
> `/v1/generate`が既定`penalty=1.3`でこの新APIを呼ぶよう配線されている
> (`ARUARU_LLM_REPETITION_PENALTY`環境変数で上書き可)。詳細は
> [CLAUDE.md](CLAUDE.md)の2026-08-10 HANDOFF参照。
>
> *English*: Added `generate_with_repetition_penalty` (CTRL-style
> repetition penalty — penalty>1.0 weakens the logits of tokens already
> seen) to `open-cuda-llm::GptModel`. The existing `generate()` is now a
> thin wrapper calling it with `penalty=1.0` (byte-identical behavior, no
> regression to existing tests/callers). This directly addresses a
> reported `aruaru-llm` bug where the base (non-fine-tuned) GPT-2's
> greedy decoding loops the same string forever (e.g. "Student: Hello").
> Added a new test on real GPT-2 124M weights,
> `repetition_penalty_reduces_degenerate_loop_on_real_gpt2_weights`,
> confirming the loop actually reproduces without the penalty and
> actually stops (producing grammatically natural text) at
> `penalty=1.3`. `aruaru-llm`'s `/v1/generate` now calls this new API
> with `penalty=1.3` by default (override via `ARUARU_LLM_REPETITION_
> PENALTY`). See the 2026-08-10 HANDOFF entry in [CLAUDE.md](CLAUDE.md).

> 📌 **最近の更新(2026-08-08)**: 下記2026-08-06の「MLA実装済み」を
> 実機で再検証(`cargo test -p opencuda-blas mla -- --nocapture` →
> `1 passed; 0 failed`、GT730実機のVulkan経路を通過)。あわせて
> FP8混合精度・DeepSeekMoEの2件を調査したが、**いずれも実装は見送り**
> (FP8: 唯一の実GPU=NVIDIA GT730〈Kepler、CC 3.5〉にFP8対応Tensor Core
> が存在せずソフトウェアエミュレーションにしかならないため。DeepSeekMoE:
> `open-cuda-llm`の`DecoderLayer`は単一の密なFFNのみで複数エキスパート・
> ルーターの実在する統合ポイントが無く、実在チェックポイントも無いため)。
> 詳細・次にすべきことは[CLAUDE.md](CLAUDE.md)の2026-08-08 HANDOFF参照。
>
> *English*: Re-verified the 2026-08-06 "MLA implemented" note on real
> hardware (`cargo test -p opencuda-blas mla -- --nocapture` →
> `1 passed; 0 failed`, exercised the GT730's real Vulkan path).
> Investigated FP8 mixed-precision and DeepSeekMoE but **declined to
> implement either**: FP8 has no real hardware support on this
> machine's only GPU (GT730, Kepler, CC 3.5 — no FP8 Tensor Cores, would
> be software emulation only); DeepSeekMoE has no genuine integration
> point since `open-cuda-llm`'s `DecoderLayer` has only a single dense
> FFN (no expert/router structure) and no real MoE checkpoint exists.
> See the 2026-08-08 HANDOFF entry in [CLAUDE.md](CLAUDE.md) for details.

> 📌 **最近の更新(2026-08-07)**: `open-cuda-llm`の`DecoderLayer`に
> 融合flash attentionカーネル(`flash_attention_with_spirv`)を実配線。
> `GptModel::set_flash_attention_spirv()`未設定時は従来の3段階
> ディスパッチへ自動フォールバック(完全後方互換)。NVIDIA GT 730実機で
> Vulkan経由の生成トークン列がCPU版とバイト単位で一致することを新規
> テストで確認、`cargo test --workspace --release`・`cargo clippy -D
> warnings`とも green。詳細は[CLAUDE.md](CLAUDE.md)のHANDOFF参照。
>
> *English*: Wired the fused flash-attention kernel
> (`flash_attention_with_spirv`) into `open-cuda-llm`'s `DecoderLayer`,
> via `GptModel::set_flash_attention_spirv()` — falls back to the
> existing 3-dispatch path when unset (fully backward compatible).
> Verified on real NVIDIA GT 730 hardware: Vulkan-generated token
> sequences are byte-identical to the CPU path. See
> [CLAUDE.md](CLAUDE.md) HANDOFF for details.

> 📌 **2026-08-06実装済み**: DeepSeek-V3のMLA(Multi-Head Latent
> Attention)にインスパイアされた低ランクKVキャッシュ圧縮
> (`opencuda-blas::mla_compress_kv`/`mla_decompress_kv`)を実装。
> 日英で技術レポート([arXiv:2412.19437](https://arxiv.org/abs/2412.19437))・
> 実装解説ブログを調査した上で、既存の実機検証済み`sgemm`基盤の上に
> 低ランク射影(圧縮93.3%削減の実績が報告されている設計)を実装し、
> 実機(GT730)でCPU/Vulkan両版の数値一致を確認済み。学習済み重みは
> 持たない(仕組みの実証、正直な開示は`CLAUDE.md`のHANDOFF参照)。
> 東芝SBM・DeepSeek技術の他リポジトリへの組み込みは引き続き検討中
> (dream-os等8リポジトリ対象)。
>
> **Implemented 2026-08-06**: low-rank KV-cache compression inspired by
> DeepSeek-V3's Multi-Head Latent Attention (MLA) —
> `opencuda-blas::mla_compress_kv`/`mla_decompress_kv`. Researched the
> technical report ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437))
> and implementation blogs in Japanese and English, then built the
> low-rank projection mechanism (the design behind DeepSeek's reported
> 93.3% KV-cache reduction) on top of the existing, real-hardware-
> verified `sgemm` backend. Verified on real hardware (GT730) that the
> CPU and Vulkan paths match numerically. Does not carry trained
> weights (this demonstrates the mechanism, not the trained
> compression quality — see the HANDOFF section of `CLAUDE.md` for the
> honest disclosure). Applying Toshiba SBM / DeepSeek techniques to the
> other 7 repos is still under consideration.

> **Updated 2026-07-25**: The dev-policy file (`CLAUDE.md`) heading was
> renamed from "Development Policy & Dev Environment Rules" to
> "Design Philosophy & Development Policy & Dev Environment Rules",
> to more clearly separate the project's design philosophy (what we
> value), development policy (how we work), and dev environment rules
> (concrete operational conventions). See `CLAUDE.md` for details.


**開発開始日: 2026-06-26**(このリポジトリのGitHub作成日)

「第二のCUDA」——Windows・macOS・Linux互換、Intel・AMD・NVIDIA互換を
目指すGPU抽象化・計算基盤(`OmniGPU`設計)。`aruaru-llm`とのSET構成
(GPU/CPU実行パイプラインの実装先)。

## これは何か

- **`opencuda-core`**: 全バックエンド共通の`GpuDevice`トレイト(CUDA
  Runtime API相当、`alloc`/`memcpy`/`launch_kernel`)。
- **`opencuda-cpu`**: CPUバックエンド(`rayon`によるデータ並列実行)。
- **`opencuda-vulkan`**: Vulkan Computeバックエンド(クロスプラット
  フォーム、Windows/Linux/Androidにネイティブ対応、macOS/iOSは
  MoltenVK経由)。GEMM/Attention/INT4・INT8量子化を実Vulkan実行で
  検証済み。
- **`opencuda-directx`**(2026-07-23新設): DirectX 12 Computeバック
  エンド(Windows専用、Vulkanと並存するオプトインバックエンド)。
  `vector_add`/`matmul`/`ChaCha20`のGPUディスパッチを実機
  (NVIDIA GT 730)で検証済み——CPU参照実装(RustCrypto `chacha20`crate
  等)との出力完全一致をテストで確認。DXGIアダプタ列挙による実際の
  ベンダー名・VRAM容量取得も実装済み。
- **`opencuda-blas`**: NumPy相当(GEMM/Attention/量子化)。
- **`open-cuda-bert`**: BERT系エンコーダのforward pass
  (multilingual-e5-small対応)。
- **`open-cuda-llm`**: vLLM相当(KVキャッシュ付き貪欲デコード)。GPT-2
  (Hugging Face `openai-community/gpt2`)の`safetensors`を読み込む
  `GptModel::load`実装済み(2026-07-25、`open-cuda-bert::BertModel::load`
  と同じ設計)。実機でGPT-2 124Mの実重みをダウンロード・ロードし、
  ランダム初期化(意味を持たない出力)と比べて明確に流暢な英語の
  貪欲デコード継続("The quick brown fox" → "es are a great way to
  get a little bit of a")を確認済み——詳細は`CLAUDE.md`HANDOFF参照。
- **`open-cuda-whisper`**(2026-07-31新設): Whisper相当(音声認識、
  マーケティング調査ロードマップ6位)。対数メルスペクトログラム抽出+
  エンコーダ(`open-cuda-bert`と同じMulti-Head Attention構成)+
  KVキャッシュ付きデコーダ(`open-cuda-llm`と同じ設計)+
  Cross-Attentionを実装。**現状ランダム初期化のMVPのみ**(学習済み
  Whisper重みのローダーは未実装、詳細は`CLAUDE.md`HANDOFF参照)。

## なぜDirectXとVulkanを両方持つか(2026-07-23の技術判断)

当初「DirectXプラグインとして開発中」という認識が示されたが、日英Web
検索で裏取りした結果、DXVK/vkd3d-proton(Valve社Protonが実際に使う技術)
はいずれも「DirectX(Windows専用API)→Vulkan(クロスプラットフォーム
API)」という変換方向であり、逆方向の実例は見つからなかった——
**クロスプラットフォーム対応という目標に対しては、既存のVulkan Compute
方針の方が技術的に近道**という結論に至った。その上で「Vulkanは残しつつ
Windows向けにDirectXを並存追加する」という方針を採用し、
`opencuda-directx`を実装した。

## 正直な開示

- **クロスプラットフォーム対応は道半ば**: Vulkan Computeは設計上
  Windows/Linux/Android対応・macOS/iOSはMoltenVK経由対応の想定だが、
  実機検証はこのマシン(Windows、NVIDIA GT 730)でのみ実施済み。
- **`opencuda-directx`のカーネルディスパッチはPhase 2として一部のみ**:
  `vector_add`・`matmul`・`ChaCha20`(暗号化部分のみ、Poly1305認証タグ
  computation は含まない)を実装。DXGIアダプタ列挙によるベンダー判定
  (`GpuVendor::Nvidia`等)も実装済みだが、`compute_capability`等の
  詳細情報はDXGIからは取得できないためプレースホルダのまま。
- **GPU圧縮・暗号化の実利益は未検証**: トンネル1フレーム程度の小さい
  ペイロードでは、Host↔Device間の転送オーバーヘッドがGPU側の演算
  優位性を相殺する可能性がある、という技術的懸念がある(実ベンチマーク
  は今後の課題)。

## このエコシステムでの関連

- [aruaru-llm](https://github.com/aon-co-jp/aruaru-llm) — このリポジトリの
  実装例(GPU/CPU実行パイプラインの利用元)。
- [RS-LinkFusion](https://github.com/aon-co-jp/RS-LinkFusion) — GPU圧縮/
  暗号化アクセラレータの利用検討元(`opencuda-directx`のChaCha20カーネル)。
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — 開発ルールの正本。

## ビルド・テスト

```bash
cargo build --workspace
cargo test --workspace

# DirectX 12実機テスト(Windows専用、real-dx12 feature)
cargo test -p opencuda-directx --features real-dx12
```

### 自分のGPUで実際に動くか試す(2026-07-27追記)

「まず何か1つ動かして確認したい」場合は、`examples/`配下の各サブクレート
(ワークスペースメンバー)を`cargo run -p <名前>`で実行する。特に
`vulkan_info`は、実機のVulkan物理デバイス(GPUベンダー名・VRAM容量)を
列挙して表示するだけの最小構成なので、「自分の環境でGPUが検出できるか」
を確認する最初の1コマンドとして最適:

```bash
cargo run -p vulkan_info
```

他の例(`matmul`・`matmul_vulkan_real`・`vector_add`・
`vector_add_vulkan`・`vector_add_vulkan_real`・`vector_add_omniir`)も
同様に`cargo run -p <名前>`で実行できる。各ベンダー(Intel/AMD/nVIDIA等)
の対応状況一覧は`OmniGPU-Design.md`§8.5を参照。

## ライセンス

Apache-2.0

## 2026-09-21: NPU層の再設計メモ / NPU-layer redesign notes

日本語: 実機(OPPO Reno11 A / moto g53y)での測定とLiteRT/NNAPIの調査結果を`DEVELOPMENT-NEXT.md`に記録。 / English: measurements on real phones and the LiteRT/NNAPI survey are recorded in `DEVELOPMENT-NEXT.md`.

## v1.3.0(2026-10-10): かすれ補正・字形照合・AI文脈補完の拡張 / v1.3.0: faded-text rescue, glyph matching, extended AI context completion

### 日本語
- `open-cuda-llm::QwenModel` に追加: バッチ prefill(`forward_prefill_per_position`、非公開)、**`score_continuations(prefix, continuations)`**(文脈のあとに各候補が続く対数尤度の和。文脈の KV キャッシュは 1 回だけ作って巻き戻して再利用)、**`top_next_tokens(prefix, k)`**、**`token_logprobs(ids)`**(各トークンの対数確率。誤読の検出用)。重み共有(tie_word_embeddings)の語彙ロジットは `opencuda_blas::sgemm`(埋め込み表を転置コピーせず `C' = E·Xᵀ`)で計算し、スレッド版と数値一致を試験で確認(試験 13 件)。MLA 圧縮が有効なモデルは未対応。
- 用途: `aruaru-llm` の OCR 補完(`/v1/ocr/scan` / `fill` / `propose`)。字形の照合は `opencuda_blas::sgemm` で、`aruaru-llm` の `/v1/ocr/glyph-rank` から呼ぶ(約 9,200 字形 × 289 位置で 1 升約 14 ms、CPU デバイス)。
- 実測: 言語モデルの計算量(2 × パラメータ数 × トークン数)が補完の速度を決める。1 か所あたり 10〜20 秒(Qwen2.5-1.5B・CPU)。未了: Qwen2.5-3B 以上(分割 safetensors)の読み込み、バッチ prefill の GPU 経路、候補採点の並列化。

### English
- Added to `open-cuda-llm::QwenModel`: batched prefill (`forward_prefill_per_position`, private), **`score_continuations(prefix, continuations)`** (summed log-likelihood of each continuation after a prefix; the prefix KV cache is built once and rewound per candidate), **`top_next_tokens(prefix, k)`**, **`token_logprobs(ids)`** (per-token log-probabilities, for misread detection). The tied-embedding vocabulary logits use `opencuda_blas::sgemm` (`C' = E·Xᵀ` without a transposed copy of the embedding table), verified identical to the threaded version (13 tests). MLA-compressed models are not supported.
- Used by: OCR completion in `aruaru-llm` (`/v1/ocr/scan` / `fill` / `propose`). Glyph matching is one `opencuda_blas::sgemm` call from `aruaru-llm`'s `/v1/ocr/glyph-rank` (~14 ms per cell for ~9,200 glyphs × 289 positions on the CPU device).
- Measured: model compute (2 × parameters × tokens) sets the completion speed — 10–20 s per spot (Qwen2.5-1.5B, CPU). Open: loading Qwen2.5-3B and larger (sharded safetensors), a GPU path for batched prefill, parallel candidate scoring.

## JSON処理の使用方針: `parse_secure_sonic`(速い型+安全)と`sonic-rs`は使える所では全て使う(2026-10-14、ユーザー指示)
正本は[`RS-JSON`](https://github.com/aon-co-jp/RS-JSON)(`README.md`・`PORTING.md`)。

**方針**: JSONを読む所は、使える所では**全て** `rust-json`(RS-JSON)の `parse_secure_sonic`(`fast`機能、結果は`sonic_rs::Value`)を使う。
`sonic-rs` を直接使える所(型付き構造体への読み込みなど)も同様。使えない所は `parse_secure`(結果は`serde_json::Value`)を使う。
外部から来る信頼できない入力(HTTP本文・ネットワーク越しのファイル・アップロード)は、**必ずサイズ・深さ・重複キーの上限つき**で読む。

**実測(7 MB・4万件)**: `serde_json` 200 ms / `parse_secure` 217 ms / **`parse_secure_sonic` 83 ms(約2.4倍速い、防御込み)** / `sonic-rs`そのまま 33 ms(約6倍、防御なし)。
`serde_json::Value`に読む限りSIMDでもほぼ速くならない(188 ms)。**速さの源は、結果を連続した領域に詰め込む値の型**。

**使える所 / 使えない所**
- ✅ 読み取り中心で、JSONを大きく・頻繁に読む所(キャッシュJSON、設定、APIの応答、ログ、インデックス)。
- ✅ 外部からの入力の入口(速さより防御が目的。`parse_secure`でも可)。
- ⚠️ `serde_json::Value`を関数間で受け渡している既存コードは、型を`sonic_rs::Value`に直す書き換えが要る(`JsonValueTrait`/`JsonContainerTrait`のメソッド)。書き換えずに`serde_json::Value`へ読むと速くならない。
- ❌ WASM(`rust_json::light`、依存ゼロ)、PHP/JavaScript側のJSON処理(Rustの部品は使えない)。
- 注意: 依存先の`sonic-rs`は内部で`unsafe`を使う(自前コードに`unsafe`は無い)。小さなJSON(数KB)では体感差が出ない。速度目的の書き換えは、実測で効く所から。

**導入**: `Cargo.toml`に `rust-json = { path = "../RS-JSON", features = ["fast"] }`。使用例は`RS-JSON/README.md`。

**このリポジトリでの適用状況**: 未適用(方針のみ)。コードの切り替えは、JSONを大きく・頻繁に読む所から実測して順に行う。

---

作者 / Author: 石 塚 正 浩（MASAHIRO ISHIZUKA） (https://github.com/aon-co-jp)

## API 方針: REST 全廃・gRPC / Connect へ (2026-11-04)

### 日本語
- **REST は全廃する。新しい API は REST で作らない。** ブラウザや `curl` から呼ぶ API は **Connect の単純呼び出し**(`POST /<パッケージ>.<Service>/<Method>`、`content-type: application/json`、本文は JSON)、サービス間は **gRPC / gRPC-web**(RPoem の管理API `open_runo.admin.v1.Admin` と同じ流儀)。ストリーミングは、必要になってから。
- **既存の REST は、触るときに順に移す。** 移したら旧 REST の経路は消す。移行が終わるまで残るが、**新規には追加しない**。
- **JSON は RS-JSON に統一**する(読み込みは `from_str_secure` / `parse_secure` でサイズ・深さ・重複キーの上限つき、書き出しは `rust_json::fast`)。ブラウザの WASM に `prost` は入れない(サイズが増えるため)。
- **例外(自分たちの API ではなく、相手の規格が決めているもの)**: ACME、OAuth/OIDC のコールバック、第三者(GitHub など)からの Webhook、ヘルスチェック、静的ファイル配信、OTLP など。これらは規格どおりの HTTP を受ける。
- 参考実装: `RPoem`(`open_runo.admin.v1.Admin`)、`open-easy-web`(`server/src/connect.rs`、`proto/open_easy_web/admin/v1/admin.proto`)。
- ※ このリポジトリのコードが既に方針どおりかどうかは、各リポジトリの現状による。この節は方針の記録であり、コードの変更を意味しない。

### English
- **REST is being abolished. New APIs are not built as REST.** APIs called from browsers or `curl` use **Connect unary calls** (`POST /<package>.<Service>/<Method>`, `content-type: application/json`, JSON body); service-to-service calls use **gRPC / gRPC-web** (the same style as RPoem's admin API `open_runo.admin.v1.Admin`). Streaming only when it is actually needed.
- **Existing REST endpoints are migrated one by one, when they are touched.** The old REST route is removed once migrated. Until migration is finished they remain, but **no new ones are added**.
- **JSON is unified on RS-JSON**: reading with `from_str_secure` / `parse_secure` (size, depth and duplicate-key limits), writing with `rust_json::fast`. `prost` is not put into browser WASM (it would grow the binary).
- **Exceptions (shapes dictated by someone else's standard, not our own API)**: ACME, OAuth/OIDC callbacks, webhooks from third parties (e.g. GitHub), health checks, static file serving, OTLP, and so on. These accept plain HTTP as the standard requires.
- Reference implementations: `RPoem` (`open_runo.admin.v1.Admin`) and `open-easy-web` (`server/src/connect.rs`, `proto/open_easy_web/admin/v1/admin.proto`).
- Note: whether the code in a given repository already follows this policy depends on that repository. This section records the policy; it does not mean the code has been changed.

### Deutsch (KI-Übersetzung, redaktionell überarbeitet; noch nicht von Muttersprachlern geprüft)
- **REST wird vollständig abgeschafft; neue APIs entstehen nicht mehr als REST.** APIs, die aus dem Browser oder per `curl` aufgerufen werden, verwenden **unäre Connect-Aufrufe** (`POST /<Paket>.<Service>/<Methode>`, `content-type: application/json`, JSON-Body). Zwischen Diensten kommen **gRPC bzw. gRPC-web** zum Einsatz, im selben Stil wie die Admin-API von RPoem (`open_runo.admin.v1.Admin`). Streaming wird erst ergänzt, wenn es tatsächlich gebraucht wird.
- **Bestehende REST-Endpunkte werden schrittweise umgestellt, und zwar immer dann, wenn man ohnehin an ihnen arbeitet.** Sobald ein Endpunkt umgestellt ist, wird die alte REST-Route entfernt. Bis zum Abschluss der Migration bleiben die übrigen bestehen, es werden aber **keine neuen mehr hinzugefügt**.
- **JSON wird einheitlich über RS-JSON verarbeitet**: Gelesen wird mit `from_str_secure` bzw. `parse_secure` (mit Obergrenzen für Größe, Verschachtelungstiefe und doppelte Schlüssel), geschrieben mit `rust_json::fast`. `prost` wird nicht in das Browser-WASM aufgenommen, da es die Datei vergrößern würde.
- **Ausnahmen (Schnittstellen, deren Form ein fremder Standard vorgibt und die nicht zu unserer eigenen API gehören)**: ACME, OAuth/OIDC-Callbacks, Webhooks von Drittanbietern (z. B. GitHub), Health-Checks, die Auslieferung statischer Dateien, OTLP und Ähnliches. Diese nehmen, wie vom jeweiligen Standard verlangt, gewöhnliches HTTP entgegen.
- Referenzimplementierungen: `RPoem` (`open_runo.admin.v1.Admin`) und `open-easy-web` (`server/src/connect.rs`, `proto/open_easy_web/admin/v1/admin.proto`).
- Hinweis: Ob der Code eines einzelnen Repositorys diese Richtlinie bereits erfüllt, hängt vom jeweiligen Repository ab. Dieser Abschnitt dokumentiert lediglich die Richtlinie; er besagt nicht, dass der Code schon angepasst wurde.

### Русский (перевод выполнен с помощью ИИ и отредактирован; носителями языка пока не проверен)
- **От REST мы полностью отказываемся; новые API больше не создаются в стиле REST.** API, к которым обращаются из браузера или через `curl`, используют **унарные вызовы Connect** (`POST /<пакет>.<Service>/<Method>`, `content-type: application/json`, тело запроса — JSON). Для взаимодействия между сервисами применяются **gRPC и gRPC-web** — в том же стиле, что и административный API RPoem (`open_runo.admin.v1.Admin`). Потоковая передача данных добавляется лишь тогда, когда в ней действительно возникает необходимость.
- **Существующие REST-эндпоинты переводятся поэтапно — по мере того, как с ними приходится работать.** После перевода старый REST-маршрут удаляется. Пока миграция не завершена, оставшиеся эндпоинты сохраняются, но **новые не добавляются**.
- **Работа с JSON унифицирована на основе RS-JSON**: чтение — через `from_str_secure` или `parse_secure` (с ограничениями на размер, глубину вложенности и повторяющиеся ключи), запись — через `rust_json::fast`. `prost` в браузерный WASM не включается, поскольку это увеличило бы размер файла.
- **Исключения (интерфейсы, форму которых определяет чужой стандарт, а не наш собственный API)**: ACME, обратные вызовы OAuth/OIDC, вебхуки сторонних сервисов (например, GitHub), проверки работоспособности, раздача статических файлов, OTLP и тому подобное. Такие интерфейсы принимают обычный HTTP, как того требует соответствующий стандарт.
- Эталонные реализации: `RPoem` (`open_runo.admin.v1.Admin`) и `open-easy-web` (`server/src/connect.rs`, `proto/open_easy_web/admin/v1/admin.proto`).
- Примечание: соответствует ли код конкретного репозитория этой политике, зависит от самого репозитория. Данный раздел лишь фиксирует политику и не означает, что код уже изменён.

## API 方針の補足: 用語の整理と使い分け (2026-11-04)

### 日本語
**1つに統一するものではなく、別の層の話です。** 「REST の代わり」になるのは Connect・gRPC・GraphQL だけで、WASM は画面を作る技術です。

| 用語 | 何か | REST の代わりか |
|---|---|---|
| REST | URL と HTTP メソッドで操作を表す API の作り方 | 全廃する側 |
| **Connect** | `POST /サービス名/メソッド名` に JSON を送る形式。`curl` でもブラウザの `fetch` でも呼べる | はい(**採用**) |
| **gRPC** | 同じ「サービス/メソッド」の考え方を、バイナリ(protobuf)と HTTP/2 で行う形式。速く、ストリーミングも得意 | はい(**サービス同士に採用**) |
| gRPC-web | gRPC のブラウザ向けの変形版 | はい(必要なら) |
| GraphQL | 欲しい項目を問い合わせる形式 | 一部だけ(**問い合わせ(読み取り)専用**) |
| WASM | Rust をブラウザの中で動かす仕組み | いいえ(画面を作る技術) |

- **ブラウザの画面や `curl` から呼ぶ API は Connect**(POST、JSON を RS-JSON で読み書き)。ブラウザの WASM は素の gRPC を話せず、Connect なら `fetch` のままで、`prost` を足さずに済む(画面のサイズも増えない)。
- **サービス同士の通信(サーバー間、重い処理、ストリーミング)は gRPC。** 多くは Connect で足りるので、ストリーミングが必要になった所だけ gRPC に上げる。
- **GraphQL は、まとめて問い合わせたいときの読み取り専用。** 書き込み(操作)には使わない。
- **WASM は画面を作る技術として、そのまま使う。** 画面から呼ぶ API が Connect になっただけ。
- 外部の規格が形を決めるもの(ACME、OAuth のコールバック、第三者の Webhook、ヘルスチェック、静的ファイル、OTLP)は、規格どおりの HTTP のまま。
- **迷ったら Connect。**

### English
**These are not alternatives to unify; they sit at different layers.** Only Connect, gRPC and GraphQL can replace REST; WASM is a way to build the UI.

| Term | What it is | Replaces REST? |
|---|---|---|
| REST | API style that expresses operations through URLs and HTTP methods | The thing being abolished |
| **Connect** | `POST /Service/Method` with a JSON body; callable from `curl` and from a browser `fetch` | Yes (**adopted**) |
| **gRPC** | The same service/method idea over binary (protobuf) and HTTP/2; fast, good at streaming | Yes (**for service-to-service**) |
| gRPC-web | A browser-friendly variant of gRPC | Yes (if needed) |
| GraphQL | A query language: ask for exactly the fields you want | Partly (**read-only queries**) |
| WASM | Runs Rust inside the browser | No (a UI technology) |

- **APIs called from browser screens or `curl` use Connect** (POST, JSON read and written with RS-JSON). Browser WASM cannot speak plain gRPC; with Connect it stays a `fetch`, and `prost` is not needed (the binary does not grow).
- **Service-to-service calls (server to server, heavy work, streaming) use gRPC.** Connect is enough in most places; move a call to gRPC only where streaming is actually needed.
- **GraphQL is read-only**, for queries that gather several pieces of data. It is not used for writes (operations).
- **WASM stays as the way to build the UI.** The API it calls simply became Connect.
- Interfaces whose shape an outside standard dictates (ACME, OAuth callbacks, third-party webhooks, health checks, static files, OTLP) stay plain HTTP.
- **When in doubt, use Connect.**

### Deutsch (KI-Übersetzung, sprachlich überarbeitet; noch nicht von Muttersprachlern geprüft)
**Es geht nicht darum, eine einzige Technik auszuwählen – die Begriffe liegen auf unterschiedlichen Ebenen.** Als Ersatz für REST kommen nur Connect, gRPC und GraphQL in Frage; WASM dagegen ist eine Technik, mit der die Benutzeroberfläche gebaut wird.

| Begriff | Was ist das? | Ersatz für REST? |
|---|---|---|
| REST | Ein API-Stil, bei dem Operationen über URLs und HTTP-Methoden ausgedrückt werden | wird abgeschafft |
| **Connect** | `POST /Dienst/Methode` mit JSON im Body; aufrufbar mit `curl` ebenso wie mit `fetch` im Browser | ja (**gewählt**) |
| **gRPC** | Dasselbe Prinzip aus Dienst und Methode, aber über binäres protobuf und HTTP/2; schnell und gut für Streaming geeignet | ja (**zwischen Diensten**) |
| gRPC-web | Eine für den Browser angepasste Variante von gRPC | ja (bei Bedarf) |
| GraphQL | Abfragesprache, mit der man genau die benötigten Felder anfordert | teilweise (**nur für lesende Abfragen**) |
| WASM | Führt Rust-Code direkt im Browser aus | nein (Technik für die Oberfläche) |

- **APIs, die von Browseroberflächen oder per `curl` aufgerufen werden, verwenden Connect** (POST; JSON wird mit RS-JSON gelesen und geschrieben). WASM im Browser kann kein reines gRPC sprechen. Mit Connect bleibt es bei einem gewöhnlichen `fetch`, und `prost` ist nicht nötig – die WASM-Datei wird dadurch nicht größer.
- **Für die Kommunikation zwischen Diensten (Server zu Server, rechenintensive Aufgaben, Streaming) wird gRPC eingesetzt.** In den meisten Fällen reicht Connect aus; auf gRPC wird nur dort umgestellt, wo Streaming tatsächlich gebraucht wird.
- **GraphQL dient ausschließlich lesenden Abfragen**, wenn mehrere Informationen gebündelt abgefragt werden sollen. Für Schreibzugriffe (Operationen) wird es nicht verwendet.
- **WASM bleibt die Technik für die Oberfläche.** Geändert hat sich lediglich, dass die aufgerufene API nun auf Connect beruht.
- Schnittstellen, deren Form durch einen externen Standard festgelegt ist (ACME, OAuth-Callbacks, Webhooks von Drittanbietern, Health-Checks, statische Dateien, OTLP), bleiben gewöhnliches HTTP.
- **Im Zweifel: Connect.**

### Русский (перевод выполнен с помощью ИИ и литературно отредактирован; носителями языка пока не проверен)
**Это не варианты, из которых нужно выбрать один, а технологии разных уровней.** Заменить REST могут только Connect, gRPC и GraphQL; WASM же — это способ создания пользовательского интерфейса.

| Термин | Что это | Заменяет REST? |
|---|---|---|
| REST | Стиль построения API, при котором операции выражаются через URL и HTTP-методы | отменяется |
| **Connect** | `POST /Сервис/Метод` с телом в формате JSON; вызывается как из `curl`, так и из браузерного `fetch` | да (**выбран**) |
| **gRPC** | Тот же принцип «сервис — метод», но поверх двоичного формата protobuf и HTTP/2; работает быстро и хорошо подходит для потоковой передачи | да (**для связи между сервисами**) |
| gRPC-web | Адаптированный для браузера вариант gRPC | да (при необходимости) |
| GraphQL | Язык запросов: клиент запрашивает ровно те поля, которые ему нужны | частично (**только запросы на чтение**) |
| WASM | Позволяет выполнять код на Rust прямо в браузере | нет (технология для интерфейса) |

- **API, которые вызываются из браузерных интерфейсов или через `curl`, строятся на Connect** (POST; JSON читается и записывается через RS-JSON). Браузерный WASM не умеет работать с «чистым» gRPC. С Connect достаточно обычного `fetch`, а `prost` не требуется — размер WASM-файла не увеличивается.
- **Для обмена между сервисами (сервер — сервер, ресурсоёмкие задачи, потоковая передача) применяется gRPC.** В большинстве случаев достаточно Connect; на gRPC переходят только там, где потоковая передача действительно необходима.
- **GraphQL используется только для чтения**, когда нужно получить сразу несколько связанных данных. Для записи (операций) он не применяется.
- **WASM по-прежнему остаётся технологией для интерфейса.** Изменилось лишь то, что вызываемый им API теперь построен на Connect.
- Интерфейсы, форма которых задана внешним стандартом (ACME, обратные вызовы OAuth, вебхуки сторонних сервисов, проверки работоспособности, статические файлы, OTLP), остаются обычным HTTP.
- **Если есть сомнения — выбирайте Connect.**

## 2026-11-19 大きなAIモデルを動かす取り組みの現状 / Running bigger AI models: status and plan

### 日本語(やさしい説明)
**いまできること**
- 大きなAIモデルは、重さが1つのファイルでなく「いくつかに分かれたファイル」で配られることがあります。これを**自動で全部ダウンロードして、読み込める**ようになりました(`aruaru-llm` と `open-cuda`)。
- 本物の Qwen2.5-3B(約6GB、2つに分割)を、このパソコン(メモリ32GB、GPUなし)で**実際に読み込んで文章を作れる**ことを確かめました。読み込み約48秒、文章づくりは1秒に約0.6語(遅いけれど動きます)。
- 読み込みは「ファイルを丸ごとメモリに入れず、必要な所だけ読む」やり方に変えたので、メモリの使いすぎが減りました。

**調べて分かったこと**
- 東芝の「シミュレーテッド分岐」や富士通の「デジタルアニーラ」は、「たくさんの選択肢から一番よい組み合わせを探す」道具です。AIの文章づくりを直接速くする道具ではありません。使うなら「どの部品をどこに置くか」「どの部分を何ビットに縮めるか」を決める所です(研究論文はまだ少ない)。
- 「折り畳み(Model Folding)」は、似た部分をまとめてモデルを小さくする方法です。DeepSeek の MLA(記憶の圧縮)・MoE(専門家を必要な分だけ使う)・FP8(数を小さく表す)も同じ仲間の工夫です。
- 実験: 学習済みの Qwen に後からMLA圧縮をかけると、記憶を25%減らしただけで品質が落ち(迷い度 7.1→12.7)、50%減らすと文が壊れました。**後付けのMLAは使えない**と分かりました(DeepSeekは最初からその形で学習しています)。

**これから作るもの(確認なしで順に進めます)**
1. 重みを小さい数(f16や量子化)のまま使って、7B級(約12GB)もこのパソコンで動かす。
2. 層ごと・専門家ごとに「何ビットにするか」を自動で決める(まずは普通の計算、あとで焼きなましやSBも試す)。
3. 似た層をまとめる「折り畳み」を試し、品質と速さを必ず測る。
4. 速さの測定を毎回記録し、実用になるまで作る・試す・直すを何度も繰り返す。

**約束**: 未了の作業は、危険でない限り**いちいち確認せず進めます**。ただし、取り返しのつかない削除や、秘密の公開はしません。結果が悪かったときは、悪かったと正直に書きます。

### English (plain words, easy to read)
**What works now**
- Big AI models are often shipped as **several files** instead of one. The program can now **download all the parts and load them** (`aruaru-llm` and `open-cuda`).
- We loaded the real Qwen2.5-3B model (about 6 GB, 2 parts) on this PC (32 GB memory, no GPU) and **it wrote text**. Loading took about 48 seconds; writing was about 0.6 words per second (slow, but it works).
- Loading now reads only the parts it needs instead of copying whole files into memory, so it uses less memory.

**What we learned**
- Toshiba's "Simulated Bifurcation" and Fujitsu's "Digital Annealer" help **pick the best combination from many choices**. They do not make an AI write faster by themselves. They could help decide where to put each part, or how many bits each part gets (papers on this are still few).
- "Model folding" means merging similar parts to make a model smaller. DeepSeek's MLA (memory compression), MoE (use only the needed experts) and FP8 (small numbers) are cousins of this idea.
- Our test: adding MLA compression to an already-trained Qwen cut memory by 25% but made quality worse (perplexity 7.1 to 12.7), and a 50% cut broke the text. **Adding MLA afterwards does not work** (DeepSeek trains with it from the start).

**What we will build next (we go on without asking)**
1. Keep weights in small numbers (f16 or quantized) so 7B-size models (about 12 GB) can run on this PC.
2. Choose bits per layer and per expert automatically (plain math first, annealing or SB later).
3. Try "folding" similar layers, and always measure quality and speed.
4. Record every speed test, then build, test and fix again and again until it is truly useful.

**Promise**: for unfinished work we **go on without asking each time**, unless it is dangerous. We never do irreversible deletes or publish secrets. If a result is bad, we say so honestly.

## 2026-11-19 大きなAIを小さなパソコンで動かす / Running big AI on a small PC (10 languages, easy words)

### 日本語
**できたこと**: 大きなAIは、重さが「いくつかのファイル」に分かれていることがあります。それを自動でぜんぶダウンロードして、読めるようになりました。本物のAI(Qwen 3B、約6GB)を、GPUのないふつうのパソコンで動かして、文章を作れました。ちょっと遅いけれど、ちゃんと動きます。
**わかったこと**: 東芝や富士通の「最適な組み合わせを探す機械」は、AIを直接速くする道具ではありません。「どの部品を何ビットに縮めるか」を決める手伝いに使えるかもしれません。作ったあとの記憶の圧縮(MLA)は、品質が落ちるので使えませんでした。
**これから**: AIの重さを小さい数で持って、もっと大きなAI(約12GB)も動かします。「確認しないで進める」ので、止まらずに作って、試して、直します。

### English
**Done**: Big AI models often come in several files. The program now downloads every part and loads them. We ran a real AI (Qwen 3B, about 6 GB) on an ordinary PC without a GPU, and it wrote text. It is a bit slow, but it works.
**Learned**: The "best-combination finder" machines from Toshiba and Fujitsu do not make AI faster by themselves. They might help decide how many bits each part gets. Squeezing the memory afterwards (MLA) hurt quality, so we cannot use it.
**Next**: Keep the AI's weights in small numbers so even bigger AI (about 12 GB) can run. We go on without asking each time: build, test and fix again and again.

### 简体中文
**已完成**:大型AI的权重常被拆成好几个文件。现在程序能自动全部下载并读取。我们在没有显卡的普通电脑上运行了真正的AI(Qwen 3B,约6GB),它能写出文字。虽然有点慢,但是能用。
**发现**:东芝和富士通那种"找最佳组合的机器"不能直接让AI变快,也许能帮忙决定每个部分用几位数。事后压缩记忆(MLA)会让质量变差,所以不能用。
**接下来**:用更小的数字保存权重,让更大的AI(约12GB)也能运行。我们不会每次都问,而是不停地做、测、改。

### 繁體中文
**已完成**:大型AI的權重常被拆成好幾個檔案。現在程式能自動全部下載並讀取。我們在沒有顯示卡的一般電腦上執行了真正的AI(Qwen 3B,約6GB),它能寫出文字。雖然有點慢,但可以用。
**發現**:東芝和富士通那種「找最佳組合的機器」不能直接讓AI變快,也許能幫忙決定每個部分用幾位元。事後壓縮記憶(MLA)會讓品質變差,所以不能用。
**接下來**:用更小的數字保存權重,讓更大的AI(約12GB)也能執行。我們不會每次都詢問,而是不停地做、測、改。

### 한국어
**해낸 일**: 큰 AI는 무게(가중치)가 여러 파일로 나뉘어 있을 때가 많아요. 이제 프로그램이 전부 자동으로 내려받아 읽을 수 있어요. 그래픽카드가 없는 보통 컴퓨터에서 진짜 AI(Qwen 3B, 약 6GB)를 돌려 글을 만들었어요. 조금 느리지만 잘 돼요.
**알게 된 것**: 도시바와 후지쯔의 "가장 좋은 조합을 찾는 기계"는 AI를 직접 빠르게 하지는 않아요. 각 부분을 몇 비트로 줄일지 정하는 데 도움이 될 수는 있어요. 나중에 기억을 줄이는 방법(MLA)은 품질이 떨어져서 쓸 수 없었어요.
**앞으로**: 무게를 더 작은 숫자로 들고 있어서 더 큰 AI(약 12GB)도 돌릴 거예요. 매번 묻지 않고 만들고, 시험하고, 고치기를 계속해요.

### Deutsch
**Geschafft**: Große KI-Modelle bestehen oft aus mehreren Dateien. Das Programm lädt jetzt alle Teile selbst herunter und liest sie. Wir haben eine echte KI (Qwen 3B, etwa 6 GB) auf einem normalen PC ohne Grafikkarte laufen lassen, und sie hat Text geschrieben. Etwas langsam, aber es klappt.
**Gelernt**: Die „Maschinen, die die beste Kombination suchen“ von Toshiba und Fujitsu machen eine KI nicht von selbst schneller. Sie könnten helfen zu entscheiden, wie viele Bits jeder Teil bekommt. Das nachträgliche Zusammendrücken des Gedächtnisses (MLA) machte die Qualität schlechter, deshalb nutzen wir es nicht.
**Als Nächstes**: Die Gewichte in kleineren Zahlen speichern, damit noch größere KI (etwa 12 GB) läuft. Wir machen ohne Rückfrage weiter: bauen, testen, verbessern, immer wieder.

### Français
**Fait** : Les grosses IA sont souvent livrées en plusieurs fichiers. Le programme télécharge maintenant toutes les parties et les lit. Nous avons fait tourner une vraie IA (Qwen 3B, environ 6 Go) sur un PC ordinaire sans carte graphique, et elle a écrit du texte. C'est un peu lent, mais ça marche.
**Appris** : Les « machines qui cherchent la meilleure combinaison » de Toshiba et Fujitsu ne rendent pas l'IA plus rapide toutes seules. Elles pourraient aider à choisir combien de bits donner à chaque partie. Compresser la mémoire après coup (MLA) a abîmé la qualité, donc nous ne l'utilisons pas.
**Ensuite** : Garder les poids en plus petits nombres pour faire tourner de plus grosses IA (environ 12 Go). Nous continuons sans demander à chaque fois : construire, tester, corriger, encore et encore.

### Español
**Hecho**: Las IA grandes suelen venir en varios archivos. Ahora el programa descarga todas las partes y las lee. Ejecutamos una IA real (Qwen 3B, unos 6 GB) en un PC normal sin tarjeta gráfica y escribió texto. Es un poco lento, pero funciona.
**Aprendido**: Las «máquinas que buscan la mejor combinación» de Toshiba y Fujitsu no hacen más rápida a la IA por sí solas. Podrían ayudar a decidir cuántos bits da cada parte. Comprimir la memoria después (MLA) empeoró la calidad, así que no lo usamos.
**Siguiente**: Guardar los pesos con números más pequeños para que corran IA aún más grandes (unos 12 GB). Seguimos sin preguntar cada vez: construir, probar y arreglar una y otra vez.

### Português
**Feito**: As IAs grandes costumam vir em vários arquivos. Agora o programa baixa todas as partes e as lê. Rodamos uma IA de verdade (Qwen 3B, cerca de 6 GB) num PC comum sem placa de vídeo, e ela escreveu texto. É um pouco lento, mas funciona.
**Aprendido**: As «máquinas que procuram a melhor combinação» da Toshiba e da Fujitsu não deixam a IA mais rápida sozinhas. Elas podem ajudar a decidir quantos bits cada parte recebe. Comprimir a memória depois (MLA) piorou a qualidade, então não usamos.
**Próximo**: Guardar os pesos em números menores para rodar IAs ainda maiores (cerca de 12 GB). Seguimos sem perguntar toda vez: construir, testar e consertar, de novo e de novo.

### Русский
**Сделано**: Большие ИИ часто поставляются в нескольких файлах. Теперь программа сама скачивает все части и читает их. Мы запустили настоящий ИИ (Qwen 3B, около 6 ГБ) на обычном компьютере без видеокарты, и он написал текст. Немного медленно, но работает.
**Узнали**: «Машины, которые ищут лучшую комбинацию» от Toshiba и Fujitsu сами по себе ИИ не ускоряют. Возможно, они помогут решить, сколько бит дать каждой части. Сжатие памяти задним числом (MLA) ухудшило качество, поэтому мы его не используем.
**Дальше**: Хранить веса в меньших числах, чтобы запускать ещё более крупные ИИ (около 12 ГБ). Мы продолжаем без вопросов каждый раз: строим, проверяем, исправляем снова и снова.

## 2026-11-19 限界前チェックポイント(大きなモデル対応、別アカウントでの再開用)

**完了(すべて main に push 済み)**
- 分割重み(`model.safetensors.index.json`+複数ファイル)の取得(`aruaru-llm` `model_catalog.rs`、単一ファイルが404のときだけ・ストリーム書き込み・導入判定対応)と読み込み(`open-cuda-llm` `qwen_arch.rs`、mmap)。モックと合成データのテスト成功。
- 実機確認(32GB・GPUなし・CPU): Qwen2.5-3B(2分割・6.17GB)を読み込み生成成功(読込約48秒、約0.6語/秒、ppl 8.5)。1.5Bは ppl 7.08。
- 後付けMLA(PCA較正)は実用外: d_c=96でKV−25%だが ppl 12.7、d_c=64で崩れる(ppl 245)。MLAは学習時から入れる設計でないと効かない。
- FP8読み込み `QwenModel::load_fp8`: 量子化を二分探索化・ホスト並列化し、行列積を融合(`opencuda-blas`)。1.5Bで読込340→21秒、生成1067→69秒、ppl 7.27(f32は7.08)、メモリ約1/4。ただしf32より約2.8倍遅い。
- 調査: 東芝SB・富士通デジタルアニーラは組合せ最適化の道具でLLM推論を直接速めない。LLMでの先行例はTree-Ising(重みのQUBO修復、自己申告値)のみ。エキスパートごとのビット配分は線形/整数計画(GEMQ・BitsMoE)。「折り畳み」はModel Folding(ICLR 2025)・Layer Collapse・Sliding-Window Merging・GPTailor。DeepSeekのMLA・MoE遅延読み込みは `deepseek_arch.rs` に実装済み(実在チェックポイントでの通しは未確認)。
- クラウドセッションに調査文書 `docs/LARGE-MODEL-TECHNIQUES.md`(ブランチ `docs/large-model-techniques`)を依頼済み(終了通知は来ない。`aruaru-llm` のブランチを確認すること)。

**未了(確認なしで順に進める方針)**
1. 層ごとの感度測定(層を飛ばす/隣接層を平均する「折り畳み」)と、ビット配分の自動決定(まず貪欲法、次に焼きなまし・SB)。`QwenModel` に層スキップ機構を足して `qwen_mla_bench` 系の例で ppl と速度を測る。
2. FP8がf32より遅い問題(約2.8倍): SIMD化・f16保持・行単位スケール見直し。
3. 7B級(約12〜15GB)の実読み込み: FP8読み込みで32GB機に載るかを実測(要ダウンロード許可内の公開モデル)。
4. 実3BのFP8読み込み実測、`pl2pl llm setup` への分割モデル・FP8の反映、`aruaru-llm` のカタログに3Bの実機確認の記載。
5. 10言語文書は現状分を追記済み。上記の結果が出たら更新する。翻訳はAI作成でネイティブ検証前。

**注意**: `F:\open-cuda` の作業ブランチは `perf/serial-small-gemm`(`HEAD:main` を指定して push している)。`F:\aruaru-llm` は `Cargo.lock` と `data/` に未コミットの変更がある(別作業のもの、コミットしない)。モデルは `F:\aruaru-llm\models\qwen2.5-{1.5b,3b}-instruct`(サイズが大きいので消さない)。測定スクリプト: `open-cuda/crates/open-cuda-llm/examples/qwen_mla_bench.rs`(`QWEN_FP8=1` でFP8、引数にd_c)。

## 2026-11-19 まとめ: 大きなAIを小さなパソコンで / Summary: big AI on a small PC

### 日本語(やさしい説明)
**できたこと**
- 重さが「いくつかのファイル」に分かれた大きなAIを、自動でぜんぶダウンロードして読めるようになりました(`aruaru-llm`・`open-cuda`)。
- 本物の Qwen2.5-3B(約6GB)を、GPUのないふつうのパソコンで動かして文章を作れました(読み込み約48秒、1秒に約0.6語)。
- 重さを「FP8」という小さい数で持つ方法を作りました。メモリが約4分の1になり、品質はほぼ同じです(迷い度 7.08→7.27)。速くする工夫で、読み込みは340秒→21秒、文章づくりは1067秒→69秒になりました。まだ普通の方法より約2.8倍遅いので、メモリが足りないときの手段です。

**試してわかったこと(正直な結果)**
- 後から記憶を圧縮する「MLA」は、品質が落ちるので使えません(25%減で迷い度 7.1→12.7、50%減で文が壊れる)。
- 「折り畳み(層を飛ばす)」を1.5Bで試しました。1層ずつ飛ばすと、真ん中あたりの層(5・10・12・15番など)は飛ばしてもほとんど平気でしたが、1番・24番・26番は大事でした。3層飛ばすと迷い度7.19(元7.08)で、速さは約7%だけ上がり、答えが少し不正確になりました。6層以上飛ばすと壊れます。**学び直しなしの層飛ばしは、効果が小さい**とわかりました。
- 東芝の「シミュレーテッド分岐」や富士通の「デジタルアニーラ」は、「いちばん良い組み合わせを探す」道具で、AIを直接速くはしません。使うなら「どの層を飛ばすか」「何ビットにするか」を決める所ですが、まだ試していません。

**これから(確認せず順に進めます)**
1. 層を飛ばす組み合わせを、焼きなましやSBで探して、貪欲法と比べる。
2. FP8を速くする(SIMD、f16保持)。
3. 7B(約12〜15GB)をFP8で32GBのパソコンに載せられるか測る。
4. `pl2pl llm setup` に分割モデルとFP8を反映する。

**約束**: 危なくない未了は、いちいち聞かずに進めます。取り返しのつかない削除や秘密の公開はしません。悪い結果も正直に書きます。

### English (plain and friendly)
**What works now**
- Big AI models are often split into several files. The program now downloads every part and loads them (`aruaru-llm` and `open-cuda`).
- We ran the real Qwen2.5-3B model (about 6 GB) on an ordinary PC with no GPU, and it wrote text (about 48 seconds to load, about 0.6 words per second).
- We built a way to keep the weights as small "FP8" numbers. Memory drops to about a quarter and quality stays almost the same (perplexity 7.08 to 7.27). After speed fixes, loading went from 340 s to 21 s and writing from 1067 s to 69 s. It is still about 2.8 times slower than the normal way, so it is for when memory is short.

**What we tested (honest results)**
- Squeezing the memory afterwards with MLA does not work: it hurts quality (25% smaller gave perplexity 7.1 to 12.7; 50% smaller broke the text).
- We tried "folding" by skipping layers on the 1.5B model. Skipping one layer at a time showed that many middle layers (5, 10, 12, 15 and so on) barely matter, while layers 1, 24 and 26 matter a lot. Skipping three layers gave perplexity 7.19 (from 7.08), only about 7% faster, with slightly less accurate answers. Skipping six or more breaks the model. **Skipping layers without retraining gives only a small gain.**
- Toshiba's Simulated Bifurcation and Fujitsu's Digital Annealer find the best combination among many choices. They do not speed up the AI by themselves. They could help choose which layers to skip or how many bits to use, but we have not tried that yet.

**What comes next (we go on without asking)**
1. Search for the best set of layers to skip with annealing or SB, and compare with the greedy way.
2. Make FP8 faster (SIMD, f16 storage).
3. Measure whether 7B (about 12 to 15 GB) fits a 32 GB PC with FP8.
4. Bring sharded models and FP8 into `pl2pl llm setup`.

**Our promise**: for safe unfinished work we go on without asking each time. We never do irreversible deletes or publish secrets, and we report bad results honestly.

## 2026-11-19 最後の追記: 飛ばす層の探し方 / Last note: choosing which layers to skip

### 日本語(やさしい説明)
- **やったこと**: AIの「飛ばしてもよい層」を、3つ選ぶ問題として2通りで探しました。(1)貪欲法=いちばん良い層を1つずつ足す。(2)焼きなまし=選び方を少しずつ入れ替えて探す。同じ試行回数(78回)で比べました。
- **結果(Qwen2.5-1.5B)**: 貪欲法は {5, 12, 15} 番で迷い度 7.187(元は 7.08)。焼きなましは {12, 15, 16} 番で 7.438。**この小さな問題では、焼きなましは貪欲法に勝てませんでした。** 5個の場合の比較は途中です。
- **気をつけること**: 1つの短い文だけで選んでいるので、その文にだけ合った選び方かもしれません。別の文でも確かめる必要があります。
- **ずっと言えること**: 東芝SBや富士通デジタルアニーラは、層がもっと多い大きなAIや、ビット配分のように選択肢が爆発する問題で試す価値があります。今回の小さな問題では、ふつうの方法で足ります。
- **アップロード**: コードと文書は GitHub の `main` に push しました。VPS には、きれいなクローンだけ安全に更新します(再ビルドや再起動はしません)。
- **これから(確認せず進めます)**: 5個の比較を終える、別の文で確かめる、FP8を速くする、7Bを測る、`pl2pl llm setup` に反映する。

### English (plain and friendly)
- **What we did**: We looked for three "layers that are safe to skip" in two ways. (1) Greedy: add the best layer one at a time. (2) Annealing: swap choices little by little. Both got the same number of tries (78).
- **Result (Qwen2.5-1.5B)**: Greedy picked layers {5, 12, 15} with perplexity 7.187 (the original is 7.08). Annealing picked {12, 15, 16} with 7.438. **On this small problem, annealing did not beat greedy.** The comparison for five layers is still running.
- **A caution**: We chose using one short sentence, so the choice may fit only that sentence. We must check with other sentences.
- **What still holds**: Toshiba's Simulated Bifurcation and Fujitsu's Digital Annealer are worth trying on bigger models with many more layers, or on problems like bit allocation where the choices explode. For this small problem, the plain method is enough.
- **Upload**: Code and docs are pushed to `main` on GitHub. On the VPS we only update clean clones safely (no rebuild, no restart).
- **Next (we go on without asking)**: finish the five-layer comparison, check with other sentences, make FP8 faster, measure 7B, and bring it into `pl2pl llm setup`.

## 2026-11-20 追記: 7Bも動いた / Update: 7B runs too

### 日本語(やさしい説明)
- **7Bが32GBのパソコンで動きました**: Qwen2.5-7B-Instruct(4つに分かれた約15GB)を、FP8で読み込んで文章を作れました。メモリに収まり、読み込みは約3.5分、文章づくりは1語あたり約8秒(とても遅いけれど、動きます)。品質の目安の迷い度は、評価用の文で8.6、別の文で12.2でした。1.5Bの別の文(12.1)とほぼ同じ値です。
- **層を飛ばす方法(折り畳み)の最終結果**: 1つの文だけで飛ばす層を選ぶと、別の文で悪くなりました(5・12・15番を飛ばすと別の文で約15%悪化、5層飛ばすとさらに悪化)。速くなるのは最大でも約14%です。**学び直しなしでは、実用的な効果は小さい**です。貪欲法と焼きなましは、この小さな問題では大差ありませんでした。
- **`pl2pl llm setup` について**: 分割モデルの対応は `aruaru-llm` の `main` に入りました。`pl2pl` が使う配布版(v0.2.5)にはまだ入っていないので、新しい版を出すまで、3Bは今までどおり自動で小さいモデルに下がります。新版を出したら3B・7Bも選べるようにします。
- **これから(確認せず進めます)**: FP8をもっと速くする(SIMDなど)、新しい版の公開、クラウドで頼んだ調査文書の確認。

### English (plain and friendly)
- **7B runs on a 32 GB PC**: We loaded Qwen2.5-7B-Instruct (about 15 GB in four parts) as FP8 and it wrote text. It fits in memory; loading took about 3.5 minutes and writing took about 8 seconds per word (very slow, but it works). The quality score (perplexity) was 8.6 on the test sentence and 12.2 on other sentences, about the same as the 1.5B model on other sentences (12.1).
- **Final result for layer skipping ("folding")**: Choosing layers to skip from a single sentence got worse on other sentences (skipping layers 5, 12 and 15 was about 15% worse on other sentences, and skipping five was worse still). The speed gain is at most about 14%. **Without retraining, the practical gain is small.** On this small problem, greedy search and annealing made little difference.
- **About `pl2pl llm setup`**: Sharded-model support is now on `main` in `aruaru-llm`. The released build that `pl2pl` uses (v0.2.5) does not have it yet, so until a new release, 3B still falls back to a smaller model as before. After the new release, 3B and 7B can be chosen.
- **Next (we go on without asking)**: make FP8 faster (SIMD and more), publish a new release, and check the survey document requested from the cloud.

## 2026-11-20 現状と完成予定: 重みの精度(f32 / FP8 / INT8 / INT4) / Status and plan: weight precision

### 日本語
**現状(実装済み・main)**
- `Precision`(F32 / Fp8 / Int8 / Int4)と層ごとの精度表 `PrecisionTable`。環境変数 `QWEN_PRECISION="int8,1:f32,24:fp8"`(最初の無印が既定、`層:精度`が上書き)。`QwenModel::load_precision` で読む。`QWEN_KEEP_F32` も従来どおり。
- `opencuda-blas` に INT8/INT4 重みの融合CPU行列積(`sgemm_int8_weight` / `sgemm_int4_weight`)。行列全体を展開せず列ブロックごとに並列で積和し、同じスケールを共有するブロックでは係数へ畳み込む。f64参照との相対誤差の試験つき(INT8 0.3%、INT4 6.4%)。
- 実測(Qwen2.5-1.5B、CPUのみ、評価文の迷い度 / 別の文 / 生成秒): f32 7.08 / – / 25、FP8 7.27 / 12.17 / 29.8、**INT8 7.37 / 12.94 / 27.2**、**INT4(128要素グループ)10.55 / 15.37 / 29.0**。INT4を1行1グループにすると迷い度26.4に悪化(グループが粗すぎ)。INT8の最初の版は81秒だったが、除算の除去で27秒になった。
- 正直な注意: 3Bと7BのINT8/INT4は未測定。INT4は品質が落ちる(約+3.5)ので、メモリが足りない時の手段。速さはf32と同程度で、速くはならない(メモリは重みが約1/4〜1/8)。

**完成予定(この順)**
1. 層ごとの感度に基づいて精度表を自動で決める(感度の高い層だけf32/FP8、他はINT4)。
2. KVキャッシュの量子化。
3. 小さい行列のSIMD化によるFP8/INT8/INT4の高速化。
4. 起動時にRAMを調べ、モデルと精度表を自動選択(`aruaru-llm` の `ModelsCompatible` を拡張)。
5. `F64Ref`オラクルを試験に追加(FP64以上は試験専用、本番では使わない)。

### English
**Status (implemented, on main)**
- `Precision` (F32 / Fp8 / Int8 / Int4) and a per-layer `PrecisionTable`. Set `QWEN_PRECISION="int8,1:f32,24:fp8"` (the bare item is the default, `layer:precision` overrides). Load with `QwenModel::load_precision`. `QWEN_KEEP_F32` still works.
- `opencuda-blas` has fused CPU GEMMs for INT8/INT4 weights (`sgemm_int8_weight` / `sgemm_int4_weight`). They never expand the whole matrix; they accumulate per column block in parallel and fold the scale into the coefficient when a block shares one scale. Tested against an f64 reference (relative error INT8 0.3%, INT4 6.4%).
- Measured (Qwen2.5-1.5B, CPU only; perplexity on the eval sentence / on other text / generation seconds): f32 7.08 / – / 25, FP8 7.27 / 12.17 / 29.8, **INT8 7.37 / 12.94 / 27.2**, **INT4 (128-element groups) 10.55 / 15.37 / 29.0**. With one group per row INT4 got much worse (ppl 26.4; groups too coarse). The first INT8 version took 81 s; removing a per-element integer division brought it to 27 s.
- Honest notes: INT8/INT4 on 3B and 7B are not measured yet. INT4 costs quality (about +3.5 ppl), so it is for when memory is short. Speed is about the same as f32, not faster; memory for weights drops to about 1/4 (INT8) or 1/8 (INT4).

**Planned (in this order)**
1. Choose the precision table automatically from per-layer sensitivity (only sensitive layers stay f32/FP8, the rest INT4).
2. KV-cache quantization.
3. SIMD for small matrices to speed up FP8/INT8/INT4.
4. Detect RAM at startup and pick the model and table automatically (extend `ModelsCompatible` in `aruaru-llm`).
5. Add an `F64Ref` oracle to the tests (FP64 and wider are test-only, never used in production).


## 2026-11-20 現状と完成予定(10か国語) / Status and plan in 10 languages
(AI翻訳、ネイティブ確認前 / AI translation, not yet checked by native speakers)

### 日本語
**現状**: 重みの精度を層ごとに選べます(`QWEN_PRECISION="int8,1:f32,24:fp8"`、F32 / FP8 / INT8 / INT4)。CPUだけで実測済み。1.5B: f32 迷い度7.08、FP8 7.27、INT8 7.37(27秒)、INT4 10.55(29秒)。3B: INT8 8.05(40秒)、INT4 10.76(46秒)。重みメモリはINT8で約1/4、INT4で約1/8。速さはf32と同程度。
**完成予定**: ①層の感度から精度表を自動で決める ②KVキャッシュの量子化 ③FP8/INT8/INT4のSIMD高速化 ④起動時にRAMを調べてモデルと精度表を自動選択 ⑤7BのINT4測定 ⑥`pl2pl`の新版公開。

### English
**Status**: Weight precision is selectable per layer (`QWEN_PRECISION="int8,1:f32,24:fp8"`; F32 / FP8 / INT8 / INT4), measured on CPU only. 1.5B: f32 perplexity 7.08, FP8 7.27, INT8 7.37 (27 s), INT4 10.55 (29 s). 3B: INT8 8.05 (40 s), INT4 10.76 (46 s). Weight memory is about 1/4 (INT8) and 1/8 (INT4); speed is about the same as f32.
**Planned**: (1) choose the precision table automatically from layer sensitivity; (2) KV-cache quantization; (3) SIMD speed-up for FP8/INT8/INT4; (4) detect RAM at startup and pick model and table; (5) measure 7B INT4; (6) release a new `pl2pl`.

### 简体中文
**现状**:权重精度可按层选择(`QWEN_PRECISION="int8,1:f32,24:fp8"`,F32 / FP8 / INT8 / INT4),仅在CPU上实测。1.5B:f32困惑度7.08,FP8 7.27,INT8 7.37(27秒),INT4 10.55(29秒)。3B:INT8 8.05(40秒),INT4 10.76(46秒)。权重内存约为INT8的1/4、INT4的1/8,速度与f32相当。
**计划**:①按层敏感度自动决定精度表 ②KV缓存量化 ③FP8/INT8/INT4的SIMD加速 ④启动时检测内存并自动选择模型与精度表 ⑤测量7B的INT4 ⑥发布新版`pl2pl`。

### 繁體中文
**現況**:權重精度可依層選擇(`QWEN_PRECISION="int8,1:f32,24:fp8"`,F32 / FP8 / INT8 / INT4),僅在CPU上實測。1.5B:f32困惑度7.08,FP8 7.27,INT8 7.37(27秒),INT4 10.55(29秒)。3B:INT8 8.05(40秒),INT4 10.76(46秒)。權重記憶體約為INT8的1/4、INT4的1/8,速度與f32相當。
**計畫**:①依層敏感度自動決定精度表 ②KV快取量化 ③FP8/INT8/INT4的SIMD加速 ④啟動時偵測記憶體並自動選擇模型與精度表 ⑤測量7B的INT4 ⑥發布新版`pl2pl`。

### 한국어
**현황**: 가중치 정밀도를 층마다 고를 수 있습니다(`QWEN_PRECISION="int8,1:f32,24:fp8"`, F32 / FP8 / INT8 / INT4). CPU만으로 실측했습니다. 1.5B: f32 혼란도 7.08, FP8 7.27, INT8 7.37(27초), INT4 10.55(29초). 3B: INT8 8.05(40초), INT4 10.76(46초). 가중치 메모리는 INT8에서 약 1/4, INT4에서 약 1/8이며 속도는 f32와 비슷합니다.
**예정**: ①층별 민감도로 정밀도 표 자동 결정 ②KV 캐시 양자화 ③FP8/INT8/INT4 SIMD 가속 ④시작 시 RAM을 확인해 모델과 정밀도 표 자동 선택 ⑤7B INT4 측정 ⑥`pl2pl` 새 버전 공개.

### Deutsch
**Stand**: Die Gewichtspräzision ist pro Schicht wählbar (`QWEN_PRECISION="int8,1:f32,24:fp8"`; F32 / FP8 / INT8 / INT4), nur auf der CPU gemessen. 1.5B: f32-Perplexität 7,08, FP8 7,27, INT8 7,37 (27 s), INT4 10,55 (29 s). 3B: INT8 8,05 (40 s), INT4 10,76 (46 s). Der Gewichtsspeicher sinkt auf etwa 1/4 (INT8) bzw. 1/8 (INT4); die Geschwindigkeit entspricht etwa f32.
**Geplant**: ①Präzisionstabelle automatisch aus der Schichtempfindlichkeit bestimmen ②KV-Cache-Quantisierung ③SIMD-Beschleunigung für FP8/INT8/INT4 ④beim Start RAM prüfen und Modell/Tabelle wählen ⑤7B mit INT4 messen ⑥neue `pl2pl`-Version veröffentlichen.

### Français
**État** : la précision des poids se choisit par couche (`QWEN_PRECISION="int8,1:f32,24:fp8"` ; F32 / FP8 / INT8 / INT4), mesurée sur CPU seulement. 1.5B : perplexité f32 7,08, FP8 7,27, INT8 7,37 (27 s), INT4 10,55 (29 s). 3B : INT8 8,05 (40 s), INT4 10,76 (46 s). La mémoire des poids tombe à environ 1/4 (INT8) et 1/8 (INT4) ; la vitesse est proche de f32.
**Prévu** : ①choisir automatiquement la table de précision selon la sensibilité des couches ②quantification du cache KV ③accélération SIMD de FP8/INT8/INT4 ④détecter la RAM au démarrage et choisir modèle et table ⑤mesurer le 7B en INT4 ⑥publier une nouvelle version de `pl2pl`.

### Español
**Estado**: la precisión de los pesos se elige por capa (`QWEN_PRECISION="int8,1:f32,24:fp8"`; F32 / FP8 / INT8 / INT4), medida solo en CPU. 1.5B: perplejidad f32 7,08, FP8 7,27, INT8 7,37 (27 s), INT4 10,55 (29 s). 3B: INT8 8,05 (40 s), INT4 10,76 (46 s). La memoria de los pesos baja a cerca de 1/4 (INT8) y 1/8 (INT4); la velocidad es parecida a f32.
**Previsto**: ①decidir la tabla de precisión automáticamente según la sensibilidad de las capas ②cuantización de la caché KV ③aceleración SIMD de FP8/INT8/INT4 ④detectar la RAM al iniciar y elegir modelo y tabla ⑤medir 7B en INT4 ⑥publicar una nueva versión de `pl2pl`.

### Português
**Estado**: a precisão dos pesos pode ser escolhida por camada (`QWEN_PRECISION="int8,1:f32,24:fp8"`; F32 / FP8 / INT8 / INT4), medida apenas na CPU. 1.5B: perplexidade f32 7,08, FP8 7,27, INT8 7,37 (27 s), INT4 10,55 (29 s). 3B: INT8 8,05 (40 s), INT4 10,76 (46 s). A memória dos pesos cai para cerca de 1/4 (INT8) e 1/8 (INT4); a velocidade é parecida com a do f32.
**Planejado**: ①definir a tabela de precisão automaticamente pela sensibilidade das camadas ②quantização do cache KV ③aceleração SIMD de FP8/INT8/INT4 ④detectar a RAM na inicialização e escolher modelo e tabela ⑤medir o 7B em INT4 ⑥publicar uma nova versão do `pl2pl`.

### Русский
**Состояние**: точность весов выбирается по слоям (`QWEN_PRECISION="int8,1:f32,24:fp8"`; F32 / FP8 / INT8 / INT4), измерено только на CPU. 1.5B: перплексия f32 7,08, FP8 7,27, INT8 7,37 (27 с), INT4 10,55 (29 с). 3B: INT8 8,05 (40 с), INT4 10,76 (46 с). Память весов примерно 1/4 (INT8) и 1/8 (INT4); скорость близка к f32.
**Планы**: ①автоматически определять таблицу точности по чувствительности слоёв ②квантование KV-кэша ③SIMD-ускорение FP8/INT8/INT4 ④при запуске проверять ОЗУ и выбирать модель и таблицу ⑤измерить 7B в INT4 ⑥выпустить новую версию `pl2pl`.

### 実測追記: 層感度による精度表の自動決定 / Measured: sensitivity-based precision table (2026-11-20)
日本語: Qwen2.5-1.5Bで、全層INT8を基準に各層を1つだけINT4にして評価文の迷い度を測った(`QWEN_AUTOTABLE=K`)。基準7.37に対し、INT4で大きく悪化する層は1番(9.21)・2番(10.03)・27番(7.92)で、他の層は7.2〜7.6でほぼ変わらない(6番は7.23でむしろ良い)。悪化の小さい10層(6,20,7,9,10,18,14,17,13,11)をINT4、残りをINT8にした表は、評価文7.351・別の文13.04(全INT8は7.372・12.94)で、品質はほぼ同じ。注意: 1つの短い文で測っているので、別の文で確かめるまで過信しない(層のノイズも大きい)。次はINT4の層数を増やした時の品質とメモリの関係を測る。
English: On Qwen2.5-1.5B, starting from all-INT8, we set one layer at a time to INT4 and measured perplexity on the eval sentence (`QWEN_AUTOTABLE=K`). Against the 7.37 baseline, INT4 hurts a lot only on layers 1 (9.21), 2 (10.03) and 27 (7.92); the other layers stay at 7.2-7.6 (layer 6 is even slightly better at 7.23). A table with the 10 least-sensitive layers (6,20,7,9,10,18,14,17,13,11) in INT4 and the rest INT8 gives 7.351 on the eval sentence and 13.04 on other text (all-INT8: 7.372 / 12.94), i.e. about the same quality. Caution: this uses one short sentence, so do not over-trust it until checked on other text (per-layer noise is large). Next: measure quality and memory as the number of INT4 layers grows.

## 2026-11-21 精度の自動選択と省メモリ(open-cuda・aruaru-llm)
- **重みの精度**: CPUだけの32GB機では、重みをINT8で持つのが実用的です(FP8より約4倍速く、7Bが1トークン約2秒)。中間層だけINT4にもできますが、別の文では約17%悪化しました。FP8は遅いので既定にしません。
- **自動選択**: `PrecisionTable::auto_for_memory` が、f32、全層INT8、中間層INT4の順に、空きRAMの7割に収まる最初の表を選びます。`aruaru-llm` では環境変数 `ARUARU_LLM_PRECISION`(未設定=f32、`auto`、または `int8,1:f32` のような表)で指定します。
- **KVのINT8**: `QWEN_KV_INT8=1` は丸めの影響だけを測る試験で、迷い度は約1%の悪化でした。メモリはまだ減りません。
- **未了**: KVを実際にINT8で保持する実装、実サーバーでの `auto` の通し確認。翻訳はAIによるもので、母語話者の確認はまだです。

## 2026-11-21 Automatic precision selection and low memory (open-cuda, aruaru-llm)
- **Weight precision**: on a CPU-only 32 GB machine, INT8 weights are the practical choice (about 4x faster than FP8; 7B runs at roughly 2 seconds per token). INT4 on the middle layers is possible but was about 17% worse on held-out text. FP8 is slow, so it is not the default.
- **Automatic choice**: `PrecisionTable::auto_for_memory` picks the first table that fits 70% of free RAM, in the order f32, all-INT8, INT4 on middle layers. In `aruaru-llm`, set the environment variable `ARUARU_LLM_PRECISION` (unset = f32, `auto`, or a table such as `int8,1:f32`).
- **INT8 KV cache**: `QWEN_KV_INT8=1` only simulates the rounding; perplexity got about 1% worse. Memory is not reduced yet.
- **Open**: actually storing the KV cache as INT8, and an end-to-end run of `auto` on a real server. Translations are AI-made and not yet checked by native speakers.

## 2026-11-21 精度自动选择与节省内存(open-cuda、aruaru-llm)
- **权重精度**:在只有CPU的32GB机器上,用INT8保存权重最实用(比FP8快约4倍,7B模型每个词元约2秒)。也可以只把中间层改为INT4,但在另一段文字上质量下降约17%。FP8较慢,因此不作为默认。
- **自动选择**:`PrecisionTable::auto_for_memory` 按"f32、全部INT8、中间层INT4"的顺序,选出第一个能放进可用内存70%的方案。在 `aruaru-llm` 中用环境变量 `ARUARU_LLM_PRECISION` 指定(未设置=f32、`auto`,或形如 `int8,1:f32` 的表)。
- **KV缓存INT8**:`QWEN_KV_INT8=1` 只是模拟舍入的影响,困惑度约变差1%,内存尚未减少。
- **未完成**:真正以INT8保存KV缓存,以及在真实服务器上完整测试 `auto`。翻译由AI完成,尚未经母语者确认。

## 2026-11-21 精度自動選擇與節省記憶體(open-cuda、aruaru-llm)
- **權重精度**:在只有CPU的32GB機器上,以INT8保存權重最實用(比FP8快約4倍,7B模型每個詞元約2秒)。也可以只把中間層改為INT4,但在另一段文字上品質下降約17%。FP8較慢,因此不作為預設。
- **自動選擇**:`PrecisionTable::auto_for_memory` 依「f32、全部INT8、中間層INT4」的順序,選出第一個能放進可用記憶體70%的方案。在 `aruaru-llm` 中以環境變數 `ARUARU_LLM_PRECISION` 指定(未設定=f32、`auto`,或形如 `int8,1:f32` 的表)。
- **KV快取INT8**:`QWEN_KV_INT8=1` 只是模擬捨入的影響,困惑度約變差1%,記憶體尚未減少。
- **未完成**:真正以INT8保存KV快取,以及在真實伺服器上完整測試 `auto`。翻譯由AI完成,尚未經母語人士確認。

## 2026-11-21 정밀도 자동 선택과 메모리 절약 (open-cuda, aruaru-llm)
- **가중치 정밀도**: CPU만 있는 32GB 컴퓨터에서는 가중치를 INT8로 두는 것이 실용적입니다(FP8보다 약 4배 빠르고, 7B 모델이 토큰당 약 2초). 중간 층만 INT4로 바꿀 수도 있지만, 다른 문장에서는 품질이 약 17% 나빠졌습니다. FP8은 느려서 기본값으로 쓰지 않습니다.
- **자동 선택**: `PrecisionTable::auto_for_memory`가 f32, 전 층 INT8, 중간 층 INT4 순으로, 남은 RAM의 70%에 들어가는 첫 번째 구성을 고릅니다. `aruaru-llm`에서는 환경 변수 `ARUARU_LLM_PRECISION`으로 지정합니다(미설정=f32, `auto`, 또는 `int8,1:f32` 같은 표).
- **KV 캐시 INT8**: `QWEN_KV_INT8=1`은 반올림의 영향만 재는 시험이며, 혼란도(perplexity)는 약 1% 나빠졌습니다. 메모리는 아직 줄지 않았습니다.
- **미완료**: KV 캐시를 실제로 INT8로 저장하는 구현, 실제 서버에서 `auto`를 끝까지 돌려 보는 시험. 번역은 AI가 했으며 원어민 확인은 아직입니다.

## 21-11-2026 Elección automática de precisión y poca memoria (open-cuda, aruaru-llm)
- **Precisión de los pesos**: en un equipo de 32 GB sin GPU, lo práctico es guardar los pesos en INT8 (unas 4 veces más rápido que FP8; un modelo 7B tarda unos 2 segundos por token). También se puede usar INT4 solo en las capas centrales, pero con otro texto la calidad empeoró cerca de un 17 %. FP8 es lento, así que no es el valor por defecto.
- **Elección automática**: `PrecisionTable::auto_for_memory` elige, en el orden f32, todo INT8, INT4 en las capas centrales, la primera tabla que cabe en el 70 % de la RAM libre. En `aruaru-llm` se indica con la variable de entorno `ARUARU_LLM_PRECISION` (sin definir = f32, `auto`, o una tabla como `int8,1:f32`).
- **Caché KV en INT8**: `QWEN_KV_INT8=1` solo simula el efecto del redondeo; la perplejidad empeoró cerca de un 1 %. La memoria aún no se reduce.
- **Pendiente**: guardar de verdad la caché KV en INT8 y ejecutar `auto` de principio a fin en un servidor real. Las traducciones las hizo una IA y aún no las han revisado hablantes nativos.

## 21/11/2026 Escolha automática de precisão e pouca memória (open-cuda, aruaru-llm)
- **Precisão dos pesos**: num computador de 32 GB sem GPU, guardar os pesos em INT8 é a escolha prática (cerca de 4 vezes mais rápido que FP8; um modelo 7B leva uns 2 segundos por token). Também é possível usar INT4 só nas camadas do meio, mas com outro texto a qualidade piorou cerca de 17 %. O FP8 é lento, por isso não é o padrão.
- **Escolha automática**: `PrecisionTable::auto_for_memory` escolhe, na ordem f32, tudo em INT8, INT4 nas camadas do meio, a primeira tabela que cabe em 70 % da RAM livre. No `aruaru-llm`, indica-se com a variável de ambiente `ARUARU_LLM_PRECISION` (não definida = f32, `auto`, ou uma tabela como `int8,1:f32`).
- **Cache KV em INT8**: `QWEN_KV_INT8=1` apenas simula o efeito do arredondamento; a perplexidade piorou cerca de 1 %. A memória ainda não diminui.
- **Pendente**: guardar de fato o cache KV em INT8 e executar o `auto` de ponta a ponta num servidor real. As traduções foram feitas por uma IA e ainda não foram revistas por falantes nativos.

## 2026-11-21 追記: KVキャッシュを実際にINT8で保持
- `QWEN_KV_INT8=1` で、KVキャッシュを1ヘッド1トークンごとのスケール付きINT8として**実際に保持**するようにした(`open-cuda` 2ba4b5f)。Qwen2.5-1.5Bで迷い度は7.079→7.159(約+1%)、別の文では12.1→11.8(誤差の範囲)。KVのメモリはf32の約1/4(head_dim 64の試験で3倍以上の縮小を確認)。
- 未了: `ARUARU_LLM_PRECISION=auto` を実サーバーで通して確認すること。翻訳はAIが作成し、ネイティブ話者による確認は未了。
