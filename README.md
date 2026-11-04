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
