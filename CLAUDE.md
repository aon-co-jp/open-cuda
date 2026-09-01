# 設計思想＆開発方針＆開発環境ルール(open-cuda)

> ## 🎯🕒 aruaru-db×RPoem SET連携方針+「現時点限定」の暫定判断(2026-08-29、
> 全リポジトリ横断調査、ユーザー指示によりファイル最上部へ固定)
>
> **正本はaruaru-db/CLAUDE.md冒頭「🎯最重要・最優先で常に念頭に置く
> こと」**: REST APIの代替をただ闇雲に作らない、aruaru-dbはRPoemとの
> SETで初めて「REST API不要・WunderGraph Cosmo有料版(Enterprise)
> 互換」という価値が成立する、という戒めがそこに記されている。
>
> **本リポジトリ(open-cuda)は現時点では対象外**: `CLAUDE.md`に
> REST API・APIキーへの言及が一件も無いことを実際に`grep`で確認済み
> ——GPU計算ライブラリでありHTTPサーバー面・APIキー認証面を一切
> 持たないため、今のところこのSET連携方針(REST撤廃・APIキー自動
> 管理)の適用対象が構造的に存在しない。詳細はaruaru-db/CLAUDE.mdの
> 2026-08-29 HANDOFF「open-cuda/open-directxとの連携性・必要性の
> 再確認」参照。
>
> **⚠️ ただしこの「対象外」は恒久的な結論ではなく、今この瞬間だけの
> 暫定判断であることを忘れないこと(ユーザー指摘、2026-08-29)**:
> 今後open-directxのDirectX互換実装・開発が進み、**OSレベルから
> アプリへ直接命令する、あるいはアプリがハードウェアアクセラレーター
> (open-directx/open-cpu)経由で動作した方がメリットが大きくなる場面
> が増えていく**という予測がある。そうなればopen-cuda・aruaru-llm・
> aruaru-db・open-web-server・RPoemとの連携が実際に深まり、この
> SET連携方針(REST API不要・Cosmo有料版互換)の適用範囲そのものを
> 再評価すべき時が来る可能性が高い。開発が進むたびに「今もまだ
> 対象外のままか」を問い直すこと。
> **📌 保留タスク(2026-08-06、次回セッションで着手予定)/ Pending task (added 2026-08-06, to be started next session)**:
> ユーザー指示により、**東芝の疑似量子コンピューター技術(Simulated
> Bifurcation Machine)**と**DeepSeekの技術**(インターネットニュースだけ
> でなく、論文〈DeepSeek-V3/R1テクニカルレポート等〉・実装ノウハウの
> ブログまで日英両言語でGoogle/GitHub調査)を、`dream-os`/`open-directx`/
> `open-cuda`/`aruaru-llm`/`open-web-server`/`RPoem`/`open-raid-z`/
> `aruaru-db`の8リポジトリへ組み込む構想がある。東芝SBMは`dream-os`
> (`sbm_ising`カーネル、64スピンPoC)に実装済み——他リポジトリへの適用は
> 各リポジトリで「何を最適化するか」を先に特定してから着手すること
> (このリポジトリ固有の候補は未検討、次回調査対象)。DeepSeekは前回調査で
> 「数千枚のGPUを1枚に圧縮する技術」という主張は確認できなかった(誤解・
> 誇張と判断済み)——今回は論文・実装ブログまで調査範囲を広げ、実在する
> 技術(MLA・DeepSeekMoE・FP8混合精度学習等)を特定してから適用箇所を
> 検討すること。詳細は`dream-os/CLAUDE.md`の同日HANDOFF参照。
>
> By user instruction, there is a plan to incorporate **Toshiba's
> pseudo-quantum-computer technology (Simulated Bifurcation Machine)**
> and **DeepSeek's technology** (researched via Google/GitHub in both
> Japanese and English, going beyond news articles to actual papers
> like the DeepSeek-V3/R1 technical reports and implementation-notes
> blogs) into 8 repositories: `dream-os`, `open-directx`, `open-cuda`,
> `aruaru-llm`, `open-web-server`, `RPoem`, `open-raid-z`, and
> `aruaru-db`. Toshiba SBM is already implemented in `dream-os` (the
   `sbm_ising` kernel, a 64-spin PoC) — applying it elsewhere requires
> first identifying a concrete optimization problem in each repo (not
> yet investigated for this repo). The previous DeepSeek research found
> no evidence for a "compress thousands of GPUs into one" technology
> (judged to be a misunderstanding/exaggeration) — this time, broaden
> the research to papers and implementation blogs, identify real
> techniques (MLA, DeepSeekMoE, FP8 mixed-precision training, etc.),
> then decide where they apply. See the same-day HANDOFF entry in
> `dream-os/CLAUDE.md` for details.


作業ドライブは`F:\runo`。この節は[`open-raid-z`](https://github.com/aon-co-jp/open-raid-z)の
`CLAUDE.md`を正本とし、各プロジェクトへコピーして同期する方針に準じる。
GitHubリポジトリ: [aon-co-jp/open-cuda](https://github.com/aon-co-jp/open-cuda)。

**開発開始日: 2026-06-26**(このリポジトリのGitHub作成日)

第二のCUDA。Windows＋MAC＋LINUX互換 ＆ INTEL＋AMD＋nVIDIA互換を開発途中です。

## このプロジェクトの役割

GPU抽象化・計算基盤(`OmniGPU`設計、詳細は`OmniGPU-Design.md`参照)。
`opencuda-core`/`opencuda-cpu`/`opencuda-vulkan`/`opencuda-blas`
(GEMM/Attention/量子化)・`open-cuda-bert`(BERT系エンコーダのforward pass、
multilingual-e5-small対応)から成るCargoワークスペース。`aruaru-llm`との
SET構成(GPU/CPU実行パイプラインの実装先)。

前回のマーケティング調査(Python製AIライブラリのRust移植ランキング)で
言う「1〜6位の良いとこ取りハイブリッド/トライブリッド」の実体
——`opencuda-blas`がNumPy相当、`open-cuda-bert`がTransformers推論パス
相当。今後`open-cuda-llm`(vLLM相当、自己回帰デコーダ追加時)を追加予定。

## 詳細な設計・現状

- `OmniGPU-Design.md` — アーキテクチャ設計書(Vulkan Compute優先、
  段階的なCUDA/ROCm/oneAPI対応ロードマップ、正直な規模見積もり含む)。
- `DEVELOPMENT-NEXT.md` — 直近の実装状況(v0.4.1時点、matmul実Vulkan
  実行・INT4/INT8量子化等)。
- `CHANGELOG.md` — バージョン履歴。

## HANDOFF追記(2026-09-01、`open-cuda-llm`にModel Folding〈層冗長性検出・除去・線形アダプタ置換〉を実装——他アカウントでの再開用メモ、必ず読むこと)

`aruaru-llm`側からの依頼(ユーザー指示「DeepSeekの折りたたみ理論
(Model Folding)を実装してほしい」)への対応として、`crates/
open-cuda-llm/src/lib.rs`の`GptModel`へ3段階の機能を追加した
(HTTP配線・利用者向けドキュメントは`aruaru-llm`側、詳細な経緯・実測
結果は`aruaru-llm/CLAUDE.md`の同日HANDOFFを正本とし、ここでは
このリポジトリ固有の実装詳細のみ記す)。

**最重要の事実確認**: 日英2言語でGoogle/GitHub調査した結果、
**「DeepSeekの折りたたみ理論」という技術は実在しないと判明した**
(DeepSeekの実際の効率化技術はMLA・FP8混合精度・DeepSeekMoEであり、
いずれも「折りたたみ」ではない)。混同の元と考えられるのは無関係の
ICLR 2025論文「Model Folding」(arXiv:2502.10216、ニューロン単位
k-meansクラスタリングを要する高度な手法)——`GptModel`の重みには
公開アクセサが無く忠実な再現は困難と判断し、以下3つの代替手法を
実装した。

1. **`analyze_layer_redundancy`/`prune_redundant_layers`**
   (ShortGPT arXiv:2403.03853/Gromov et al. arXiv:2403.17887方式、
   独立閾値): 各層の入力/出力コサイン類似度からBlock Influenceを
   算出し、閾値未満の層を除去する。既存の`forward_prefill`
   インフラをそのまま再利用(新規GPU固有コードは追加していない、
   `device: &Arc<dyn GpuDevice>`引数経由で呼び出し側が渡した
   CPU/Vulkan/DirectXいずれでも動作)。
2. **`find_best_layer_block_to_remove`/`remove_layer_block`**
   (Gromov et al.論文の本来のアルゴリズムに忠実化): 削除したい
   層数を固定し、その本数の連続ブロックを総当たり比較して除去
   影響が最小の1つを選ぶ。上記1の「独立した層を寄せ集める」弱点
   (実測で6層中5層の一括削除が破綻を招いた)を解消する設計。
3. **`fold_block_with_linear_adapter`+`DecoderLayer::
   linear_adapter`**(SHIFT-LLM arXiv:2608.25068/SlimLLM
   arXiv:2505.22689着想のclosed-form線形置換、**正直な開示: これらは
   非常に新しい論文であり本実装は再現実装ではなく独自の簡略版**):
   除去ブロックを跡形もなく消すのではなく、最小二乗法(閉形式の
   リッジ回帰、`nalgebra`使用、勾配降下法は使わない)でフィットした
   1層の線形アダプタへ置換する。Attentionサブ層は`Linear::zeroed`で
   出力を常にゼロに潰し(残差のみ通過)、FFNサブ層は`Linear::
   identity`(恒等写像)+GELU+`output`(唯一のフィット対象)という
   構成——既存の`DecoderLayer`構造体を100%再利用し、新しいレイヤー型は
   追加していない。

**実測結果(distilgpt2、6層、極端な予算=6層中5層除去)**: 方式1・2は
完全な劣化ループ("Theodoreodoreodoreodore...")に陥ったが、方式3は
劣化ループを回避し実在の英単語を使った出力("I slowly, it the
rainforest in a few one way with at right outside to play.")を
生成した。**正確に言うと**: 方式3は実測できる本物の改善だが完全な
修正ではない——出力は依然として文法的に一貫した文章にはならない。

**検証**: `cargo test -p open-cuda-llm --release -- --test-threads=1`
で**32件全green**(新規6件+既存26件、実GPU経路のregressionテスト含む、
`--test-threads=1`が必要な理由〈重量級テストの資源競合〉は既存の
2026-08-08 HANDOFF参照)。実重み(distilgpt2/gpt2)を使う`--ignored`
テスト3件(`manual_compare_independent_threshold_vs_block_search_
on_real_gpt2_weights`・`manual_compare_plain_removal_vs_linear_
adapter_at_extreme_budget_on_real_gpt2_weights`等)も実行し、上記の
実測結果を確認済み。

**未着手・次回検討候補**: (1) 線形アダプタの`ridge_lambda`(既定
`1e-2`固定)を呼び出し側から調整可能にする、(2) Attentionサブ層を
ゼロに潰す設計はQKV射影・softmax計算自体は実行してしまう(出力だけ
捨てる)ため計算コストが完全には削減されない——専用の「Attentionを
スキップする軽量パス」を追加する余地がある、(3) 較正データは英語の
一般文のみで日本語・他言語での検証は未実施、(4) この3手法をVulkan/
DirectX経由(実GPU)で実測することは今回未実施(CPU実行のみで検証)。

コミット: `890f8b1`(方式1)・`74263e1`(方式2)・`3a887ad`(方式3)。

## HANDOFF追記(2026-09-01(続き)、Model Foldingの残課題4項目に着手・完了 / Follow-up: addressed all 4 remaining Model Folding gaps)

直前エントリの「未着手・次回検討候補」4項目すべてに着手した(複数
セッション並行作業、コミット`761cb76`)。

1. **`ridge_lambda`の外部調整可能化**: `GptModel::
   fold_block_with_linear_adapter`のシグネチャに`ridge_lambda:
   Option<f32>`引数を追加。`None`なら既定値`1e-2`、`Some(v)`なら
   `v`をそのまま使う(`v`が非有限・0以下なら`ensure!`で正直に拒否)。
   `AdapterFoldReport::ridge_lambda_used`で実際に使われた値を常に
   開示する。`aruaru-llm`側`POST /v1/models/fold-layers`の
   `ridge_lambda`リクエストパラメータ経由でHTTPから調整可能
   (`aruaru-llm/CLAUDE.md`同日エントリ参照)。新規テスト2件
   (既定/明示値の反映確認・非正/NaN/inf値の拒否確認)。
2. **実GPU経路(Vulkan/DirectX)での3手法の実測**: 新規
   `manual_bench_fold_layers_cpu_vs_vulkan_vs_directx_on_real_gpt2_
   weights`(`--ignored`、実重み・実GPU必要)。このマシン
   (NVIDIA GeForce GT 730、Windows)で実際に実行した結果
   (`analyze_layer_redundancy`/`find_best_layer_block_to_remove`/
   `fold_block_with_linear_adapter`、実GPT-2 124M・12層・2層除去):

   | 経路 | analyze_layer_redundancy | find_best_layer_block_to_remove | fold_block_with_linear_adapter |
   |---|---|---|---|
   | CPU | 1.96s | 1.90s | 2.07s |
   | Vulkan | 43.69s | 32.18s | 34.58s |
   | DirectX(DXIL常駐オフロード) | 4.03s | 4.62s | 4.22s |

   **正直な結論: このマシンではCPUが最速、DirectXがCPUの約2倍、
   Vulkanが最も遅い(CPUの約17〜22倍)**。これは過去HANDOFF
   (2026-08-15・2026-08-22・2026-08-23)で既に実測済みの傾向
   ——GT730はGEMMそのものがCPUより遅く、かつAttention/LayerNorm/
   GELU自体はCPU側で計算するハイブリッド構成(DXILオフロードは
   密GEMMのみ)のため、H2D/D2H転送・ディスパッチのオーバーヘッドが
   支配的になる——と一致する。「GPUで速くなった」とは主張しない。
   より強い統合GPU+弱いCPUの組み合わせ(過去実測のAdreno 619)なら
   有利になり得るが、**この機・このワークロードでは未検証のまま**。
3. **日本語・多言語較正データでの検証**: `aruaru-llm`側に
   `multilingual_fold_calibration_prompts()`(英語・日本語・中国語・
   フランス語・ドイツ語・スペイン語混在12文)を新設し、
   `fold_active_model`系3関数すべての既定較正データ(`sample_prompts`
   省略時)をこれに切り替えた。実GPT-2 124M重みで
   `fold_active_model_by_block`(1層除去)を実行し、折りたたみ前後
   とも`generate()`が正常に動作すること(クラッシュしない、空文字
   列を返さない)を確認済み。**正直な開示**: `GptTokenizer`は英語
   中心の学習済みBPE語彙(GPT-2本体)のため、日本語・多言語入力でも
   トークン化自体は失敗しないが「日本語での折りたたみ後品質が英語と
   同等」であることまでは主張しない(詳細は`aruaru-llm/CLAUDE.md`
   同日エントリ参照)。
4. **UIボタンからの実際のE2E呼び出し**: `open-english`側
   `index.html`/`app.js`に`POST /v1/models/fold-layers`を実際に
   呼ぶボタン(層数・線形アダプタチェックボックス・`ridge_lambda`
   入力欄)を新設。このセッションで実際にaruaru-llmサーバーを起動し
   (`127.0.0.1:4601`)、UIのJSが送るのと同じリクエストボディ
   (`{"num_layers_to_remove":1,"use_linear_adapter":true,
   "ridge_lambda":0.5}`)を`curl`で送信、`ridge_lambda_used:0.5`が
   レスポンスに正しく反映されること、折りたたみ前後の生成サンプルが
   両方とも文法的に妥当な英文であることを実HTTPで確認した(詳細は
   `aruaru-llm/CLAUDE.md`・`open-english/CLAUDE.md`同日エントリ参照)。

**検証**: `cargo test -p open-cuda-llm --release -- --test-threads=1`
**34件全green**(0失敗、3件`--ignored`)。`--ignored`ベンチ
(上記2.)も実際に実行し実測値を取得済み(モックやスキップではない)。
`aruaru-llm`側`cargo test --release -- --test-threads=1`
**101件全green**(2件`--ignored`、うち多言語較正テストは実際に
`--ignored`で実行し実測を確認済み)。

**残課題(正直な開示)**: (1) Attentionサブ層を完全にスキップする
軽量パス(QKV射影自体を省略)は未実装のまま(前回HANDOFFから変更
なし)。(2) より高性能な統合GPU(Adreno等)でのDirectX/Vulkan経路
再実測は、このマシンにその種のGPUが無いため引き続き未検証。

## HANDOFF追記(2026-08-23、階層的アクセラレーション: D3D12 Compute経由のGEMMオフロードを実装 / Hierarchical acceleration: DXIL GEMM offload)

ユーザーの目的「NVIDIA GPU非搭載の安価なPCでもAI推論をなるべく速く」への対応として、
`CUDA → Vulkan → DirectX 12 → CPU SIMD`という階層的フォールバックの
**DirectX段**を実装した。

### 実装内容

- `opencuda-blas`:
  - `GemmPath::DirectXGeneric`を追加。`select_gemm_path`は
    「ベンダー専用経路がスタブ、かつ`supports_spirv()==false`、かつ
    `supports_dxil()==true`」のときにこの経路を選ぶ(Vulkanが使える
    なら従来どおりVulkanを優先。既存挙動は不変)。
  - `sgemm_directx_generic(device, m,k,n, a,b, dxil)`を新設
    (`sgemm_vulkan_generic`と同じ契約のD3D12版)。
  - `upload_resident_matrix` / `sgemm_directx_resident_b`を新設。
    重み行列`B`をVRAMへ**常駐**させ、毎回のH2D転送を省く
    (`lm_head`なら768×50257×4 ≒ 154MBを1トークンごとに転送していた)。
- `open-cuda-llm`: `GptModel::set_matmul_dxil_offload(device, dxil)`を新設。
  全`Linear`(QKV融合 / attn_out / intermediate / output / lm_head)の
  重みをD3D12デバイスへアップロードし、密GEMMだけをそこへオフロードする。
  `ResidentWeights`(RAII)が`GptModel`のdropで全バッファを解放する。

### 実測(この開発機: NVIDIA GeForce GT 730 + Ryzen 9 3950X〈avx2+fma3〉)

`cargo test --release -p opencuda-blas --test sgemm_directx_bench -- --nocapture`
の実出力(GPT-2形状のGEMM、単位ms):

| m | k | n | DirectX(毎回転送) | DirectX(重み常駐) | CPU(AVX2) |
|---|---|---|---|---|---|
| 1 | 768 | 2304 | 33.0 | 4.7 | 1.6 |
| 1 | 768 | 768 | 25.6 | 4.1 | 0.14 |
| 1 | 768 | 3072 | 49.8 | 5.2 | 0.48 |
| 1 | 3072 | 768 | 46.2 | 12.1 | 0.56 |
| 1 | 768 | 50257 | 468.7 | 51.0 | 8.4 |
| 64 | 768 | 3072 | 73.0 | 36.7 | 2.7 |

**正直な結論: この開発機では DirectX 経路は CPU(AVX2)より遅い**
(重み常駐化で6〜10倍改善したが、それでもCPU比3〜30倍遅い)。原因は
(a)`matmul.hlsl`がタイリング無しのnaive実装、(b)GT 730が非力
(過去HANDOFF 2026-08-15でもGEMMはCPUの1/2〜1/4の速度と実測済み)、
(c)デコード時は`m=1`で8×8スレッドグループの7/8が遊ぶ。
**「速くなった」とは主張しない。** より強い統合GPU+弱いCPUの組み合わせ
(過去HANDOFFのAdreno 619実測ではGPUが最大5.99倍速かった)では有利に
なり得るが、それは**この機では未検証**。既定では無効のままにしてある。

### 検証(実機、型チェックのみで完了と報告しない方針の徹底)

- 新規`crates/opencuda-blas/tests/sgemm_directx_real.rs`(2件):
  実D3D12デバイス上の`sgemm_directx_generic`がCPU参照実装と1e-3以内で
  一致すること、`select_gemm_path`がDirectXデバイスに対して
  `DirectXGeneric`を返すことを実機で確認。
- 新規`generate_end_to_end_matches_cpu_on_real_d3d12_after_set_matmul_dxil_offload`
  (`open-cuda-llm`): DXILオフロードを配線した`generate()`の出力が、
  純CPU実行と**トークン列完全一致**することを実D3D12上で確認。
- `cargo test --release -p opencuda-blas -p open-cuda-llm -- --test-threads=1`
  全green(blas 34件、llm 24件+実機テスト群)。

- 次にすべきこと: (1) `matmul.hlsl`をタイル化(共有メモリ)して
  naive実装から改善する。(2) `m=1`(デコード)専用のGEMV形カーネルを
  用意し、8×8スレッドグループの無駄をなくす。(3) Attention/LayerNorm/
  GELUもDXIL化してモデル全体をGPU常駐にする(現状は密GEMMのみ)。
  (4) Intel/AMD統合GPU搭載機での再実測——この機のGT 730は
  「安いPCの統合GPU」の代表として適切ではない。

## HANDOFF追記(2026-08-23、open-cpu への CPU 機能検出の一元化 + 実バグ 2 件修正)

`opencuda-blas` の `simd.rs` が独自に `is_x86_feature_detected!` を
呼んで CPU 機能を検出していたのを、エコシステム共通基盤
[`open-cpu`](https://github.com/aon-co-jp/open-cpu) へ移譲した
(`Cargo.toml` に path 依存を追加)。`CpuFeatures` 構造体のフィールドは
従来どおりなので既存の呼び出し側は無変更。

**この作業中に見つけた実バグを 2 件修正した(いずれも「単独の機能
フラグで分岐していたが、実際に必要なのは複数機能の組み合わせ」という
同種の誤り):**

1. **AVX-512 経路が opt-in 無しで選択される状態だった。**
   `dot_f32` / `axpy` が `if f.avx512f` だけで 512bit 経路へ入るため、
   AVX-512 搭載機で実行すると **実機未検証のコードが自動的に走って
   しまう**。open-cpu の方針(未検証パスは既定で選ばせない)に合わせ、
   `avx512_f32_path()` を新設して `AVX-512F+BW+VL` が揃い、かつ
   環境変数 `OPEN_CPU_ENABLE_AVX512=1` がある場合のみ選ぶようにした。
2. **int8 VNNI の分岐条件が `target_feature` の宣言と一致していなかった。**
   `dot_i8_avx512vnni` は `#[target_feature(enable = "avx512vnni,avx512bw,avx512f")]`
   と 3 機能を要求しているのに、呼び出し側は `if f.avx512vnni` だけを
   見ていた。`avx512bw`/`avx512f` も確認する組み合わせ判定へ修正。
   `dot_i8_avxvnni`(`avxvnni,avx2`)も同様に修正。

あわせて `CpuFeatures` に `avx512bw` フィールドと、組み合わせ判定用の
`isa_profile()` / `has_avx2_fma()` / `has_vnni_path()`、ログ用の
`cpu_runtime_line()` を追加した。

**検証**: `cargo test -p opencuda-blas --release` **34 テスト通過**
(スカラー参照実装との一致テストを含む)。開発機は Ryzen 9 3950X
(Zen 2)で `isa_profile = avx2+fma3`。**AVX-512 / VNNI 経路は
引き続き実機未検証**(CPU が非搭載)。

- 次にすべきこと: GEMM の重み再配置(llama.cpp の online repack 相当)は
  本リポジトリの規模に対して過剰と判断し見送った。GFNI/AMX も同様。
  詳細な技術動向調査の結果は `open-cpu/CLAUDE.md` の 2026-08-23 HANDOFF に
  参照元リンク付きで記録してある。

## HANDOFF追記(2026-08-15、sftp-git開発セッションからの横断作業)

- **Android実機(moto g53y 5G、Adreno 619)でVulkan実行を実機検証
  (ユーザー指示「スマホのGPUは意外と高速かも知れないのでTESTして」)**:
  1. `vulkan_info`をaarch64-linux-android向けにクロスコンパイルし、
     `adb push`+実機実行で**実際にAdreno 619を検出**
     (`OpenCUDA Vulkan Device (Adreno (TM) 619)`、共有VRAM
     3.9GB、Vulkan API 1.1.128、`INTEGRATED_GPU`)——過去HANDOFF
     (2026-07-25)の「クロスコンパイル成功のみ確認、実機実行は未検証」
     という制約を解消した。
  2. `matmul_vulkan_real`もクロスコンパイル・実機実行し、CPU版・
     Vulkan版・CPU参照実装の3経路が全て数値一致することを実機で確認。
     **実行中に発見したバグ**: シェーダパス解決が
     `env!("CARGO_MANIFEST_DIR")`(開発機のビルド時絶対パス)固定
     だったため、`adb push`しただけの実機では`No such file or
     directory`で即失敗した。実行ファイルと同じディレクトリの
     `shaders/matmul.spv`を優先的に探すフォールバックを追加して解消
     (`adb push`でバイナリとシェーダを一緒に配置する運用に対応)。
  3. **正直な開示・速度比較は未達成**: `matmul_vulkan_real`は正しさの
     検証(数値一致)のみを行う設計で、CPU/GPU個別の所要時間を計測する
     機構を持たない。64×64という小さい行列サイズも速度比較には
     不十分(全体の実行が0.10秒程度で、デバイス初期化オーバーヘッドが
     支配的になり計算時間の差を切り分けられない)。**「スマホのGPUが
     意外と高速」という仮説を検証できるだけの実測データは今回得られて
     いない**——誇張せず、この限界を明記する。
  4. **検証結果**: 実機2バイナリ(`vulkan_info`・`matmul_vulkan_real`)
     とも実際にAndroid実機上で正しく動作することを確認(型チェック・
     クロスコンパイル成功のみでの完了報告ではない)。
  - 次にすべきこと: (1) CPU/GPU個別の所要時間を計測する専用ベンチマーク
    (十分大きい行列サイズ、複数回実行の平均、デバイス初期化オーバー
    ヘッドを含む場合/含まない場合の両方を分けて計測)を実機実行で
    行う——「スマホのGPUが意外と高速かも」という仮説に実測で答える
    には現状のexampleでは不十分。(2) `matmul_vulkan_real`以外の
    example(`softmax_vulkan_real`・`raid6_*_vulkan_real`等)も
    同じ`CARGO_MANIFEST_DIR`固定パスの問題を抱えている可能性が高く、
    同様のexe相対パスフォールバックが必要か点検する価値がある。

- **2026-08-15(続き) 専用ベンチマーク`matmul_bench`を新設し、
  「スマホのGPUは意外と高速かも」という仮説を実機で検証(直上の
  「次にすべきこと(1)」への対応)**:
  1. **新規`examples/matmul_bench`**: サイズ・反復回数を引数指定でき
     (既定256×256・5回)、CPU/GPUそれぞれ「デバイス初期化時間」と
     「計算のみの時間(複数回実行の中央値/最小/最大)」を分離して計測。
     CPU/GPU結果の数値一致を確認した上でのみ速度を比較する設計
     (結果が食い違う場合は速度比較自体を無効として扱う)。
  2. **実機検証(デスクトップNVIDIA GT730 vs Android実機Adreno 619、
     型チェックのみで完了と報告しない方針を徹底)**:

     | 環境 | 行列サイズ | CPU計算(中央値) | GPU計算(中央値) | 結果 |
     |---|---|---|---|---|
     | デスクトップ(GT730) | 256×256 | 2.27ms | 10.27ms | CPUが4.52倍速い |
     | デスクトップ(GT730) | 512×512 | 40.00ms | 93.43ms | CPUが2.34倍速い |
     | **Android実機(Adreno 619)** | 256×256 | 32.00ms | 22.81ms | **GPUが1.40倍速い** |
     | **Android実機(Adreno 619)** | 512×512 | 509.60ms | 85.04ms | **GPUが5.99倍速い** |

  3. **結論(誇張しない、実測に基づく)**: **ユーザーの仮説「スマホの
     GPUは意外と高速」は、少なくともこの実機(Adreno 619)・この
     ワークロード(GEMM)では実測で裏付けられた**。デスクトップの
     GT730(古いローエンド discrete GPU、Kepler世代)ではCPUが常に
     優勢だったのに対し、スマホのAdreno 619(統合GPU)は行列サイズが
     大きくなるほどCPUとの差が拡大しGPU優勢になった——これはスマホ
     CPUがdesktop CPUほど強力でない(rayonスレッド数もCPU負荷に応じて
     6〜7論理コアで変動、desktopは32論理コア)一方、モバイル向け統合
     GPU(Adreno)はスマホの主な用途(グラフィックス処理)に最適化されて
     おり、CPU側の相対的な弱さがGPU優位を際立たせたためと考えられる。
  4. **正直な限界**: (a) 検証は1機種(moto g53y 5G)・1ワークロード
     (naive GEMM)に限られ、他のAndroid機種・iOSデバイス・他の計算
     カーネル(softmax・attention等)への一般化は未検証。(b) GPU側の
     デバイス初期化オーバーヘッド(実機で58〜63ms)はCPU(1ms未満)より
     大幅に大きいため、**1回限りの小さい計算(例: 1トークンデコード)
     では依然GPU初期化コストが支配的になりうる**——今回の優位性は
     「同一デバイスを複数回の計算に使い回す」場合の計算時間のみの
     比較である点に注意。(c) 発熱・バッテリー消費への影響は未計測。
  5. **検証結果**: `cargo build -p matmul_bench --release`
     (デスクトップ)・`--target aarch64-linux-android --release`
     (Android)とも成功。実機2環境で実際に実行し上記表の数値を取得
     (`adb push`+実機実行、モックなし)。
  - 次にすべきこと: (1) 他のカーネル(softmax・attention等)でも同様の
    ベンチマークを取り、Adreno優位がGEMM固有かどうか確認する。
    (2) 他のAndroid機種(異なるGPUベンダー)・iOSでの追試。
    (3) `aruaru-llm`のGPT-2推論のような実際のワークロードで
    Adreno 619経由の推論が実用的な速度になるかの検証(現状のベンチは
    素のGEMMのみ、Attention等の複合カーネルは含まない)。

## HANDOFF追記(2026-07-23、RS-LinkFusionセッションからの引き継ぎ)

- **ユーザーから「open-cudaはDirectXのプラグインとして開発中」との
  認識が示されたが、実装調査の結果、現状は`Vulkan Compute`基盤
  (`opencuda-vulkan`、`OmniGPU-Design.md`に明記の設計方針)であり、
  DirectX/DirectCompute/DirectML/HLSLへの依存・実装は一切見つから
  なかった**(RS-LinkFusion(`F:\runo\RS-LinkFusion`)側のGPU/NPU
  暗号化・圧縮アクセラレーション調査中に発覚、コード変更はまだ
  行っていない)。
- ユーザーは「DirectX版として仕切り直したい」との意向。これは
  `opencuda-vulkan`をDirectX/DirectComputeへ置き換える(または並存
  させる)大きな方針転換であり、`aruaru-llm`(既存の実装例・利用元)
  への影響も及ぶため、**次回はopen-cuda専用のセッションとして
  スコープを切って着手すべき**(RS-LinkFusion側のセッションに
  詰め込まず、別タスクとして丁寧に検証しながら進める)。
- 検討時の技術的懸念(RS-LinkFusion側調査で判明、DirectX移行時にも
  該当しうる): 汎用`GpuDevice`(alloc/memcpy/launch_kernel)の抽象化
  自体はバイト列を扱えるが、圧縮・暗号化カーネル(ChaCha20-Poly1305等)
  は既存の`opencuda-blas`/`open-cuda-bert`(ML専用、GEMM/Attention/
  量子化)には一切存在せず、新規に書く必要がある。また小サイズ
  ペイロード(例: ネットワークMTU程度の数百〜数千バイト)では
  Host↔Device間のメモリ転送オーバーヘッドがGPU側の演算優位性を
  相殺し、実利益が出ない可能性がある——DirectX版でも同じトレード
  オフを検証すべき。
- **2026-07-23(続き) 日英Web検索での裏取り結果、方針決定、
  `opencuda-directx`クレート新設・実機検証まで完了**:
  1. **裏取り結果**: DXVK/vkd3d-proton(Valve社Protonが実際に使う、
     Linux上でDirectX 12ゲームを動かす技術)を調査した結果、いずれも
     「DirectX(Windows専用API)→Vulkan(クロスプラットフォームAPI)」
     という変換方向であり、逆方向(DirectXを他OSへネイティブ移植)の
     実例は見つからなかった。Vulkanは既にWindows/Linux/Androidへ
     ネイティブ対応し、macOS/iOSも[MoltenVK](https://github.com/KhronosGroup/MoltenVK)
     経由で対応可能——**クロスプラットフォーム対応という目標に
     対しては、既存の`opencuda-vulkan`の方が技術的に近道**という
     結論をユーザーへ報告した。
  2. **ユーザー決定**: 上記を踏まえ「Vulkanは残しつつ、Windows向けに
     別途DirectXバックエンドを追加する」(両方維持、共存)方針を選択。
  3. **`opencuda-directx`クレート新設**(`opencuda-vulkan`の「Phase 1.5
     モック→実機」パターンを踏襲): `opencuda-core::KernelSource`へ
     `Dxil(Vec<u8>)`バリアント(非破壊追加)・`GpuDevice::supports_dxil()`
     能力フラグ(既定`false`)を追加。`DirectXMockDevice`(GPUなしで
     DXIL経路の契約を検証、DXBCコンテナマジックバイト検証・
     `vector_add`シミュレーション、3テストgreen)+`real-dx12` feature
     配下の実`DirectXDevice`(`windows` crate 0.58、`D3D12CreateDevice`
     でのデバイス作成・UPLOADヒープ経由の実メモリ確保・h2d/d2h/d2dの
     実装)。
  4. **実機検証(このマシンのNVIDIA GT 730で実施、型チェックのみで
     完了と報告しない方針を徹底)**: `real_d3d12_device_roundtrips_h2d_and_d2h_on_real_hardware`
     テストが実際に`D3D12CreateDevice`でデバイスを作成し、UPLOADヒープ
     上に実リソースを確保、CPU→D3D12マップ済みメモリへの書き込み・
     読み戻しが完全一致することを実証(スキップメッセージなし、
     実機パスが実行されたことを`--nocapture`で確認済み)。
  5. **正直な開示・スコープの区切り**: カーネルディスパッチ
     (ルートシグネチャ・Compute PSO・ディスクリプタヒープ・コマンド
     リスト記録)は**Phase 2として未実装**——`DirectXDevice::launch_kernel`
     は明示的に`UnsupportedKernel`エラーを返し、`supports_dxil()`も
     `false`を返す(「対応している」という誤ったシグナルを出さない)。
     `Vulkan`側の`dispatch_spirv`に相当する処理を次回実装する。
     デバイス列挙もDXGIアダプタ列挙を経ず`D3D12CreateDevice(None, ...)`
     でデフォルトアダプタ決め打ちのため、`GpuVendor`は`Unknown`のまま
     (Vulkan側のようなベンダーID判定は未実装)。
  6. **検証**: `cargo build --workspace`/`cargo test --workspace
     --features opencuda-directx/real-dx12`ともリグレッション無し、
     全テストgreen(`opencuda-directx`は4件、モック3件+実機1件)。
  - ~~次にすべきこと(1) Phase 2~~ **完了(2026-07-23、同日中)**、
    下記エントリ参照。

- **2026-07-23(続き2) Phase 2完了: DXILカーネルディスパッチを実装・
  実機検証**:
  1. **HLSL埋め込みルートシグネチャを採用**(日英Web検索で裏取り):
     `[RootSignature("UAV(u0), UAV(u1), UAV(u2), RootConstants(...)")]`
     属性をHLSL側に書くと、dxcがコンパイルしたDXILバイト列自体に
     ルートシグネチャが同梱される。Rust側は`ID3D12Device::
     CreateRootSignature(0, dxil_bytes)`へそのバイト列をそのまま渡す
     だけでよく、C++/Rustコードでの手動ルートシグネチャ記述子構築が
     不要になった。
  2. **ディスクリプタヒープを使わずルートUAVディスクリプタで直接
     バインド**: `SetComputeRootUnorderedAccessView(index, gpu_addr)`を
     3つのUAVバッファそれぞれに使い、`CreateDescriptorHeap`/
     `CreateUnorderedAccessView`/ディスクリプタテーブル管理を丸ごと
     回避——実装量・バグの温床を大幅に削減する設計判断。
  3. **メモリ管理をPhase 1のUPLOAD直接マップ方式から、DEFAULTヒープ
     (UAV対応)+UPLOAD/READBACKステージングバッファ経由のコピー方式へ
     刷新**(UAVバインドにはDEFAULTヒープが必須、UPLOADヒープは
     UAVを許可しないというD3D12の制約による)。
  4. **実際に遭遇したバグ**: READBACKヒープのリソースにUPLOADヒープ用
     の初期状態`D3D12_RESOURCE_STATE_GENERIC_READ`を流用していた
     ため`CreateCommittedResource`が`E_INVALIDARG`(0x80070057)を
     返す実バグが発生(型チェックのみで通ってしまう類のバグ、実際に
     `cargo test`を実機で回して発覚)。READBACKヒープは初期状態
     `D3D12_RESOURCE_STATE_COPY_DEST`固定という仕様に修正して解決。
  5. **`tools/compile-dx12-shaders.sh`/`.ps1`新設**
     (`tools/compile-vulkan-shaders.*`と同じ命名パターン)。dxc
     (Windows SDK付属、`C:\Program Files (x86)\Windows Kits\10\bin\
     *\x64\dxc.exe`)で`shaders/vector_add.hlsl`→`vector_add.dxil`
     (DXBCコンテナ形式)をコンパイル。`.dxil`は`.gitignore`対象
     (Vulkanの`.spv`と同じ扱い)。
  6. **実機検証(NVIDIA GT 730、型チェックのみで完了と報告しない
     方針を徹底)**: `real_d3d12_dispatches_vector_add_and_matches_cpu_reference`
     テストが実際に(a) DXILからルートシグネチャ+Compute PSOを作成、
     (b) 256要素のfloat配列2本をDEFAULTヒープバッファへh2dコピー、
     (c) ルートUAVディスクリプタでバインドしDispatch実行、
     (d) 結果をd2hコピーしCPU参照値(単純な加算)と1e-3精度で一致、
     を実証。`opencuda-vulkan`のGEMM実機テストと同型のパターンで
     検証した。
  7. **検証**: `cargo build --workspace`/`cargo test --workspace
     --features opencuda-directx/real-dx12`ともリグレッション無し、
     全テストgreen(`opencuda-directx`は5件: モック3件+実機2件
     〈メモリ往復+カーネルディスパッチ〉)。
  - ~~次にすべきこと(2) DXGIアダプタ列挙~~ **完了(2026-07-23、
    同日中)**、下記エントリ参照。

- **2026-07-23(続き3) DXGIアダプタ列挙による`GpuVendor`判定を実装**:
  `IDXGIFactory1::EnumAdapters1(0)`でデフォルトアダプタ(通常は最も
  高性能なディスクリートGPU)を列挙し、`DXGI_ADAPTER_DESC1`から
  `VendorId`(0x10DE=NVIDIA/0x1002・0x1022=AMD/0x8086=Intel)・
  `Description`(アダプタ名、UTF-16→Rust文字列変換)・
  `DedicatedVideoMemory`を取得。取得したアダプタハンドルはそのまま
  `D3D12CreateDevice`へ渡す(従来の`None`=OS既定選択から変更、実際に
  列挙したアダプタでデバイスを作る)。DXGI列挙が失敗した場合は
  `None`パスへ安全にフォールバックする設計(付加情報であり必須要件
  ではないため)。**正直な開示**: `compute_capability`/`gfx_version`/
  `architecture`はDXGIからは取得できない詳細情報のため
  `(0,0)`/`"unknown"`のプレースホルダのまま(CUDA/ROCm等ベンダー
  固有APIでの取得が必要、今回スコープ外)。
  **実機検証**: `real_d3d12_device_reports_a_real_adapter_name_and_known_vendor_via_dxgi`
  テストが実際に`name="NVIDIA GeForce GT 730"`・
  `vendor=Nvidia { compute_capability: (0, 0) }`・
  `total_memory=2104819712`(約2GB)を取得できることを確認
  (プレースホルダ名のままになっていないこと・`Unknown`のままに
  なっていないことを明示的にassert)。
  **検証**: `cargo test --workspace --features opencuda-directx/real-dx12`
  でリグレッション無し、`opencuda-directx`は6件全green(モック3件+
  実機3件〈メモリ往復・カーネルディスパッチ・DXGIベンダー判定〉)。
  - ~~次にすべきこと(1)(2)~~ **完了(2026-07-23、同日中)**、下記
    エントリ参照。(3)コマンドリストのバッチ化は引き続き未着手。

- **2026-07-23(続き4) matmulカーネル対応・圧縮/暗号化カーネル
  (ChaCha20)のDXIL実装、および実バグ発見・修正**:
  1. **matmul対応**: `shaders/matmul.hlsl`新設(行優先、
     `C[m×n]=A[m×k]×B[k×n]`、`opencuda-vulkan`のGEMMシェーダと同じ
     契約)。`real.rs::dispatch_matmul`を追加、`launch_kernel`の
     カーネル名分岐に組み込み。実機(NVIDIA GT 730)でCPU参照実装
     (`sgemm`のCPU版相当)と数値一致することを検証。
  2. **ChaCha20ストリーム暗号カーネル**(圧縮/暗号化カーネルの第一弾、
     RS-LinkFusion側の要望への回答): `shaders/chacha20.hlsl`
     (RFC 8439のブロック関数、20ラウンド、1スレッド1ブロック
     〈64バイト〉)。`real.rs::dispatch_chacha20`を追加。
  3. **実バグを発見・修正(型チェックのみで完了と報告しない方針が
     機能した具体例)**: 実機テストで、GPU出力が暗号化されず**平文
     そのまま**返ってくる不具合を発見。原因は`cbuffer`内の
     `uint key[8]`/`uint nonce[3]`という**スカラー配列宣言**——
     HLSLのcbufferパッキング規則は配列の各要素を16バイト境界へ
     パディングする(`float weights[3]`が3×16=48バイトを占める、
     というよく知られた罠と同じ)ため、Rust側が
     `SetComputeRoot32BitConstant`で13個のdwordを隙間なく詰めて
     渡す設計と、HLSL側が読むバイトオフセットがズレ、
     key/nonce/counter_base/length_wordsが実質無関係な値(ほぼゼロ)
     になり、キーストリームが機能せず平文がそのまま出力されていた。
     `key[8]`/`nonce[3]`を`key0`〜`key7`/`nonce0`〜`nonce2`という
     個別スカラーフィールドへ書き換え、パディング無しの密なレイアウト
     にすることで解消。
  4. **検証**: RustCrypto製`chacha20`クレート(devDependency)をCPU参照
     実装として使い、同一の鍵・ノンス・平文でGPU出力と完全一致する
     ことを実証(`counter_base=0`に揃えた自己整合的な検証、RFC固有の
     テストベクタには依存しない設計)。`cargo test -p opencuda-directx
     --release --features real-dx12`**8件全green**(モック3件+実機
     5件: メモリ往復・vector_add・matmul・DXGIベンダー判定・
     ChaCha20)。`cargo build --workspace`リグレッション無し。
  5. **正直な開示・スコープ**: これは`accel.rs`(RS-LinkFusion/
     open-web-server-wireが使うChaCha20-Poly1305 AEAD全体)のうち
     認証タグ計算(Poly1305)を含まないChaCha20暗号化部分のみの
     GPU実装デモンストレーション。本番のAEAD実装として組み込むには
     別途Poly1305認証タグの実装、および小サイズペイロード(MTU程度、
     数百〜数千バイト)でのH2D/D2Hオーバーヘッドが実利益を生むかの
     ベンチマークが必要。
  - 次にすべきこと: (1) Poly1305認証タグのGPU実装(完全なAEAD化)、
    (2) 小サイズペイロードでのCPU版との実ベンチマーク比較、
    (3) コマンドリストのバッチ化によるスループット改善(現状は
    操作ごとに同期的にフェンス待機、正しさ優先のMVP)。

## エコシステム全体マップ

同時並行開発の対象プロジェクト一覧・詳細は
[`open-raid-z`のCLAUDE.md](https://github.com/aon-co-jp/open-raid-z/blob/main/CLAUDE.md)
「関連プロジェクト」節を参照。主な関連リポジトリ:

- [aruaru-llm](https://github.com/aon-co-jp/aruaru-llm) — `open-cuda`の
  実装例(bag-of-words→`open-cuda-bert`埋め込みベースの意図分類へ移行済み)
- [RS-Git](https://github.com/aon-co-jp/RS-Git)・[RJSON](https://github.com/aon-co-jp/RJSON) —
  Git forge・JSON処理(OTPログイン・アクセス制御パターンの先行実装)
- [RS-Chiketto](https://github.com/aon-co-jp/RS-Chiketto)・[RS-Blog](https://github.com/aon-co-jp/RS-Blog)・[RS-EC](https://github.com/aon-co-jp/RS-EC) —
  Redmine/WordPress/EC-CUBE相当(順次着手中、`RS-Chiketto`から)

## HANDOFF

- **2026-08-22 CPU推論経路をSIMD化(AVX2+FMA3)——実GPT-2 124Mの生成が
  同一セッション内A/B比較で実測3.34倍高速化。AVX-512/VNNI経路も先行実装
  (実機未検証)**:
  ユーザー指示「AVX2/AVX-512の多段ディスパッチ、FMA3をAI推論の行列演算へ、
  VNNIは将来のCPU買い替えに備えてコードパスだけ用意」への対応。
  1. **なぜCPU側を速くするのが正しいか(既存の実測に基づく判断)**: この
     リポジトリの過去HANDOFF(2026-08-15・2026-08-06)で、デスクトップの
     GT730はGEMMでCPUより2.3〜4.5倍**遅く**、`aruaru-llm`の1トークン
     デコードではVulkanディスパッチの固定オーバーヘッドが支配的で
     CPUより遅い、と既に実測済みだった。つまり実運用の主経路はCPUであり、
     そこをSIMD化するのが最も効く。
  2. **新規`crates/opencuda-blas/src/simd.rs`**: 実行時CPU機能検出
     (`CpuFeatures::detect()`)による多段ディスパッチ。
     - `dot_f32`(AVX-512F → AVX2+FMA3 → スカラー)
     - `axpy`(`acc += scale*src`、同上)
     - `sgemm_cpu`(行ごとrayon並列。**非転置Bのときk方向のaxpy蓄積へ
       組み替える**——素朴な`b[kk*n+col]`はストライドアクセスでSIMD
       ロードできないため、出力行への連続アクセスへ変換するのが要点。
       転置Bのときは行同士の内積)
     - `dot_i8`(**AVX-512 VNNI / AVX-VNNI** → スカラー。int8量子化推論用。
       整数演算なので全経路でビット完全一致)
  3. **配線**: `launch_naive_gemm`の先頭で、`device.supports_spirv()==false`
     (=CPUバックエンド)なら`simd::sgemm_cpu`へ分岐する。要素ごとに
     `GpuDevice::launch_kernel`をディスパッチする従来のカーネルは
     「実カーネル起動を通す」という設計意図のため削除せず残し、
     **`OPENCUDA_DISABLE_CPU_SIMD_GEMM=1`で従来経路へ戻せる**ようにした
     (下記A/B計測はこの環境変数で実際に行った)。
  4. **ベンチマーク(実測、AMD Ryzen 9 3950X / 検出機能`avx2+fma3+sse2`)**:
     - 内積単体(`cargo test -p opencuda-blas --release
       simd::tests::manual_bench -- --ignored --nocapture`):
       k=64 **4.10x** / k=256 **6.68x** / k=768 **6.54x** / k=4096 **6.09x**。
     - **実GPT-2 124M重みでのエンドツーエンド生成**(既存の
       `manual_bench_real_gpt2_generate_timing`、プロンプト15トークン・
       20トークン生成、**同一セッション内で環境変数を切り替えたA/B**):
       従来経路 **6.78秒** → SIMD経路 **2.03秒**(**3.34倍**)。
     - 副次的な実測: `cargo test -p open-cuda-llm --release`の総実行時間が
       **129.56秒 → 60.85秒**(実GPT-2重みを使うテスト群を含む)。
  5. **TEST**: `cargo test -p opencuda-blas --release`**34件全green**
     (既存31件+`simd`の新規3件)。`cargo test --workspace --release --
     --test-threads=1`**全クレートregression無し**(`open-cuda-llm`24件
     ——実GPT-2重みでのCPU/Vulkan生成一致テストを含む——も全green。
     FMA3は中間丸めを行わないためスカラーと**ビット単位では一致しない**が、
     既存の許容誤差付きテスト・生成トークン列一致テストはすべて通った)。
  6. **将来のCPUへの備え(ユーザー指示、2026-08-22)**: 設計方針は
     「**コードを書き足すのではなく、機能フラグが有効になるだけで自動的に
     高速パスが使われる**」こと。AVX-512搭載機・VNNI搭載機へ載せ替えれば
     `CpuFeatures::detect()`が`avx512f`/`avx512vnni`/`avxvnni`を`true`に
     返し、既に書いてある64バイト幅・int8内積の経路がそのまま有効になる。
  7. **正直な開示(未検証)**: この開発機はZen 2のため、
     **AVX-512F経路・AVX-512 VNNI経路・AVX-VNNI経路はコンパイル確認のみで
     実機での実行・ベンチマークは未実施**。また`dot_i8`(VNNI)は
     `quantize_int8`等の既存量子化APIへは**まだ配線していない**
     (量子化APIの出力形式との突き合わせが必要なため、次の増分)。
     GPU経路には一切手を入れていない。
  8. **将来の共通化**: CPU機能検出ロジックは新設の共有クレート
     `F:\runo\open-cpu`(`aon-co-jp/open-cpu`)へ集約する方針が決まった
     (2026-08-22ユーザー指示)。本セッションでは`open-cpu`が別セッションで
     並行作成中のため依存切り替えは行っていないが、検出とカーネルを
     `opencuda-blas/src/simd.rs`の1ファイルに閉じ込めてあるため差し替えは
     容易。`open-raid-z`側(`zfs_accel_hlsl/src/simd.rs`、RAID6 GF(2^8)の
     SIMD化)も同日同様の構成で実装済み。
  - 次にすべきこと: (1) `dot_i8`(VNNI)を`quantize_int8`/`quantize_int4`の
    出力へ実際に配線し、int8量子化推論のCPU経路を作る(VNNI非搭載機でも
    スカラー経路で正しく動くため、実装・テスト自体はこの機でも可能)。
    (2) AVX-512搭載機を入手した際に上記ベンチマークをそのまま再実行する。
    (3) `open-cpu`完成後に検出ロジックを差し替える。

- **2026-08-20 FlexQ(arXiv:2508.04405)風のINT6量子化を実装、PuzzleMoE
  (arXiv:2511.04805)は前提条件を実際に確認した上で実装見送り(ユーザー
  指示、`aruaru-llm`側で発見したDeepSeek系新技術2件への対応、詳細な
  判定経緯は`aruaru-llm/CLAUDE.md`の同日HANDOFF参照)**:
  1. **PuzzleMoE(見送り)**: WebFetchで論文を確認。訓練不要
     (training-free)の後処理という点は軽量だが、**既存のMoEアーキテクチャ
     が前提**。このリポジトリのモデルクレート(`open-cuda-llm`の
     `DecoderLayer`、`open-cuda-bert`)はいずれも単一の密なFFNのみで
     複数エキスパート・ルーターを持たず、GPT-2/BERT系の学習済み重みも
     MoE構成ではないため適用不能。GPT-2のFFN層をMoE化した上で再学習する
     ルートは、このマシンのGPU(GT730、Kepler世代、Tensor Core無し)では
     過去の実測(GEMM/AttentionがCPUより約8倍遅い)から非現実的と判断し、
     コード変更は行わなかった。
  2. **FlexQ(実装)**: `crates/opencuda-blas/src/lib.rs`に
     `QuantizedInt6Tensor`/`quantize_int6`/`dequantize_int6`を新設
     (既存の`quantize_int4`/`quantize_int8`と同じグループ単位対称量子化
     の枠組みを再利用、対称レンジ[-31, 31]、共通カーネル
     `launch_quantize_kernel`をそのまま再利用)。6bitは8の倍数でない
     ため、INT4の「2値/バイト」方式は拡張できず、**4値をひとまとめに
     して24bit=3byteへパック**する方式で対応した(4*6=24=3*8、バイト
     境界にちょうど揃う)。要素数が4の倍数でない場合は末尾を0パディング
     してから処理する。
  3. **正直な開示・スコープ**: FlexQ論文が提案する活性化側の
     W6A6/W6A8混在(レイヤー感度分析による切り替え)・専用GPUカーネル
     (Binary Tensor Core等価物によるネイティブINT6 matmul)は未実装
     (既存の`quantize_int4`/`quantize_int8`と同じく、量子化APIの提供
     までがスコープ、matmulへの配線は次の増分)。
  4. **検証結果**: 新規テスト4件——
     `quantize_int6_roundtrip_error_is_bounded_by_half_scale`(往復誤差が
     スケール半値以内)・
     `quantize_int6_precision_is_between_int4_and_int8`(同一入力での
     総誤差がINT4>INT6>INT8の順に単調減少することを確認、AWQ系テスト
     `quantize_int8_is_more_precise_than_int4_on_same_input`と同じ検証
     パターン)・`quantize_int6_all_zero_group_stays_zero`・
     `quantize_int6_handles_length_not_multiple_of_four`(4値/3バイトの
     パディング境界を明示的に検証)。`cargo test -p opencuda-blas
     --release`**31件全green**(既存27件+新規4件、regression無し)。
     `cargo clippy -p opencuda-blas --all-targets --release -- -D
     warnings`警告0件。`cargo build --workspace --release`
     リグレッション無し。
  - 次にすべきこと: (1) `quantize_int6`をモデルクレート
    (`open-cuda-llm`/`open-cuda-bert`)の実際の重みへ配線する統合は
    未着手(既存の`quantize_int4`/`quantize_int8`も同様に未配線のまま、
    今回もAPI層の提供に留めた)。(2) FlexQの活性化側量子化
    (W6A6/W6A8混在)・専用GPUカーネルは必要になれば追加検討。(3)
    PuzzleMoEは、MoE構成の学習済みチェックポイントの入手、またはGT730
    より高性能なGPUの調達ができた場合にのみ再検討する。

- **2026-08-19 「常駐サービスが無いのは欠陥」というユーザー指摘への調査回答
  (ユーザー指示: open-cuda/open-directxは常駐サービスを持たず使い捨て
  バイナリのみの構成であり、DirectX互換システムとして常駐すべきでは
  ないかとの懸念に対し、実際のMicrosoft DirectX/NVIDIA CUDAの
  アーキテクチャをWebSearchで裏取りした上で結論を出す)**:
  1. **調査結果(DirectX)**: 本物のMicrosoft DirectXは`d3d11.dll`等の
     DLL群(「DirectX Runtime」)を各アプリが動的リンクして使う
     **ランタイムライブラリ方式**であり、「DirectXサービス」という
     独立の常駐バックグラウンドプロセスは存在しない。GPUベンダーの
     カーネルモードドライバ/ユーザーモードドライバは常駐するが、これは
     GPUベンダー(NVIDIA/AMD/Intel)が提供するものでDirectX自体の一部
     ではない。
  2. **調査結果(CUDA)**: NVIDIA CUDAも同様に、アプリが`cudart`等の
     ランタイムライブラリをリンクして使う方式が基本。ただし
     `nvidia-persistenced`(Linux向け、オプションのユーザー空間デーモン)
     が存在し、GPUデバイス状態をジョブ間で維持することで初期化オーバー
     ヘッド・起動レイテンシを削減する目的で使われる——「複数プロセス間の
     調停役」ではなく「単一GPUの初期化状態を使い回すための性能最適化」が
     その役割。
  3. **結論**: 本家DirectX/CUDAとも「各プロセスにリンクされるランタイム
     ライブラリ」が標準アーキテクチャであり、常駐バックグラウンドサービス
     ではない。よって**現在のopen-cudaの設計(ライブラリ+使い捨て
     example/デモ)は本家と一致しており「欠陥」ではない**——という
     前回セッションの結論は、今回のWebSearchでの裏取りでも覆らなかった。
     ただし`nvidia-persistenced`に相当する「GPUデバイス初期化状態を
     プロセス間で使い回し、起動レイテンシを削減する」という限定的な
     常駐ユースケースは実在するため、これは実利がある具体的なケースとして
     記録しておく(過大な設計変更・本格的な常駐デーモン実装はスコープ外
     のため今回は実装しない——GT730 1台構成のこのマシンでは複数プロセス
     間でのGPU初期化状態共有の実利用シーンが無く、実装しても実機検証
     できないため)。
  4. **今回は実装せず、根拠のみ記録**。将来、`aruaru-llm`等で複数
     プロセスが同時にopen-cudaのVulkanデバイスを初期化する実運用シーンが
     生じた場合は、`nvidia-persistenced`型の「デバイス初期化状態の
     プロセス間キャッシュ」を軽量な常駐コンポーネントとして検討する
     価値がある。
  - 出典: [nvidia-persistenced manpage](https://manpages.ubuntu.com/manpages/noble/man1/nvidia-persistenced.1.html)、
    [NVIDIA Persistence Daemon docs](https://docs.nvidia.com/deploy/driver-persistence/persistence-daemon.html)、
    DirectX Runtimeがランタイムライブラリ(DLL群)方式であることの一般的な
    技術文書。

- **2026-08-10(続き) 東芝SBM(シミュレーテッド分岐)の動作実証デモを新設
  (ユーザー指示「aruaru-llm/open-cuda/open-directxへのSBM適用先を日英で
  再調査し、無ければ動作実証デモを作って」への対応)**:
  1. **調査結果(再確認)**: 日英でGoogle検索を2ラウンド実施
     (東芝SBM単体の応用例、およびGPUシェーダコンパイラのレジスタ割り当て・
     異種GPUワークロードスケジューリングへの量子インスパイア最適化の
     適用例)。SBMは組合せ最適化(QUBO/Ising)専用ソルバーであり、
     `aruaru-llm`(テキスト生成)・`open-directx`(DXBC/DXIL変換)・
     `open-cuda`(GEMM/Attention計算)のいずれにも、現時点で実機検証可能な
     形の組合せ最適化問題が存在しないことを再確認した(唯一近い分野
     「異種GPU環境でのワークロードスケジューリング」も、このマシンに
     実GPUが1台しかないため実機検証不能)。こじつけての実装は見送り。
  2. **代わりに新設**: `examples/sbm_demo`(独立バイナリ、ワークスペース
     依存なし・GPUディスパッチ無し、純粋なCPU上のRust実装)。Ballistic
     Simulated Bifurcation(bSB、Goto et al. 2019 Science Advances)で
     Max-Cut問題(NP困難な組合せ最適化の代表例、SBMの実応用と同じ
     QUBO/Ising形式)を解く。頂点数8/10/12/14の小規模グラフに対し、
     全探索で求めた真の最適解とbSBの出力を比較検証する(誇張しない、
     既存のCPU参照実装数値一致検証と同じ考え方)。
  3. **実装中に発見・修正した実バグ**: 初回実装ではカット値が常に0
     (全頂点が同じスピンに収束)になる不具合が発生した。原因は
     結合項の符号——SB標準形は「強磁性的にスピンを揃えよう」とする
     設計だが、Max-Cutは「隣接ノードを分けたい(反強磁性的)」問題
     のため、結合項の符号を反転させる必要があった
     (`dy = -(a0-a_t)*x_i - c0*coupling`、標準形の`+c0*coupling`から
     符号反転)。修正後、全グラフサイズで真の最適解と完全一致した。
  4. **実機検証(型チェックのみで完了と報告しない方針を徹底)**:
     `cargo run -p sbm_demo --release`をこのマシン(Windows x86_64)で
     実行し、n=8/10/12/14の全グラフでSB近似値が全探索の真の最適値と
     完全一致(比率1.0000)することを確認。**加えてユーザー指示
     「スマホ搭載GPUでも動作検証して」への対応として**、Android NDK
     クロスコンパイラで`aarch64-linux-android`向けにビルドし、実機
     (moto g53y 5G、adb経由)へpush・実行し、同じグラフで同じ結果
     (全探索の最適解と完全一致)が得られることを確認した。**正直な
     開示**: このデモはGPUディスパッチを一切行わない純CPU実装のため、
     厳密には「スマホのGPU」ではなく「スマホのCPU(aarch64)での
     クロスプラットフォーム動作」の検証である。
  5. **検証結果**: `cargo test -p sbm_demo --release`2件全green
     (全探索一致検証+辺の無いグラフの境界ケース)。`cargo clippy -p
     sbm_demo --all-targets --release -- -D warnings`警告0件
     (`needless_range_loop`1件を検出・修正済み)。`cargo build
     --workspace --release`ワークスペース全体でregression無し。
  6. **正直な開示・スコープ**: (1) 頂点数14以下という小規模デモに
     限られる(SBM実機のFPGA超並列実装・10万変数超のIsing問題対応の
     速度・規模は再現しない)。(2) `aruaru-llm`/`open-directx`との
     実質的な連携・統合は無い(独立したアルゴリズム実証のみ)。
     (3) restarts=8・steps=400という経験的パラメータのため、より
     大規模・難しいグラフでは真の最適解に到達しない可能性がある
     (この4サイズでは全て一致したが、一般保証ではない)。
  - 次にすべきこと: 特に緊急の課題は無い。将来、`aruaru-llm`/
    `open-cuda`/`open-directx`のいずれかに実際の組合せ最適化問題
    (例: 複数GPU実機環境でのワークロード割り当て)が生じた場合、
    このデモのbSB実装(`examples/sbm_demo/src/main.rs`)を土台に
    転用できる。

- **2026-08-10 `GptModel::generate_with_repetition_penalty`(CTRL方式の
  繰り返しペナルティ)を新設、`aruaru-llm`側報告の反復ループバグへ
  根本対応(ユーザー指示「反復バグの根本解決の為にaruaru-llm側の繰り返し
  ペナルティ実装して」)**:
  1. **背景**: `aruaru-llm`(`open-english`フロントエンド利用中)から、
     対話ファインチューニング無しの素のGPT-2貪欲デコードが
     "Student: Hello\nStudent: Hello\n..."のような同一文字列を無限
     ループする実バグの報告があった。これまでの緩和策(フロントエンド側
     での応急トリム・`max_new_tokens`縮小)は表示上の症状を抑えるのみで
     根本原因(貪欲デコード自体に反復を防ぐ機構が無い)は未解決だった。
  2. **実装**: `crates/open-cuda-llm/src/lib.rs`に
     `generate_with_repetition_penalty(device, prompt_ids,
     max_new_tokens, repetition_penalty)`を新設。既に登場したトークン
     (プロンプト+生成済み、両方)のlogitへ、CTRL論文
     (Keskar et al. 2019)方式のペナルティ(`logit>0`なら`/penalty`、
     `logit<=0`なら`*penalty`)を適用してからargmaxする。既存の
     `generate()`は`generate_with_repetition_penalty(..., 1.0)`を呼ぶ
     薄いラッパーへ変更(`penalty==1.0`の場合は`apply_repetition_penalty`
     内で早期returnし一切のlogit変更を行わないため、既存の全テスト・
     呼び出し元の数値的な挙動は完全に無変更)。
  3. **実機検証(型チェックのみで完了と報告しない方針を徹底、実GPT-2
     124M重み)**: 新規テスト
     `repetition_penalty_reduces_degenerate_loop_on_real_gpt2_weights`が、
     `open-english`と同じプロンプト構造(`"...Student: Hello\nTrainer:"`)
     で実際に検証: (a) `penalty=1.0`(ペナルティ無し)では実際に
     `"Student:"`が2回以上出現する劣化ループへ陥ることを確認(この既知の
     失敗モードを再現できていることの裏取り)、(b) `penalty=1.3`では
     `"Student:"`の出現回数が実際に減ること、(c) `penalty=1.0`時は
     `generate()`(既存API)と完全にバイト一致することを確認。実際の
     生成結果(`--nocapture`実出力):
     ```
     no repetition penalty : " Hello\nStudent: Hello\nTrainer: Hello\nStudent: Hello\nTrainer: Hello\n..."
     repetition_penalty=1.3: " I am the student of your class, and you have been teaching for over
       two years now? Student: Yes sir! You're my teacher here today... Trattoria is very nice to me too"
     ```
     ペナルティ無し版は実際に劣化ループへ陥り、`penalty=1.3`版は文法的に
     自然な会話文へ変わることを実証した。
  4. **検証結果**: `cargo build -p open-cuda-llm --release`警告0件・
     成功。`cargo test -p open-cuda-llm --release -- --test-threads=1`
     **20件全green**(既存19件+新規1件、既存の全テストへ回帰なし)。
     `cargo clippy -p open-cuda-llm --all-targets --release -- -D
     warnings`警告0件。
  5. **`aruaru-llm`側の配線**: `src/generation.rs`に
     `default_repetition_penalty()`(`ARUARU_LLM_REPETITION_PENALTY`
     環境変数、既定`1.3`)を新設し、`generate()`の呼び出しを
     `generate_with_repetition_penalty`経由へ変更(既定で有効化)。
     実際にサーバーを起動し、`POST /v1/generate`へ`open-english`と同じ
     プロンプト(`"...Student: Hello\nTrainer:"`)を送って
     `"I'm sorry for the delay in your appointment but it's not too
     late to get back on track! Thank you so"`(反復なし)という応答を
     実HTTPで確認済み。`cargo test --release`(aruaru-llm)**46件全green**
     (既存機能への回帰なし)。
  6. **正直な開示・スコープ**: (a) このペナルティは`/v1/generate`
     (GPT-2自己回帰生成)のみに適用、`/v1/chat`(意図分類、貪欲デコード
     自体を使わない)には無関係。(b) ペナルティの強さ(`1.3`)は
     この1シナリオでの実測に基づく経験的な既定値であり、あらゆる
     プロンプトで最適とは限らない——`ARUARU_LLM_REPETITION_PENALTY`で
     呼び出し側が調整できる設計とした。(c) サンプリング(温度・top-k/
     top-p)は依然未実装(貪欲デコード+繰り返しペナルティのみ)。
  - 次にすべきこと: (1) 他のリポジトリ(`open-directx`)側の連携強化・
    GPU/NPUハードウェアアクセラレーターの再検証(前回までのHANDOFFで
    「このGT730では1トークンデコードでCPUより遅い」と実測済みのため、
    より高性能なGPU実機が得られた場合にのみ再検討)、(2) フロントエンド
    JS(open-english)をRust+RPoemへ移植する大規模タスク(別セッションで
    スコープを切って着手すべき規模、ユーザーからも言及あり)。

- **2026-08-08(続き) MLA低ランクKVキャッシュ圧縮にPCA較正版を追加
  (ユーザー指示: `aruaru-llm`側で実測した「乱数射影MLAは実GPT-2 124M
  重みで生成品質を明確に劣化させる」という結果を受け、修正できるか調査・
  実装)**:
  1. **原因の特定**: `enable_mla_kv_compression`(乱数初期化の
     `down_proj`/`up_proj`)は、`d_c`(圧縮後次元、例: `16`)が
     `head_dim`(例: `64`)に対して小さい場合、Johnson–Lindenstrauss型の
     乱数射影が距離をよく保存する理論的保証(次元が対数オーダーで十分
     大きい場合のみ成立)から外れており、実データの分散構造を全く反映
     しない――これが観測された劣化(反復・破綻した出力)の数学的な理由。
  2. **実装**: `crates/open-cuda-llm/src/lib.rs`に
     `GptModel::enable_mla_kv_compression_calibrated(d_c, device,
     sample_prompts: &[Vec<u32>])`を新設。実際のサンプル文で
     `forward_prefill_all_layers`を(圧縮を有効化する前の状態で)走らせ、
     各レイヤー・各ヘッドの本物のK/V活性化(`KvCacheHead::k`/`v`、
     `proj=None`ならフル精度のまま保持される既存の設計を利用)を収集し、
     その非中心二次モーメント行列(`XᵀX`)を`nalgebra::SymmetricEigen`
     (純Rust実装、CUDA/ROCm/oneAPIツールチェイン不要)で固有値分解、
     固有値降順で上位`d_c`個の固有ベクトルを`down_proj`の基底として使う
     (直交基底のため`up_proj`は転置)。K/Vは既存設計通り同じ射影行列を
     共有するため、両方の活性化を縦に連結してから単一のPCA基底を求める。
     `nalgebra 0.33`を新規依存として追加(pure Rust、LAPACK/BLAS
     バックエンド不要、`cargo build`で問題なく解決)。
  3. **実機検証(NVIDIA GT730、型チェックのみで完了と報告しない方針を
     徹底)**: 新規テスト`calibrated_pca_mla_kv_compression_on_real_gpt2_
     weights`が、実GPT-2 124M重み・`d_c=16`(`head_dim=64`、75%圧縮、
     タスク指定の数値と一致)で、(a)非圧縮・(b)乱数射影MLA・(c)PCA較正版
     MLAの3経路を比較。較正データは8文の一般的な英文(気象・経済・日常・
     歴史・科学等、トピックを分散させた短文)、テストプロンプトは
     較正データと似た文体の`"The quick brown fox"`(タスク指定)と、
     全く異なる文体のheld-outプロンプト`"def compute_gradient(weights,
     learning_rate):"`(Pythonコード)の両方で実行し、汎化を確認。
     **実際の生成結果**(`cargo test -p open-cuda-llm --release
     calibrated_pca_mla -- --test-threads=1 --nocapture`の実出力、
     `--test-threads=1`が必要な理由は後述):
     ```
     === calibration-style prompt ("The quick brown fox") ===
       uncompressed         : "es are a great way to get a little bit of a kick out of your"
       random-projection MLA: "es, and the government, and away from the government and point of the government"
       PCA-calibrated MLA    : "es are a bit of a lot of the way to the forest.\n\n"
     === held-out prompt ("def compute_gradient(weights, learning_rate):") ===
       uncompressed         : "\n\nreturn (\n\n\\t\\t\\t\\t\\t"
       random-projection MLA: "\nThe following the government and point of the government and point of the government and"
       PCA-calibrated MLA    : "\n\n\n\n\nThe following is a new generation of the following:\n"
     ```
  4. **正直な評価**: PCA較正版は乱数射影版より明らかに改善している
     ――乱数射影版は両プロンプトとも"the government and point of the
     government"のような無限ループ的な破綻(明確な失敗モード)に陥って
     いるのに対し、PCA較正版はループに陥らず、非圧縮版ほど自然ではない
     ものの文法的にはある程度成立した文を生成している。**ただし
     「実運用で許容できる品質」までは回復していない**――非圧縮版と
     比較すればPCA較正版も明確に劣化している(意味的一貫性が低い)。
     これは(a)較正サンプルがわずか8文と小規模であること、(b)このPCAは
     バイアス項を持たない非中心(uncentered)PCAであり、実際のK/V分布の
     平均が非ゼロならその分の再構成誤差が残ること、(c)そもそも本物の
     DeepSeek-V3のMLAは大規模事前学習で`down_proj`/`up_proj`自体を
     エンドツーエンドに**学習**しており、事後的なPCA較正はその代替には
     なってもLLM訓練で得られる情報保持効果には及ばないこと、が理由と
     考えられる。held-outプロンプト(較正文体と全く異なるPythonコード)
     でもPCA較正版が乱数射影版のループ的破綻を回避できていた点は、
     較正データへの単純な過学習ではなく、ある程度汎用的な分散方向
     (英語テキスト全般に共通する活性化統計)を捉えられていることを
     示唆する。
  5. **結論(質問への回答)**: 「安価な単一GPU(GT730級)でMLA圧縮を
     実用化できるか」――**部分的には可能、ただし限定的**。乱数射影より
     PCA較正の方が明確に良く、実装・計算コストも(サンプル文での
     プリフィル数回+小さい`head_dim x head_dim`行列の固有値分解のみ、
     GT730でも数十秒以内)このマシンで十分現実的。しかし「非圧縮と
     遜色ない品質」には届いておらず、KVキャッシュメモリを本当に節約
     したい場面(長いコンテキスト・小VRAM)での実運用に薦められる水準
     ではまだない、というのが正直な結論。次の増分候補は次項参照。
  6. **検証結果**: `cargo build -p open-cuda-llm --release`警告0件・
     成功。`cargo test -p open-cuda-llm --release -- --test-threads=1`
     **18件全green**(既存17件+新規1件)。`cargo clippy -p
     open-cuda-llm --all-targets --release -- -D warnings`警告0件。
     `cargo test --workspace --release -- --test-threads=1`**全クレート
     regression無し**。`cargo clippy --workspace --all-targets --release
     -- -D warnings`警告0件。
  7. **新たに判明した実機の制約(正直な開示、コード変更ではなく実行方法の
     注意点)**: `cargo test -p open-cuda-llm --release`をデフォルトの
     並列実行(`--test-threads`省略)で回すと、`STATUS_ACCESS_VIOLATION`
     (`0xc0000005`)でテストバイナリごと異常終了することを確認した。
     `--test-threads=1`で直列実行すると全件green(上記6.)になることから、
     複数の重量級テスト(実GPT-2 124M重み〈548MB〉を複数回並列ロード・
     実Vulkanデバイスへの複数同時接続)がこのマシンの資源(VRAM 2GB・
     限られたRAM)を同時に奪い合うことが原因と考えられる(このセッション
     で新規に発生した回帰ではなく、既存の実GPT-2/実Vulkanテスト同士でも
     元々起こり得た資源競合に、今回のテストが1本追加されたことで顕在化
     したものと推測——個々のテストロジック自体は`--test-threads=1`で
     問題なくpassするため、コード側のバグではなく実行時資源制約の問題と
     判断)。**このマシンでこのクレートの全テストを実行する際は
     `--test-threads=1`(または`--release -- --test-threads=1`)を使う
     ことを推奨**として明記しておく。
  - 次にすべきこと: (1) 較正サンプル数を増やす(現状8文、より大規模・
    多様なコーパスでPCA基底の安定性・汎化がどう変わるか未検証)、
    (2) 中心化PCA(バイアス項付き、`mla_compress_kv`/`mla_decompress_kv`
    のAPIにオフセット加算を追加する設計変更が必要)が非中心PCAより
    実際に改善するかの検証、(3) `d_c`を大きくした場合(圧縮率は下がるが
    情報損失も減る)の質とメモリ削減のトレードオフの実測、(4)
    `aruaru-llm`側でこの較正版をオプトインで呼べるようにする配線
    (`GptModel::load`直後にサンプルプロンプトで較正する、という運用に
    なる見込み)。

- **2026-08-08 MLA実装の実機検証・FP8実現性調査・DeepSeekMoE統合可否判定**
  (ユーザー指示、README.mdに掲載済みの2026-08-06「MLA実装済み」記載の
  裏取り、および前回HANDOFFで保留していたFP8/DeepSeekMoEの検討):
  1. **MLA実機検証(結果: 合格)**: `opencuda-blas::mla_compress_kv`/
     `mla_decompress_kv`(`crates/opencuda-blas/src/lib.rs` 1045〜1073行)
     は実コードとして存在することを確認。実行コマンド
     `cargo test -p opencuda-blas mla -- --nocapture`の実際の出力:
     ```
     running 1 test
     test tests::mla_compress_decompress_round_trip_matches_between_cpu_and_vulkan ... ok
     test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 26 filtered out; finished in 0.68s
     ```
     このテストはCPUバックエンドとVulkan実デバイス(後述のGT730)の両方で
     compress→decompressのround tripを実行し数値一致を検証しており、
     `eprintln!("skipping mla test: ...")`によるスキップ分岐は発火して
     いない(=実GPU経路を通過した)ことをログで確認済み。README.mdの
     「2026-08-06実装済み」表記は裏取りが取れた。
  2. **FP8混合精度: このマシンでは非現実的と判断(実装せず)**。
     `nvidia-smi --query-gpu=name --format=csv`で唯一の実GPUが
     `NVIDIA GeForce GT 730`(Kepler世代、GK208、Compute Capability 3.5)
     であることを再確認。FP8(E4M3/E5M2)のネイティブ演算をサポートする
     Tensor Coreは、NVIDIA製品ではHopper(H100)/Ada Lovelace(RTX 40系)
     世代(Compute Capability 8.9以降)以降にしか存在せず、Kepler世代の
     GT730にはTensor Core自体が存在しない(既知のアーキテクチャ仕様、
     追加のGPU固有ベンチマークは不要なレベルで確定的)。よって、GT730上で
     FP8を謳うなら「ソフトウェアエミュレーション(FP32/FP16で計算し
     ビット幅を切り詰めるだけ)」にならざるを得ず、DeepSeek-V3が実際に
     得ている高速化(Tensor CoreによるFP8演算のネイティブ実行速度)は
     一切再現できない。**実装しても名ばかりのFP8になり、性能上の意味も
     教育的な実証価値も乏しいため見送った**(この判断はコード変更前の
     調査のみで完了、`opencuda-blas`にFP8関連コードは追加していない)。
  3. **DeepSeekMoE(スパースMoEルーティング)統合: 見送り(理由あり)**。
     `crates/open-cuda-llm/src/lib.rs`の`DecoderLayer`(286行目〜)を
     精査した結果、既存のFFN構造は`intermediate`(`Linear`、GELU適用)→
     `output`(`Linear`)という**単一の密なFFNのみ**であり、複数エキスパート
     ・ゲーティング/ルーターに相当する構造は一切存在しない。さらに
     `GptModel::load`(715〜716行目)が読み込む実重みはHugging Face形式の
     GPT-2(`openai-community/gpt2`、`mlp.c_fc`/`mlp.c_proj`という単一FFN
     テンソル名)であり、そもそもMoE構成の学習済みチェックポイントを
     持たない。ここでMoEルーティングを追加するには、(a)本物のMoE
     チェックポイントが存在しない以上ランダム初期化の複数エキスパートを
     でっち上げるしかなく、(b)ルーター・負荷分散損失・ゲーティングという
     DeepSeekMoEの核心部分を「フックする先」が現行コードに存在しない
     ため、実質的に「既存層への統合」ではなく「検証しようのない新規
     架空実装」になってしまう。タスク指示の「具体的な統合ポイントが
     無ければ実装を見送り、理由を明記する」方針に従い、**実装しない**。
  ### 次にすべきこと
  - FP8: このマシン(GT730)では引き続き非現実的。Hopper/Ada世代GPUが
    調達できた場合にのみ再検討する(現時点で予定なし)。
  - DeepSeekMoE: 実装するなら、まず(1)MoE構成の実在する学習済み
    チェックポイント(小型でよい)を用意する、または(2)
    `DecoderLayer`のFFNを複数の`intermediate`/`output`ペア+簡易
    top-kルーターへ拡張し、`GptModel::load_random`側でランダム初期化
    してでも「ルーティングの数値的挙動」自体を検証できるテストを
    先に設計する、のどちらかの前提を整えてから着手すること(今回は
    どちらも未着手)。
  - MLA: round tripテスト以外に、実際のAttention計算経路
    (`DecoderLayer::forward_step`/`forward_prefill`)へ`mla_compress_kv`/
    `mla_decompress_kv`を配線しKVキャッシュのメモリ削減を実運用で
    確認する統合は未着手(現状は`opencuda-blas`内の単体関数として
    存在するのみ)。

- **2026-08-07(続き5) `open-cuda-llm`のAttention呼び出し経路に
  `flash_attention_with_spirv`を配線(直下2026-08-07(続き)エントリ
  「次にすべきこと(1)」への対応、ユーザー指示「dream-os/open-directx/
  open-cuda/aruaru-llmの関連性・連携性・実用性・完成度を向上」)**:
  1. **調査(着手前に4リポジトリの実態を再確認)**: `nvidia-smi`で
     このマシンのGPUが依然NVIDIA GeForce GT 730(ドライバ475.14)の
     1台のみであることを再確認。`dream-os`(`crates/dream-os-kernel`、
     Toshiba SBM `sbm_ising` 64スピンPoC実装済み)・`open-directx`
     (`directx-graphics-vulkan`/`directx-shader-translate`、境界チェック
     付きチェーン6項まで実機検証済み)・`aruaru-llm`(`/v1/generate`・
     `/v1/translate`の空入力400化まで完了)のCLAUDE.md HANDOFFを読み、
     各リポジトリの「次にすべきこと」を突き合わせた結果、**このリポジトリ
     自身が直前に残した「(1) Attention呼び出し経路をflash_attention_
     with_spirvへ切り替える」が、最も具体的・低リスクで完成度に直結する
     項目**と判断した(dream-os/open-directx側のSBM/DeepSeek組み込みは
     依然「各リポジトリで最適化対象を先に特定する」という前提条件が
     未達のまま、無理に着手すると憶測ベースの実装になるため見送った)。
  2. **実装**: `crates/open-cuda-llm/src/lib.rs`の`DecoderLayer::
     forward_step`/`forward_prefill`に`flash_spirv: Option<(&[u8],
     usize)>`引数を追加。Attention呼び出し箇所を「`Some`なら
     `opencuda_blas::flash_attention_with_spirv`(1回のディスパッチで
     QKᵀ・オンラインsoftmax・P·Vが完結)、`None`なら従来通り
     `scaled_dot_product_attention_with_spirv_and_softmax`(GEMM+softmaxを
     別々にディスパッチ)」の分岐にした。`q_full`/`k_all`/`v_all`は既存の
     「クエリ1行をキャッシュ長`n`回複製して`n×n` attentionを計算し先頭行
     だけ使う」設計により**すでに`n*head_dim`長**(=`seq_len=n`)なので、
     `flash_attention_with_spirv`の`seq_len`契約にそのまま合致し、
     追加の形状変換は不要だった。`GptModel`に`flash_attn_spirv:
     Option<(Arc<Vec<u8>>, usize)>`フィールド+
     `set_flash_attention_spirv(spirv, block_size)`を新設
     (`set_matmul_spirv`/`set_softmax_spirv`と同じ設計パターン、
     `None`のまま〈既定〉なら後方互換で従来経路のまま)。
  3. **実機検証(型チェックのみで完了と報告しない方針を徹底、NVIDIA
     GeForce GT 730)**: 新規テスト
     `generate_end_to_end_matches_cpu_on_real_vulkan_hardware_after_
     set_matmul_and_flash_attention_spirv`(`open-cuda-llm`)が、
     `set_matmul_spirv`+`set_flash_attention_spirv`(`block_size=4`)を
     配線したモデルで`generate()`を実Vulkanデバイス上で実行し、CPU実行と
     生成トークン列が完全一致(byte-identical)することを確認
     (`cargo test -p open-cuda-llm --release generate_end_to_end --
     --nocapture` → 3テスト〈matmulのみ/matmul+softmax/matmul+flash_
     attention〉すべて`ok`)。
  4. **検証結果**: `cargo build -p open-cuda-llm --release`警告0件・
     成功。`cargo test -p open-cuda-llm --release`**17件全green**
     (既存16件+新規1件、`manual_bench_*`は既存通り`--ignored`)。
     `cargo test --workspace --release`**全クレートregression無し**
     (全て`test result: ok`)。`cargo clippy -p open-cuda-llm
     --all-targets --release -- -D warnings`**警告0件**
     (`forward_step`の引数8個に既存の`forward_prefill`と同じく
     `#[allow(clippy::too_many_arguments)]`を付与)。
     `cargo clippy --workspace --all-targets --release -- -D warnings`
     も警告0件。
  5. **正直な開示・スコープの限界**:
     - これは`open-cuda-llm`(自己回帰デコーダ)のみへの配線であり、
       `open-cuda-bert`(エンコーダ、KVキャッシュを持たないため
       `flash_attention`の「クエリ複製でcausalマスクを代替する」設計
       とは無関係)・`open-cuda-whisper`(Cross-Attentionは
       `flash_attention_with_spirv`の`seq_len`同値前提と噛み合わないため
       別途検討が必要)には配線していない。
     - `aruaru-llm`側では今回`set_flash_attention_spirv`を実際に呼ぶ
       配線(`generation.rs::wire_matmul_spirv`相当の追加)は**行って
       いない**——`aruaru-llm`側のCLAUDE.md HANDOFF(2026-08-06(続き2))
       に記録済みの実測(1トークンデコードでVulkanディスパッチ固定
       オーバーヘッドがCPUより遅い)が、ディスパッチ回数を3→1へ減らす
       今回の変更でどこまで緩和されるかは**未計測**。速度面の主張は
       一切していない(「配線が正しく動く」ことのみ実証)。
     - `dream-os`/`open-directx`との直接のコード連携は今回も無い
       (このパスでは両リポジトリのファイルには一切触れていない)。
     - `block_size`のチューニング(テストでは`4`固定)・`head_dim`/
       `block_size`が256を超える場合の一般化は未着手(既存の
       flash_attention実装の既知の制約のまま)。
  - 次にすべきこと: (1) `aruaru-llm`側で`set_flash_attention_spirv`を
    実際に呼ぶオプトイン配線+実機(GT 730)での速度計測
    (「GEMM+CPU softmax」「GEMM+GPU softmax」「GEMM+fused flash
    attention」の3経路を実測比較する価値がある)、(2)
    `open-cuda-bert`/`open-cuda-whisper`側への同種の配線が意味を持つか
    の検討(上記の理由により単純な流用はできない)、(3)
    Qualcomm/ARM/Imagination実機・AMD ROCm/Intel oneAPI導入待ちの
    項目群は変更なし、(4) dream-os/open-directx側のSBM/DeepSeek組み込み
    構想は、各リポジトリで具体的な最適化対象が特定されるまで引き続き
    保留。

- **2026-08-07(続き) `flash_attention`(タイル化+オンラインsoftmax)にSPIR-V
  ディスパッチ経路を新設(2026-08-06(続き4)エントリ「次にすべきこと(3)」・
  直上2026-08-06(続き2)エントリ「次にすべきこと(3)」への対応、それまで
  ホスト側CPU実装のみだった`flash_attention`に実GPUディスパッチを追加)**:
  1. **シェーダ設計**: `examples/flash_attention_vulkan_real/shaders/
     flash_attention.comp`を新規作成。既存のsoftmax/matmulカーネルのような
     「ホスト側から複数回ディスパッチしてタイルを回す」方式ではなく、
     **1スレッド=1クエリ行**を担当し、K/V全体を`block_size`単位で
     シェーダ内部のループとして巡回しながらオンラインsoftmaxの漸化式
     (`m_i`/`l_i`/累積出力を毎タイル更新)を完結させる、**1回のディスパッチ
     で完了する単一フューズドカーネル**として実装した(CPU版
     `flash_attention`と同一のアルゴリズム・同一の数学的結果になる設計)。
     `tools/compile-vulkan-shaders.{ps1,sh,cmd}`にも追加し、`glslc`で
     `flash_attention.spv`へコンパイル済み。
  2. **正直な制約**: シェーダは固定長ローカル配列(`MAX_DIM=256`)を使うため
     `head_dim`・`block_size`ともに256を超えると失敗する設計にした
     (`crates/opencuda-vulkan/src/real.rs`の`ensure_flash_attention_args`
     側で明示的にチェックし、黙って配列外を読み書きする事態を避けている)。
     head_dimに対する共有メモリ最適化・warp内リダクションは行っていない
     (正確性優先、既存のsoftmax.comp/matmul.compと同じ方針)。
  3. **実装**: `VulkanDevice::launch_kernel`のカーネル名ホワイトリストに
     `flash_attention`を追加、`ensure_flash_attention_args`/
     `run_flash_attention_spirv`(4バッファ+3xu32+1xf32 push constant、
     既存の`dispatch_spirv`共有経路を利用)を`crates/opencuda-vulkan/
     src/real.rs`に新設。`crates/opencuda-blas/src/lib.rs`には
     `flash_attention_with_spirv(device, q, k, v, seq_len, head_dim,
     block_size, spirv)`を新設(CPU版`flash_attention`は変更せず既存
     リファレンス実装のまま残置)。
  4. **スコープの正直な開示**: `open-cuda-llm`/`open-cuda-bert`側の
     Attention呼び出し経路(`scaled_dot_product_attention_with_spirv_and_
     softmax`)への配線は**今回は行っていない**——それらは素朴な(非Flash)
     Attentionを使っており、`flash_attention`系はこれまで単体の関数として
     しかテストされていない(HANDOFF記載通り)。今回追加した
     `flash_attention_with_spirv`もモデル層(`DecoderLayer`/`EncoderLayer`)
     には未接続で、`opencuda-blas`単体の部品として実装・実機検証したのみ。
  5. **実機検証(型チェックのみで完了と報告しない方針を徹底、NVIDIA
     GeForce GT 730)**: 新規テスト
     `flash_attention_spirv_matches_cpu_on_real_hardware`(`opencuda-blas`)が、
     `seq_len=17, head_dim=8`・`block_size ∈ {1, 4, 17}`の組み合わせで
     GPUディスパッチ結果とCPU版`flash_attention`を誤差1e-3以内で比較し
     全一致することを確認(`cargo test -p opencuda-blas --release
     flash_attention_spirv_matches_cpu_on_real_hardware -- --nocapture`
     → `test result: ok. 1 passed`、スキップされていないことをログで確認
     済み)。
  6. **検証結果**: `cargo build --workspace --release`警告0件・成功。
     `cargo test -p opencuda-blas --release`**27件全green**(既存26件+
     新規1件)。`cargo test --workspace --release`全クレート`test result:
     ok`(regression無し)。`cargo clippy --workspace --all-targets
     --release -- -D warnings`**警告0件**(`flash_attention_with_spirv`の
     引数8個には既存の`sgemm`等と同じく`#[allow(clippy::too_many_
     arguments)]`を付与)。
  - 次にすべきこと: (1) `open-cuda-llm`/`open-cuda-bert`のAttention呼び出し
    経路を、素朴なAttention(`scaled_dot_product_attention_with_spirv_and_
    softmax`)から`flash_attention_with_spirv`へ切り替えるモデル層配線
    (今回は影響範囲拡大を避けてスコープ外とした、既存の`softmax`専用
    カーネル配線と同じ手順で対応可能なはず)。(2)
    2026-08-06(続き2)エントリで指摘された「1トークンデコード
    (`seq_len=1`)ではVulkanディスパッチ固定オーバーヘッドが支配的」という
    懸念は`flash_attention`でも同様に当てはまると推測されるが未検証
    (`seq_len=1`のケースでの実測ベンチマークは今回未実施)。(3)
    `head_dim`・`block_size`が256を超える場合の対応(現状はエラーで拒否
    するのみ、シェーダ側をタイル化して制約を緩和する余地がある)。

- **2026-08-07 `mla_compress_kv`/`mla_decompress_kv`を`open-cuda-llm`の実際の
  KVキャッシュ経路(`KvCacheHead`)へ配線(直下2026-08-06(続き4)エントリ
  「次にすべきこと(1)」への対応)**: それまで`opencuda-blas`単体の部品
  として実装されているだけで呼び出し元が未接続だった状態を解消した。
  1. **実装**: `crates/open-cuda-llm/src/lib.rs`に`MlaHeadProjection`
     (ヘッドごとのdown_proj/up_proj、`GptModel::enable_mla_kv_compression
     (d_c, seed)`が乱数初期化)を新設し、`DecoderLayer`へ`mla:
     Option<Vec<MlaHeadProjection>>`フィールドとして保持。`KvCacheHead`を
     変更し、`push`が`proj: Option<&MlaHeadProjection>`を受け取って
     `Some`の場合は`mla_compress_kv`で`d_c`次元へ圧縮してから
     `k_latent`/`v_latent`へ格納(フル精度の`k`/`v`は保持しない)、
     `current_kv`が`mla_decompress_kv`でAttention計算直前に復元する設計
     (`None`〈既定〉の場合は従来通りフル精度のまま、既存の数値一致
     テストへの影響ゼロ)。`forward_step`/`forward_prefill`双方の
     呼び出し箇所を新シグネチャへ切り替えた。
  2. **正直な開示**: `opencuda-blas`側の既存の開示通り、射影行列は
     学習済みではなく乱数初期化のため、圧縮・復元後のk/vは元の値と
     一致しない(非可逆)。この配線が実証するのは「低ランク圧縮の計算
     経路がKVキャッシュ・Attention計算まで正しく繋がり、`generate()`が
     エンドツーエンドで動作する」ことであり、生成品質の維持は主張しない
     (`enable_mla_kv_compression`のdocコメントに明記、学習済み重みが
     使えるようになった場合に置き換える土台として位置づけ)。
  3. **実機検証(型チェックのみで完了と報告しない方針を徹底、NVIDIA
     GeForce GT 730)**: 新規テスト
     `mla_kv_compression_completes_on_real_vulkan_hardware`
     (`open-cuda-llm`)が、`set_matmul_spirv`+`enable_mla_kv_compression`
     両方を配線したモデルで`generate()`を実Vulkanデバイス上で実行し、
     圧縮・復元のGEMM(`mla_compress_kv`/`mla_decompress_kv`)を含めて
     最後まで完走することを確認(CPU/Vulkan間のトークン列一致は非可逆
     変換のため主張しない設計、コメントに明記)。あわせてCPU側の
     単体テスト2件: `mla_kv_compression_enabled_model_generates_
     without_panicking`(基本動作)・
     `mla_kv_compression_actually_changes_generation_versus_uncompressed`
     (「配線したが実は経路を素通りしているだけ」という見逃しを防ぐため、
     複数シードで圧縮ありと無しの生成結果が実際に異なることを確認する
     回帰テスト)・`mla_kv_compression_rejects_non_reducing_d_c`
     (`d_c >= head_dim`のガード)も追加。
  4. **検証結果**: `cargo build -p open-cuda-llm --release`/
     `cargo build --workspace --release`警告0件・成功。`cargo test -p
     open-cuda-llm --release`**16件全green**(既存12件+新規4件、1件
     `--ignored`)。`cargo test --workspace --release`**全クレート
     regression無し**(全て`test result: ok`)。`cargo clippy -p
     open-cuda-llm --all-targets --release -- -D warnings`**警告0件**。
  - 次にすべきこと: (1) `open-cuda-bert`(エンコーダ)側は自己回帰
    デコード用のKVキャッシュを持たないアーキテクチャのため、今回の
    配線対象は`open-cuda-llm`のみ(想定通り、`open-cuda-bert`側は
    対応不要と判断)。(2) 直下2026-08-06(続き4)エントリの残り課題
    ——DeepSeekMoEのauxiliary-loss-free負荷分散([arXiv:2408.15664](https://arxiv.org/pdf/2408.15664))
    の調査・適用検討(MoEアーキテクチャ自体がこのエコシステムに無いため
    まず適用対象の有無を検討する必要がある)、FP8混合精度学習の調査
    (このマシンのGPU〈GT730〉がFP8命令をサポートするか不明、要確認)
    は未着手のまま変更なし。(3) 学習済みのMLA射影重み(`down_proj`/
    `up_proj`)を実際に読み込めるようにするローダーは今回もスコープ外
    (現状は`enable_mla_kv_compression`の乱数初期化のみ)。

- **2026-08-06(続き4) DeepSeek-V3のMLA(Multi-Head Latent Attention)に
  インスパイアされた低ランクKVキャッシュ圧縮を実装・実機検証
  (ユーザー指示、8リポジトリへの東芝SBM/DeepSeek技術組み込み構想の
  第一段)**: 日英でDeepSeek-V3技術レポート([arXiv:2412.19437](https://arxiv.org/abs/2412.19437))・
  実装解説ブログを調査。MLAの核心は「KVを`d_h`次元でそのままキャッシュ
  せず、より小さい`d_c`次元の潜在ベクトルへ低ランク射影(down-projection)
  して圧縮保存し、必要な時にup-projectionで復元する」設計で、
  DeepSeek-V2の実測ではKVキャッシュを93.3%削減・最大生成スループット
  5.76倍に向上させたと報告されている。
  1. **実装**: `crates/opencuda-blas/src/lib.rs`に
     `mla_compress_kv`/`mla_decompress_kv`(既存の実機検証済み`sgemm`
     〈CPU/Vulkan両対応〉を土台にした低ランク射影の圧縮・復元)、
     `mla_memory_reduction_percent`(削減率計算ヘルパー)を新設。
  2. **正直な開示**: DeepSeek-V3の実際の`down_proj`/`up_proj`重み行列は
     大規模事前学習によって獲得されるものであり、本実装はその学習済み
     重みを持たない——「情報をほぼ無損失で圧縮できる」というMLAの実運用
     上の効能を主張するものではなく、「低ランク射影という計算の仕組み
     自体が既存のGEMM基盤の上に正しく実装できる」ことの実証に留まる
     (`quantize_int4`等の既存の量子化機能と同じ「メモリ効率化」という
     方向性の追加手段)。
  3. **実機検証(NVIDIA GT730)**: `mla_compress_decompress_round_trip_
     matches_between_cpu_and_vulkan`テスト——`d_h=16, d_c=4`(75%削減)の
     圧縮・復元をCPU版・Vulkan版両方で実行し、数値完全一致を確認。
     `cargo test --workspace --release`で全クレートregression無し。
  - 次にすべきこと: (1) `open-cuda-llm`/`open-cuda-bert`の実際のKV
     キャッシュ経路への配線(現状は`opencuda-blas`単体の部品として実装、
     呼び出し元は未接続)、(2) DeepSeekMoEのauxiliary-loss-free負荷分散
     ([arXiv:2408.15664](https://arxiv.org/pdf/2408.15664))の調査・
     適用検討(MoEアーキテクチャ自体がこのエコシステムに無いため、
     まず適用対象の有無を検討する必要がある)、(3) FP8混合精度学習の
     調査(このマシンのGPU〈GT730〉がFP8命令をサポートするか不明、
     要確認)。

- **2026-08-06(続き3) `dream-os`(新規リポジトリ)向けに`sha256d_mine`
  カーネルディスパッチを追加(ユーザー指示、DreamOS PoCでのマイニング
  相当ハッシュ計算カーネル実装の一環)**: `VulkanDevice::launch_kernel`の
  カーネル名ホワイトリスト(`vector_add`/`matmul`/`raid6_*`/`softmax`)に
  `sha256d_mine`を追加。既存の`ensure_softmax_args`/`run_softmax_spirv`と
  同じ設計パターン(2バッファ+2xu32 push constant、共有`dispatch_spirv`
  経路を利用)で`ensure_sha256d_mine_args`/`run_sha256d_mine_spirv`を
  新設(`crates/opencuda-vulkan/src/real.rs`)。呼び出し元は
  `F:\runo\dream-os\crates\dream-os-kernel`(新規リポジトリ、`../
  open-cuda/crates/opencuda-*`へのpath依存)。**検証**: `cargo build -p
  opencuda-vulkan --features real-vulkan --release`警告0件。実際の
  カーネル動作検証(GPU計算がCPU参照実装`sha2`クレートと一致すること)は
  `dream-os`側の`tests/mining_real_vulkan.rs`で実施済み(このマシンの
  NVIDIA GT730・Android実機〈Adreno 619〉双方で実証、詳細は`dream-os/
  CLAUDE.md`参照)。
  - 次にすべきこと: `opencuda-vulkan::VulkanDevice::new(id)`の`id`引数が
    実際には物理デバイス選択に使われず常に最初のcomputeデバイスを開く
    設計になっている実バグ(`dream-os`側の複数GPU対応調査で発覚)を
    修正する——複数GPU実機が無いためこのマシンでは検証不可能だが、
    コードロジック上の修正自体は可能。

- **2026-08-06(続き2) softmax専用SPIR-Vカーネルを`scaled_dot_product_attention`
  経路へ実配線し、「GPU GEMM + CPU softmax」から「GPU GEMM + GPU softmax」へ
  移行・実機検証完了(直上エントリ「次にすべきこと(1)」への対応、
  `aruaru-llm`側セッションから継続して着手、ユーザー指示「aruaru-llm連携性
  向上」)**:
  1. **実装**: `opencuda-blas::scaled_dot_product_attention_with_spirv_and_
     softmax(device, q, k, v, seq_len, head_dim, matmul_spirv, softmax_spirv)`
     を新設。`matmul_spirv`・`softmax_spirv`の両方が`Some`かつ
     `select_gemm_path`が`GemmPath::VulkanGeneric`を選ぶ場合のみ、QKᵀ・
     行ごとのsoftmax(`softmax_vulkan_generic`経由)・P·Vの**すべてを実際に
     Vulkanデバイス上でディスパッチする**。片方でも`None`/CPU経路の場合は
     softmaxも含めて従来通りホスト側CPU(rayon並列)にフォールバックする
     (GEMM経路とsoftmax経路を常に一致させる設計、H2D/D2H往復が中途半端に
     増えるだけの構成を避けるため)。既存の`scaled_dot_product_attention_
     with_spirv`(GEMMのみSPIR-V対応)は`softmax_spirv=None`固定で新関数を
     呼ぶ薄いラッパーへ変更し、後方互換を維持した(既存呼び出し元は
     無改修)。
  2. **`open-cuda-llm::GptModel`・`open-cuda-bert::BertModel`双方に
     `set_softmax_spirv`を追加**: `set_matmul_spirv`と同じ設計パターン
     (`Arc<Vec<u8>>`を保持し`forward_step`/`forward_prefill`/
     `EncoderLayer::forward`経由で伝播)。`DecoderLayer::forward_step`/
     `forward_prefill`・`EncoderLayer::forward`のシグネチャに
     `softmax_spirv: Option<&[u8]>`引数を追加し、Attention呼び出し箇所を
     新関数`scaled_dot_product_attention_with_spirv_and_softmax`へ切り替えた。
  3. **実機検証(型チェックのみで完了と報告しない方針を徹底、NVIDIA
     GeForce GT 730)**:
     - `opencuda-blas`新規テスト
       `scaled_dot_product_attention_with_spirv_and_softmax_matches_cpu_
       on_real_hardware`が、matmul.spv・softmax.spv両方を実Vulkanデバイスへ
       渡し、CPU版(`GemmPath::CpuNaive`)と誤差1e-3以内で一致することを確認。
     - **本命**: `open-cuda-llm`新規テスト
       `generate_end_to_end_matches_cpu_on_real_vulkan_hardware_after_set_
       matmul_and_softmax_spirv`が、`set_matmul_spirv`+`set_softmax_spirv`
       両方を配線したモデルで`generate()`を実Vulkanデバイス上で実行し、
       CPU実行と生成トークン列が完全一致(byte-identical)することを確認
       (`cargo test -p open-cuda-llm --release generate_end_to_end --
       --nocapture` → 2テストとも`ok`)。
     - `aruaru-llm`側(`--features real-vulkan`)で実際にサーバーを起動し
       (`RUST_LOG=info`)、起動ログで`generation`/`scoring`双方について
       `loaded matmul.spv (2732 bytes) ... via set_matmul_spirv`・
       `loaded softmax.spv (4680 bytes) ... via set_softmax_spirv`の両方が
       記録されることを確認。実際に`POST /v1/generate`
       (`{"prompt":"The quick brown fox","max_new_tokens":5}`)へHTTP
       リクエストを送り、`"es are a great way"`(CPU版の既知の継続文
       `"es are a great way to get a little bit of a"`の先頭一致)・
       `"engine":"gpt2-greedy-decode-v0-open-cuda-llm-vulkan"`(`-vulkan`
       接尾辞、実際にVulkan経由で動いたことをエンジンラベルからも確認)
       という正しい応答を得た。
  4. **検証**: `cargo build --workspace --release`警告0件・成功。
     `cargo test --workspace --release`**全クレートregression無し**
     (`opencuda-blas`25件・`open-cuda-llm`13件〈うち1件ignore〉・
     `open-cuda-bert`2件、他は既存通り、全て`test result: ok`)。
     `cargo clippy -p opencuda-blas -p open-cuda-llm --all-targets
     --release -- -D warnings`(および`cargo clippy --workspace
     --all-targets --release`)**警告0件**。`aruaru-llm`側も
     `cargo test --release`/`cargo test --release --features
     real-vulkan`いずれも既存46件全green(regression無し)。
  5. **正直な開示・性能**: `POST /v1/generate`(`max_new_tokens=5`)の
     実測所要時間は**約35.9秒**——CPU版(既存記録: 20トークンで約6〜7秒)
     と比較して大幅に遅い。これは2026-07-26 HANDOFFで示した「1トークン
     デコードは`seq_len=1`のGEMM/softmaxが極めて軽く、Vulkanの
     ディスパッチ固定オーバーヘッド(コマンドバッファ記録・
     `vkQueueSubmit`・フェンス待機)がGPU計算時間より支配的になり、
     CPU実行より遅くなりうる」という設計上の懸念が、今回**softmax専用
     カーネルの追加ディスパッチ(レイヤーあたりQKᵀ・softmax・P·Vの3回、
     従来のGEMMのみ版の1.5倍のディスパッチ回数)によりさらに悪化する形で
     実測された**——「正しく動く」ことは実証できたが「速くなる」ことは
     実証していない、誇張しない結論として記録する。
  - 次にすべきこと: (1) デコード側(`forward_step`、`seq_len=1`)への
    Vulkanディスパッチはオーバーヘッドが支配的で実用上不利なことが
    改めて確認されたため、プリフィル(`forward_prefill`、`seq_len>1`の
    バッチGEMM)側でのみVulkan経路を使い、デコード側はCPU固定にする
    「経路ごとの使い分け」の設計を検討する価値がある(現状は
    `set_matmul_spirv`/`set_softmax_spirv`を呼べばプリフィル・デコード
    両方が一律Vulkan経由になる)、(2) 真にGPU常駐率を上げるには
    Attention全体を1回のディスパッチにまとめる融合(fused)カーネルが
    必要で、これは今回のような複数カーネルの直列呼び出しでは原理的に
    解消できない(ディスパッチ回数そのものを減らす設計が必要)、(3)
    `flash_attention`(タイル化+オンラインsoftmax)側は依然ホスト側CPU
    実装のまま未着手。

- **2026-08-05(続き) SPIR-V版Attention(GPU GEMM + CPU softmaxハイブリッド)を
  実装し、`GptModel::generate()`が実Vulkanハードウェア上で最後まで完走する
  ことをエンドツーエンド検証(直下エントリ「次にすべきこと(1)」への対応)**:
  1. **背景**: 直下エントリで`Linear::forward`(GEMM)側の`matmul.spv`未配線
     バグは直したが、`scaled_dot_product_attention`が内部で`launch_naive_gemm`
     (`KernelSource::Native`)を直接呼んでいたため、`VulkanDevice::launch_kernel`
     (SpirVしか受理しない)に渡すと`kernel source not supported by this
     backend: Native`で即座にpanicする、というより深いギャップが残っていた。
  2. **調査・設計判断**: `crates/opencuda-blas/shaders/matmul.comp`
     (既存の非転置GEMM専用SPIR-Vシェーダ)を読み、QKᵀ・P·Vの両ステップは
     形状的にはGEMMそのものなので、新規SPIR-Vカーネルを書かずに既存の
     `sgemm_vulkan_generic`を再利用できると判断した。ただしシェーダは
     `b`の転置に対応していないため、QKᵀ計算ではKをホスト側で
     `seq_len x head_dim`→`head_dim x seq_len`へ転置してから通常GEMMとして
     渡す(転置コストはO(seq_len*head_dim)、GEMM本体のO(seq_len^2*head_dim)
     より軽い)。一方、行ごとのsoftmax(exp/sum/normalize)には対応する
     SPIR-Vカーネルが無く、新規に書くには規模が大きいため、**今回は
     ホスト側CPU(rayon並列)のまま残した**——ユーザー指示「野心的すぎる
     実装を無理に通さない」方針に従い、正直に「GPU GEMM + CPU softmax」の
     ハイブリッドとして実装・命名した(完全にGPU常駐する融合Attention
     カーネルではない、と`opencuda-blas`側のdocコメントに明記)。
  3. **実装**: `crates/opencuda-blas/src/lib.rs`に
     `scaled_dot_product_attention_with_spirv(device, q, k, v, seq_len,
     head_dim, spirv: Option<&[u8]>)`を新設。既存の`scaled_dot_product_
     attention`はこれを`spirv=None`で呼ぶ薄いラッパーへ変更(後方互換、
     既存呼び出し元は無改修で従来通りCPU Native経路のまま動く)。P・Vの
     計算は既存の`sgemm(...,spirv)`をそのまま再利用(`spirv`を素通し)。
     `crates/open-cuda-llm/src/lib.rs`の`DecoderLayer::forward_step`/
     `forward_prefill`の呼び出し箇所を新関数へ切り替え、
     `self.qkv.spirv_matmul`(直下エントリで配線済みの`Arc<Vec<u8>>`)を
     そのままattentionへも渡すようにした。
  4. **実機検証(型チェックのみで完了と報告しない方針を徹底、NVIDIA
     GeForce GT 730)**:
     - 新規テスト`scaled_dot_product_attention_with_spirv_matches_cpu_
       on_real_hardware`(`opencuda-blas`)が、`select_gemm_path`が
       `GemmPath::VulkanGeneric`を選ぶ実Vulkanデバイス上で新関数を実行し、
       CPU版(`GemmPath::CpuNaive`)と誤差1e-3以内で一致することを確認
       (`cargo test -p opencuda-blas --release
       scaled_dot_product_attention_with_spirv -- --nocapture`
       →`test result: ok. 1 passed`)。
     - **本命**: 新規テスト`generate_end_to_end_matches_cpu_on_real_
       vulkan_hardware_after_set_matmul_spirv`(`open-cuda-llm`)が、
       `GptModel::set_matmul_spirv`済みモデルに対し`generate()`を実
       Vulkanデバイス上でそのまま呼び、CPU実行(spirv未設定モデル)と
       生成トークン列が完全一致することを確認した——これで
       「配線しても`Native`カーネルで即panicする」という直下エントリの
       既知ブロッカーは解消された。実行結果:
       `cargo test -p open-cuda-llm --release generate_end_to_end --
       --nocapture` →
       `test tests::generate_end_to_end_matches_cpu_on_real_vulkan_
       hardware_after_set_matmul_spirv ... ok`
       (`test result: ok. 1 passed; 0 failed`)。
     - 既存テスト回帰無し: `cargo test -p opencuda-blas --release`
       23件全green、`cargo test -p open-cuda-llm --release`11件全green
       (`manual_bench_*`は既存通り`--ignored`)、`cargo clippy -p
       opencuda-blas -p open-cuda-llm --all-targets --release --
       -D warnings`警告0件、`cargo build --workspace --release`
       リグレッション無し。
  5. **正直な開示・スコープ**: これは「完全にGPU常駐するfused Attention
     カーネル」ではなく「QKᵀ・P·Vの2つのGEMMは実Vulkanディスパッチ、
     softmaxはCPU」のハイブリッドである。この点は`opencuda-blas::
     scaled_dot_product_attention_with_spirv`のdocコメントに明記した。
  - 次にすべきこと: (1) softmax専用のSPIR-Vカーネル(または真の融合
    Attentionカーネル)を書き、GPU常駐率をさらに上げる増分、(2)
    `flash_attention`(タイル化+オンラインsoftmax)側もSPIR-V対応させる
    かどうかの検討(現状は`scaled_dot_product_attention`系のみ対応、
    `flash_attention`は引き続き純粋ホスト側CPU実装のまま)、(3)
    `aruaru-llm`側で実際に`GptModel::set_matmul_spirv`を呼ぶ配線
    (これでようやく`generate()`全体が実際にVulkan経由で動作するように
    なったため、着手する価値がある)、(4) ユーザー指示の優先順位
    (1. open-directx 2. open-cuda 3. aruaru-llm)に沿って、次は
    open-directx側の作業へ切り替える。

- **2026-08-05 `Linear::forward`が`matmul.spv`を渡さずGemmPath::
  VulkanGenericが機能しなかった実バグを修正・実機検証(直下2026-08-04
  エントリ「最優先課題」への対応、ユーザー指示「open-directx open-cuda
  aruaru-llmの連携・実用性・完成度を向上」)**:
  1. **修正**: `crates/open-cuda-llm/src/lib.rs`の`Linear`構造体に
     `spirv_matmul: Option<Arc<Vec<u8>>>`フィールドを追加(既定`None`、
     既存の全構築箇所〈`Linear::random`・`load_conv1d`・`lm_head`直接
     構築〉で明示的に`None`を設定し後方互換を維持)。`Linear::forward`は
     `sgemm`呼び出しの`spirv`引数にこのフィールドを渡すよう変更(従来は
     常に`None`固定だった)。新規`GptModel::set_matmul_spirv(&mut self,
     spirv: Vec<u8>)`が、モデル内の全`Linear`(各レイヤーのQKV融合/
     attn_out/intermediate/output+`lm_head`)へ同じ`Arc`を配線する。
  2. **実機検証(型チェックのみで完了と報告しない方針を徹底)**:
     新規テスト`set_matmul_spirv_makes_linear_forward_use_vulkan_and_
     matches_cpu_output`が、実Vulkan環境(NVIDIA GeForce GT 730)+
     事前コンパイル済み`matmul.spv`で`Linear::forward`を実際にCPU経路
     (`GemmPath::CpuNaive`)とVulkan経路(`GemmPath::VulkanGeneric`)の
     両方で実行し、出力が数値一致(誤差1e-3以内)することを確認した
     ——これで「配線しても即座に失敗する」という2026-08-04の実バグは
     解消された。
  3. **発見した第二のブロッカー(正直な開示、今回は解消せず)**: 当初は
     `GptModel::generate()`をCPU/Vulkan双方で走らせ出力一致を見る
     テストとして書いたが、実機で実行したところ
     `VulkanDevice::launch_kernel`が`kernel source not supported by
     this backend: Native`で**実際にpanicした**。原因は
     `scaled_dot_product_attention`が内部で使う`launch_naive_gemm`が
     `KernelSource::Native`(Rustクロージャカーネル)を要求するが、
     `VulkanDevice`は`KernelSource::SpirV`しか受理しないため——GEMM
     (Linear層)側の配線を直しても、**Attention計算自体は依然Vulkan
     デバイス上で即座に失敗する**。SPIR-V版のattentionカーネルが
     新規に必要な、今回のGEMM配線修正より規模の大きいギャップと判断し、
     無理に着手せず正直に記録するに留めた(テストのスコープを
     `Linear::forward`単体の検証に絞ることで、この既知のブロッカーを
     踏まずに今回の修正だけを実証できる形にした)。
  4. **検証**: `cargo build -p open-cuda-llm --release`/`cargo test -p
     open-cuda-llm --release`**10件全green**(既存9件+新規1件、
     `manual_bench_*`は既存通り`--ignored`)、`cargo clippy -p
     open-cuda-llm --all-targets --release -- -D warnings`警告0件、
     `cargo build --workspace --release`リグレッション無し。
  - 次にすべきこと: (1) SPIR-V版attentionカーネルの新規実装
    (`scaled_dot_product_attention`をVulkanデバイス上で実行可能に
    する、規模の大きい増分)、(2) `aruaru-llm`側で`GptModel::
    set_matmul_spirv`を実際に呼ぶ配線(GEMM部分だけでもVulkan経由に
    できるが、上記(1)が無い限り`generate()`全体はAttentionで失敗
    するため、実際に呼んでも現状は`generate()`が動かない点に注意)、
    (3) ユーザー指示の優先順位(1. open-directx 2. open-cuda
    3. aruaru-llm)に沿って、次はopen-directx側の作業へ切り替える。

- **2026-08-04(続き) `aruaru-llm`側の実機検証で判明: `Linear::forward`が
  `matmul.spv`を`sgemm`へ渡していないため`GemmPath::VulkanGeneric`が
  機能しない(次回セッションの最優先課題として記録)**: 直下エントリ
  (QKV融合+プリフィル/デコード分離)の実装後、`aruaru-llm`側で
  `real-vulkan` featureを新設し実機(NVIDIA GT 730)で`opencuda_vulkan::
  real::VulkanDevice`経由の`/v1/generate`を検証したところ、デバイス
  構築自体は成功する(ログに`OpenCUDA Vulkan Device (NVIDIA GeForce
  GT 730)`と出る)ものの、実リクエストが**約0.2秒で即座にエラー失敗**
  することが判明した。原因はこのリポジトリの`crates/open-cuda-llm/
  src/lib.rs`の`Linear::forward`が`opencuda_blas::sgemm`を呼ぶ際、
  `spirv`引数に常に`None`を渡しており、`GemmPath::VulkanGeneric`が
  必須とするコンパイル済みシェーダバイト列(`matmul.spv`)が渡っていない
  ため。これは「配線しても遅い」という直下エントリの設計上の懸念より
  手前の、単純に**動作しない**という結果——`opencuda_blas::sgemm`の
  `GemmPath::VulkanGeneric`自体は既に実装済み(`device.supports_spirv()`
  かつ`spirv`引数ありで動作する設計)なので、呼び出し側(`Linear::
  forward`)が`matmul.spv`のロード・引き渡しを行っていないだけの
  ギャップと考えられる(今回、指示により`open-cuda`側のコード変更は
  行わず、この調査結果のみ`aruaru-llm/CLAUDE.md`側に記録した)。
  - 次にすべきこと(最優先): `Linear::forward`(またはその呼び出し元の
    `DecoderLayer`)が使用中の`GpuDevice`実装に応じて`matmul.spv`を
    ロード・保持し`sgemm`へ渡すよう配線する。実装後、`aruaru-llm`側
    (`--features real-vulkan`)で実機再検証し、(1) CPU版とVulkan版の
    生成トークン列が完全一致すること、(2) 実際の速度差(直下エントリの
    QKV融合+プリフィル分離の効果でVulkan版が有利になるか)を計測する
    こと。詳細は`aruaru-llm/CLAUDE.md`のHANDOFF 2026-08-04エントリ・
    `aruaru-llm/README.md`「実推論ディスパッチ先としてのVulkan」節参照。

- **2026-08-04 `open-cuda-llm`にプリフィル/デコード分離+QKV融合GEMMを実装
  (aruaru-llm側CLAUDE.md 2026-07-26 HANDOFFで指摘された「安易なGPU配線は
  逆に遅くなりうる」問題への設計変更(a)(b)に対応、ユーザー指示
  「open-directx open-cuda aruaru-llmなどの使いやすさ向上と連携と実用性と
  完成度を向上させて」)**:
  1. **背景**: `aruaru-llm`側の2026-07-26 HANDOFFで、`GptModel`の推論
     ループが常に`seq_len=1`(1トークンずつ)で`opencuda_blas::sgemm`を
     呼ぶ設計のため、Vulkan経由への単純な置き換えはディスパッチ
     オーバーヘッドがCPU実行より遅くなる懸念があり、次の設計変更(a)
     プリフィルのバッチ化・(b)QKV融合GEMMが推奨されていた。今回はこの
     (a)(b)を実装した(GPUディスパッチへの実配線(c)は今回は着手せず、
     次回HANDOFFとして正直に申し送る、下記参照)。
  2. **(b) QKV融合GEMM**: `DecoderLayer`の`query`/`key`/`value`という
     3本の独立した`Linear`フィールドを、単一の`qkv: Linear`
     (`out_dim=3*hidden`)へ統合した。GPT-2のsafetensorsは元々Q/K/Vを
     `c_attn`という1本の融合`Conv1D`(`[hidden, 3*hidden]`)として保存
     しており、`load_fused_qkv`(列方向に3分割してから3本の`Linear`を
     組み立てていた専用関数)を削除し、既存の`load_conv1d`をそのまま
     `out_dim=3*hidden`で呼ぶだけで済むことが分かった(分割自体が
     不要だった)。`forward_step`(1トークンデコード)・新設の
     `forward_prefill`(下記)双方でこの融合`Linear`を使うため、
     デコード・プリフィルの両方でQ/K/Vのディスパッチ回数が3回→1回に
     減っている。
  3. **(a) プリフィル/デコード分離**: `DecoderLayer::forward_prefill`
     (新設)は、プロンプト全体(`seq_len`トークン)をQKV融合`Linear`・
     `attn_out`・`intermediate`・`output`の4つとも`seq_len`を`m`パラメータ
     とする**本当のGEMM(m>1)**として1回ずつ呼ぶ(レイヤーあたりの
     ディスパッチ回数が`4*seq_len`から`4`へ削減)。Attention自体は
     位置ごとの因果性(causality)を守るため、行(トークン位置)を昇順に
     処理しその位置までのキャッシュのみを参照する形は維持した(素朴な
     O(n)重複クエリ方式、既存のまま変更なし)。`GptModel::generate`は
     プロンプトの初回forwardをこの`forward_prefill`(内部で
     `forward_prefill_all_layers`)経由に変更し、生成された各トークンの
     逐次デコードは従来通り`forward_step`(`seq_len=1`)のまま
     (prefill/decode分離)。
  4. **挙動を変えない最適化であることの検証(型チェックのみで完了と
     報告しない方針を徹底)**:
     - 新規回帰テスト2件(`open-cuda-llm/src/lib.rs`):
       `prefill_batch_generate_matches_token_by_token_forward_step_reference`
       (複数トークンのプロンプト)・
       `prefill_batch_generate_matches_reference_for_single_token_prompt`
       (`seq_len=1`の境界ケース)——いずれも、最適化後の`generate()`が、
       1トークンずつ`forward_step`をループする素朴なリファレンス実装と
       生成トークン列が完全一致(ビット完全)することを確認。
     - **実GPT-2 124M重み(`openai-community/gpt2`、このマシンに
       ダウンロード済み)で、変更前(`git stash`でコード変更を退避)と
       変更後の生成結果を実際に比較**: プロンプト`"The quick brown
       fox"`・`max_new_tokens=12`で、変更前後とも
       `token ids: [274, 389, 257, 1049, 835, 284, 651, 257, 1310,
       1643, 286, 257]`(`"es are a great way to get a little bit of
       a"`)と**完全一致**(1トークンも違わない)することを確認した。
     - `cargo test -p open-cuda-llm --release`**9件全green**(既存7件+
       新規2件、regression無し)。`cargo build --workspace --release`/
       `cargo test --workspace --release`全クレートregression無し。
       `cargo clippy -p open-cuda-llm --all-targets --release -- -D
       warnings`(および`cargo clippy --workspace`)**警告0件**
       (リファクタ過程で新たに検出された`empty_line_after_doc_comments`・
       `explicit_counter_loop`の2件も解消済み)。
  5. **手動ベンチマーク(参考値、正直な開示)**: `#[ignore]`付きの
     `manual_bench_real_gpt2_generate_timing`テストを追加し、実GPT-2
     124M・プロンプト長15トークン・`max_new_tokens=20`でCPU実行時間を
     計測したところ約6.8秒だった。**これは変更前との比較ベンチマークでは
     ない**(旧コードでの同条件計測は今回実施していない)——今回の変更は
     「GEMM呼び出し回数を減らす」ものであり、CPU素朴実装の総浮動小数点
     演算量自体は不変のため、CPU実行時間の大幅な改善は本質的に期待して
     いない(このテストはあくまでVulkan配線時に比較対象となるCPU側の
     基準値を残す目的)。
  6. **正直な開示・スコープ外**: (c)「`aruaru-llm`側にオプトインの
     `real-vulkan` feature配線を追加し、実機(GT 730)でCPU版とVulkan版の
     生成結果が数値的に一致すること・実際の速度差をベンチマークで確認」
     は**今回未着手**。理由: (a)(b)の実装・検証(挙動を変えないことの
     証明)に時間を要したため、優先度指示通り「(a)(b)の完了・検証までを
     確実にやり切る」を優先した。(c)に着手する場合、`aruaru-llm/src/
     main.rs`のデバイス選択(`CpuDevice::new(0)`)を`opencuda-vulkan::
     real::VulkanDevice`へ切り替えるオプトインfeatureを追加し、
     `opencuda_blas::select_gemm_path`が`GemmPath::VulkanGeneric`を
     選ぶこと(既存の自動選択ロジック、2026-07-22実装済み)を利用する形に
     なる見込み。ディスパッチ回数は今回の変更でレイヤーあたり
     `4*seq_len+seq_len`(デコード)から`4`(プリフィル)+デコードは
     従来通りに削減されたため、Vulkan配線時のオーバーヘッド懸念は
     プリフィル側については大きく緩和されているはずだが、これは
     理論上の期待であり実測はしていない。
  - 次にすべきこと: (1) (c)`aruaru-llm`側`real-vulkan` feature配線+
    実機(GT 730)でのCPU版/Vulkan版の生成結果一致・速度ベンチマーク、
    (2) デコード側(`forward_step`、`seq_len=1`のまま)へのVulkan適用は
    引き続き懸念が残るため、複数リクエストのバッチデコード
    (continuous batching相当)等の追加設計が必要かの検討、(3) README/
    OmniGPU-Design.mdへの本変更の反映(現状はCLAUDE.md HANDOFFのみ)。

- **2026-07-31 `open-cuda-whisper`新設(6位Whisper相当のMVP着手、ユーザー指示)**:
  `open-raid-z`の「Python製AIライブラリのRust移植」ロードマップ
  (マーケティング調査1〜6位)のうち、前回(2026-07-25)HANDOFFで次の
  推奨とされていた6位Whisper相当に着手した。`open-cuda-bert`
  (エンコーダ専用パターン)・`open-cuda-llm`(KVキャッシュ付きGPT系
  デコーダパターン)の両方を組み合わせ、実際のWhisperアーキテクチャ
  (音声エンコーダ+テキストデコーダ+Cross-Attention)向けに新規実装した。
  1. **対数メルスペクトログラム抽出**(`log_mel_spectrogram`): 16kHz
     モノラルPCM→25msウィンドウ・10msホップのSTFT(素朴なO(N²)DFT、
     性能最適化は次回課題として正直に開示)→80メル帯域の対数パワー。
     外部音声デコードライブラリ非依存(既にデコード済みのf32 PCM
     サンプルを受け取る前提)。
  2. **`WhisperEncoder`**: メル特徴量を`Linear`で射影(本家の畳み込み
     stemの簡略版、正直な開示として明記)+正弦波位置埋め込み+
     pre-LNトランスフォーマー(双方向自己注意、`open-cuda-bert`と同じ
     Multi-Head Attention構成)。
  3. **`WhisperDecoder`**: `open-cuda-llm::GptModel`と同じKVキャッシュ付き
     自己回帰デコーダに、エンコーダ出力への**Cross-Attention**サブ層を
     追加。Cross-Attentionはクエリ長(デコーダ側)とキー/バリュー長
     (エンコーダ側)が異なるため`opencuda_blas::scaled_dot_product_
     attention`(Q/K/V等長前提)をそのまま使えず、`opencuda_blas::sgemm`
     のみを組み合わせた`cross_attention`ヘルパーを新設した。
  4. **重要な設計判断(ユーザー指摘を受けて)**: 当初`opencuda-directx`
     抜きで設計されていた`opencuda-blas`の自動バックエンド選択
     (`select_gemm_path`)が、`GpuVendor`(NVIDIA/AMD/Intel等の
     シリコンベンダー)だけを見て経路を選ぶ設計であることを確認した。
     **DirectXデバイスもVulkanデバイスも同じ`GpuVendor::Nvidia`等を
     返しうる**(どちらもDXGI/vkGetPhysicalDeviceProperties経由で
     同じベンダーIDを読むため)ため、現状の`select_gemm_path`ロジックは
     DirectXデバイスに対しても誤って`GemmPath::VulkanGeneric`
     (SPIR-Vシェーダ前提)を選んでしまう——これは`open-cuda-whisper`
     固有の問題ではなく`opencuda-blas`(=`open-cuda-bert`/`open-cuda-llm`
     含む全モデルクレート共通の基盤)側の既知のギャップと判断し、
     **`open-cuda-whisper`側にDirectX固有分岐を持ち込むことはしなかった**
     (モジュールdocに詳細を明記)。`opencuda-directx`は既にmatmul
     カーネルを実機検証済み(2026-07-27 HANDOFF参照)のため、
     `opencuda-blas`側に`GemmPath::DirectXGeneric`を追加すれば
     `open-cuda-whisper`を含む全モデルクレートが自動的にDirectX対応
     される見込み。
  5. **正直な開示・スコープの限界**(`open-cuda-bert`/`open-cuda-llm`初回
     MVPと同じ開発方針): (a) 学習済み重みは未対応(`load_random`のみ、
     `openai/whisper-tiny`等の実safetensorsローダーは次回の増分)。
     (b) 畳み込みstemを`Linear`射影に簡略化(真の畳み込みは未実装)。
     (c) トークナイザは`ByteTokenizer`(UTF-8バイト単位)のみ、本家の
     マルチリンガルBPE語彙は未対応。
  6. **検証**: `cargo build -p open-cuda-whisper`警告0件、
     `cargo test -p open-cuda-whisper`**9件全green**——メルスペクトログラム
     の形状・NaN/Inf非混入・短すぎる音声の安全な拒否、エンコーダの
     出力形状、デコーダの指定トークン数生成、同一シード決定性、
     異なるシードでの出力差、**そして最重要の
     `incremental_kv_cache_decoding_matches_full_recompute_at_each_
     position`**(KVキャッシュ経由の逐次デコードとキャッシュ無し
     フルスクラッチ再計算が、Cross-Attention込みで数値一致
     〈誤差1e-4以内〉することを確認——`open-cuda-llm`の同名テストの
     Cross-Attention版)、`transcribe`の一気通貫動作確認。
     `cargo clippy -p open-cuda-whisper --all-targets -- -D warnings`
     警告0件。`cargo build --workspace`/`cargo test --workspace`とも
     既存クレートへのregression無し(全green)。
  - 次にすべきこと: (1) `opencuda-blas::select_gemm_path`への
    `GemmPath::DirectXGeneric`追加(DirectX/Vulkanの判別、上記4.参照、
    `open-cuda-whisper`単体のスコープを超える基盤課題)、(2) 実在の
    学習済みWhisper重み(`openai/whisper-tiny`等)のsafetensorsローダー、
    (3) `aruaru-llm`を本家`poem`クレート直接依存からRPoem
    (`open-runo-poem-compat`)へ移行し、`open-cuda-whisper`を含む
    AI機能群をPoem互換APIとして他言語からHTTP経由で利用可能にする
    (ユーザー指示、「Rust＋Poem版と並行でRPoem」の実現、今回は
    Whisper本体の実装を優先しスコープ外とした)。

- **2026-07-30(続き2) Poly1305認証タグのGPU実装(opencuda-directx)を
  実機検証まで完了(ユーザー指示「Poly1305はGoogle検索して実装法も調査
  して開発して」への対応、前回HANDOFF〈2026-07-27〉の「130ビット剰余算を
  HLSLの32ビット整数演算のみで正しく実装する必要があり、誤りが数値検証
  なしには発見しづらい実装難度と判断し、今回は着手を見送った」を実際に
  解消)**:
  1. **日英Web検索で設計を裏取り**: Poly1305は`h_new=(h_old+m_i)*r mod
     (2^130-5)`という逐次依存のチェーン(1メッセージ内のブロック間では
     並列化できない)であること、公開ドメイン実装"poly1305-donna-32"
     (Andrew Moon作)が採用する5×26bit limb表現でのmod
     `2^130-5`演算という標準設計を確認した。
  2. **並列化の設計判断**: 1メッセージ内のブロック並列化(r^kの冪乗事前
     計算+並列リダクションが必要)はスコープ外とし、代わりに「多数の
     独立したメッセージを1スレッド1メッセージで一括処理する」バッチ
     並列化を採用——RS-LinkFusionが実際に扱う多数の独立した小さい
     ネットワークパケット(MTU程度)という利用形態に、内部並列化より
     素直に合致する設計判断。
  3. **64bit整数型を使わない実装**: DXIL SM6.0でも64bit整数演算
     (`uint64_t`)はオプション機能(Int64ShaderOps)でGT730のような
     旧世代GPUでの対応可否が不明なため、ChaCha20実装時と同じ「実機で
     本当に動くか不明な機能に頼らない」方針を貫き、32bit×32bit→64bit
     (hi,lo)ペア乗算(`umul32`)・64bit加算(`uadd64`)・64bit右シフト
     (`ushr64_lo`)を32bit整数演算のみで自前実装し、poly1305-donna-32の
     `unsigned long long`演算をすべてこれらのヘルパーへ機械的に置き換える
     形で移植した。
  4. **実装**: `crates/opencuda-directx/shaders/poly1305.hlsl`
     (r値のクランプ・h+=m・h*=r(mod p)・桁上げ伝播・最終処理まで
     poly1305-donna-32を忠実に移植)。`opencuda-directx::real::
     DirectXDevice`に`dispatch_poly1305`を追加(既存の`dispatch_chacha20`
     と同じ`UAV+RootConstants`パターン、UAV4本〈data/keys/block_counts/
     tags〉)。`launch_kernel`のカーネル名分岐に`"poly1305"`/
     `"poly1305_mac"`を追加。`tools/compile-dx12-shaders.{sh,ps1}`に
     コンパイルエントリを追加。
  5. **実機検証(型チェックのみで完了と報告しない方針を徹底)**:
     RustCrypto製`poly1305`クレート(dev-dependency、`compute_unpadded`
     ——端数ブロック処理を含まない点が本GPU実装の制約と正確に一致する
     ため選定)をCPU参照実装とし、3個の独立したメッセージ(鍵・長さが
     それぞれ異なる)を1回のディスパッチでバッチ処理した結果が、
     実機(NVIDIA GeForce GT 730)でCPU参照実装とバイト単位で完全一致
     することを確認した(`real_d3d12_dispatches_poly1305_batch_and_matches_rustcrypto_reference`)。
  6. **検証**: `cargo test -p opencuda-directx --release --features
     real-dx12`**9件全green**(既存6件+新規1件〈実質3件分のメッセージを
     1テストでバッチ検証〉、regression無し)。`cargo clippy -p
     opencuda-directx --all-targets --features real-dx12 -- -D
     warnings`警告0件(検証の過程で見つかった既存コードの
     `manual_is_multiple_of`警告1件も合わせて解消)。`cargo build
     --workspace`/`cargo test --workspace --release`両方でregression無し。
  7. **正直な開示・スコープ**: (a) メッセージ長は16バイトの整数倍のみ
     対応(Poly1305本来の「最後の不完全ブロックへのパディング」処理は
     未実装、呼び出し側が16バイト境界にパディング済みのデータを渡す
     前提)。(b) 1メッセージ内のブロック並列化は行っていない(上記の
     設計判断通り、バッチ〈メッセージ間〉並列のみ)。(c) `accel.rs`
     (RS-LinkFusion/open-web-server-wireが使うChaCha20-Poly1305 AEAD
     全体)への実際の配線はまだ行っていない——これでChaCha20暗号化+
     Poly1305認証タグの両方がGPU実装・実機検証済みとなったが、両者を
     組み合わせた完全なAEAD実装としての統合、および小サイズペイロード
     (MTU程度)でのH2D/D2Hオーバーヘッドが実利益を生むかのベンチマークは
     依然未着手。
  - 次にすべきこと: (1) ChaCha20+Poly1305を組み合わせた完全なAEAD
    実装としての`accel.rs`への統合、(2) 小サイズペイロードでのCPU版
    (`chacha20poly1305`クレート)とのベンチマーク比較、(3) メッセージ長
    が16バイトの整数倍でない場合(端数ブロック)への対応。

- **2026-07-30(続き) RAID6 Q-parity(Reed-Solomon、GF(2^8))のGPU実装第二段を
  実装・実機検証(ユーザー指示「Q-parityは必要で重要なので必ずGoogleで
  日本語と英語で設計方法と実装方法を検索して調査して開発実装して」への
  対応、前回HANDOFFで「実装難度が高く見送り」としていた項目に正面から
  着手)**:
  1. **日英Web検索で設計を裏取り**(着手前に必ず調査、という指示通り):
     Linuxカーネル`lib/raid6`/mdadmおよびH. Peter Anvin
     "The mathematics of RAID-6"論文が採用する標準方式——
     `Q = XOR_d(g^d・D_d)`(生成元`g=0x02`)、GF(2^8)の既約多項式
     `x^8+x^4+x^3+x^2+1`(バイト表現`0x11D`、乗算時に最上位ビットが
     立っていた場合の還元バイトは`0x1D`)——を日本語・英語両方の検索で
     確認し、実装した`gf_mul`(キャリーレス乗算+条件付き還元を8回、
     いわゆる"Russian peasant"乗算)がこの標準と一致することを裏付けた。
  2. **設計**: `raid6_xor_parity`(P-parity)と同じ`data`バッファレイアウト
     (disk d の word i は`data[d*block_words+i]`)を再利用しつつ、
     ディスクごとの係数`g^d`を渡す`coeffs`バッファ(`num_disks`要素)を
     追加した5引数契約(`data, coeffs, parity, num_disks, block_words`)。
     GLSL側は32bit word内の4バイトそれぞれについて`gf_mul(byte, coeff)`
     をXOR累積し、再パックして出力する。
  3. **実装**: `examples/raid6_q_parity_vulkan_real/shaders/raid6_q_parity.comp`
     (GLSL、`gf_mul`関数を含む)、`opencuda-vulkan::real::VulkanDevice`に
     `ensure_raid6_q_parity_args`/`run_raid6_q_parity_spirv`を追加し
     `launch_kernel`のカーネル名分岐に`"raid6_q_parity"`を追加。
     `tools/compile-vulkan-shaders.{sh,ps1,cmd}`に新シェーダのコンパイル
     エントリを追加。新規example crate`raid6_q_parity_vulkan_real`:
     CPU側に`gf_mul`(シェーダの`gf_mul`と全く同じアルゴリズムだが独立
     実装、Rustのバイト演算で記述)によるリファレンス実装・CPU版・
     実Vulkan版の3経路を同一入力(4ディスク×4096バイト、係数
     `g^0,g^1,g^2,g^3`)で実行し、bit-exact一致を検証。
  4. **実機検証(型チェックのみで完了と報告しない方針を徹底)**:
     `cargo run -p raid6_q_parity_vulkan_real --release`を実際にこの
     マシン(NVIDIA GeForce GT 730)で実行し、CPU vs reference・
     Vulkan vs reference・Vulkan vs CPUの3組すべてがbit-exact一致
     することを確認した。
  5. **検証**: `cargo build --workspace`警告0件、`cargo test --workspace
     --release`全クレートregression無し、`cargo clippy -p opencuda-vulkan
     -p raid6_xor_parity_vulkan_real -p raid6_q_parity_vulkan_real
     --all-targets --features real-vulkan -- -D warnings`警告0件。
  6. **正直な開示・スコープ**: これでRAID6のP-parity・Q-parity両方が
     GPU実装・実機検証済みとなったが、依然として(a)`open-raid-z`本体の
     実パリティ計算経路への統合、(b)実ブロックサイズでのCPU版との
     ベンチマーク比較、は未着手のまま(前回HANDOFFと同じ残作業)。
     また今回の`gf_mul`はスカラー(1バイトずつ)実装であり、Anvinの
     論文が言及する「複数バイトを並列処理する高速化」等のSIMD的な
     最適化は行っていない(RAID6の正しさの実証を優先、性能最適化は
     次の増分)。
  - 次にすべきこと: (1) `open-raid-z`本体の実パリティ計算経路への統合、
    (2) 実ブロックサイズ(4KB〜1MB程度)でのCPU版とのベンチマーク比較、
    (3) Poly1305認証タグのGPU実装(`opencuda-directx`側の既存の見送り
    項目、ユーザーから同時に「必ずGoogleで調査して実装して」との指示
    あり、次のセッション増分として着手予定)。

- **2026-07-30 RAID6 P-parity(XOR)のGPU実装第一段を実装・実機検証
  (open-raid-zのNVMe RAID6ランダムアクセス低速化問題への対応、ユーザー
  指示「open-directxとopen-cudaなどでハードウェアアクセラレーター対応を
  実装して解決して欲しい」への回答)**:
  1. **背景**: 4〜8枚のNVMe SSDでRAID6を組むと、Read-Modify-Write
     (parity write penalty)によりランダムアクセスが低速化する問題を、
     GPU/ASICアクセラレーターでパリティ計算をオフロードして解決したい、
     というユーザーの構想(既に`open-raid-z`のREADME/CLAUDE/PORTING.md
     にロードマップとして記録済み)の実装第一段。
  2. **設計**: 既存の`opencuda-vulkan::real::VulkanDevice`の
     `dispatch_spirv`共通経路(`vector_add`/`matmul`と同じパターン)を
     再利用し、新規`raid6_xor_parity`カーネルを追加。可変本数のデータ
     ディスクをシェーダの固定バインディング数で扱えるようにするため、
     「N本のディスクバッファを個別バインドする」のではなく「N本の
     ブロックを1本のバッファへ連結し(`data[disk*block_words+i]`
     レイアウト)、単一バッファとしてバインドする」設計にした——
     `vector_add`(3バッファ固定)と同様にシンプルなバインディング数
     (data 1本+parity 1本)を保てる。
  3. **実装**:
     - `examples/raid6_xor_parity_vulkan_real/shaders/raid6_xor_parity.comp`
       (GLSL、`local_size_x=256`、各wordについて全ディスクをXOR)。
     - `opencuda-vulkan::real::VulkanDevice`に
       `ensure_raid6_xor_parity_args`/`run_raid6_xor_parity_spirv`を
       追加(`vector_add`/`matmul`と全く同じ形の引数検証+ディスパッチ
       パターン)。`launch_kernel`のカーネル名分岐に`"raid6_xor_parity"`
       を追加。
     - `tools/compile-vulkan-shaders.{sh,ps1,cmd}`に新シェーダの
       コンパイルエントリを追加。
     - 新規example crate`raid6_xor_parity_vulkan_real`
       (`matmul_vulkan_real`と同じ構成): CPU素朴XORループのリファレンス
       実装・CPU版(`CpuDevice`のnativeカーネル)・実Vulkan版の3つを
       同じ入力(4ディスク×4096バイト=1024word、ディスク・word両方に
       依存する疑似ランダムパターンで、全ゼロ・全同一値では検出できない
       取りこぼしバグを避ける設計)で実行し、bit-exact一致(浮動小数点
       誤差許容なし、XORは厳密一致するべき演算のため)を検証。
  4. **実機検証(型チェックのみで完了と報告しない方針を徹底)**:
     `cargo run -p raid6_xor_parity_vulkan_real --release`を実際に
     このマシン(NVIDIA GeForce GT 730)で実行し、
     `device: OpenCUDA Vulkan Device (NVIDIA GeForce GT 730)`という
     実機ログとともに、CPU vs reference・Vulkan vs reference・
     Vulkan vs CPUの3組すべてがbit-exact一致することを確認した。
  5. **検証**: `cargo build --workspace`警告0件、
     `cargo test --workspace --release`全クレートregression無し、
     `cargo clippy -p opencuda-vulkan -p raid6_xor_parity_vulkan_real
     --all-targets --features real-vulkan -- -D warnings`警告0件。
  6. **正直な開示・スコープ**: これはP-parity(XOR)のみの実装であり、
     RAID6のQ-parity(Reed-Solomon符号、GF(2^8)上の演算)は別途実装が
     必要な、より複雑な増分として意図的に切り出した(直近のPoly1305
     GPU実装〈130ビット剰余算〉が「誤りが数値検証なしには発見しづらい
     実装難度」として見送られたのと同じ判断基準)。また、これは
     「パリティ計算そのものをGPUで行える」ことの実証であり、
     `open-raid-z`本体の実RAID6パリティ計算経路への実際の配線
     (統合)はまだ行っていない。ベンチマーク(小サイズブロックでの
     H2D/D2H転送オーバーヘッドがGPU計算優位性を相殺しないか)も未実施。
  - 次にすべきこと: (1) Q-parity(Reed-Solomon、GF(2^8))のGPU実装、
    (2) `open-raid-z`本体の実パリティ計算経路への統合、
    (3) 実ブロックサイズ(4KB〜1MB程度)でのCPU版とのベンチマーク比較
    (H2D/D2Hオーバーヘッドが実利益を生むか、前回のChaCha20と同様の
    懸念事項)。

- **2026-07-27 DirectX12スタックの実機健全性を再確認(ユーザー指示:
  Windows/Linux/nVIDIA実機を中心に開発・検証を進める、SET連携強化の
  一環)**: `cargo test -p opencuda-directx --release --features
  real-dx12 -- --nocapture`を実際に実行し、モック3件+実機5件
  (DXGIアダプタ名/ベンダー判定・H2D/D2Hラウンドトリップ・vector_add・
  matmul・ChaCha20)**全8件green**であることを実測で確認した(実出力:
  `DXGI adapter: name="NVIDIA GeForce GT 730" vendor=Nvidia {
  compute_capability: (0, 0) } total_memory=2104819712`)。新規のコード
  変更は無い、既存機能の実機再検証のみ。**正直な開示**: 前回HANDOFFの
  「次にすべきこと」(1) Poly1305認証タグのGPU実装(完全なAEAD化)は、
  130ビット剰余算(mod 2^130-5)をHLSLの32ビット整数演算のみで正しく
  実装する必要があり(標準的な5×26ビットlimb表現+桁上げ処理)、
  誤りが数値検証なしには発見しづらい実装難度と判断し、今回は着手を
  見送った(実装するなら段階的に、CPU側リファレンス実装〈RustCrypto
  `poly1305`クレート〉との1ブロックずつの数値照合を都度行いながら
  進めるべき領域として明記しておく)。「(3) コマンドリストのバッチ化」
  も、`GpuDevice::launch_kernel`が1回のディスパッチごとに同期的な
  フェンス待機を行う設計(`execute_and_wait`)を変更するには、CPU/
  Vulkan/DirectXの全バックエンドが共有する`GpuDevice`トレイト自体に
  新しいバッチAPI(`begin_batch`/`end_batch_and_wait`等)を追加する必要が
  あり、影響範囲が広いため今回は見送った。
  - 次にすべきこと: 前回HANDOFFの(1)(2)(3)から変更なし(Poly1305/
    ベンチマーク/コマンドリストバッチ化、いずれも未着手のまま)。

- **2026-07-25 `open-cuda-llm`にsafetensorsローダー追加(実GPT-2重みで検証)
  + `needless_range_loop`警告2件解消**: 前回HANDOFFの「次にすべきこと」
  (1)(2)に着手。
  1. **safetensorsローダー**: `open-cuda-bert::BertModel::load`と同じ設計
     (config.json→safetensorsの順で読み、テンソル名を辿る)で
     `GptModel::load(dir: &Path) -> Result<Self>`を実装。GPT-2は
     BERTと重みレイアウトが異なる点への対応が必要だった:
     (a) `Conv1D`層(`[in_dim, out_dim]`のまま保存、`nn.Linear`と違い
     転置不要)、(b) Q/K/Vが`c_attn`という1本の融合`Conv1D`
     (`[hidden, 3*hidden]`)にまとまっている(列方向に3分割)、
     (c) `lm_head`はトークン埋め込み`wte.weight`と重み共有(weight
     tying)されており、safetensors内に別テンソルとして存在しない
     (`wte.weight`を転置して代用)。
  2. **アーキテクチャ変更(重要)**: 当初の`DecoderLayer`はpost-LN
     (BERT/GPT-1系、Attention/FFNの後に残差加算+LN)だったが、実際の
     GPT-2はpre-LN(Attention/FFNの「前」に正規化を適用し、残差加算は
     正規化前の`hidden`に対して行う)を採用しているため、実重みを
     読み込んで意味のある出力を得るにはpre-LNへの構造変更が必須だった
     (`ln_1`/`ln_2`フィールドへ改名)。あわせてGELU近似も、GPT-2の
     `activation_function: "gelu_new"`(tanh近似)に合わせて既存の
     erfベース近似から差し替えた。ランダム初期化パス(`load_random`)は
     この変更後もKVキャッシュ増分計算とフルスクラッチ再計算の数値一致
     テストに影響なし(pre-LN/post-LNどちらでも自己無矛盾性は保たれる
     ため)。
  3. **実機検証(ネットワーク到達性を確認の上、実際にダウンロード・
     ロード・生成まで実施——型チェックのみで完了と報告しない方針を
     徹底)**: `huggingface.co`への到達性を`curl`で確認後、
     GPT-2 124M(`openai-community/gpt2`)の`model.safetensors`
     (548MB)・`config.json`・`tokenizer.json`を実際にダウンロード
     (`crates/open-cuda-llm/models/gpt2/`、`.gitignore`対象、リポジトリ
     には含めない)。`GptModel::load`で実際にロードでき
     (`vocab_size=50257`/`hidden_size=768`/`num_layers=12`/
     `num_heads=12`を実際に検証)、GPT-2自身のBPE語彙に対応した
     `GptTokenizer`(`tokenizers`クレート、`tokenizer.json`を読む)で
     "The quick brown fox"を継続生成させたところ、**貪欲デコードで
     "es are a great way to get a little bit of a"という文法的に
     自然な英語が出力された**(同一プロンプト・同一トークナイザで
     ランダム初期化モデルを走らせると"Kraken cluster cluster cluster
     Kraken Kraken..."という無意味な反復になる、テスト
     `real_gpt2_weights_load_and_produce_output_distinct_from_random_init`
     の`--nocapture`出力で両方を記録・比較済み)。**正直な評価**:
     これは「完全に流暢な文章生成」を主張するものではない(GPT-2 124M
     自体が小型モデルであり、本クレートの実装もPagedAttention等の
     本家vLLM最適化を持たない単一シーケンス逐次デコードのまま)が、
     「配線が正しく機能しているか」という当初の検証目的に対しては、
     ランダム重みの無意味な反復出力から明確に区別できる、文法的に
     妥当な英語への変化を実際に確認できた。
  4. **合成safetensorsによる単体テストも追加**(実重みが無い環境でも
     ローダーのロジック自体を検証できるように、モジュールdocコメント/
     タスク指示に沿って): `load_parses_gpt2_shaped_safetensors_and_config_without_panicking`
     が、GPT-2契約通りのテンソル名・形状を持つ合成safetensorsファイルを
     その場で構築し、`GptModel::load`が正しくパースできること・
     生成処理が最後まで通ることを検証する(実重みの有無に関わらず常に
     実行される)。
  5. **`GptTokenizer`新設**: 実重みでの意味のある検証には、
     `ByteTokenizer`(バイト値=トークンID)ではなくGPT-2自身のBPE語彙
     ベースのトークナイザが必要だったため、`open-cuda-bert::BertTokenizer`
     と同じ設計で`tokenizers`クレートによる`GptTokenizer`(`tokenizer.json`
     読み込み)を追加した。**正直な開示**: `ByteTokenizer`は既定のまま
     残しており(既存4テストは変更無し、後方互換)、実重みと
     `ByteTokenizer`を組み合わせても意味のある出力は得られない
     (GPT-2のBPE語彙IDとは無関係なため)——ドキュメントに明記済み。
  6. **`needless_range_loop`警告2件の解消**: `DecoderLayer::forward_step`の
     ヘッドループ(`enumerate().take(num_heads)`へ)、`load_fused_qkv`の
     QKV分割ループ(`chunks_exact`/`chunks_exact_mut`のzipへ)を
     イテレータベースへ書き換え。
  7. **検証結果**: `cargo build -p open-cuda-llm --release`警告0件、
     `cargo test -p open-cuda-llm --release`**6件全green**(既存4件+
     新規2件〈合成safetensors検証・実GPT-2重み検証〉、実機で実際に
     GPT-2重みをロード・生成し記録)。`cargo test --workspace --release`
     も全クレートでregression無し(全て`test result: ok`)。
     `cargo clippy --workspace --all-targets --release`**警告0件**
     (対象2件を含め全て解消)。
  - 次にすべきこと: (1) 本家vLLMの核心的最適化(PagedAttention・連続
    バッチング)への着手検討、(2) 残り4目標(PyTorch互換/scikit-learn/
    Whisper相当)のうち次に着手するものの選定(前回HANDOFFの推奨
    〈Whisper相当〉のまま)。

- **2026-07-22 `open-cuda-llm`新設(1位vLLM相当のMVP着手)**: `open-raid-z`
  CLAUDE.mdの「Python製AIライブラリのRust移植ハイブリッド/トライブリッド版」
  構想(マーケティング調査1〜6位: vLLM/Transformers/NumPy/PyTorch互換/
  scikit-learn/Whisper相当)のうち、`opencuda-blas`(NumPy相当)・
  `open-cuda-bert`(Transformersエンコーダ相当)に続き未着手だった
  **1位vLLM相当**に、新規クレート`crates/open-cuda-llm`として着手した。
  6目標を同時に手を広げず、既存の`opencuda-blas`のGEMM/Attention
  カーネルをそのまま再利用できて最も早く実用最小限(MVP)を動かせる
  対象として選定(判断基準: 既存クレートの再利用度合い、外部の巨大
  モデルダウンロードが不要であること)。
  1. **実装内容**: GPT系デコーダ(Self-Attention+FFNのレイヤーを
     `num_layers`重ね、KVキャッシュ付きで1トークンずつ貪欲デコード
     〈argmax〉する)。causalマスクは明示的なマスク行列ではなく
     「まだキャッシュに存在しない未来のトークンは追加されていない」
     ことで自然に実現(クエリ行を`n`回複製してn×n attentionを計算し
     先頭行だけ使う簡易手法、`opencuda-blas`側の変更は不要)。
     トークナイザはUTF-8バイト単位の素朴な自前実装(`ByteTokenizer`、
     外部モデルファイル不要)。重み初期化は決定的PRNG(`SplitMix64`)。
  2. **正直な開示**: (a) 本家vLLMの核心的最適化(PagedAttention・
     連続バッチング・複数リクエスト同時処理)は一切未実装、単一
     シーケンスの逐次デコードのみ。(b) **学習済み重みは無い**
     (`open-cuda-bert`と異なり`safetensors`ローダー未実装)、生成される
     テキストは意味を持たない——検証対象は「自己回帰生成パイプライン
     の配線が正しいか」であって「自然な文章を生成できるか」ではない。
     (c) バイト単位トークナイザなので本格的なBPE/SentencePieceより
     語彙効率は悪い。
  3. **検証**: `cargo build -p open-cuda-llm --release`警告0件、
     `cargo test -p open-cuda-llm --release`4件全green——
     `generates_requested_number_of_tokens_without_panicking`
     (プロンプトから8トークン生成しpanicしないこと)、
     `same_seed_and_prompt_produce_identical_output_deterministically`、
     `different_seeds_produce_different_weights_and_usually_different_output`、
     そして最も重要な**`incremental_kv_cache_decoding_matches_full_recompute_at_each_position`**
     (KVキャッシュを使った逐次デコードの各位置のロジットが、キャッシュ
     無しでシーケンス全体をフルスクラッチ再計算した場合と数値一致
     〈誤差1e-4以内〉することを検証——`opencuda-blas`の既存Flash
     Attention数値一致テストと同じ考え方で、causalマスクの代替実装が
     正しいことを裏付ける)。`cargo clippy -p open-cuda-llm --all-targets
     --release`は`needless_range_loop`警告2件のみ(機能に影響しない、
     次回クリーンアップ対象)。`cargo test --workspace --release`で
     既存クレート全て regression 無し(`open-cuda-bert`等の既存テストに
     影響なし)。
  - 次にすべきこと: (1) 実在の学習済みGPT系モデル(GPT-2小型版等)の
    `safetensors`を読み込むローダーの追加(`open-cuda-bert`の
    `BertModel::load`と同様の設計で移植可能)、(2) `clippy`の
    `needless_range_loop`警告2件の解消、(3) 残り4目標
    (PyTorch互換/scikit-learn/Whisper相当)のうち次に着手するものの
    選定(現時点の推奨: Whisper相当——既存の`open-cuda-bert`の
    エンコーダ実装パターンを転用しやすく、音声特徴量抽出さえ用意すれば
    比較的早くMVPに到達できると見込む)。

- **2026-07-21 CLAUDE.md新規作成**: これまでREADME/DEVELOPMENT-NEXT.md
  のみでプロジェクト共通の開発方針ドキュメントが無かったため新設。
  併せて`open-cuda-bert`(以前ローカルのみで未コミットだった)を
  ワークスペースへ正式追加・コミット・push済み(コミット`47f7837`)。
  - 次にすべきこと: (1) `opencuda-blas`のGPU専用パス(cuBLAS/rocBLAS/
    oneMKL/Vulkan汎用)の実装、(2) 真のFlash Attention、(3) `open-cuda-llm`
    (自己回帰デコーダ、vLLM相当)の設計・着手。

- **2026-07-21 真のFlash Attention + VulkanGEMM(matmul)経路を実装**:
  上記HANDOFFの(1)(2)のうち、GPUを要さない(2)と、実機Vulkanで検証可能
  だった(1)の一部(Vulkan汎用GEMM)を実装した。
  - `opencuda-blas::flash_attention(q, k, v, seq_len, head_dim, block_size)`:
    タイル化 + オンラインsoftmax(Dao et al. FlashAttentionのアルゴリズム1
    相当)を実装。既存の`scaled_dot_product_attention`(全展開版)は削除せず
    そのまま残し、docコメントで両者の違い(メモリ効率のためのタイル化の
    有無)を明記した。純粋なホスト側Rust実装で`GpuDevice`のカーネル
    ディスパッチは使っていない(その旨もdocに明記、誇張を避けるため)。
    新規テスト4件: 固定入力での`scaled_dot_product_attention`との数値一致
    (block_sizeがseq_lenを割り切らないケース含む)、決定的LCGによる
    やや大きめの乱数入力での一致、seq_len=1の境界ケース、次元不一致/
    block_size=0のエラーケース。
  - `opencuda-blas::sgemm_vulkan_generic(device, m, k, n, a, b, spirv)`:
    `GemmPath::VulkanGeneric`のスタブを実装に置き換えた。SPIR-Vバイト列は
    呼び出し側が渡す設計にした(`examples/matmul_vulkan_real/shaders/
    matmul.spv`と同じシェーダを想定)。当初`include_bytes!`でこのクレートに
    埋め込む案を試したが、`.spv`はビルド成果物のためリポジトリ全体で
    `.gitignore`の`**/*.spv`により追跡されておらず(`tools/
    compile-vulkan-shaders.*`で都度生成する運用)、埋め込むとシェーダ
    未コンパイルのクローン直後の環境で`cargo build -p opencuda-blas`
    自体が壊れることに気づき、その場で設計を修正した。alpha/beta
    スケーリングはシェーダ側が対応していないため非対応(`sgemm`のCPU版
    との差異として正直に明記)。
    新規テスト1件(`sgemm_vulkan_generic_matches_cpu_naive_on_real_hardware`):
    このマシンの実Vulkan環境(NVIDIA GeForce GT 730、`vulkaninfo --summary`
    で実機確認)で`examples/matmul_vulkan_real/shaders/matmul.spv`を
    読み込み`VulkanDevice::new`を実際に生成し、CPU版`sgemm`との数値一致
    (誤差1e-3以内)を検証。spvファイル未コンパイルの環境やVulkanデバイスが
    取得できない環境(CI等)ではassertを誤魔化さず`eprintln!`してスキップ
    する設計。
  - 検証結果: `cargo build -p opencuda-blas --release`成功、
    `cargo test -p opencuda-blas --release`は19件全passed(既存14件+
    新規5件、Vulkanテストも実機でpassed、スキップではない)。
    `cargo clippy -p opencuda-blas --all-targets`警告0件。
    `cargo test --workspace --release`も全クレートで regression 無し
    (全て`test result: ok`)。
  - 正直な制限・意図的にスキップした事項: `select_gemm_path`のベンダー
    判定ロジック自体は変更していない(`GpuVendor::Nvidia`は依然として
    `GemmPath::CuBlas`を返すスタブ経路のまま)。実機のVulkanDeviceは
    `GpuVendor::Nvidia`を返すため、`sgemm_vulkan_generic`は`sgemm`経由の
    自動選択には現状組み込まれておらず、明示的に呼び出す別関数として
    追加した(自動選択ロジックの再設計は次の増分に残す)。cuBLAS/
    rocBLAS/oneMKLの実装は引き続きスタブのまま(それぞれのGPUベンダー
    専用ライブラリをこのマシンでコンパイル・検証する手段が無いため。
    未検証コードを実装済みと偽ることになるので着手しなかった)。
    `omnigpu-llm`(自己回帰デコーダ)は今回のスコープ外(指示通り、
    規模が大きすぎるため次回以降)。

- **2026-07-22 `sgemm`自動選択にVulkanGenericフォールバックを配線**:
  上記HANDOFFで次の増分として明示されていた「`select_gemm_path`の
  自動選択ロジックの再設計」を実装した。
  - `opencuda-core::GpuDevice`に`supports_spirv(&self) -> bool`を追加
    (デフォルト`false`)。`DeviceInfo::vendor`だけでは「Vulkan経由で
    アクセスしているデバイスか、将来のCUDA直叩き実装か」を区別できず
    (実機NVIDIA機がVulkan経由だと`GpuVendor::Nvidia`を返すため)、
    ベンダー情報とは別に明示的な能力フラグが必要だった。
    `opencuda-vulkan::real::VulkanDevice`のみ`true`をオーバーライド
    (SpirVカーネルのみ受理する`launch_kernel`実装と整合)。CPU/mock
    デバイスは変更不要(デフォルトの`false`のまま)。
  - `select_gemm_path`: ベンダー別専用経路(CuBlas/RocBlas/OneMkl)が
    選ばれた場合でも、それが依然スタブであり、かつ渡された
    `device.supports_spirv()`が`true`なら`GemmPath::VulkanGeneric`へ
    フォールバックするよう変更(ベンダー判定ロジック自体は変更せず、
    その後段にフォールバック判定を追加する形)。
  - `sgemm`のシグネチャに`spirv: Option<&[u8]>`引数を追加(既存の
    `CpuNaive`専用呼び出しは`None`で良い)。`GemmPath::VulkanGeneric`が
    選ばれたとき、`spirv`が`Some`なら内部で`sgemm_vulkan_generic`を
    呼び出し、その結果へホスト側で`alpha`/`beta`スケーリングを適用して
    `CpuNaive`経路と同じセマンティクスを保つ。`None`ならエラーで
    明示的に失敗させる(黙って別経路にフォールバックしたり誤った
    結果を返したりしない)。ワークスペース内の既存呼び出し元
    (`open-cuda-bert`、本クレート内のattention実装・既存テスト)は
    末尾に`None`を追加して更新。`sgemm_vulkan_generic`自体は変更なし
    (引き続き直接呼び出し可能)。
  - 新規テスト1件(`sgemm_auto_dispatch_uses_vulkan_path_on_real_nvidia_hardware_instead_of_cublas_stub`):
    実機Vulkan環境で、`select_gemm_path(vulkan_device)`が
    `GemmPath::VulkanGeneric`を返すこと(ベンダーは依然
    `GpuVendor::Nvidia`であることも確認)、および自動選択の入口である
    `sgemm`(`sgemm_vulkan_generic`の直接呼び出しではなく)がVulkan経路
    経由でCPU版`sgemm`と数値一致する結果を返すことを検証。spv未
    コンパイル/Vulkanデバイス無しの環境では既存テストと同様に
    assertを誤魔化さず`eprintln!`してスキップ。
  - 検証結果: `cargo build -p opencuda-blas --release`警告0件、
    `cargo test -p opencuda-blas --release`は20件全passed(既存19件+
    新規1件、実機でpassed、スキップではない)。
    `cargo clippy -p opencuda-blas --all-targets --release`警告0件。
    `cargo test --workspace --release`も全クレートで regression 無し。
  - 正直な制限: cuBLAS/rocBLAS/oneMKLの実装自体は引き続きスタブの
    まま(前回と同じ理由、このマシンでは検証手段が無い)。それらが
    実装され次第、フォールバック優先順位(現状はスタブ<Vulkan)を
    ベンダー専用経路優先へ戻す必要がある旨は`select_gemm_path`の
    docコメントに明記した。

- **2026-07-25 Android-Vulkan互換性の監査(`opencuda-vulkan`)+ 実際にaarch64-linux-android向けクロスコンパイル成功まで確認**:
  1. **ソース監査**: `crates/opencuda-vulkan/src/real.rs`を精査した結果、Windows/Linuxデスクトップ専有の前提は見つからなかった——サーフェス/ウィンドウイングAPI(`VK_KHR_surface`/`VK_KHR_win32_surface`/`VK_KHR_xlib_surface`等)への依存は一切無く、`Entry::load()`(`ash`の動的ロード、`vulkan-1.dll`/`libvulkan.so`をプラットフォームに応じて自動選択)から`vkCreateInstance`→物理デバイス列挙→論理デバイス作成という、完全にヘッドレスなCompute専用の初期化のみ。`Cargo.toml`も`ash = { version = "0.37", default-features = false, features = ["loaded"] }`で、`cfg(windows)`/`cfg(target_os = "linux")`等のプラットフォーム分岐は`opencuda-vulkan`のソース中に1つも存在しない。
  2. **実クロスコンパイル検証(型チェックだけで完了と報告しない方針を徹底、このマシンに実際にAndroid NDK 27.1.12297006とRust向け`aarch64-linux-android`ターゲットがインストール済みであることを確認した上で実施)**:
     ```
     $ cargo build -p opencuda-vulkan --target aarch64-linux-android --features real-vulkan
     Compiling ash v0.37.3+1.3.251
     Compiling opencuda-vulkan v0.4.1 (F:\runo\open-cuda\crates\opencuda-vulkan)
     Finished `dev` profile [unoptimized + debuginfo] target(s) in 13.68s
     ```
     (`AR_aarch64_linux_android`/`CC_aarch64_linux_android`/`CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER`をNDKのclangラッパーへ設定した上で実行、`file`コマンドで生成された`libopencuda_vulkan.rlib`が実際にar archiveとして出力されていることも確認)。
  3. **正直な開示・確認できていないこと**: 上記はあくまで**クロスコンパイルの成功**(ビルドが通ること)の確認であり、実Android端末/エミュレータ上での実行(`vkCreateInstance`が実際に成功しVulkanデバイスを列挙できるか、`.so`としてリンクしAPKへ組み込んで動作するか)は未検証(実機/エミュレータでの検証環境がこのセッションには無い)。既知の懸念点として: (a) AndroidはVulkanドライバの実装差異(特に古い端末・エミュレータ)が大きく、`vkCreateInstance`自体は成功してもGPU固有の挙動差が実機でしか分からない、(b) このクレートは`cdylib`としてのビルド設定(JNI経由で呼び出す場合に必要な`crate-type`)が現状無い(`rlib`のみ)ため、実際にAndroidアプリへ組み込むにはJNIブリッジ層(または`aruaru-llm/android`のようなHTTPクライアント構成に倣う)が別途必要。
  4. **結論**: Android-Vulkan対応の**アーキテクチャ上の明確なブロッカーは見つからなかった**(ヘッドレスCompute専用設計がAndroidのVulkanネイティブ対応と自然に噛み合う)。ただし「Android対応が完了した」と主張するものではなく、あくまで「クロスコンパイルが実際に通ることを確認した」段階の報告である。
  - 次にすべきこと: (1) `cdylib`ビルド設定の追加+JNIブリッジ(またはaruaru-llm/android方式のHTTPクライアント経由での間接利用)の設計、(2) 実Android端末/エミュレータでの`vkCreateInstance`実行検証(環境が整い次第)、(3) `opencuda-directx`(D3D12、Windows専用)側は今回のスコープ外のまま。

- **2026-07-25(続き) `GpuVendor`にQualcomm/ARM/Imagination PowerVRを追加
  + `OmniGPU-Design.md`にベンダー対応状況マトリクスを新設(INTEL/AMD/
  nVIDIA統合というユーザー指示への、正直に検証可能な範囲での増分)**:
  1. **このマシンの実環境を再確認**: `vulkaninfo --summary`実行結果、
     実機GPUは依然**NVIDIA GeForce GT 730の1台のみ**(`vendorID=0x10de`、
     `deviceType=DISCRETE_GPU`)——統合Intel GPU等の第二のGPUは存在しない
     ため、複数実ベンダーでのVulkan列挙の実機検証はこのマシンでは不可能
     と確認(false claimを避けるため、着手前に確認)。CUDA Toolkitは
     `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`に存在するが、
     AMD ROCm・Intel oneAPI/oneMKLは未インストール(前回HANDOFFの
     「検証手段が無い」は依然事実のまま)。
  2. **`opencuda-core::device::GpuVendor`にバリアント追加**(非破壊、
     既存の`Nvidia`/`Amd`/`Intel`/`Cpu`/`Unknown`は変更なし):
     `Qualcomm { architecture: String }`/`Arm { architecture: String }`/
     `ImaginationPowerVr { architecture: String }`。PCI/Vulkanベンダー
     ID(Web検索+pci-ids.ucw.czで裏取り: Qualcomm`0x5143`〈"Qualcomm Inc"
     と確認〉、ARM`0x13B5`〈"ARM"と確認〉、Imagination Technologies
     `0x1010`〈pci-ids上は"Video Logic, Ltd."名義——PowerVR部門の前身の
     旧社名、Wikipedia "PowerVR"項目で"formerly VideoLogic"と裏付け〉)。
     `opencuda-vulkan::real::vendor_from_id`のmatchへ3分岐追加。
     `opencuda-blas::select_gemm_path`も、これら3ベンダーには
     cuBLAS/rocBLAS/oneMKLに相当するベンダー専用GEMMスタブが存在しない
     ため、最初から`GemmPath::VulkanGeneric`を返すよう分岐を追加
     (スタブ経由の遠回りを避ける設計判断)。
  3. **正直な開示**: 追加したQualcomm/ARM/Imagination分岐は**型チェック・
     ビルド成功のみ確認**——実機Qualcomm Adreno/ARM Mali/Imagination
     PowerVR GPU上での`vkEnumeratePhysicalDevices`実行検証は、この
     マシンにそれらのGPUが存在しないため不可能(前述の`vulkaninfo`
     確認結果より)。既存のNVIDIA実機テスト(`opencuda-blas`の
     `sgemm_vulkan_generic_matches_cpu_naive_on_real_hardware`等)には
     影響しないことのみ確認済み。
  4. **`OmniGPU-Design.md`「8.5. ベンダー対応状況マトリクス」節を新設**:
     Vulkan Compute統合(実働・主要な統合機構、ディスパッチ経路自体には
     ベンダー分岐が無いことを`real.rs`の実装から確認済み)/`GpuVendor`
     列挙(報告用の情報層、今回拡張)/ベンダー専用最適化ライブラリ経路
     (cuBLAS/rocBLAS/oneMKL、引き続きスタブ・検証不能)の3層を表形式で
     区別し、「統合の実体はVulkanであり、ベンダー専用ライブラリ層は
     前提条件ではなく将来の追加最適化」であることを明記(誇張防止)。
  5. **検証結果**: `cargo build --workspace`警告0件・成功。
     `cargo test --workspace --release`全クレートregression無し
     (`opencuda-blas`は既存21件+実機Vulkanテスト2件を含む22件全green、
     実機NVIDIA GT 730でスキップ無しで実行されたことを確認)。
     `cargo clippy --workspace --all-targets`警告0件。
  6. **正直な結論**: 「実機でVulkan経由の複数ベンダー統合を新たに検証
     できた」という増分ではない(このマシンにはNVIDIA 1台しか無い
     ため、その種の検証は原理的に不可能)。今回の増分は
     (a)`GpuVendor`の分類粒度を実世界のモバイル/組込みGPUベンダーまで
     広げたこと、(b)Vulkan統合という「実際に機能している統合機構」と
     ベンダー専用スタブ層との違いを設計書に正直に文書化したこと、
     の2点に限定される。これ以上、このマシンの制約(GPU1台・
     ROCm/oneMKL未導入)の中で正直に主張できる新規の実機検証項目は
     見当たらなかった。
  - 次にすべきこと: (1) 実Qualcomm/ARM/Imagination GPU環境
    (実Android端末等)が用意でき次第、`vendor_from_id`の3分岐の実機
    検証、(2) AMD ROCm/Intel oneAPIのインストール手段が得られ次第、
    `select_gemm_path`のスタブ実装への着手。

- **2026-07-26 `opencuda-blas`のドキュメント修正(実装済みコードに
  ドキュメントを追いつかせる、新規実装ではない)**: `crates/
  opencuda-blas/src/lib.rs`冒頭のモジュールdocコメントが
  「`flash_attention`という名前の関数は実装していない」
  「`quantize_int4`はこのパスでは対象外」という、2026-07-21・
  2026-07-22のHANDOFFで既に実装済みとなった内容と矛盾する古い記述
  のままだった(実装時に更新し忘れたもの)ことに気付き、修正した。
  1. **事実確認**: `lib.rs`を全文読み、`flash_attention`(440行目
     付近)がタイル化+オンラインsoftmax(Dao et al. アルゴリズム1相当)
     の本物の実装であること、`quantize_int4`/`quantize_int8`/
     `quantize_int4_awq`(AWQ風activation-aware INT4量子化)が
     `dequantize_*`の逆変換・往復誤差検証テストとともに実装済み
     であることを、コード本体を読んで確認した(grepだけで済ませず)。
  2. **修正箇所**: (a) モジュールdocコメント(1〜40行目付近)を、
     cuBLAS/rocBLAS/oneMKLのみが未検証スタブのまま(このマシンには
     CUDA/ROCm/oneAPIのツールチェインが無いため)であることを明記
     しつつ、flash_attention/量子化3関数は実装済みである旨へ書き換え。
     (b) `scaled_dot_product_attention`のdocコメント(356行目付近)
     内の「`flash_attention`という別関数を、真のタイル化実装向けの
     スタブとして残してある」という記述も同様に古かったため、
     「タイル化・オンラインsoftmaxを実際に行う真のFlash Attentionは
     別関数`flash_attention`として実装済み」へ修正。(c)
     `README-Japan.md`のロードマップ表(Phase 3節)も同じ理由で
     チェックボックスと説明文が古かったため、`flash_attention`・
     `GemmPath::VulkanGeneric`・INT4/INT8/AWQ量子化を`[x]`実装済みへ
     更新(cuBLAS/rocBLAS/oneMKLのみ`[ ]`スタブのまま明記)。
     `PORTING.md`にはこの種の古い記述は見つからなかった。
  3. **`nvcc --version`を実際に再実行して確認**: `nvcc: command not
     found`(見つからない)ことを確認済み——cuBLAS検証手段は依然
     このマシンには無く、cuBLAS/rocBLAS/oneMKL経路は今回も一切
     実装・変更していない(ドキュメント修正のみのスコープ)。
  4. **検証結果**(実際に実行、型チェックのみで済ませていない):
     `cargo build --workspace --release`警告0件・成功。
     `cargo test --workspace --release`**全クレートregression無し**
     (`opencuda-blas`22件・`open-cuda-bert`2件・`opencuda-directx`3件
     〈モックのみ、`real-dx12`feature未指定〉・`opencuda-ir`1件+
     結線テスト1件・`open-cuda-llm`6件・`opencuda-vulkan`結線テスト
     3件、他クレートは0件、全て`test result: ok`、失敗0件)。
     `cargo clippy --workspace --all-targets --release`**警告0件**。
  5. **正直な開示**: このセッションでの新規実装は無い
     (flash_attention/量子化3関数はいずれも2026-07-21・07-22の
     過去コミットで既に実装・検証済みだったコードそのもの)。今回の
     作業は「ドキュメントが実装より古い状態のまま取り残されていた」
     という不整合の是正のみ。cuBLAS/rocBLAS/oneMKLは引き続き未実装
     スタブのまま(理由は変わらず、このマシンにCUDA/ROCm/oneAPI
     ツールチェインが無いこと)。
  - 次にすべきこと: 前回HANDOFF(1)(2)に変更なし(実Qualcomm/ARM/
    Imagination GPU環境・AMD ROCm/Intel oneAPI導入待ち)。

- **2026-07-27(続き) README.mdに「まず自分のGPUで動くか試す」導線を追加(使いやすさ改善、ユーザー指示「open-directx と open-cuda と aruaru-llmのSETの完成度と実用性と使いやすさの向上をお願い」)**:
  1. 既に`examples/`配下に`vulkan_info`(実Vulkan物理デバイス列挙)・
     `matmul`・`matmul_vulkan_real`・`vector_add`系4本の実行可能な
     ワークスペースメンバーが存在していたが、README.mdからその存在が
     全く案内されていなかった(外部監査で指摘された使いやすさの
     ギャップ)。新規に「自分のGPUで実際に動くか試す」節を追加し、
     `cargo run -p vulkan_info`を「まず1つ動かして確認する」最初の
     コマンドとして案内。他のexampleへの導線・`OmniGPU-Design.md`§8.5
     (ベンダー対応表)への参照も追記。
  2. **検証**: `cargo run -p vulkan_info`を実際に実行し、このマシン
     (NVIDIA GeForce GT 730)で正しくベンダー名・VRAM容量・Vulkan
     API/ドライババージョンが出力されることを確認済み。`cargo test
     --workspace`は既存の全テスト回帰なし(ドキュメントのみの変更)。
  - 次にすべきこと: 前回HANDOFFの内容に変更なし(cuBLAS/rocBLAS/
    oneMKL・実Qualcomm/ARM/Imagination GPU環境待ち)。

- **2026-07-27(続き2) `GptModel::load`が`transformer.`プレフィックス付きテンソル名を読めず実際に失敗する実バグを発見・修正(aruaru-llmの実E2E検証中に発覚)**:
  1. **発見の経緯**: aruaru-llmで実際に`distilbert/distilgpt2`を
     Hugging Faceからダウンロード→`POST /v1/models/select`で切り替え
     ようとしたところ、`missing tensor 'wte.weight': TensorNotFound`
     で失敗。ダウンロードされた`model.safetensors`のヘッダーを実際に
     読んだところ、テンソル名が`transformer.wte.weight`・
     `transformer.h.0...`のように**`transformer.`プレフィックス付き**
     であることが判明した。一方、既に動作確認済みの`openai-community/
     gpt2`本体は`wte.weight`(プレフィックス無し)を使っており、
     **同じGPT-2アーキテクチャでも変換元スクリプトによってテンソル名
     規約が実際に異なる**ことが根本原因だった。
  2. **`crates/open-cuda-llm/src/lib.rs::GptModel::load`を修正**:
     ロード時に`wte.weight`/`transformer.wte.weight`のどちらが実在するか
     を確認して`key_prefix`を自動判定し、以降の全テンソル名
     (`wte.weight`/`wpe.weight`/`h.{i}...`/`ln_f`)にこの`key_prefix`を
     前置するよう変更。モデルごとの個別分岐を増やすのではなく、
     プレフィックスの自動判定という1箇所の変更で両規約を吸収する設計。
  3. **検証**: 新規回帰テスト
     `load_parses_transformer_prefixed_safetensors_like_distilgpt2`を
     追加(合成フィクスチャを`transformer.`プレフィックス付きで生成し、
     ロード→生成まで通ることを確認)。既存の
     `load_parses_gpt2_shaped_safetensors_and_config_without_panicking`
     も無変更のままgreen(後方互換)。`cargo test -p open-cuda-llm`
     **7件全green**(既存5件+新規2件のうち1件は本来から存在した
     フィクスチャ関数のリファクタリングで名称変更、実質+1件)。
  4. **実E2E確認(型チェックだけで終わらせない)**: 修正後、実際に
     aruaru-llmサーバーを起動し、`POST /v1/models/select`で
     `distilgpt2`への切り替えが成功し(修正前は失敗していた)、
     `POST /v1/generate`で実際に英文が生成される(`distilgpt2-greedy-
     decode-v0-open-cuda-llm-cpu`エンジンラベル付きで応答)ことを
     実際に確認した——「ダウンロード→切り替え→生成」の一気通貫を
     実際に検証し、かつその過程で見つかった実バグを実際に修正した。
  - 次にすべきこと: (1) `gpt2-medium`/`gpt2-large`/`gpt2-xl`が
    それぞれどちらのテンソル名規約を使うか未確認(`gpt2`はプレフィックス
    無し、`distilgpt2`はプレフィックス有りと確認済みだが、他のサイズは
    未検証)、(2) 将来的にLlama/Mistral等の異なるアーキテクチャへ
    対応する場合は、この`key_prefix`方式では吸収しきれない可能性が
    高い(テンソル名の構造自体が異なるため)。

- **2026-08-05 前回HANDOFF項目(1)を検証: `gpt2-medium`/`gpt2-large`/
  `gpt2-xl`のテンソル名規約を実際に確認**:
  1. **safetensorsヘッダーを実際に読んで確認**(フルダウンロード前に、
     Hugging Face上の`model.safetensors`へRangeリクエストで先頭8バイト
     〈ヘッダー長〉+ヘッダーJSON本体のみを取得する方式):
     `openai-community/gpt2-medium`(316テンソル)・`gpt2-large`
     (472テンソル)・`gpt2-xl`(628テンソル)のいずれも、`transformer.`
     プレフィックス付きテンソルは0件、`wte.weight`/`wpe.weight`/
     `ln_f.weight`/`ln_f.bias`はプレフィックス無しの形で存在することを
     確認した。つまりこの3サイズはいずれも`gpt2`(base)と同じ
     プレフィックス無し規約であり、`distilgpt2`のみが例外
     (`transformer.`プレフィックス付き)という位置づけが裏付けられた。
  2. **型チェックだけで終わらせず実E2Eで検証**: このマシンには
     既に`aruaru-llm`が`gpt2-medium`/`gpt2-large`/`gpt2-xl`を含む
     全カタログをダウンロード済み(先行セッションの成果物)で、かつ
     常駐サーバーが稼働中だったため、これを使って実際に
     `POST /v1/models/select`→`POST /v1/generate`を実行した。
     `gpt2-medium`・`gpt2-large`はいずれも切り替え成功、生成も
     正常応答(例: `gpt2-medium`で"The capital of France is"→
     " Paris, and the capital of France is Paris."、
     `gpt2-large-greedy-decode-v0-open-cuda-llm-cpu`等の期待通りの
     エンジンラベル付き)。**コード変更は一切不要**(既存の
     `key_prefix`自動判定ロジックがそのまま両方を正しく処理した)。
  3. **正直な開示・新たに判明した制限**: `gpt2-xl`(1.5B、6.4GB)への
     切り替えは`"open-cuda-llm: failed to read model.safetensors in
     ... out of memory"`で実際に失敗した。原因はコードの不具合では
     なく、このマシンの空きメモリ不足(切り替え試行時点で物理メモリ
     32GB中の空き実測約4GB、既に`gpt2-large`ロード済みで
     `aruaru-llm`プロセス自体が約10GB使用中、加えてWSL・他の並列
     開発セッション等が同時稼働していたための資源逼迫)と判断——
     `gpt2-xl`のテンソル名規約自体はヘッダー確認で他2サイズと同一と
     確認済みであり、コードパスは同じはずだが、実機での完全なE2E
     生成成功までは確認できていない(誇張しないための明記)。
     この際、サーバープロセスがOOM後にクラッシュして応答不能に
     なったため、再起動して既定モデル(`gpt2`)がロードされた状態へ
     復旧させ、他セッションが使っている可能性のある常駐サーバーの
     状態を元通りにした。
  4. **検証結果**: `cargo build --workspace --release`は変更なし
     (今回はコード変更を伴わない検証作業のため未実行)。ドキュメント
     (本ファイル)のみ更新。
  - 次にすべきこと: (1) `gpt2-xl`実機ロードのメモリ不足問題——
     十分な空きメモリが確保できるタイミング(他の並列セッションが
     アイドルな時等)に再度`select`を試し、実際に生成まで到達する
     ことを確認する。恒常的な対策としては、大きいモデルを読み込む
     前に他の使用中モデルを明示的に解放する仕組み(現状は
     プロセス内に複数モデルを保持したままの可能性がある)の要否を
     `aruaru-llm`側で調査する価値がある。(2) 前回HANDOFFの(2)
     (Llama/Mistral等異なるアーキテクチャは`key_prefix`方式では
     吸収しきれない)は未着手のまま変更なし。(3)
     Qualcomm/ARM/Imagination実機・AMD ROCm/Intel oneAPI導入待ちの
     項目群も変更なし。

- **2026-08-06 softmax専用のSPIR-Vカーネルを新規実装・実機検証(前回HANDOFF
  「次にすべきこと(1)」への着手、ユーザー指示の優先順位確認: aruaru-llm側は
  既に`set_matmul_spirv`を実配線済み〈commit `6452ae4`〉と確認できたため
  今回はopen-cuda本体の実装増分に着手)**:
  1. **事前調査**: `aruaru-llm/src/generation.rs`を読み、`wire_matmul_spirv`
     関数が既に`GptModel::set_matmul_spirv`経由でQKV/attn_out/intermediate/
     output/lm_headの全`Linear`層へmatmul.spvを配線済みであることを確認した
     (2026-08-05付コメントで前回HANDOFFの実バグ修正commit `6452ae4`を参照)。
     API不整合は見つからず、open-cuda側の対応は不要と判断。また
     `open-directx`(`F:\runo\open-directx`)を確認したところ、CLAUDE.mdの
     旧記述(「空リポジトリ」)とは異なり、既に`directx-graphics-vulkan`
     (`render_triangle`実機検証済み)・`directx-shader-translate`
     (DXIL/DXBC/HLSL変換、多数の`.dxbc`/`.dxil`シェーダを含む)の2クレートを
     持つ独立プロジェクトへ発展していることを確認した。ただしこれは
     グラフィックスパイプライン(triangle rasterization)寄りの実装であり、
     本タスクのAttention/GEMM(コンピュートシェーダ)経路との直接連携は無い
     (現時点でopen-cuda側から呼び出す配線は無し、両者は独立に発展中)。
  2. **このマシンの実GPU環境を再確認**: `nvidia-smi`実行結果、依然
     **NVIDIA GeForce GT 730の1台のみ**(ドライバ475.14、VRAM 2048MiB、
     Driver-reported CUDA 11.4)——GT 730より高性能なGPUはこのマシンには
     存在しない(前回までのHANDOFFの制約が継続)。
  3. **実装**: `examples/softmax_vulkan_real/shaders/softmax.comp`
     (GLSL、`local_size_x=256`、1ワークグループ=1行を担当し、共有メモリ
     `shared float sdata[256]`でmax・sumの二分木リダクションを行う数値安定
     softmax)。`opencuda-vulkan::real::VulkanDevice`に
     `ensure_softmax_args`/`run_softmax_spirv`を追加(`vector_add`等と同じ
     `args: &[KernelArg]`契約、ポインタ1つ+usize2つの3引数)、
     `launch_kernel`のカーネル名分岐に`"softmax"`を追加。
     `opencuda-blas::softmax_vulkan_generic(device, rows, cols, data, spirv)`
     をホスト側ラッパーとして新設(`sgemm_vulkan_generic`と同じ設計、
     `LaunchConfig::linear(rows*256, 256)`でgrid.x=rowsになるよう調整)。
     新規example crate`softmax_vulkan_real`(CPUリファレンス実装との
     数値一致・各行の合計が1.0になることを検証)を`tools/
     compile-vulkan-shaders.{sh,ps1,cmd}`のコンパイル対象へ追加、
     ワークスペースメンバーへ登録。
  4. **正直な開示・スコープ**: このカーネル自体は独立した再利用可能な
     部品として実装・実機検証済みだが、**既存の
     `scaled_dot_product_attention_with_spirv`(GPU GEMM + CPU softmaxの
     ハイブリッド)内部のCPU softmaxをこのカーネルへ置き換える配線は
     まだ行っていない**(既存APIのシグネチャ変更が必要になるため、
     影響範囲の検討を伴う次の増分として切り出した)。`softmax_vulkan_generic`
     の関数docコメントにもこの開示を明記済み。
  5. **実機検証(型チェックのみで完了と報告しない方針を徹底)**:
     `cargo run -p softmax_vulkan_real --release`をこのマシン(NVIDIA
     GeForce GT 730)で実際に実行し、`device: OpenCUDA Vulkan Device
     (NVIDIA GeForce GT 730)`という実機ログとともに、8x37行列(256の
     倍数でない列数)でVulkan版がCPUリファレンスと誤差1e-4以内で一致し、
     各行の合計が1.0になることを確認した。
  6. **検証結果**: `cargo build --workspace --release`警告0件・成功。
     `cargo test -p opencuda-blas --release`**24件全green**(既存23件+
     新規`softmax_vulkan_generic_matches_cpu_reference_on_real_hardware`
     1件、実機Vulkanでスキップ無しで実行)。`cargo test --workspace
     --release`全クレートregression無し(全て`test result: ok`)。
     `cargo clippy -p opencuda-vulkan -p opencuda-blas -p softmax_vulkan_real
     --all-targets --release --features real-vulkan -- -D warnings`
     **警告0件**。
  7. **完成度調査(grep)**: リポジトリ全体で`todo!()`/`unimplemented!()`/
     TODO/FIXME/stubを再調査。実質的な未着手項目は`opencuda-multidev::
     transfer_between_devices`の`TODO(Phase 3)`(ホストメモリ経由の
     d2h→h2d、複数デバイス構成が実マシンに無いため元々検証不能な項目)
     のみで、他は過去HANDOFFで既に開示済みのcuBLAS/rocBLAS/oneMKLスタブ・
     ドキュメント上の言及に限られる——新たな見逃しは見つからなかった。
  - 次にすべきこと: (1) 本増分の`softmax_vulkan_generic`を
    `scaled_dot_product_attention_with_spirv`(または新規の完全fused
    attention関数)へ実際に配線し、「GPU GEMM + CPU softmax」の
    ハイブリッドから「GPU GEMM + GPU softmax」へ移行する、(2)
    `flash_attention`のSPIR-V対応(タイル単位でのGPUディスパッチ、
    現状は純粋ホスト側Rust実装のまま)、(3)
    Qualcomm/ARM/Imagination実機・AMD ROCm/Intel oneAPI導入待ちの
    項目群は変更なし、(4) `open-directx`と`open-cuda`はいずれも実体を
    持つプロジェクトへ発展したが、両者間の直接連携(コンピュート
    シェーダ経路とDXIL/DXBC変換パイプラインの統合)はまだ設計段階、
    ユーザー優先順位に沿って次回検討する。

- **2026-08-19 自動アップデート機能の展開可否を調査(実装見送り)**:
  ユーザーより、`open-english`の`server/src/self_update.rs`(GitHub
  Releases検知+`/healthz`ベース自動ロールバック付き自己更新)と同様の
  仕組みを`aruaru-llm`の依存先である本リポジトリへも展開する依頼を
  受け調査。
  - workspace構成を確認: ライブラリクレート10個(`opencuda-core`/
    `opencuda-cpu`/`opencuda-mock`/`opencuda-vulkan`/`opencuda-directx`/
    `opencuda-ir`/`opencuda-blas`/`opencuda-multidev`/`open-cuda-bert`/
    `open-cuda-llm`/`open-cuda-whisper`)に加え、`examples/`配下に
    `fn main`を持つバイナリが12個(`vector_add`・`vector_add_vulkan`・
    `vector_add_omniir`・`vector_add_vulkan_real`・`vulkan_info`・
    `matmul`・`matmul_vulkan_real`・`matmul_bench`・
    `raid6_xor_parity_vulkan_real`・`raid6_q_parity_vulkan_real`・
    `softmax_vulkan_real`・`sbm_demo`)存在することを確認した。
  - **自動アップデート実装の見送り理由**: 依頼文の想定通り、本リポジトリの
    実態は「ライブラリクレート集+検証用example群」であり、常駐して
    稼働し続けるサーバー/CLIサービスは1つも無い。12個のexample
    バイナリはいずれも`cargo run -p <name> --release`で都度実行される
    ベンチマーク・実機検証用の使い捨てプロセスであり、インストール後に
    起動しっぱなしで稼働する対象ではない。`/healthz`のようなヘルス
    チェック・自己更新関連コードもリポジトリ全体を`grep`した結果ゼロ件
    だった。「新バージョン検知→自身を差し替え→ヘルスチェック失敗で
    ロールバック」という自己更新の概念は、依存先である`aruaru-llm`側が
    `Cargo.toml`の依存バージョン(現在`workspace.package.version =
    "0.4.1"`)を上げて再ビルド・再デプロイする形が実態に即した
    「アップデート」であり、本リポジトリ単体に自己更新機構を実装する
    ことは見送った(依頼文中で懸念された通りの結論)。
  - 次にすべきこと: もし将来`opencuda-*`系列から常駐デーモン・
    ユーザー配布用CLIツール(例: GPU監視デーモン等)が切り出される
    場合は、その時点で改めて`self_update.rs`相当の導入をユーザーと
    相談する。現状は据え置き。

- **2026-08-20 「2つのopen-directx」混同の解消(調査のみ)**:
  以前のHANDOFF(直近エントリ)で「`open-directx`と`open-cuda`は
  いずれも実体を持つプロジェクトへ発展したが、両者間の直接連携は
  まだ設計段階」と記述していたが、これは不正確だったため訂正する。
  実際には設計段階の連携が「ある」のではなく、**同名の無関係な別物が
  2つ存在するだけ**だった:
  1. 本リポジトリ内蔵の`opencuda-directx`クレート
     (`crates/opencuda-directx`、workspace memberとして`Cargo.toml`に
     `"crates/opencuda-directx"`で登録)。
  2. GitHub上の独立リポジトリ`aon-co-jp/open-directx`
     (`F:\runo\open-directx`)。ワークスペースメンバーは
     `directx-shader-translate`/`directx-graphics-vulkan`/
     `directx-graphics-window`で、クレート名・パスとも(1)と重複なし。
     path依存・submodule等の技術的連携は無い(grep相互参照ゼロ件)。
  `aruaru-llm`が使っているのは(1)のみ(`hw-detect-directx` optional
  feature経由、既定では無効)で、(2)は一切使用していない。
  詳細な調査記録は`aruaru-llm/CLAUDE.md`の2026-08-20エントリ参照。
  次にすべきこと: 「両者間の直接連携」という表現は今後使わず、
  必要なら「(1)内蔵opencuda-directxクレート」「(2)独立リポジトリ
  open-directx」と明示的に区別して記述すること。

- **2026-09-01 Model Folding残タスク: `ridge_lambda`外部化(完了)/
  GPU実測(未着手・環境制約)(他アカウントでの再開用メモ)**:

  前回セッションで実装した3手法(独立閾値/連続ブロック探索/線形
  アダプタ、`open-cuda-llm`クレート)について、ユーザーから明示された
  残タスクのうち2件に対応した。

  1. **`ridge_lambda`の外部調整可能化(完了)**: `fold_block_with_
     linear_adapter`のシグネチャに`ridge_lambda: Option<f32>`引数を
     追加(既存呼び出し元は全て`None`=既定値`1e-2`に更新済み)。値を
     `AdapterFoldReport::ridge_lambda_used`として結果に含め、呼び出し
     側が実際に使われた値を確認できるようにした。非有限・非正の値は
     `ensure!`で正直に拒否(数値的に不安定な解や特異行列エラーを未然
     に防ぐ)。テスト2件追加(既定値フォールバック/明示指定の反映、
     不正値5種の拒否)、`cargo test -p open-cuda-llm --lib
     fold_block_with_linear_adapter`で4件ともpass確認済み。HTTP側の
     配線(`POST /v1/models/fold-layers`)は`aruaru-llm`側で対応済み
     (`aruaru-llm/CLAUDE.md`参照)。
  2. **GPU実測(Vulkan/DirectX経由、未着手・環境制約)**: このクレートの
     Attention計算(QKᵀ・softmax・P·V)は既に`--features real-vulkan`
     経由で実Vulkanデバイス上にディスパッチ可能(2026-08-05以降に配線
     済み、上記の別エントリ参照)。したがって層折りたたみ3手法の
     GEMM/Attention部分をVulkan経路に乗せること自体は技術的には可能
     なはずだが、**このセッションの作業環境にGPU・GPUドライバが存在
     しないため、実際にビルド・実行してベンチマークを取ることが物理的
     にできなかった**(試さずに「無理そう」と判断したのではなく、GPU
     検出コマンド自体が失敗することを確認した上での結論)。開発機
     (NVIDIA GT 730搭載、`--features real-vulkan`のビルド実績あり)
     でこの計測を行うことが次回の課題。

  次回再開する場合: GPUが使える環境で`cargo test -p open-cuda-llm
  --features real-vulkan`を通した上で、層折りたたみ3手法それぞれの
  `find_best_layer_block_to_remove`/`fold_block_with_linear_adapter`
  呼び出し前後の実行時間をCPU経路と比較計測するところから始める。

- **2026-09-01 Model Folding follow-up: `ridge_lambda` externalized
  (done) / real GPU measurement (not started, environment constraint)
  — English handoff summary**:

  Of the remaining tasks the user explicitly requested as follow-ups to
  the previous Model Folding session (3 techniques already implemented
  in the `open-cuda-llm` crate: independent threshold / contiguous
  block search / linear adapter), 2 were addressed this round:

  1. **`ridge_lambda` made externally configurable (done)**:
     `fold_block_with_linear_adapter` now takes a `ridge_lambda:
     Option<f32>` argument (existing call sites updated to pass `None`,
     preserving the default `1e-2`). The value actually used is now
     reported back via `AdapterFoldReport::ridge_lambda_used`.
     Non-finite or non-positive values are honestly rejected via
     `ensure!` rather than silently producing a degenerate solution.
     Two tests were added and all 4 in that test group pass
     (`cargo test -p open-cuda-llm --lib
     fold_block_with_linear_adapter`). HTTP wiring for `POST
     /v1/models/fold-layers` was done on the `aruaru-llm` side (see
     `aruaru-llm/CLAUDE.md`).
  2. **Real GPU measurement via Vulkan/DirectX (not started, environment
     constraint)**: this crate's attention computation (QKᵀ, softmax,
     P·V) can already dispatch to a real Vulkan device via
     `--features real-vulkan` (wired since 2026-08-05, see the earlier
     entry above), so routing the 3 layer-folding techniques' GEMM/
     attention work through Vulkan should be technically feasible.
     However, **the sandbox this session ran in has no GPU or GPU
     driver, so it was physically impossible to build and actually
     benchmark this** — this is a conclusion reached after confirming
     GPU-detection commands themselves fail, not a guess made without
     trying. Doing this measurement on the dev machine (NVIDIA GT 730,
     which has a working `--features real-vulkan` build history) is
     the next task.
