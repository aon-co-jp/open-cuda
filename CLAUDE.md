# 開発方針＆開発環境ルール(open-cuda)

作業ドライブは`F:\runo`。この節は[`open-raid-z`](https://github.com/aon-co-jp/open-raid-z)の
`CLAUDE.md`を正本とし、各プロジェクトへコピーして同期する方針に準じる。
GitHubリポジトリ: [aon-co-jp/open-cuda](https://github.com/aon-co-jp/open-cuda)。

**開発開始日: 2026-06-26**(このリポジトリのGitHub作成日)

第二のCUDA。Windows＋MAC＋LINUX互換 ＆ INTEL＋AMD＋nVIDIA互換を開発途中です。

## このプロジェクトの役割

GPU抽象化・計算基盤(`OmniGPU`設計、詳細は`OmniGPU-Design.md`参照)。
`opencuda-core`/`opencuda-cpu`/`opencuda-vulkan`/`opencuda-blas`
(GEMM/Attention/量子化)・`opencuda-bert`(BERT系エンコーダのforward pass、
multilingual-e5-small対応)から成るCargoワークスペース。`aruaru-llm`との
SET構成(GPU/CPU実行パイプラインの実装先)。

前回のマーケティング調査(Python製AIライブラリのRust移植ランキング)で
言う「1〜6位の良いとこ取りハイブリッド/トライブリッド」の実体
——`opencuda-blas`がNumPy相当、`opencuda-bert`がTransformers推論パス
相当。今後`opencuda-llm`(vLLM相当、自己回帰デコーダ追加時)を追加予定。

## 詳細な設計・現状

- `OmniGPU-Design.md` — アーキテクチャ設計書(Vulkan Compute優先、
  段階的なCUDA/ROCm/oneAPI対応ロードマップ、正直な規模見積もり含む)。
- `DEVELOPMENT-NEXT.md` — 直近の実装状況(v0.4.1時点、matmul実Vulkan
  実行・INT4/INT8量子化等)。
- `CHANGELOG.md` — バージョン履歴。

## エコシステム全体マップ

同時並行開発の対象プロジェクト一覧・詳細は
[`open-raid-z`のCLAUDE.md](https://github.com/aon-co-jp/open-raid-z/blob/main/CLAUDE.md)
「関連プロジェクト」節を参照。主な関連リポジトリ:

- [aruaru-llm](https://github.com/aon-co-jp/aruaru-llm) — `open-cuda`の
  実装例(bag-of-words→`opencuda-bert`埋め込みベースの意図分類へ移行済み)
- [RGit](https://github.com/aon-co-jp/RGit)・[RJSON](https://github.com/aon-co-jp/RJSON) —
  Git forge・JSON処理(OTPログイン・アクセス制御パターンの先行実装)
- [RS-Chiketto](https://github.com/aon-co-jp/RS-Chiketto)・[RS-Blog](https://github.com/aon-co-jp/RS-Blog)・[RS-EC](https://github.com/aon-co-jp/RS-EC) —
  Redmine/WordPress/EC-CUBE相当(順次着手中、`RS-Chiketto`から)

## HANDOFF

- **2026-07-21 CLAUDE.md新規作成**: これまでREADME/DEVELOPMENT-NEXT.md
  のみでプロジェクト共通の開発方針ドキュメントが無かったため新設。
  併せて`opencuda-bert`(以前ローカルのみで未コミットだった)を
  ワークスペースへ正式追加・コミット・push済み(コミット`47f7837`)。
  - 次にすべきこと: (1) `opencuda-blas`のGPU専用パス(cuBLAS/rocBLAS/
    oneMKL/Vulkan汎用)の実装、(2) 真のFlash Attention、(3) `opencuda-llm`
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
