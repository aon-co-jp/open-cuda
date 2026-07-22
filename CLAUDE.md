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
    (`opencuda-bert`、本クレート内のattention実装・既存テスト)は
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
