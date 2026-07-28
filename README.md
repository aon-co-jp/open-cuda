# open-cuda

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
- **`opencuda-bert`**: BERT系エンコーダのforward pass
  (multilingual-e5-small対応)。
- **`opencuda-llm`**: vLLM相当(KVキャッシュ付き貪欲デコード)。GPT-2
  (Hugging Face `openai-community/gpt2`)の`safetensors`を読み込む
  `GptModel::load`実装済み(2026-07-25、`opencuda-bert::BertModel::load`
  と同じ設計)。実機でGPT-2 124Mの実重みをダウンロード・ロードし、
  ランダム初期化(意味を持たない出力)と比べて明確に流暢な英語の
  貪欲デコード継続("The quick brown fox" → "es are a great way to
  get a little bit of a")を確認済み——詳細は`CLAUDE.md`HANDOFF参照。

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
