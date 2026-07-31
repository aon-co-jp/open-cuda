# DEVELOPMENT NEXT

## v0.4.1 現在地(未リリース)

`opencuda-blas`にINT4/INT8量子化(`quantize_int4`/`quantize_int8`)を実装した。
グループ単位の対称量子化で、要素ごとの量子化演算は`GpuDevice::launch_kernel`
経由の実カーネルディスパッチ(CPUバックエンドでは`rayon`並列)、INT4のニブル
パッキング(2値/バイト)はバイト共有の書き込み競合を避けるためホスト側で行う。
`dequantize_int4`/`dequantize_int8`による逆変換も実装し、ラウンドトリップ誤差・
奇数長パディング・全ゼログループ・グループ境界・INT4よりINT8が高精度なこと等を
検証する単体テスト8件を追加(`opencuda-blas`のテストは計14件、全green)。
`cargo clippy --workspace --all-targets`は引き続き警告0件。

### 正直な制限

- `quantize_int4`/`quantize_int8`はCPUバックエンド上の実装のみ。GPU側
  (Vulkan/CUDA/ROCm)カーネルとしての量子化はまだ無い。
- aruaru-llmの`scoring.rs`はまだこの量子化APIを使っていない(bag-of-words
  ドット積スコアリングのまま)。

## v0.4.0 現在地

v0.4.0 では、ローカルLLM推論の中核演算である matmul を実Vulkanで動かした。

- `opencuda-vulkan::real::VulkanDevice` が `matmul`/`matmul_f32` カーネル(SPIR-V)を実行できるようになった。A(M×K)・B(K×N)・C(M×N) は行優先(row-major)、(m, k, n) は push constantで渡す。
- `examples/matmul_vulkan_real` で CPUバックエンド(rayon naive matmul)と実Vulkan Compute(naive matmul shader)を同じ入力行列で実行し、ホスト側リファレンス値・CPU結果・Vulkan結果の3者が一致することを確認する経路を追加。
- `vector_add`/`matmul` のVulkanディスパッチ処理を `dispatch_spirv` に共通化。
- `cargo clippy --workspace --all-targets` は引き続き警告0件を維持。
- このセッションでは実Vulkan環境（NVIDIA GeForce GT 730）が利用可能で、64×64行列のmatmulをCPU/Vulkan両方で実行し、誤差1e-3以内で一致することを実機確認できた。
- 性能計測はまだ行っていない(naive実装、タイリング等の最適化は未着手)。

## 次の候補

### v0.4.1 以降の候補(未着手)

- matmulのタイリング/共有メモリ最適化(性能改善、正確性を保ったまま)。
- より大きな行列サイズでの検証(現状は64×64のみ)、非正方行列・非16の倍数サイズの境界条件テスト。
- Flash Attention や量子化(INT4/INT8)など、OmniGPU-Design.md の `omnigpu-blas` ロードマップにある次のAIカーネルの検討。
- `omnigpu-ir`(OmniIR)経由でVulkan matmulを起動する経路(現状はCPU native lowerのみ`vector_add_f32`対応、matmulは未対応)。

## リサーチノート: GPU1枚運用に向けた圧縮技術の調査(2026-07-22)

ユーザーから「疑似量子コンピューター技術・折り畳み理論でGPU1枚運用を
再現する」という着想の元ネタ調査依頼を受けて実施(WebSearch使用)。
`opencuda-blas`/`open-cuda-llm`/`aruaru-llm`への応用可能性まで含めて
整理する。**誇張を避けるため、事実関係を先に正直に切り分ける**。

### 事実確認: 東芝の「疑似量子コンピューター」はLLM圧縮技術ではない

2026年4月に東芝が発表した第3世代シミュレーテッド分岐(SB)アルゴリズム
(SQBM+)は、GPU/FPGA上で量子アニーリングの原理を模倣し**組合せ最適化
問題**(創薬候補探索・配送ルート・ポートフォリオ設計等の離散最適化)を
高速に解く技術で、第2世代比100倍高速・成功確率が数%→約100%に向上した
というもの([東芝公式発表](https://www.global.toshiba/ww/technology/corporate/rdc/rd/topics/26/2604-01.html)、
[日経記事](https://www.nikkei.com/article/DGXZQOUC0640B0W6A400C2000000/))。
富士通デジタルアニーラ・NECベクトルアニーリングと同系統の「量子
インスパイアード」技術。**「富士通の100倍」という報道は量子アニーリング
ハードウェア対比の話であり、LLMをGPU1枚に圧縮する技術ではない**。
Transformerの推論・学習とは別分野なので、このまま`open-cuda-llm`へ
転用できる技術ではないと判断する(現時点で不採用)。

### 実在する近縁技術: CompactifAI(量子インスパイアード・テンソルネットワーク圧縮)

ユーザーの言う「折り畳み理論」に対応しうる実在研究として、Multiverse
Computing社の**CompactifAI**がある。LLMの重み行列をテンソルネットワーク
(MPO: Matrix Product Operator)分解で圧縮する手法で、ニューロン数や
精度を直接削るのではなく層間の相関構造を圧縮する
([AI Business Review解説記事](https://aibr.jp/2025/03/17/%E5%A4%A7%E8%A6%8F%E6%A8%A1%E8%A8%80%E8%AA%9E%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AE%E6%A5%B5%E9%99%90%E5%9C%A7%E7%B8%AE%EF%BC%9A%E9%87%8F%E5%AD%90%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%91%E3%82%A4%E3%82%A2/))。
量子コンピューティング研究由来のテンソルネットワーク数学をLLM圧縮に
転用した2025年の研究で、「疑似量子×折り畳み×圧縮」という組み合わせに
実際に対応する。ただしMPO分解後の再学習(fine-tune)が必要で実装コストが
高く、本ワークスペースでの検証は小型モデルからの実験段階が必要。

### 2026年時点で実用段階にある単一GPU技術

- **AWQ/GPTQ/GGUF系の活性化考慮量子化**: FP16比1/4のVRAMで70B級モデルを
  動かす主流手法([BIZON](https://bizon-tech.com/blog/best-gpu-llm-training-inference)、
  [MarkTechPost 2026-07-19](https://www.marktechpost.com/2026/07/19/best-local-llms-you-can-run-on-a-single-24gb-gpu-in-2026-qwen-gemma-mistral-deepseek-compared/))。
- **PowerInfer**: ニューロン活性化のべき乗則局所性を利用し、高頻度
  活性ニューロンをGPU常駐・低頻度分をCPUオフロードすることで消費者
  GPU1枚上で最大11倍高速化([arXiv:2312.12456](https://arxiv.org/pdf/2312.12456))。

### 応用の優先順位(採用、2026-07-22決定)

ユーザーとの合意により、以下の順で着手する:

1. **①AWQ改良**: 既存の`opencuda-blas::quantize_int4`/`quantize_int8`
   (対称・グループ単位)を、活性化統計に基づいて重要チャネルを高精度
   のまま残す非対称・活性化考慮型の量子化へ拡張する。既存クレートの
   直接拡張で着手コストが最も低い。
2. **②PowerInfer型オフロード**: `open-cuda-llm`(KVキャッシュ付き
   GPT系デコーダMVP、2026-07-22着手)と`opencuda-cpu`が既に両方
   存在するため、ニューロン活性化頻度を追跡して高頻度層をGPU常駐・
   低頻度層をCPUオフロードする戦略をKVキャッシュ実装と並行して
   組み込める。
3. **③CompactifAI型テンソルネットワーク圧縮の実験実装**: MPO分解+
   再学習が必要な研究レベルの手法のため、小型モデルでの精度劣化計測
   から始める実験クレート(`opencuda-compress`案)として、①②より
   後回しにする。

**東芝SBM型シミュレーテッド分岐ソルバーは今回は保留**: 話題性は
あるが、上記の通りLLM圧縮という目的に直結しないため。将来的に
「量子化ビット配分の最適化」や「構造化プルーニングのマスク探索」
といった離散最適化サブ問題への応用余地はあるが、投機的な研究アイデア
の域を出ず、現時点では①②③より優先度を下げる。
