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

## NPU層の再設計メモ(2026-09-21、実機3種での検証と一次資料調査に基づく)

ユーザー指示「スマホ版のNPUがあれば計算にも使用して」「open-cpu・open-directx・open-cuda・aruaru-llmを見直して開発し直して」を受け、
Android実機(OPPO Reno11 A / moto g53y 5G / arrows We2 Plus予定)とエミュレータで、NPU/GPU/CPUの実測を行った。

### 実測で分かった事実(誇張しない)
- **端末のNNAPI加速器は端末次第**。moto g53y(Snapdragon 480+, SM4350)は`nnapi-reference`(Android標準のCPU実装)しか無く、
  NNAPI経由のNPU/DSP利用は不可能(実機で列挙して確認、`NnapiProbe`)。QualcommはNNAPIのNPUドライバーを出していない。
- **以前の「NNAPIで3〜8倍速い」は(比較相手が悪く)誤認だった。ただしOPPOでは、正しい基準でも一括計算で6〜7倍の本物の加速が別途確認できた(下表)**。比較相手が素朴なKotlinのループで、速かった実体はTFLite自身のCPUカーネル
  (NNAPIなしでも同じ速度)。CPU側にJITウォームアップも無く、測定が偏っていた。以後は「NNAPIなしのTFLite CPU」を基準にし、
  加速器を名前指定して測り、1.3倍以上速く品質ゲートも通った場合だけ「効いている」と判定する(`MatVecSelector`)。
- 1組のコサイン類似度のような小さな計算は、NNAPI起動のオーバーヘッドで常にCPUが速い。行列×ベクトルのような大きな計算だけが対象。

### 一次資料(2026-09調査)
- **NNAPIはAndroid 15で非推奨**。現在の標準は**LiteRT(旧TFLite)のCompiledModel API + ベンダー別アクセラレーター**
  (Qualcomm AI Engine Direct/QNN: HTP v69/v73/v75/v79/v81、MediaTek NeuroPilot: Dimensity 7300/8300/9000/9200/9300/9400/9500、
  Google Tensor、Samsung Exynos、Intel OpenVINO)。Android API 31+、arm64のみ。ベンダーのランタイム(例: libQnnHtp.so)は
  Play Feature Delivery/PODAIで配布、モデルはAOT(端末SoC別に事前コンパイル)またはオンデバイスJIT。
  出典: https://developers.google.com/edge/litert/next/npu , https://developers.google.com/edge/litert/next/mediatek
- LiteRT 2.x(`com.google.ai.edge.litert:litert-api:2.2.0`)は`org.tensorflow.lite.nnapi.NnApiDelegate`を持たない
  → NNAPI(旧)とLiteRT(新)は同一アプリで併存できない。移行は「NNAPI維持のビルド」と「LiteRTビルド」を分けるか、段階的に切り替える。
- **「折りたたみ」の正式名は Model Folding**(Wang et al., ICLR 2025, arXiv:2502.10216)。似たニューロンをk-meansでまとめる
  データ不要・再学習不要のモデル圧縮(ResNet18/LLaMA-7Bで検証)。実装: https://github.com/nanguoyu/model-folding-universal
  → aruaru-llmの`idle_background_fold`(Model Folding準備)はこの手法に対応する名前であり、今後の本命候補。
- **DeepSeekが少ないGPUで動かす技術**: MLA(KVキャッシュの低ランク圧縮)、DeepSeekMoE(256エキスパート中8のみ活性)、
  FP8(128x128タイル単位スケール)、DualPipe(通信と計算の重ね合わせ)、Engram(2026年の条件付きメモリ、別論文)。
  出典: https://arxiv.org/abs/2412.19437 , https://github.com/deepseek-ai/FlashMLA
- 東芝SQBM+は組合せ最適化技術でLLM圧縮ではない(上記2026-07-22の結論のまま)。

### 端末別の見立て(検証待ち)
| 端末 | SoC | 見立て |
|---|---|---|
| moto g53y 5G | Snapdragon 480+ (SM4350) | NNAPI加速器なし(確定)。LiteRTのQNN対応(v69以降)にも該当しない見込み。GPU(Adreno 619)はVulkan Compute経由が本命 |
| OPPO Reno11 A | MediaTek Dimensity 7050 (MT6877V), APU 550 | **実機検証済み(2026-09-21)**: NNAPIに`mtk-neuron_shim`/`mtk-mdla_shim`/`mtk-dsp_shim`が実在。行列16384x768・64クエリ一括でfp16が**TFLite CPU(4スレッド)比 6〜7倍**(77.9ms→11.0ms、誤差3.6e-4、上位10件一致1.0)。1クエリなど小さい計算はCPUが速い(起動オーバーヘッド)。`mtk-dsp_shim`はCPUと同等でNPUの利得なし。int8は近似(誤差約2%)で、現状の実装ではfp16より遅い(ホスト側の量子化変換が律速) |
| arrows We2 Plus M06 | Snapdragon 7s Gen 2 | Hexagon NPUあり。LiteRT QNN対応(HTP世代)に該当するかは実機で確認が必要 |

### 見直しの方針(4リポジトリ)
1. **open-cpu**: x86専用だった検出を全アーキテクチャ(x86/aarch64)のインベントリへ拡張済み(`inventory()`)。次は、aarch64向けのNEON/dotprodカーネル。
2. **open-cuda**: `GpuDevice`(背骨)の外側に「アクセラレーター種別(CPU/GPU/NPU/DSP)と能力交渉」の層を設ける。NPUバックエンドは
   Rust側ではなくAndroid側(LiteRT/NNAPI)に置き、Rustからは結果を受け取る境界を定義する(NPUのAPIはJava/NDK側に閉じているため)。
3. **aruaru-llm**: `/v1/runtime`に加え、端末の診断結果(`HardwareReport`)を取り込み、推薦モデル・量子化方式(FP16/int8/int4)を端末能力で選ぶ。
4. **open-directx**: Windows側のDirectML NPU(Intel/AMD/Qualcomm NPU)を同じ能力交渉に載せる候補(未着手)。
5. 圧縮(Model Folding、int4/int8量子化、MoE型の部分ロード)は、実機のRAM/NPU能力に合わせて選ぶ。
