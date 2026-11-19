# open-cuda お引越しガイド

> 🎯🕒 **移植時の前提(2026-08-29)**: aruaru-dbはRPoemとのSETで「REST
> API不要・Cosmo有料版互換」の価値が成立する(正本: aruaru-db/
> CLAUDE.md冒頭)が、**open-cudaは現時点ではGPU計算ライブラリでHTTP面が
> 無くこの方針の対象外**(実際に`grep`で確認済み)。**ただしこれは今
> この瞬間だけの暫定判断**——open-directxのDirectX互換実装が進み、
> OSレベル命令やハードウェアアクセラレーター(open-directx/open-cpu)
> 経由の動作がメリットを持つ場面が増えれば、この対象外判断を再評価
> すること。

> **2026-07-25 更新**: 開発方針ファイル(`CLAUDE.md`)の見出しを
> 「設計思想＆開発方針＆開発環境ルール」へ改名しました
> (設計思想・開発方針・開発環境ルールを明確に区別)。移設先でも
> `CLAUDE.md`の内容を必ず確認してください。


他プロジェクトへ`open-cuda`の設計パターンを移植する際の要点をまとめる。

## -1. `open-cuda-llm::DeepseekModel`(本物のDeepSeek-V2/V3 MLA、2026-09-13新設)

世界中の言語(英語・日本語・中国語)でのGoogle検索・GitHub調査
(2026-09-13)に基づき`deepseek_arch.rs`を新設した。移植時の要点:

- **実チェックポイントのフィールド名**(`deepseek-ai/
  DeepSeek-V2-Lite-Chat/config.json`実物で確認済み):
  `q_lora_rank`(小型モデルは`null`)、`kv_lora_rank`、
  `qk_nope_head_dim`、`qk_rope_head_dim`、`v_head_dim`。
  テンソル名: `q_a_proj`/`q_a_layernorm`/`q_b_proj`(query低ランク、
  `q_lora_rank=null`なら`q_proj`直結)、`kv_a_proj_with_mqa`/
  `kv_a_layernorm`/`kv_b_proj`(KV低ランク、MLAの核心)、`o_proj`。
- **forward順序**: down-proj→RMSNorm→up-projで各ヘッドを展開し、
  「RoPE無しnope部分」と「RoPE有りrope専用部分(KVは全ヘッドMQA的に
  共有)」に分割するdecoupled RoPE。詳細擬似コードは
  `deepseek_arch.rs`のモジュールdoc参照。
- **既存の共有Attentionヘルパーが使えない**: `q`/`k`次元
  (`qk_nope_head_dim+qk_rope_head_dim`)と`v`次元(`v_head_dim`)が
  非対称(実チェックポイントで192 vs 128)なため、
  `opencuda_blas::scaled_dot_product_attention`(単一head_dim前提)は
  再利用不可——素朴なCPUループでQKᵀ・softmax・P·Vを直接計算する必要が
  ある。
- **2026-09-13(続き)追記: MoEは実装済み**——上記の「意図的にスコープ外」
  は当初の判断で、その後ユーザー指示によりDeepSeekMoE(共有エキスパート+
  top-kルーティング)を実装した。テンソル名: `mlp.gate.weight`
  (ルーター)・`mlp.shared_experts.{gate_proj,up_proj,down_proj}`
  (常時計算)・`mlp.experts.{0..n}.{gate_proj,up_proj,down_proj}`
  (ルーティングされる個々のエキスパート)。ルーティング擬似コード:
  `scores=softmax(x@gate.T)` → top-k選択 → (`norm_topk_prob`なら
  再正規化) → `routed_scaling_factor`乗算 → 選択エキスパート出力を
  重み付き加算 → `shared_experts`出力を加算。推論専用実装のため
  auxiliary loss計算・expert-parallelism分散シャーディング等の学習
  専用ロジックは省略(DeepSeek公式推論実装にも存在しない)。
  **なお未対応のまま(誇張しない)**: aux-loss-free補正
  (`gate.e_score_correction_bias`、V3系)・group-limited routing
  (`n_group`/`topk_group`、V3系)・`scoring_func="sigmoid"`(V3系)——
  `load()`は`scoring_func != "softmax"`を明示的に拒否する。
  absorb最適化・YaRN RoPE・Attentionコア/MoEのvulkan/DXILディスパッチ
  も未対応。V2-Lite相当(`scoring_func="softmax"`・`n_group=1`)の
  構成は読める設計だが、実機ダウンロード検証(15.7Bパラメータ、この
  開発機では推論実行は不可能)はまだ行っていない。
- `aruaru-llm`側の連携(`deepseek_generation.rs`): Qwenの
  `QWEN_CATALOG`のような自動ダウンロードカタログは意図的に設けて
  いない(上記の理由で「ダウンロードすれば動く」と言えるリポジトリが
  無いため)。かわりにローカルディレクトリを直接指定する
  `POST /v1/deepseek/select { "dir": "..." }`を追加した。

## 0. `open-cuda-llm::GptModel`のModel Folding機能(2026-09-01新設)

`analyze_layer_redundancy`/`prune_redundant_layers`/
`find_best_layer_block_to_remove`/`remove_layer_block`/
`fold_block_with_linear_adapter`(`crates/open-cuda-llm/src/lib.rs`)
は、GPT-2互換の自己回帰デコーダを持つ他プロジェクトへもそのまま移植
できる設計(`GptModel`固有の内部構造〈`DecoderLayer`/`Linear`/
`LayerNorm`〉に依存するが、外部APIとしては`device: &Arc<dyn
GpuDevice>`+`sample_prompts: &[Vec<u32>]`のみを要求する)。移植先で
同じ機能が必要になった場合、この一式をそのままコピーすれば動作する
はず(利用側のHTTP配線は`aruaru-llm/src/generation.rs`・
`src/main.rs`が参考実装)。**「DeepSeekの折りたたみ理論」という技術は
実在しない**という調査結果と、実装した代替手法の詳細・実測結果は
`CLAUDE.md`の2026-09-01 HANDOFF追記に集約してある——移植先で同じ
依頼(「Model Foldingを実装して」「DeepSeekの折りたたみを実装して」)
を受けた場合、同じ調査をゼロからやり直す前に必ずこの記録を参照する
こと。

## 1. `GpuDevice`トレイト(バックエンド非依存の設計)

CUDA Runtime API相当の最小契約(`alloc`/`free`/`memcpy_h2d`/
`memcpy_d2h`/`memcpy_d2d`/`launch_kernel`/`synchronize`)+能力フラグ
(`supports_spirv`/`supports_dxil`、デフォルト`false`)。新しいハード
ウェアバックエンドを追加する際は、この契約を実装し、`KernelSource`
enumへ新しいバリアント(例: `Dxil(Vec<u8>)`)を**非破壊で追加**する
(既存バックエンドのコードは無変更のまま動き続ける)。

```rust
pub trait GpuDevice: Send + Sync {
    fn info(&self) -> &DeviceInfo;
    fn alloc(&self, bytes: usize) -> Result<DevicePtr>;
    fn free(&self, ptr: DevicePtr) -> Result<()>;
    fn memcpy_h2d(&self, dst: DevicePtr, src: &[u8]) -> Result<()>;
    fn memcpy_d2h(&self, dst: &mut [u8], src: DevicePtr) -> Result<()>;
    fn memcpy_d2d(&self, dst: DevicePtr, src: DevicePtr, bytes: usize) -> Result<()>;
    fn launch_kernel(&self, kernel: &CompiledKernel, cfg: &LaunchConfig, args: &[KernelArg]) -> Result<()>;
    fn synchronize(&self) -> Result<()>;
    fn supports_spirv(&self) -> bool { false }
    fn supports_dxil(&self) -> bool { false }
}
```

## 2. 「モック→実機」の2段階実装パターン(移植先でも踏襲推奨)

新しいGPUバックエンドを追加する際、いきなり実機実装から始めない。

1. **Phase 1: モックデバイス**(ハードウェア無しで動く、GPUなしの
   CI環境でも契約〈カーネルソース種別の受理・拒否〉を検証できる)。
2. **Phase 1.5〜2: 実機実装**(`real-vulkan`/`real-dx12`のような
   Cargo featureで隔離、既定オフ)。実機が無い環境では自動スキップ
   する(`eprintln!`でスキップ理由を表示、テストを偽装しない)。

`opencuda-vulkan`(`VulkanMockDevice`→`real::VulkanDevice`)、
`opencuda-directx`(`DirectXMockDevice`→`real::DirectXDevice`)いずれも
この構成。

## 3. HLSL cbufferの配列パディングの罠(DirectX/HLSLを使う移植先すべてに該当)

`cbuffer`内で`uint key[8]`のようなスカラー配列を宣言すると、**各要素が
16バイト境界へパディングされる**(`float weights[3]`が3×16=48バイトを
占める、というよく知られたHLSLの罠)。Rust側で`SetComputeRoot32BitConstant`
により隙間なく詰めたdword列を渡す設計と組み合わせると、HLSL側が読む
バイトオフセットとズレ、値が実質ゼロになる——GPU暗号化カーネルの実装で
「出力が暗号化されず平文のまま返る」という形で実際に発覚した
(`opencuda-directx`のChaCha20カーネル、コミット`ec6acf1`)。

**回避策**: cbuffer内では配列宣言を避け、`key0`〜`key7`のような個別
スカラーフィールドとして宣言する(密なレイアウトになりRust/C++側の
詰め込みと一致する)。

```hlsl
// NG: 各要素が16バイトにパディングされる
cbuffer Constants : register(b0) { uint key[8]; };

// OK: 密なレイアウト
cbuffer Constants : register(b0) {
    uint key0; uint key1; uint key2; uint key3;
    uint key4; uint key5; uint key6; uint key7;
};
```

## 4. ルートシグネチャのHLSL埋め込み(DirectX 12移植時の簡略化テクニック)

`[RootSignature("UAV(u0), UAV(u1), RootConstants(num32BitConstants=N, b0)")]`
属性をHLSLシェーダー自体に書くと、`dxc`コンパイル時にルートシグネチャが
DXILバイト列へ同梱される。Rust側は`ID3D12Device::CreateRootSignature`
へそのDXILバイト列をそのまま渡すだけでよく、C++/Rust側で手動の
ルートシグネチャ記述子構築が不要になる。またディスクリプタヒープを
経由せず`SetComputeRootUnorderedAccessView`でUAVバッファを直接ルート
ディスクリプタとしてバインドすれば、ディスクリプタヒープ管理という
別のバグの温床を避けられる(`opencuda-directx`で採用した設計)。

## 5. DXGIアダプタ列挙によるベンダー判定

`D3D12CreateDevice(None, ...)`(アダプタ未指定)はOS既定のアダプタを
選ぶだけで、ベンダー名やVRAM容量は取得できない。実際のベンダー情報
(NVIDIA=0x10DE/AMD=0x1002・0x1022/Intel=0x8086のPCIeベンダーID)を
得るには`IDXGIFactory1::EnumAdapters1(0)`→`DXGI_ADAPTER_DESC1`を
経由し、取得したアダプタハンドルをそのまま`D3D12CreateDevice`へ渡す。
DXGI列挙が失敗しても`None`パス(OS既定選択)へ安全にフォールバックする
設計にすること(付加情報であり必須要件ではないため)。

## 6. GPU圧縮/暗号化を検討する際の正直な注意

小サイズペイロード(ネットワークMTU程度、数百〜数千バイト)では、
Host↔Device間の転送オーバーヘッドがGPU側の演算優位性を相殺し、実利益が
出ない可能性がある。GPU暗号化カーネルを移植・統合する前に、対象
ペイロードサイズでの実ベンチマークを取ってから判断すること
(`RS-LinkFusion`側`accel.rs`統合時に判明した懸念、詳細は同リポジトリの
CLAUDE.md参照)。

## 7. RAID6パリティ計算カーネルの移植パターン(2026-07-30追加)

`opencuda-vulkan`の`raid6_xor_parity`/`raid6_q_parity`カーネルは、可変本数の
データディスクを「1本の連結バッファ」としてバインドする設計(個別バッファ
本数をシェーダの固定バインディング数に依存させない)。他プロジェクトで
同様の「N個の入力を1カーネルで処理したい」場面があれば、この連結バッファ
方式を踏襲すると良い。Q-parity(Reed-Solomon)のGF(2^8)乗算は
`gf_mul`関数(Russian peasant乗算、既約多項式`0x11D`)としてシェーダ内に
自己完結しており、他言語(HLSL等)への移植もアルゴリズムをそのまま
書き写せる。

## 8. 64bit整数型に依存しないGPU実装パターン(2026-07-30追加、Poly1305)

DXIL SM6.0でも64bit整数演算(`uint64_t`)はオプション機能
(Int64ShaderOps)で、旧世代GPUでの対応可否が不明な場合がある。
`opencuda-directx`のPoly1305実装(`shaders/poly1305.hlsl`)は、32bit×32bit
→64bit(hi,lo)ペア乗算(`umul32`)・64bit加算(`uadd64`)・64bit右シフト
(`ushr64_lo`)を32bit整数演算のみで自前実装することでこの制約を回避した。
64bit整数の対応可否が不明なターゲットへ暗号/大整数演算を移植する場合の
パターンとして参考にできる。

## 9. DeepSeek-V3のMLA風の低ランクKVキャッシュ圧縮(2026-08-06追加)

`opencuda-blas::mla_compress_kv`/`mla_decompress_kv`は、既存の実機検証
済み`sgemm`(CPU/Vulkan両対応)を土台に、down-projection(`d_h→d_c`)/
up-projection(`d_c→d_h`)という低ランク射影を実装したもの。
`open-cuda-llm::GptModel::enable_mla_kv_compression(d_c, seed)`で
オプトイン的にKVキャッシュ経路へ配線済み(既定は従来通りフル精度、
後方互換)。**正直な開示**: 射影行列はランダム初期化のみで学習済み
重みを持たないため、圧縮は非可逆(生成品質を保持しない)——この配線が
実証するのは「計算経路が`generate()`まで正しく繋がっていること」で
あり、「DeepSeek実運用の圧縮品質を再現すること」ではない。他プロジェクト
へ移植する場合も、学習済み射影重みを別途用意しない限り同じ限界が
付随する点に注意。

## 繰り返しペナルティ(`GptModel::generate_with_repetition_penalty`、2026-08-10新設)

対話ファインチューニング無しの素のGPT-2貪欲デコードが同一文字列を無限
ループする既知の劣化モードへの対応。`open-cuda-llm::GptModel`に、既に
登場したトークン(プロンプト+生成済み)のlogitへCTRL方式のペナルティ
(`logit>0`なら`/penalty`、`logit<=0`なら`*penalty`)を適用してから
argmaxする`generate_with_repetition_penalty(device, prompt_ids,
max_new_tokens, penalty)`を追加した。既存の`generate()`は`penalty=1.0`
で呼ぶ薄いラッパー(`penalty==1.0`なら早期returnし一切のlogit変更を
行わないため、既存呼び出し元の挙動は完全に無変更)。

移植手順:
1. 呼び出し側を`generate()`から`generate_with_repetition_penalty(...,
   penalty)`へ切り替える(`penalty=1.0`のままなら挙動は変わらない)。
2. 経験的な既定値は`1.3`(`aruaru-llm`側の実測、`open-english`と同じ
   プロンプト構造での実GPT-2 124M重み検証に基づく)——プロンプト・
   ユースケースが異なる場合は再調整が必要な点に注意。
3. サンプリング(温度・top-k/top-p)は組み合わせていない(貪欲デコード+
   繰り返しペナルティのみ)。

## F16/F64 GPUディスパッチの実装パターン(`hgemm`/`dgemm`、2026-09-05新設)

要素サイズがf32(4バイト)と異なる精度をVulkan Computeでディスパッチする
際の2つの異なる移植パターン:

1. **F16(half): 2要素パッキング方式**(`crates/opencuda-blas/shaders/
   hgemm.comp`相当は`examples/hgemm_vulkan_real/shaders/hgemm.comp`
   参照): GLSLの`unpackHalf2x16`/`packHalf2x16`(core、追加のVulkan
   拡張不要)で、half 2要素を1つのuintへパックしたバッファを読み書き
   する。1スレッドが出力ワード1つ(連続する2要素分)をまとめて計算・
   書き込みすることで、別スレッドが同じワードの別半分を同時に書く
   データ競合を避ける設計——このため呼び出し側は`k`・`n`が偶数である
   ことを保証する必要がある(奇数なら明示的エラー、黙って誤った結果を
   返さない)。シェーダ内部の演算自体はf32(ネイティブFP16 ALUが無い
   前提のソフトウェア変換)。
2. **F64(double): ネイティブ型方式**(`examples/dgemm_vulkan_real/
   shaders/dgemm.comp`): GLSLの`double`型はstd430で8バイト/要素の
   単純配列としてそのままバインドできるため、F16のようなパッキングは
   不要——`matmul.comp`のf32版と全く同じ構造で要素サイズだけ変える
   だけで済む。ただし実行には**物理デバイス/ドライバの`shaderFloat64`
   機能サポートが必須**(`GpuDevice::supports_f64_shader()`、新設の
   能力フラグ)。`VulkanDevice::new`が論理デバイス作成前に
   `vkGetPhysicalDeviceFeatures`を問い合わせ、対応している場合のみ
   `enabled_features`でこの機能を有効化する(未対応デバイスへ
   無条件で要求すると`vkCreateDevice`自体が失敗するため)。

**移植時の判断基準**: 対象の精度がGLSL/SPIR-Vのネイティブ型として
存在する(`double`・`bool`等)場合はパターン2(ネイティブ型+能力フラグ
確認)を、存在しない(`half`はGLSL 4.20以降core型として存在するが
`unpackHalf2x16`系関数はf32算術ベースであり、真のFP16 ALU実行では
ない)場合や、対象GPUでの拡張サポートが不確実な場合はパターン1
(パッキング+ソフトウェア変換)を検討するとよい。

**正直な開示**: このマシン(NVIDIA GeForce GT 730、Kepler世代)は
`shaderFloat64`を実際にサポートしていた(意外な発見、事前の想定は
「コンシューマGPUはFP64が遅い」という速度の話であり、機能対応可否
とは別問題だった)。他のベンダー・世代での対応状況は未検証。F128は
GPUハードウェアが原理的に存在しないため、この2パターンいずれも適用
対象外(恒久的にCPU参照実装のみ)。

## 現状(2026-07-30)

`opencuda-core`/`opencuda-cpu`/`opencuda-vulkan`/`opencuda-directx`/
`opencuda-blas`/`open-cuda-bert`/`open-cuda-llm`から成るCargoワークスペース。
`opencuda-directx`はPhase 2まで実装済み(vector_add/matmul/ChaCha20/
Poly1305の実機ディスパッチ)。`opencuda-vulkan`にRAID6 P-parity(XOR)/
Q-parity(Reed-Solomon)カーネルを追加、実機検証済み。詳細な到達状況は
`CLAUDE.md`のHANDOFF節を参照。

## `chain_n_buffer`汎用Nバッファディスパッチを`opencuda-vulkan`へ追加(2026-09-12、open-directx連携)

`open-directx`側(`directx-shader-translate`)がDXBC→SPIR-V翻訳する
「RegExprチェーン」カーネル(`yuv444_to_g`のような4バッファ以上を
読み書きするもの、および今後のMED予測器〈left/top/topleft/output〉の
ような固定本数ではないバッファ数のカーネル)を実Vulkanで検証しようと
した際、`VulkanDevice::launch_kernel`が`"vector_add"`/`"matmul"`等
カーネル名ごとにバッファ本数を決め打ちでディスパッチする実装だった
ため、4バッファ以上のカーネルは構造検証(SPIR-Vの形のみ確認)止まりに
なっていた(`open-directx/PORTING.md`に記録済みの既知の制約)。

内部で実際にVulkanディスクリプタ/コマンドバッファを組み立てる
`dispatch_spirv`関数自体は元々`buffers: &[vk::Buffer]`という可変長
引数を受け取る、バッファ本数に汎用対応した実装だった(呼び出し側の
公開APIだけがカーネル名ごとに固定本数へ絞り込んでいた)。そのため
今回追加したのは、この既存の汎用性を実際に引き出す薄い公開エント
リポイントのみ:

- `ensure_chain_n_buffer_args`: 引数を「`KernelArg::Ptr`がN個
  (呼び出し側がSPIR-Vのbinding順に並べる)+最後に`KernelArg::Usize(n)`
  (要素数)」という契約で検証し、Vulkanバッファハンドルの`Vec`を返す
  (最低2引数=バッファ1本+nのみ検証、上限本数は決め打ちにしない)。
- `run_chain_n_buffer_spirv`: 上記を`dispatch_spirv`へそのまま渡す。
- `VulkanDevice::launch_kernel`のカーネル名ディスパッチに
  `"chain_n_buffer"`/`"chain_n_buffer_f32"`を追加。

`vector_add`/`matmul`等の既存カーネル名の挙動・引数契約は一切変更
していない(完全に加算的な変更)。`cargo test -p opencuda-vulkan
--features real-vulkan`: 既存テストに回帰無し。

**実際の効果**: `open-directx`側の`yuv444_to_g_real_vulkan.rs`
テストを、この新APIを使う形へ書き換えたところ、以前は構造検証止まり
だったGチャンネル(4バッファ)カーネルが**実GT730ハードウェアで
256/256要素の数値一致でpassするようになった**(詳細は
`open-directx/PORTING.md`「FFv1 step 1」以降の節を参照)。

**Honest scope (English)**: this is a purely additive change — a new
kernel-name dispatch path (`chain_n_buffer`/`chain_n_buffer_f32`) that
exposes the buffer-count-generic `dispatch_spirv` internals through a
thin public entry point. No existing kernel name's behavior or argument
contract changed. This resolves the "generic N-buffer dispatch" item
that `open-directx/PORTING.md` had recorded as its top-priority next
step for real-GPU-verifying chain kernels with more than 3 buffers.

## `QwenModel`へPCA較正版MLA風KVキャッシュ圧縮を移植(2026-09-13追加)

セクション9(`GptModel`のMLA風低ランクKVキャッシュ圧縮)の続き。
`QwenModel::enable_mla_kv_compression`(ランダム射影版)は既に実装
済みだったが、そのdocコメント自身が「PCA較正版の`QwenModel`移植は
今回のスコープ外(次の増分)」と明記していた——今回その次の増分を
実施した。

`GptModel::enable_mla_kv_compression_calibrated`(2026-08-08新設、
非中心PCAで実際のK/V活性化統計から射影を較正する版)と同じ設計
(`lib.rs`の`pca_top_directions`をそのまま再利用、直交基底のため
`up_proj=down_proj`の転置)を、GQA(クエリヘッド数よりKVヘッド数が
少ない場合がある)に対応する形で`QwenModel::enable_mla_kv_compression_
calibrated`として移植した。`GptModel`版は`forward_prefill_all_layers`
(バッチ処理)で較正データを集めるが、`QwenModel`にはまだその一括版が
無いため`forward_step`をプロンプトのトークン数ぶん逐次呼び出す形に
した(結果は数学的に同じ、実行効率のみの違い)。

新規テスト2本(`QwenConfig::tiny`の合成ランダム重みで検証——実学習済み
Qwen重みは本セッションでは未使用、`GptModel`版の`calibrated_pca_mla_
kv_compression_on_real_gpt2_weights`が実GPT-2重み任意配置時のみ実行
される既存の枠組みと同じ制約): 較正が成功し生成まで完走すること、
および無効な入力(`d_c>=head_dim`、空プロンプト列、較正データ不足、
既圧縮モデルへの再較正)を正しく拒否すること。

`cargo test -p open-cuda-llm`: 全緑(57件、up from 55)。`cargo clippy`:
`open-cuda-llm`自体はクリーン(依存クレート`opencuda-vulkan`に既存の
無関係な`chunks_exact`系lint3件があるのみ、このセッションでは変更
していない)。

**正直な開示(このセクション9系全体に通底する制約、再確認)**: これは
DeepSeek-V2/V3の実際のMLA(学習時から低ランク射影+decoupled RoPEを
組み込んだアーキテクチャ、実チェックポイントの`kv_a_proj_with_mqa`/
`kv_b_proj`/`q_a_proj`/`q_b_proj`等の専用テンソルが必要)ではない——
**「MLA風」**、つまり既に標準Attentionで学習済みのモデルへ事後的に
低ランクKVキャッシュ圧縮を後付けする手法である。ユーザーから
「DeepSeekのMLA実装」という依頼があった際の候補としては、本当の
DeepSeek-V2/V3チェックポイントを読み込む新規アーキテクチャモジュール
(`qwen_arch.rs`と同じパターンの`deepseek_arch.rs`)を追加する方が
本来の意味での「MLA実装」に近いが、これは重みローダー・KVキャッシュ
構造・GPU側matmulディスパッチにまたがる大きな新規アーキテクチャ追加
であり、今回はセクション9系の既存トラジェクトリ(既存コードが自ら
「次の増分」と明記していた具体的なタスク)を優先して完了させた。
`deepseek_arch.rs`の新設(実チェックポイント対応)は、別途まとまった
セッションとして今後の課題に残す。

**関連する既発見(混同の解消、参考)**: `open-cuda-llm::GptModel::
analyze_layer_redundancy`(2026-09-01新設)のモジュールdocに、
「DeepSeekの折りたたみ理論」という依頼を受けて調査した結果**「DeepSeekの
foldingという技術は実在しない」**(DeepSeekの実際の効率化技術はMLA・
FP8混合精度・DeepSeekMoEであり、「折りたたみ」ではない;混同の元は
無関係の「Model Folding」論文〈ICLR 2025、Wang et al.〉)という結論が
既に記録されている——今回のセッションで同じ疑問が改めて挙がったため、
既存の調査結果をここに再度リンクして参照しやすくしておく。

## 追記(2026-09-21): NPU/モバイルGPUの移植メモ / NPU and mobile-GPU porting notes

**日本語**: モバイルNPUは端末ごとに経路が違う: MediaTek=NNAPIの`mtk-*_shim`(実機OPPOで確認)、Qualcomm=NNAPIドライバー無し(LiteRTのQNN/HTP、v69以降)、Tensor/Exynos=各ベンダー。RustのGpuDevice抽象にNPUを直接載せず、Android側(LiteRT/NNAPI)で実行して結果を受ける境界を定義する方針。モバイルGPUはVulkan Compute(実機: Adreno 619 / Mali-G68 MC4がVulkan対応と確認)で統合済みの設計を維持。詳細は`DEVELOPMENT-NEXT.md`。

**English**: Mobile NPU paths differ per device: MediaTek = NNAPI `mtk-*_shim` (verified on OPPO), Qualcomm = no NNAPI driver (LiteRT QNN/HTP, v69+), Tensor/Exynos = vendor stacks. Rather than putting NPUs directly under the Rust `GpuDevice` abstraction, run them on the Android side (LiteRT/NNAPI) and define a result boundary. Mobile GPUs stay on Vulkan Compute (Adreno 619 and Mali-G68 MC4 confirmed Vulkan-capable). See `DEVELOPMENT-NEXT.md`.

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

## JSON handling policy: use `parse_secure_sonic` (fast type + secure) and `sonic-rs` everywhere they can be used (2026-10-14, user instruction)
The source of truth is [`RS-JSON`](https://github.com/aon-co-jp/RS-JSON) (`README.md`, `PORTING.md`).

**Policy**: wherever JSON is read, use `parse_secure_sonic` from `rust-json` (RS-JSON, the `fast` feature; the result is `sonic_rs::Value`) **everywhere it can be used**.
Use `sonic-rs` directly where possible too (e.g. reading into typed structs). Where it cannot be used, use `parse_secure` (result: `serde_json::Value`).
Untrusted input from outside (HTTP bodies, files over the network, uploads) **must always be read with size / depth / duplicate-key limits**.

**Measured (7 MB, 40,000 records)**: `serde_json` 200 ms / `parse_secure` 217 ms / **`parse_secure_sonic` 83 ms (about 2.4x faster, with the defences)** / raw `sonic-rs` 33 ms (about 6x, no defences).
Reading into `serde_json::Value` gives almost no speed-up even with SIMD (188 ms). **The speed comes from the value type that packs the result into contiguous memory.**

**Where it can / cannot be used**
- ✅ Read-heavy code that parses large or frequent JSON (cache JSON, config, API responses, logs, indexes).
- ✅ Entrances for outside input (the goal there is defence rather than speed; `parse_secure` is fine).
- ⚠️ Existing code that passes `serde_json::Value` between functions needs its type changed to `sonic_rs::Value` (methods from `JsonValueTrait` / `JsonContainerTrait`). Reading into `serde_json::Value` without that change is not faster.
- ❌ WASM (`rust_json::light`, zero dependencies) and the JSON handling on the PHP / JavaScript side (Rust components cannot be used there).
- Note: the `sonic-rs` dependency uses `unsafe` internally (our own code has none). For small JSON (a few KB) the difference is not noticeable; do speed-motivated rewrites where measurements show a benefit.

**Adoption**: in `Cargo.toml`, `rust-json = { path = "../RS-JSON", features = ["fast"] }`. Example in `RS-JSON/README.md`.

**Status in this repository**: Not yet applied (policy only). Code will be switched over place by place, starting where measurements show large or frequent JSON reads.

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

## JSON の読み込み(2026-11-06)
JSON を読む所は、設定ファイル(`config.json`)・safetensors のインデックスはすでに `sonic_rs`。残っていた safetensors のヘッダー(`deepseek_arch::read_safetensors_header`)を `serde_json::from_slice` から `sonic_rs::from_slice` に替え、ヘッダー長の欄で巨大な領域を確保しないよう上限(256 MiB)を追加(壊れた/悪意あるファイル対策)。モデルのファイルはローカルの信頼できる入力なので `rust-json` への依存は足していない(リポジトリ間の Cargo 依存を避ける方針)。試験を2件追加、`cargo test -p open-cuda-llm --lib` 全通過(75件)。

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
