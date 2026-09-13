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
