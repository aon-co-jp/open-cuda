# open-cuda お引越しガイド

> **2026-07-25 更新**: 開発方針ファイル(`CLAUDE.md`)の見出しを
> 「設計思想＆開発方針＆開発環境ルール」へ改名しました
> (設計思想・開発方針・開発環境ルールを明確に区別)。移設先でも
> `CLAUDE.md`の内容を必ず確認してください。


他プロジェクトへ`open-cuda`の設計パターンを移植する際の要点をまとめる。

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

## 現状(2026-07-30)

`opencuda-core`/`opencuda-cpu`/`opencuda-vulkan`/`opencuda-directx`/
`opencuda-blas`/`open-cuda-bert`/`open-cuda-llm`から成るCargoワークスペース。
`opencuda-directx`はPhase 2まで実装済み(vector_add/matmul/ChaCha20/
Poly1305の実機ディスパッチ)。`opencuda-vulkan`にRAID6 P-parity(XOR)/
Q-parity(Reed-Solomon)カーネルを追加、実機検証済み。詳細な到達状況は
`CLAUDE.md`のHANDOFF節を参照。
