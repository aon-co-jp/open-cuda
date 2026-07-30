# 設計思想＆開発方針＆開発環境ルール(open-cuda)

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
  は既存の`opencuda-blas`/`opencuda-bert`(ML専用、GEMM/Attention/
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
  実装例(bag-of-words→`opencuda-bert`埋め込みベースの意図分類へ移行済み)
- [RS-Git](https://github.com/aon-co-jp/RS-Git)・[RJSON](https://github.com/aon-co-jp/RJSON) —
  Git forge・JSON処理(OTPログイン・アクセス制御パターンの先行実装)
- [RS-Chiketto](https://github.com/aon-co-jp/RS-Chiketto)・[RS-Blog](https://github.com/aon-co-jp/RS-Blog)・[RS-EC](https://github.com/aon-co-jp/RS-EC) —
  Redmine/WordPress/EC-CUBE相当(順次着手中、`RS-Chiketto`から)

## HANDOFF

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

- **2026-07-25 `opencuda-llm`にsafetensorsローダー追加(実GPT-2重みで検証)
  + `needless_range_loop`警告2件解消**: 前回HANDOFFの「次にすべきこと」
  (1)(2)に着手。
  1. **safetensorsローダー**: `opencuda-bert::BertModel::load`と同じ設計
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
     (`crates/opencuda-llm/models/gpt2/`、`.gitignore`対象、リポジトリ
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
     ベースのトークナイザが必要だったため、`opencuda-bert::BertTokenizer`
     と同じ設計で`tokenizers`クレートによる`GptTokenizer`(`tokenizer.json`
     読み込み)を追加した。**正直な開示**: `ByteTokenizer`は既定のまま
     残しており(既存4テストは変更無し、後方互換)、実重みと
     `ByteTokenizer`を組み合わせても意味のある出力は得られない
     (GPT-2のBPE語彙IDとは無関係なため)——ドキュメントに明記済み。
  6. **`needless_range_loop`警告2件の解消**: `DecoderLayer::forward_step`の
     ヘッドループ(`enumerate().take(num_heads)`へ)、`load_fused_qkv`の
     QKV分割ループ(`chunks_exact`/`chunks_exact_mut`のzipへ)を
     イテレータベースへ書き換え。
  7. **検証結果**: `cargo build -p opencuda-llm --release`警告0件、
     `cargo test -p opencuda-llm --release`**6件全green**(既存4件+
     新規2件〈合成safetensors検証・実GPT-2重み検証〉、実機で実際に
     GPT-2重みをロード・生成し記録)。`cargo test --workspace --release`
     も全クレートでregression無し(全て`test result: ok`)。
     `cargo clippy --workspace --all-targets --release`**警告0件**
     (対象2件を含め全て解消)。
  - 次にすべきこと: (1) 本家vLLMの核心的最適化(PagedAttention・連続
    バッチング)への着手検討、(2) 残り4目標(PyTorch互換/scikit-learn/
    Whisper相当)のうち次に着手するものの選定(前回HANDOFFの推奨
    〈Whisper相当〉のまま)。

- **2026-07-22 `opencuda-llm`新設(1位vLLM相当のMVP着手)**: `open-raid-z`
  CLAUDE.mdの「Python製AIライブラリのRust移植ハイブリッド/トライブリッド版」
  構想(マーケティング調査1〜6位: vLLM/Transformers/NumPy/PyTorch互換/
  scikit-learn/Whisper相当)のうち、`opencuda-blas`(NumPy相当)・
  `opencuda-bert`(Transformersエンコーダ相当)に続き未着手だった
  **1位vLLM相当**に、新規クレート`crates/opencuda-llm`として着手した。
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
     (`opencuda-bert`と異なり`safetensors`ローダー未実装)、生成される
     テキストは意味を持たない——検証対象は「自己回帰生成パイプライン
     の配線が正しいか」であって「自然な文章を生成できるか」ではない。
     (c) バイト単位トークナイザなので本格的なBPE/SentencePieceより
     語彙効率は悪い。
  3. **検証**: `cargo build -p opencuda-llm --release`警告0件、
     `cargo test -p opencuda-llm --release`4件全green——
     `generates_requested_number_of_tokens_without_panicking`
     (プロンプトから8トークン生成しpanicしないこと)、
     `same_seed_and_prompt_produce_identical_output_deterministically`、
     `different_seeds_produce_different_weights_and_usually_different_output`、
     そして最も重要な**`incremental_kv_cache_decoding_matches_full_recompute_at_each_position`**
     (KVキャッシュを使った逐次デコードの各位置のロジットが、キャッシュ
     無しでシーケンス全体をフルスクラッチ再計算した場合と数値一致
     〈誤差1e-4以内〉することを検証——`opencuda-blas`の既存Flash
     Attention数値一致テストと同じ考え方で、causalマスクの代替実装が
     正しいことを裏付ける)。`cargo clippy -p opencuda-llm --all-targets
     --release`は`needless_range_loop`警告2件のみ(機能に影響しない、
     次回クリーンアップ対象)。`cargo test --workspace --release`で
     既存クレート全て regression 無し(`opencuda-bert`等の既存テストに
     影響なし)。
  - 次にすべきこと: (1) 実在の学習済みGPT系モデル(GPT-2小型版等)の
    `safetensors`を読み込むローダーの追加(`opencuda-bert`の
    `BertModel::load`と同様の設計で移植可能)、(2) `clippy`の
    `needless_range_loop`警告2件の解消、(3) 残り4目標
    (PyTorch互換/scikit-learn/Whisper相当)のうち次に着手するものの
    選定(現時点の推奨: Whisper相当——既存の`opencuda-bert`の
    エンコーダ実装パターンを転用しやすく、音声特徴量抽出さえ用意すれば
    比較的早くMVPに到達できると見込む)。

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
     (`opencuda-blas`22件・`opencuda-bert`2件・`opencuda-directx`3件
     〈モックのみ、`real-dx12`feature未指定〉・`opencuda-ir`1件+
     結線テスト1件・`opencuda-llm`6件・`opencuda-vulkan`結線テスト
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
  2. **`crates/opencuda-llm/src/lib.rs::GptModel::load`を修正**:
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
     も無変更のままgreen(後方互換)。`cargo test -p opencuda-llm`
     **7件全green**(既存5件+新規2件のうち1件は本来から存在した
     フィクスチャ関数のリファクタリングで名称変更、実質+1件)。
  4. **実E2E確認(型チェックだけで終わらせない)**: 修正後、実際に
     aruaru-llmサーバーを起動し、`POST /v1/models/select`で
     `distilgpt2`への切り替えが成功し(修正前は失敗していた)、
     `POST /v1/generate`で実際に英文が生成される(`distilgpt2-greedy-
     decode-v0-opencuda-llm-cpu`エンジンラベル付きで応答)ことを
     実際に確認した——「ダウンロード→切り替え→生成」の一気通貫を
     実際に検証し、かつその過程で見つかった実バグを実際に修正した。
  - 次にすべきこと: (1) `gpt2-medium`/`gpt2-large`/`gpt2-xl`が
    それぞれどちらのテンソル名規約を使うか未確認(`gpt2`はプレフィックス
    無し、`distilgpt2`はプレフィックス有りと確認済みだが、他のサイズは
    未検証)、(2) 将来的にLlama/Mistral等の異なるアーキテクチャへ
    対応する場合は、この`key_prefix`方式では吸収しきれない可能性が
    高い(テンソル名の構造自体が異なるため)。
