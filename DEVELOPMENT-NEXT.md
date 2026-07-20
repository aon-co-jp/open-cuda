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
