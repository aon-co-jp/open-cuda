# Changelog

## v0.4.1 (未リリース)

### Added
- `opencuda-blas`: `quantize_int4`/`quantize_int8`を実装(`crates/opencuda-blas/src/lib.rs`)。
  グループ単位の対称量子化(`scale = max_abs / q_max`、INT4は±7、INT8は±127)。
  要素ごとの量子化演算は`GpuDevice::launch_kernel`経由の実カーネルディスパッチ
  (CPUバックエンドでは`rayon`並列)で行い、INT4のニブルパッキング(2値/バイト)は
  バイト共有の書き込み競合を避けるためホスト側で行う。`QuantizedInt4Tensor`/
  `QuantizedInt8Tensor`と、対応する`dequantize_int4`/`dequantize_int8`も追加。
- 単体テスト8件を追加(ラウンドトリップ誤差の範囲検証、奇数長パディング、
  全ゼログループ、グループ境界、空入力/group_size=0の拒否、INT4よりINT8が
  高精度なこと)。`opencuda-blas`のテストは計14件、全green。

### Verified
- `cargo build --workspace --all-targets` / `cargo test --workspace`: 全パス。
- `cargo clippy --workspace --all-targets`: 警告0件(既存の`manual_slice_size_calculation`
  ×3・`too_many_arguments`×1を`sgemm`/`launch_naive_gemm`と同じ`size_of_val`/
  `#[allow]`パターンで解消)。

### 正直な制限
- CPUバックエンドのみ。GPU側(Vulkan/CUDA/ROCm)量子化カーネルは未実装。
- `aruaru-llm`はまだこのAPIを利用していない。

## v0.4.0

### Added
- Vulkan版 `matmul` の最小実装(`crates/opencuda-vulkan/src/real.rs`): `run_matmul_spirv` / `ensure_matmul_args` を追加し、`VulkanDevice::launch_kernel` が `matmul`/`matmul_f32` カーネルを受け付けるようにした。行優先(row-major)レイアウトで A(M×K)・B(K×N)・C(M×N) を扱い、push constantで (m, k, n) を渡す。
- `vector_add` と `matmul` のパイプライン構築/ディスパッチ/後始末を共通化した `dispatch_spirv` ヘルパーを追加(記述量削減、正しさは実機で再検証済み)。
- `opencuda-core::LaunchConfig::grid2d(rows, cols, block_x, block_y)` を追加。matmul等の2次元出力カーネル向けにワークグループ数を計算する。
- `examples/matmul_vulkan_real`: CPUバックエンド(rayon naive matmul)と実Vulkan Compute(naive matmul shader)を同じ入力行列で実行し、ホスト側リファレンス値・CPU結果・Vulkan結果の3者を突き合わせる新サンプル。シェーダは `examples/matmul_vulkan_real/shaders/matmul.comp`(local_size 16x16)。
- `tools/compile-vulkan-shaders.{ps1,cmd,sh}` が `vector_add.comp` と `matmul.comp` の両方をコンパイルするよう更新。

### Verified
- `cargo check --workspace --all-targets` / `cargo clippy --workspace --all-targets`: 警告・エラーなし。
- `cargo test --workspace`: 全パス。
- **実Vulkan環境あり(NVIDIA GeForce GT 730)**: `cargo run --release -p matmul_vulkan_real` で 64×64 * 64×64 の matmul を実行し、CPU結果・Vulkan結果・ホスト側リファレンス値が全て一致(誤差 1e-3 以内)することを確認。既存の `vector_add_vulkan_real` / `vulkan_info` も再実行し、`dispatch_spirv` へのリファクタ後も同じ結果になることを確認。
- 性能計測は方針通り実施していない(正確性優先)。

## v0.3.6

### Changed
- ワークスペース版を `0.3.6` に更新。
- 実Vulkan `VulkanDevice::new` のエラー表示を改善: `vkCreateInstance`/`vkEnumeratePhysicalDevices` 失敗時の対処ヒントを追加し、物理デバイスは列挙したが compute キューが無い場合はデバイス名と種別の一覧をエラーメッセージに含めるようにした。また、`vkCreateDevice` / `vkCreateCommandPool` 失敗時に `VkInstance` / `VkDevice` のリークを防ぐよう後始末を追加。
- `opencuda-vulkan::real::VulkanDevice` に `diagnostics()` を追加し、`vulkan_info` で queue family index、device type（DISCRETE_GPU等）、API version、driver version を表示できるようにした。
- `tools/compile-vulkan-shaders.{ps1,cmd,sh}` に `glslc --version` の表示を追加し、コンパイル前にツールチェーンのバージョンを確認できるようにした。
- `cargo clippy --workspace --all-targets` の警告5件（`opencuda-ir` の `op_ref`、`opencuda-multidev` の `manual_checked_ops`、`opencuda-vulkan` の `manual_is_multiple_of` x2）を解消し、警告0件を達成。

### Verified
- `cargo check --workspace --all-targets`: 警告・エラーなし。
- `cargo clippy --workspace --all-targets`: 警告0件。
- `cargo test --workspace`: 全パス（`opencuda-ir` omniir_path、`opencuda-vulkan` vulkan_mock 含む）。
- `cargo run --release -p vector_add` / `vector_add_omniir` / `vector_add_vulkan`（Mock） / `matmul`: 全てCPU/Mock経路で正しい結果。
- **実Vulkan環境あり**: このセッションでは NVIDIA GeForce GT 730 が実際に検出され、`cargo run --release -p vulkan_info` で queue_family_index=0, device_type=DISCRETE_GPU, api_version=1.2.175, driver_version の実測値表示を確認。`tools/compile-vulkan-shaders.sh` で `glslc`(shaderc v2026.2) によるシェーダコンパイルと `cargo run --release -p vector_add_vulkan_real` の実Vulkan `vector_add` 成功を確認済み。

## v0.3.5

### Changed
- ワークスペース版を `0.3.5` に更新。
- 実Vulkan `vector_add` 成功後の仕上げとして、`VulkanDevice` の未使用フィールド警告を解消。
- PowerShell の実行ポリシーで `.ps1` が止まる環境向けに、`tools/compile-vulkan-shaders.cmd` を追加。
- 通常テスト用に `tools/test-v0.3.5.ps1` / `tools/test-v0.3.5.cmd` / `tools/test-v0.3.5.sh` を追加。
- 実Vulkan専用テスト用に `tools/test-vulkan-real-v0.3.5.ps1` / `tools/test-vulkan-real-v0.3.5.cmd` を追加。

### Verified by user on v0.3.4 base
- `cargo check --workspace --all-targets`
- `cargo run --release -p vector_add`
- `cargo run --release -p vector_add_omniir`
- `cargo run --release -p vector_add_vulkan`
- `cargo run --release -p matmul`
- `cargo run --release -p vulkan_info`
- `powershell -ExecutionPolicy Bypass -File .\tools\compile-vulkan-shaders.ps1`
- `cargo run --release -p vector_add_vulkan_real`

## v0.3.4

### Fixed
- `ash 0.37` で `vk::WriteDescriptorSet` に lifetime 引数を付けていたBUGを修正。
- `MemoryPropertyFlags` を `Debug` 表示していたBUGを修正。

## v0.3.3

### Added
- `examples/vulkan_info` を workspace member に追加。
- `examples/vector_add_vulkan_real` を workspace member に追加。
- 実Vulkan経路も `cargo check --workspace --all-targets` の確認対象へ追加。
