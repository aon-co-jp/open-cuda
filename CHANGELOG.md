# Changelog

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
