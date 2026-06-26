# Changelog

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
