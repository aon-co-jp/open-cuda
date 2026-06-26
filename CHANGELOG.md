# Changelog

## v0.3.0-dev

### Added

- `opencuda-vulkan` に optional feature `real-vulkan` を追加。
- `ash` ベースの最小実Vulkan Computeバックエンド `VulkanDevice` を追加。
- 実Vulkan版 `vector_add` サンプルを `examples/vector_add_vulkan_real` に追加。
- GLSL Compute Shader `vector_add.comp` を追加。
- Windows用 `tools/compile-vulkan-shaders.ps1` と Linux/macOS用 `tools/compile-vulkan-shaders.sh` を追加。

### Kept stable

- v0.2 の `VulkanMockDevice` は残した。
- 通常の `cargo check --workspace` は、Vulkan SDK や実GPUなしでも確認できる構成を維持。
- 実Vulkanサンプルは workspace から除外し、環境が整っている場合だけ個別に実行する。

### Current limitation

- 実Vulkanは v0.3 時点では `vector_add` 専用の correctness backend。
- 高速化、descriptor cache、pipeline cache、device-local memory staging、複数カーネル汎用化は v0.4 以降。
