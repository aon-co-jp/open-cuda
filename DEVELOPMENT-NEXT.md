# DEVELOPMENT NEXT

## v0.3.5 現在地

v0.3.5 では、v0.3.4 で確認できた実機 Vulkan Compute `vector_add` 成功を受けて、配布物を使いやすく整理した。

- `cargo check --workspace --all-targets` の対象に通常サンプル、Mock、実Vulkanサンプルを含める。
- CPU Native、OmniIR、VulkanMock、実Vulkan `vector_add` の最小経路を維持。
- PowerShell実行ポリシーで `.ps1` が止まるWindows環境向けに `.cmd` スクリプトを追加。
- `VulkanDevice` の未使用フィールド警告を解消。

## 次の候補

### v0.3.6

- 実Vulkan `vector_add` のエラー表示をさらに改善。
- `vulkan_info` に queue family index、device type、API version、driver version を表示。
- `compile-vulkan-shaders` 実行前に `glslc --version` を表示。
- `cargo clippy --workspace --all-targets` での警告削減。

### v0.4.0

- Vulkan版 `matmul` の最小実装。
- CPU matmul と Vulkan matmul の結果比較。
- まずは性能より正確性を優先する。
