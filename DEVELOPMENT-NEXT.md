# DEVELOPMENT NEXT

## v0.3.6 現在地

v0.3.6 では、実Vulkanのエラー診断とツールチェーン可視性を強化した。

- 実Vulkan `VulkanDevice::new` のエラー表示を改善（ヒント付き失敗理由、compute キュー不在時のデバイス列挙、`VkInstance`/`VkDevice` リーク防止の後始末）。
- `vulkan_info` に queue family index、device type、API version、driver version を表示。
- `tools/compile-vulkan-shaders.{ps1,cmd,sh}` 実行前に `glslc --version` を表示。
- `cargo clippy --workspace --all-targets` の警告を0件に削減。
- このセッションでは実Vulkan環境（NVIDIA GeForce GT 730）が利用可能で、`vulkan_info` と `vector_add_vulkan_real` の両方を実機で検証できた。

## 次の候補

### v0.4.0

- Vulkan版 `matmul` の最小実装。
- CPU matmul と Vulkan matmul の結果比較。
- まずは性能より正確性を優先する。
