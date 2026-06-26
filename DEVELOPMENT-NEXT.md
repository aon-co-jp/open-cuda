# OpenCUDA Development Next

## v0.3.0-dev 現在地

v0.3.0-dev では、v0.2.0 の Mock / OmniIR 経路を壊さず、実Vulkan Computeへ進むための最小実装を追加した。

通常の確認:

```powershell
cargo check --workspace
cargo run --release -p vector_add
cargo run --release -p vector_add_omniir
cargo run --release -p vector_add_vulkan
```

実Vulkan確認は環境依存なので、別サンプルとして分離している。

## 実Vulkan vector_add の実行手順

前提:

- Vulkan対応GPUまたはVulkan対応ドライバ
- Vulkan SDK、または `glslc` が使える環境
- Rust / Cargo

Windows:

```powershell
.\tools\compile-vulkan-shaders.ps1
cargo run --release --manifest-path examples\vector_add_vulkan_real\Cargo.toml
```

Linux/macOS:

```bash
./tools/compile-vulkan-shaders.sh
cargo run --release --manifest-path examples/vector_add_vulkan_real/Cargo.toml
```

期待される出力:

```text
device: OpenCUDA Vulkan Device (...)
OK: real Vulkan Compute produced correct vector_add result
c[0]=1000000, c[999999]=1000000
```

## v0.4.0で進めること

1. 実Vulkanバックエンドの `vector_add` 以外への汎用化。
2. DescriptorSet / Pipeline のキャッシュ。
3. device-local memory + staging buffer 対応。
4. `opencuda-ir` の命令セット拡張。
5. OmniIR → 本物SPIR-V生成器の設計開始。
6. `matmul` のVulkan版 correctness backend。

## 方針

- Mockは消さない。CIとGPUなし開発のために残す。
- 実Vulkanは optional feature として段階的に強くする。
- OpenCUDAのREADMEでは、動く範囲と未実装範囲を必ず分けて書く。
