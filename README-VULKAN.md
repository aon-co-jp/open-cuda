# OpenCUDA Vulkan 手順

## v0.3.5 の位置づけ

v0.3.5 は、実Vulkan Computeで `vector_add` を動かす最小経路を維持しつつ、Windowsで実行しやすい補助スクリプトを追加した版です。

## 通常確認

```powershell
cargo check --workspace --all-targets
cargo run --release -p vulkan_info
```

`vulkan_info` が成功すると、Vulkan loader、physical device、logical device、compute queue が利用できる状態です。

## SPIR-V生成

PowerShell実行ポリシーに止められない方法として、`.cmd` を推奨します。

```powershell
.\tools\compile-vulkan-shaders.cmd
```

PowerShellで実行する場合は、環境によって次のように実行します。

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\compile-vulkan-shaders.ps1
```

## 実Vulkan vector_add

```powershell
cargo run --release -p vector_add_vulkan_real
```

成功時の目標出力です。

```text
device: OpenCUDA Vulkan Device (...)
OK: real Vulkan Compute produced correct vector_add result
c[0]=1000000, c[999999]=1000000
```

## まとめ実行

```powershell
.\tools\test-v0.3.5.cmd
.\tools\test-vulkan-real-v0.3.5.cmd
```
