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

## クロスOS(2026-09-03、設計方針)

`opencuda-vulkan` は `ash` の `loaded` feature で Vulkan ローダを動的に
拾うため、**OS 依存の分岐を持たない**。到達手段は以下(詳細・一次資料は
`OmniGPU-Design.md` §11):

- **Linux**: mesa RADV/ANV / NVIDIA の Vulkan ICD(`libvulkan.so`)。
  NVIDIA GT 730 で実機検証済み。AMD/Intel 実機は未検証。
- **Windows**: `vulkan-1.dll`。実機検証済み。Vulkan が無い環境向けに
  D3D12/DXIL フォールバック(`opencuda-directx`)も実機検証済み。
- **macOS / iOS**: **MoltenVK**(Vulkan → Metal、SPIR-V → MSL 変換内蔵)を
  入れれば `opencuda-vulkan` が**新規コード無しでそのまま動くはず**。
  Vulkan SDK(LunarG)を入れると `libMoltenVK.dylib` が ICD として
  登録される。`VK_KHR_portability_subset` で一部機能が制限されるため、
  カーネル側で回避が要る場合がある。**実機 Mac での検証は未実施**
  (Android で 2026-08-15 に行ったクロスビルド+実機実行と同じ手順で
  検証予定)。
- **その他 Unix**: mesa Vulkan が動く範囲でベストエフォート。
