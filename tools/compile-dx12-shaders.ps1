# HLSLコンピュートシェーダーをDXILへコンパイルする(DirectX 12バックエンド用)。
# 使い方: pwsh tools/compile-dx12-shaders.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

$dxc = $env:DXC_BIN
if (-not $dxc) {
    $cmd = Get-Command dxc.exe -ErrorAction SilentlyContinue
    if ($cmd) {
        $dxc = $cmd.Source
    } else {
        $candidates = Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\bin\*\x64\dxc.exe" -ErrorAction SilentlyContinue
        if ($candidates) {
            $dxc = ($candidates | Sort-Object FullName -Descending | Select-Object -First 1).FullName
        } else {
            Write-Error "dxc.exe not found. Install the Windows SDK, or set DXC_BIN."
            exit 1
        }
    }
}
Write-Host "using dxc: $dxc"

$shaderDir = Join-Path $root "crates\opencuda-directx\shaders"
& $dxc -T cs_6_0 -E main (Join-Path $shaderDir "vector_add.hlsl") -Fo (Join-Path $shaderDir "vector_add.dxil")
Write-Host "OK: compiled vector_add.hlsl -> vector_add.dxil"
