$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$shaderDir = Join-Path $root "examples\vector_add_vulkan_real\shaders"
$src = Join-Path $shaderDir "vector_add.comp"
$out = Join-Path $shaderDir "vector_add.spv"

if (-not (Get-Command glslc -ErrorAction SilentlyContinue)) {
    Write-Error "glslc が見つかりません。Vulkan SDK をインストールして、glslc に PATH を通して下さい。"
}

glslc $src -o $out
Write-Host "OK: compiled $src -> $out"
