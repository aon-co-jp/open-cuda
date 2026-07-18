$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if (-not (Get-Command glslc -ErrorAction SilentlyContinue)) {
    Write-Error "glslc が見つかりません。Vulkan SDK をインストールして、glslc に PATH を通して下さい。"
}

Write-Host "glslc version:"
glslc --version

$shaders = @(
    @{ Dir = "examples\vector_add_vulkan_real\shaders"; Name = "vector_add" },
    @{ Dir = "examples\matmul_vulkan_real\shaders"; Name = "matmul" }
)

foreach ($shader in $shaders) {
    $shaderDir = Join-Path $root $shader.Dir
    $src = Join-Path $shaderDir "$($shader.Name).comp"
    $out = Join-Path $shaderDir "$($shader.Name).spv"
    glslc $src -o $out
    Write-Host "OK: compiled $src -> $out"
}
