$ErrorActionPreference = "Stop"
& "$PSScriptRoot\compile-vulkan-shaders.ps1"
cargo run --release -p vector_add_vulkan_real
Write-Host "OK: real Vulkan OpenCUDA v0.3.5 vector_add completed."
