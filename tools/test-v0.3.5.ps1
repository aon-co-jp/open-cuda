$ErrorActionPreference = "Stop"
Write-Host "== OpenCUDA v0.3.5 normal checks =="
cargo check --workspace --all-targets
cargo run --release -p vector_add
cargo run --release -p vector_add_omniir
cargo run --release -p vector_add_vulkan
cargo run --release -p matmul
cargo run --release -p vulkan_info
Write-Host "OK: normal OpenCUDA v0.3.5 checks completed."
Write-Host "To run real Vulkan vector_add: powershell -ExecutionPolicy Bypass -File .\tools\compile-vulkan-shaders.ps1; cargo run --release -p vector_add_vulkan_real"
Write-Host "Or avoid PowerShell policy: .\tools\compile-vulkan-shaders.cmd; cargo run --release -p vector_add_vulkan_real"
