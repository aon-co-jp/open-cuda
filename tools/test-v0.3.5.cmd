@echo off
setlocal
cargo check --workspace --all-targets || exit /b %errorlevel%
cargo run --release -p vector_add || exit /b %errorlevel%
cargo run --release -p vector_add_omniir || exit /b %errorlevel%
cargo run --release -p vector_add_vulkan || exit /b %errorlevel%
cargo run --release -p matmul || exit /b %errorlevel%
cargo run --release -p vulkan_info || exit /b %errorlevel%
echo OK: normal OpenCUDA v0.3.5 checks completed.
echo To run real Vulkan vector_add:
echo   tools\compile-vulkan-shaders.cmd
echo   cargo run --release -p vector_add_vulkan_real
