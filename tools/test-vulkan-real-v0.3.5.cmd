@echo off
setlocal
call "%~dp0compile-vulkan-shaders.cmd" || exit /b %errorlevel%
cargo run --release -p vector_add_vulkan_real || exit /b %errorlevel%
echo OK: real Vulkan OpenCUDA v0.3.5 vector_add completed.
