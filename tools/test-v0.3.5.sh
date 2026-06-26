#!/usr/bin/env bash
set -euo pipefail
echo "== OpenCUDA v0.3.5 normal checks =="
cargo check --workspace --all-targets
cargo run --release -p vector_add
cargo run --release -p vector_add_omniir
cargo run --release -p vector_add_vulkan
cargo run --release -p matmul
cargo run --release -p vulkan_info
echo "OK: normal OpenCUDA v0.3.5 checks completed."
echo "To run real Vulkan vector_add: ./tools/compile-vulkan-shaders.sh && cargo run --release -p vector_add_vulkan_real"
