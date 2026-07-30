#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v glslc >/dev/null 2>&1; then
  echo "glslc not found. Install Vulkan SDK or shaderc tools and add glslc to PATH." >&2
  exit 1
fi
echo "glslc version:"
glslc --version

compile() {
  local dir="$1" name="$2"
  local shader_dir="$ROOT/examples/$dir/shaders"
  glslc "$shader_dir/$name.comp" -o "$shader_dir/$name.spv"
  echo "OK: compiled $shader_dir/$name.comp -> $shader_dir/$name.spv"
}

compile vector_add_vulkan_real vector_add
compile matmul_vulkan_real matmul
compile raid6_xor_parity_vulkan_real raid6_xor_parity
