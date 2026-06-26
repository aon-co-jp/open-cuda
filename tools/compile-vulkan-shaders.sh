#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHADER_DIR="$ROOT/examples/vector_add_vulkan_real/shaders"
if ! command -v glslc >/dev/null 2>&1; then
  echo "glslc not found. Install Vulkan SDK or shaderc tools and add glslc to PATH." >&2
  exit 1
fi
glslc "$SHADER_DIR/vector_add.comp" -o "$SHADER_DIR/vector_add.spv"
echo "OK: compiled $SHADER_DIR/vector_add.comp -> $SHADER_DIR/vector_add.spv"
