#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DXC_BIN="${DXC_BIN:-}"
if [ -z "$DXC_BIN" ]; then
  if command -v dxc >/dev/null 2>&1; then
    DXC_BIN="dxc"
  else
    CANDIDATE="/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/dxc.exe"
    if [ -f "$CANDIDATE" ]; then
      DXC_BIN="$CANDIDATE"
    else
      echo "dxc not found. Install the Windows SDK (includes dxc.exe) or set DXC_BIN." >&2
      exit 1
    fi
  fi
fi
echo "using dxc: $DXC_BIN"

SHADER_DIR="$ROOT/crates/opencuda-directx/shaders"
"$DXC_BIN" -T cs_6_0 -E main "$SHADER_DIR/vector_add.hlsl" -Fo "$SHADER_DIR/vector_add.dxil"
echo "OK: compiled $SHADER_DIR/vector_add.hlsl -> $SHADER_DIR/vector_add.dxil"

"$DXC_BIN" -T cs_6_0 -E main "$SHADER_DIR/matmul.hlsl" -Fo "$SHADER_DIR/matmul.dxil"
echo "OK: compiled $SHADER_DIR/matmul.hlsl -> $SHADER_DIR/matmul.dxil"

"$DXC_BIN" -T cs_6_0 -E main "$SHADER_DIR/chacha20.hlsl" -Fo "$SHADER_DIR/chacha20.dxil"
echo "OK: compiled $SHADER_DIR/chacha20.hlsl -> $SHADER_DIR/chacha20.dxil"

"$DXC_BIN" -T cs_6_0 -E main "$SHADER_DIR/poly1305.hlsl" -Fo "$SHADER_DIR/poly1305.dxil"
echo "OK: compiled $SHADER_DIR/poly1305.hlsl -> $SHADER_DIR/poly1305.dxil"
