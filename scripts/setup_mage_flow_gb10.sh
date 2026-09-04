#!/bin/bash
# Build an isolated Mage-Flow runtime on an ARM64 GB10 host.
# The model snapshot is managed separately by download_assets_modelscope.sh.
set -euo pipefail

MAGE_SOURCE_ROOT="${MAGE_SOURCE_ROOT:-$HOME/Code/Mage/mage_flow}"
MAGE_VENV_PATH="${MAGE_VENV_PATH:-$MAGE_SOURCE_ROOT/.venv}"
MAGE_PYTHON_HEADERS_ROOT="${MAGE_PYTHON_HEADERS_ROOT:-$HOME/.cache/mage-flow/python-dev}"
SHADOW_PEFT_ROOT="${SHADOW_PEFT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [ ! -s "$MAGE_SOURCE_ROOT/requirements.txt" ]; then
  echo "Mage-Flow source not found at $MAGE_SOURCE_ROOT" >&2
  exit 2
fi

# flash-attn otherwise builds SM80/90/100/120 variants. GB10 only needs SM120.
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-120}"
export MAX_JOBS="${MAX_JOBS:-8}"

python3 -m venv "$MAGE_VENV_PATH"
"$MAGE_VENV_PATH/bin/python" -m pip install --upgrade pip
"$MAGE_VENV_PATH/bin/python" -m pip install -r "$MAGE_SOURCE_ROOT/requirements.txt"
"$MAGE_VENV_PATH/bin/python" -m pip install -e "$MAGE_SOURCE_ROOT" --no-deps
"$MAGE_VENV_PATH/bin/python" -m pip install setuptools wheel ninja 'torchao==0.18.0'
if [ -s "$SHADOW_PEFT_ROOT/pyproject.toml" ]; then
  "$MAGE_VENV_PATH/bin/python" -m pip install -e "$SHADOW_PEFT_ROOT" --no-deps
fi
MAGE_BUILD_TOOL_PATH="$MAGE_VENV_PATH/bin:$PATH"

MAGE_PYTHON_VERSION="$($MAGE_VENV_PATH/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
MAGE_SYSTEM_INCLUDE="$($MAGE_VENV_PATH/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY"))')"
if [ ! -s "$MAGE_SYSTEM_INCLUDE/Python.h" ]; then
  MAGE_LOCAL_INCLUDE="$MAGE_PYTHON_HEADERS_ROOT/usr/include/python$MAGE_PYTHON_VERSION"
  if [ ! -s "$MAGE_LOCAL_INCLUDE/Python.h" ]; then
    if ! command -v apt-get >/dev/null || ! command -v dpkg-deb >/dev/null; then
      echo "Python.h missing and apt/dpkg are unavailable" >&2
      exit 3
    fi
    MAGE_HEADER_TMP="$(mktemp -d)"
    trap 'rm -rf "$MAGE_HEADER_TMP"' EXIT
    (
      cd "$MAGE_HEADER_TMP"
      apt-get download "libpython$MAGE_PYTHON_VERSION-dev" "python$MAGE_PYTHON_VERSION-dev"
      mkdir -p "$MAGE_PYTHON_HEADERS_ROOT"
      for package in ./*.deb; do
        dpkg-deb -x "$package" "$MAGE_PYTHON_HEADERS_ROOT"
      done
    )
  fi
  export CPATH="$MAGE_LOCAL_INCLUDE:$MAGE_PYTHON_HEADERS_ROOT/usr/include${CPATH:+:$CPATH}"
fi

env PATH="$MAGE_BUILD_TOOL_PATH" \
  "$MAGE_VENV_PATH/bin/python" -m pip install --no-build-isolation flash-attn==2.8.3

"$MAGE_VENV_PATH/bin/python" - <<'PY'
import platform

import diffusers
import flash_attn
import torch
import torchao
import transformers

print(f"machine={platform.machine()}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"diffusers={diffusers.__version__}")
print(f"transformers={transformers.__version__}")
print(f"flash_attn={flash_attn.__version__}")
print(f"torchao={torchao.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
  print(f"device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
PY
