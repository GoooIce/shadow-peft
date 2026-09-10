#!/bin/bash
# vLLM 0.29.0 install attempt on GB10 (aarch64) with the repo's no-sudo
# python-dev header workaround (mirrors scripts/setup_mage_flow_gb10.sh).
#
# Failure being worked around:
#   instanttensor (vLLM dep) ships x86_64-only wheels -> sdist build ->
#   fatal error: Python.h: No such file or directory (no python3.12-dev, sudo
#   needs a password), so headers are unpacked into a user cache dir + CPATH.
set -uo pipefail
V2=$HOME/venvs/minicpm5-vllm
HDR=${MINICPM5_PYDEV_ROOT:-$HOME/.cache/minicpm5/python-dev}

echo "=== vllm install v2 start $(date -Is) host=$(hostname) ==="
echo "python: $(python3 -V)"

if [ ! -x "$V2/bin/pip" ]; then
  echo "--- creating venv $V2 ---"
  python3 -m venv "$V2"
fi

if [ ! -s "$HDR/usr/include/python3.12/Python.h" ]; then
  echo "--- unpacking python3.12-dev headers into $HDR ---"
  TMP=$(mktemp -d)
  (cd "$TMP" && apt-get download libpython3.12-dev python3.12-dev) 2>&1 | tail -4
  mkdir -p "$HDR"
  for p in "$TMP"/*.deb; do
    dpkg-deb -x "$p" "$HDR" && echo "unpacked $(basename "$p")"
  done
  rm -rf "$TMP"
fi
ls -la "$HDR/usr/include/python3.12/Python.h"

export CPATH="$HDR/usr/include/python3.12:$HDR/usr/include${CPATH:+:$CPATH}"
export MAX_JOBS=${MAX_JOBS:-8}
echo "CPATH=$CPATH"

"$V2/bin/pip" install --upgrade pip 2>&1 | tail -1
"$V2/bin/pip" install vllm==0.29.0 2>&1 | tail -60
echo "=== vllm install v2 exit=$? $(date -Is) ==="

"$V2/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__); print('torch', torch.__version__, torch.version.cuda)" 2>&1 | tail -6
echo "=== vllm probe done $(date -Is) ==="
