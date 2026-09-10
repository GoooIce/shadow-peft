#!/bin/bash
# Install llmcompressor 0.13.0 into a dedicated venv on GB10 (aarch64).
#
# Why a separate venv: llmcompressor 0.13.0 pins transformers<=5.14.1 and
# compressed-tensors==0.18.0, while the serving venv ~/venvs/minicpm5-vllm is
# vLLM 0.29.0 + compressed-tensors 0.17.0.  Installing llmcompressor in place
# would downgrade/upgrade packages the serving stack depends on.
#
# The repo-wide aarch64 workaround (instanttensor has no aarch64 wheel) is not
# needed here because llmcompressor does not depend on vLLM.
set -uo pipefail

V=${MINICPM5_QUANT_VENV:-$HOME/venvs/minicpm5-quant}
LOG=${MINICPM5_QUANT_LOG:-$HOME/minicpm5-payload/logs/install_llmcompressor.log}
mkdir -p "$(dirname "$LOG")"

{
  echo "=== llmcompressor install start $(date -Is) host=$(hostname) ==="
  python3 -V

  if [ ! -x "$V/bin/pip" ]; then
    echo "--- creating venv $V ---"
    python3 -m venv "$V"
  fi

  "$V/bin/pip" install --upgrade pip 2>&1 | tail -1
  echo "--- installing llmcompressor==0.13.0 (torch 2.13.0 wheel is in the pip cache) ---"
  "$V/bin/pip" install "llmcompressor==0.13.0" 2>&1 | tail -40

  echo "--- resolved versions ---"
  "$V/bin/python" - <<'PY'
import importlib.metadata as md
for name in ("llmcompressor", "compressed-tensors", "torch", "transformers",
             "datasets", "accelerate", "auto-round", "numpy"):
    try:
        print(f"{name}=={md.version(name)}")
    except Exception as error:  # pragma: no cover
        print(f"{name} MISSING ({error})")
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability())
PY
  echo "=== llmcompressor install done exit=$? $(date -Is) ==="
} > "$LOG" 2>&1 < /dev/null
