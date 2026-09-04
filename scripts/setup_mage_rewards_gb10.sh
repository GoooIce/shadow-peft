#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
reward_venv=${MAGE_REWARD_VENV:-"$repo_root/.venv"}
hps_source=${HPSV2_SOURCE_DIR:-"$HOME/Code/HPSv2"}
hps_commit=866735ecaae999fa714bd9edfa05aa2672669ee3

if [[ ! -x "$reward_venv/bin/python" ]]; then
  echo "missing reward Python environment: $reward_venv" >&2
  exit 1
fi
uv_bin=${UV_BIN:-$(command -v uv || true)}
if [[ -z "$uv_bin" && -x "$HOME/.local/bin/uv" ]]; then
  uv_bin="$HOME/.local/bin/uv"
fi
if [[ -z "$uv_bin" ]]; then
  echo "uv is required" >&2
  exit 1
fi

if [[ ! -d "$hps_source/.git" ]]; then
  git clone https://github.com/tgxs002/HPSv2.git "$hps_source"
fi
git -C "$hps_source" fetch origin "$hps_commit"
git -C "$hps_source" checkout --detach "$hps_commit"

"$uv_bin" pip install --python "$reward_venv/bin/python" \
  pillow torchvision open_clip_torch ftfy regex einops timm webdataset \
  braceexpand pandas requests clint sentencepiece fsspec scipy transformers safetensors
"$uv_bin" pip install --python "$reward_venv/bin/python" --no-deps -e "$hps_source"
"$uv_bin" pip install --python "$reward_venv/bin/python" --no-deps -e "$repo_root"

"$reward_venv/bin/python" - <<'PY'
import hpsv2
import open_clip
import torch
import torchvision

print(
    {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "hpsv2": getattr(hpsv2, "__version__", "source-checkout"),
        "open_clip": open_clip.__version__,
        "cuda": torch.cuda.is_available(),
    }
)
PY
