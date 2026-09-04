#!/usr/bin/env bash
# Add reward-only Python packages to the proven Mage environment without letting
# a resolver replace its torch/torchvision/transformers stack.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mage_venv=${MAGE_VENV:-"$HOME/Code/Mage/mage_flow/.venv"}
hps_source=${HPSV2_SOURCE_DIR:-"$HOME/Code/HPSv2"}
hps_commit=866735ecaae999fa714bd9edfa05aa2672669ee3

if [[ ! -x "$mage_venv/bin/python" ]]; then
  echo "missing Mage Python environment: $mage_venv" >&2
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

before=$(
  "$mage_venv/bin/python" -c \
    'import torch, torchvision, transformers; print(torch.__version__, torchvision.__version__, transformers.__version__)'
)
if [[ ! -d "$hps_source/.git" ]]; then
  git clone https://github.com/tgxs002/HPSv2.git "$hps_source"
fi
git -C "$hps_source" fetch origin "$hps_commit"
git -C "$hps_source" checkout --detach "$hps_commit"

"$uv_bin" pip install --python "$mage_venv/bin/python" --no-deps \
  open_clip_torch==3.3.0 ftfy==6.3.1 timm==1.0.29 webdataset==1.0.2 \
  braceexpand==0.1.7 clint==0.5.1 wcwidth==0.8.3
"$uv_bin" pip install --python "$mage_venv/bin/python" --no-deps -e "$hps_source"
"$uv_bin" pip install --python "$mage_venv/bin/python" --no-deps -e "$repo_root"

after=$(
  "$mage_venv/bin/python" -c \
    'import torch, torchvision, transformers; print(torch.__version__, torchvision.__version__, transformers.__version__)'
)
if [[ "$before" != "$after" ]]; then
  echo "refusing modified Mage core stack: before=$before after=$after" >&2
  exit 2
fi

"$mage_venv/bin/python" - <<'PY'
import hpsv2
import open_clip
import torch
from shadow_lab.image_reward_models import HPSv21Scorer, OpenCLIPH14Scorer

print(
    {
        "torch": torch.__version__,
        "hpsv2": getattr(hpsv2, "__version__", "source-checkout"),
        "open_clip": open_clip.__version__,
        "scorers": [HPSv21Scorer.__name__, OpenCLIPH14Scorer.__name__],
        "cuda": torch.cuda.is_available(),
    }
)
PY
