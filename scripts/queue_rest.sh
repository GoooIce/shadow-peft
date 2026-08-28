#!/bin/bash
# Self-driving queue after the 1.2B scale arm:
#   1. wait for rungB_1p2b_base verdict
#   2. 230M-Base (HF) scale arm
#   3. exp4 heterogeneity arms (A1/A3/B)
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "[queue] waiting for 1.2B arm ..."
while [ ! -f runs/exp1/rungB_1p2b_base/verdict.json ]; do sleep 60; done
echo "[queue] 1.2B done:"; cat runs/exp1/rungB_1p2b_base/verdict.json

echo "[queue] downloading LFM2.5-230M-Base from HF"
$PY - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("LiquidAI/LFM2.5-230M-Base", local_dir=__import__("os").path.expanduser("~/models/LFM2.5-230M-Base"))
print("230M-Base downloaded")
EOF

bash scripts/exp1_scale_arm.sh ~/models/LFM2.5-230M-Base 230m_base 384

echo "[queue] exp4 arms"
$PY scripts/exp4_arms.py

echo "[queue] ALL DONE"
