#!/bin/bash
# Run the BF16 smoke test in the transformers-only venv (map ticket 06).
# usage: run_bf16_smoke.sh <tag>   (tag = goooinext | gooo)
set -uo pipefail
TAG=${1:?usage: run_bf16_smoke.sh <tag>}
V1=$HOME/venvs/minicpm5-bringup
OUT=$HOME/runs/minicpm5/bf16-smoke-$TAG.json
LOG=$HOME/logs/minicpm5-bf16-smoke-$TAG.log
mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"
{
  echo "=== bf16 smoke start $(date -Is) host=$(hostname) ==="
  "$V1/bin/python" "$HOME/runs/minicpm5/scripts/bf16_smoke.py" \
      --model "$HOME/models/MiniCPM5-2B" \
      --out "$OUT" \
      --repeat \
      --eos-ablation
  echo "=== bf16 smoke exit=$? $(date -Is) ==="
} > "$LOG" 2>&1 < /dev/null
