#!/bin/bash
# Launcher for the llmcompressor NVFP4 W4A4 export (GB10 discipline: setsid +
# triple redirect so the job survives the ssh session).
set -uo pipefail

ROOT=$HOME/minicpm5-payload
PY=${MINICPM5_QUANT_PY:-$HOME/venvs/minicpm5-quant/bin/python}
MODEL=${MINICPM5_MODEL:-$HOME/models/MiniCPM5-2B}
OUT=${MINICPM5_PAYLOAD_OUT:-$ROOT/out/MiniCPM5-2B-NVFP4-W4A4}
LOG=$ROOT/logs/quantize_nvfp4_$(date +%Y%m%d-%H%M%S).log
mkdir -p "$(dirname "$LOG")" "$OUT"

exec setsid "$PY" "$ROOT/scripts/minicpm5/quantize_nvfp4_llmcompressor.py" \
  --model "$MODEL" \
  --calib "$ROOT/data/neutral/calib_wikitext103_train.jsonl" \
  --out-dir "$OUT" \
  --num-samples 512 \
  --max-seq-length 2048 \
  > "$LOG" 2>&1 < /dev/null
