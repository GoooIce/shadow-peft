#!/bin/bash
# Perplexity sweep for the W8A8 ticket: bf16 / nvfp4 W4A4 / int8 W8A8 on the two
# held-out wikitext-103 splits, through the same serving runtime and tokenizer.
#
# Mirrors ticket 10's run_ppl.sh exactly (64 sequences x 2048 tokens per split,
# --max-model-len 2049, --gpu-memory-utilization 0.30, --max-num-seqs 16, no
# --linear-backend for bf16/nvfp4 so the auto selection those numbers were
# produced with is reproduced).  The W8A8 arm is the only one that names a
# backend, because SM121 refuses every CUTLASS INT8 kernel.
set -uo pipefail

ROOT=$HOME/minicpm5-payload
W=$HOME/minicpm5-w8a8
PY=$HOME/venvs/minicpm5-vllm/bin/python
OUT=$W/out/ppl
LOG=$W/logs/ppl_sweep_$(date +%Y%m%d-%H%M%S).log
mkdir -p "$OUT" "$(dirname "$LOG")"

export PATH="$HOME/venvs/minicpm5-vllm/bin:$PATH"
HDR=$HOME/.cache/minicpm5/python-dev/usr/include
if [ -d "$HDR/python3.12" ]; then
  export CPATH="$HDR/python3.12:$HDR${CPATH:+:$CPATH}"
fi
export VLLM_WORKER_MULTIPROC_METHOD=spawn

run_arm() {
  local label=$1 model=$2 quant=$3 backend=$4
  local args=()
  if [ "$quant" != "none" ]; then args+=(--quantization "$quant"); fi
  if [ -n "$backend" ]; then args+=(--linear-backend "$backend"); fi
  for SPLIT in wikitext103_test wikitext103_validation; do
    echo "=== ppl $label $SPLIT start $(date -Is) ==="
    "$PY" "$W/scripts/ppl_vllm.py" \
      --model "$model" \
      --eval-jsonl "$ROOT/data/neutral/eval_${SPLIT}.jsonl" \
      --label "${label}_${SPLIT}" \
      --out "$OUT" \
      --max-model-len 2049 \
      --gpu-memory-utilization 0.30 \
      --max-num-seqs 16 \
      "${args[@]}"
    echo "=== ppl $label $SPLIT exit=$? $(date -Is) ==="
  done
}

{
  run_arm bf16 "$HOME/models/MiniCPM5-2B" none ""
  run_arm nvfp4_w4a4 "$ROOT/out/MiniCPM5-2B-NVFP4-W4A4" compressed-tensors ""
  run_arm int8_w8a8 "$ROOT/out/MiniCPM5-2B-INT8-W8A8" compressed-tensors triton
  echo "=== ppl sweep done $(date -Is) ==="
} > "$LOG" 2>&1 < /dev/null
