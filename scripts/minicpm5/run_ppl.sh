#!/bin/bash
# Launcher for one perplexity measurement through vLLM.
#   usage: run_ppl.sh <label> <model_dir> [quantization]
# Runs both held-out wikitext splits and writes ppl_<label>*.json under
# ~/minicpm5-payload/out/ppl/.
set -uo pipefail

LABEL=$1
MODEL=$2
QUANT=${3:-none}

ROOT=$HOME/minicpm5-payload
PY=${MINICPM5_VLLM_PY:-$HOME/venvs/minicpm5-vllm/bin/python}
OUT=$ROOT/out/ppl
LOG=$ROOT/logs/ppl_${LABEL}_$(date +%Y%m%d-%H%M%S).log
mkdir -p "$OUT" "$(dirname "$LOG")"

# GB10 environment fixes carried over from ticket 06:
#  * vLLM's kernel warmup shells out to `ninja`, which ships in the venv bin
#    and is not on PATH by default;
#  * flashinfer/triton JIT compile cuda_utils.c against Python.h, which lives
#    in the user-space python3.12-dev unpack (no sudo on these hosts).
export PATH="$HOME/venvs/minicpm5-vllm/bin:$PATH"
HDR=$HOME/.cache/minicpm5/python-dev/usr/include
if [ -d "$HDR/python3.12" ]; then
  export CPATH="$HDR/python3.12:$HDR${CPATH:+:$CPATH}"
fi

QUANT_ARG=()
if [ "$QUANT" != "none" ]; then
  QUANT_ARG=(--quantization "$QUANT")
fi

{
  echo "=== ppl $LABEL start $(date -Is) host=$(hostname) model=$MODEL quant=$QUANT ==="
  for SPLIT in wikitext103_test wikitext103_validation; do
    "$PY" "$ROOT/scripts/minicpm5/ppl_vllm.py" \
      --model "$MODEL" \
      --eval-jsonl "$ROOT/data/neutral/eval_${SPLIT}.jsonl" \
      --label "${LABEL}_${SPLIT}" \
      --out "$OUT" \
      --max-model-len 2049 \
      --gpu-memory-utilization 0.30 \
      --max-num-seqs 16 \
      "${QUANT_ARG[@]}"
    echo "--- $SPLIT done $(date -Is) ---"
  done
  echo "=== ppl $LABEL done $(date -Is) ==="
} > "$LOG" 2>&1 < /dev/null
