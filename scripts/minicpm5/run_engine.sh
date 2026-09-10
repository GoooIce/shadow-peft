#!/bin/bash
# Run one refit_gate engine session on GB10.
#   usage: run_engine.sh <label> <model_dir> <out.json> [chain.json] [extra args...]
# Chain is a JSON file (list of steps); omit or pass "" for a plain fresh run.
set -uo pipefail

LABEL=$1
MODEL=$2
OUT=$3
CHAIN=${4:-}
shift 4 || true

ROOT=$HOME/minicpm5-payload
PY=${MINICPM5_VLLM_PY:-$HOME/venvs/minicpm5-vllm/bin/python}
LOG=$ROOT/logs/refit_${LABEL}_$(date +%Y%m%d-%H%M%S).log
mkdir -p "$ROOT/logs" "$ROOT/out/refit"

# GB10 environment fixes carried over from ticket 10 (ninja on PATH, Python.h
# for JIT helpers).  The NVFP4 kernel is pinned explicitly, so PATH does not
# influence kernel selection here.
export PATH="$HOME/venvs/minicpm5-vllm/bin:$PATH"
HDR=$HOME/.cache/minicpm5/python-dev/usr/include
if [ -d "$HDR/python3.12" ]; then
  export CPATH="$HDR/python3.12:$HDR${CPATH:+:$CPATH}"
fi
export PYTHONPATH="$ROOT/scripts/minicpm5:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# The gate sends callables over collective_rpc (pointer census, dumps); vLLM
# 0.29 refuses function objects unless the pickle fallback is explicitly allowed.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

QUANT=${QUANT:-compressed-tensors}
ARGS=(--mode run --model "$MODEL" --label "$LABEL" --out "$OUT"
      --probe-set "$ROOT/out/refit/probes.json"
      --linear-backend cutlass)
if [ "$QUANT" != "none" ]; then
  ARGS+=(--quantization "$QUANT")
fi
if [ -n "$CHAIN" ]; then
  ARGS+=(--chain "$(cat "$CHAIN")")
fi

{
  echo "=== refit $LABEL start $(date -Is) host=$(hostname) model=$MODEL ==="
  "$PY" "$ROOT/scripts/minicpm5/refit_gate.py" "${ARGS[@]}" "$@"
  echo "=== refit $LABEL done $(date -Is) ==="
} > "$LOG" 2>&1 < /dev/null
echo "log: $LOG"
