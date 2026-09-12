#!/bin/bash
# QES-faithful W8A8 campaign: two seeds, sequential, goooinext only.
# Preregistration: docs/experiments/minicpm5-qes-faithful-prereg-20260910.md section 10.
set -uo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PATH="$HOME/venvs/minicpm5-vllm/bin:$PATH"
export CPATH="$HOME/.cache/minicpm5/python-dev/usr/include:$HOME/.cache/minicpm5/python-dev/usr/include/python3.12:$HOME/.cache/minicpm5/python-dev/usr/include/aarch64-linux-gnu/python3.12${CPATH:+:$CPATH}"
rm -f ~/.cache/flashinfer/*/121a/cached_ops/tmp/*.lock 2>/dev/null || true
cd ~/minicpm5-w8a8qes/scripts/minicpm5

MODEL=~/minicpm5-payload/out/MiniCPM5-2B-INT8-W8A8
DEV=~/minicpm5-search/artifacts/taskset_dev.jsonl
CONFIRM=~/minicpm5-pool4/taskset_confirm_v4.jsonl
KB=~/minicpm5-w8a8qes/out/kernel_base.safetensors
KBS=~/minicpm5-w8a8qes/scripts/minicpm5/build_kernel_payload.py
PROBES=~/minicpm5-payload/out/refit/probes.json
BF16KL=~/minicpm5-search/out/bf16_kl.json

run_seed() {
  local SEED=$1
  local OUT=~/minicpm5-w8a8qes/out/seed${SEED}
  echo "=== [$(date +%F\ %T)] seed ${SEED}: search start ==="
  python3 pilot_search.py \
    --model $MODEL --taskset $DEV --out $OUT \
    --kernel-base $KB --kernel-base-script $KBS \
    --generations 300 --candidates 50 --edits 32 --alpha 12 \
    --seed ${SEED} --tau 0.02 --verify-live --gate-halt \
    --probe-prompts $PROBES --bf16-logprobs $BF16KL \
    --stop-policy --alpha-ladder --kl-baseline 1.1040 --trend-materiality 1.5 \
    --linear-backend triton
  echo "=== [$(date +%F\ %T)] seed ${SEED}: search exit=$? (confirm-only next) ==="
  python3 pilot_search.py \
    --model $MODEL --taskset $DEV --out $OUT \
    --kernel-base $KB --kernel-base-script $KBS \
    --resume $OUT/state.json --confirm $CONFIRM --confirm-only \
    --probe-prompts $PROBES --bf16-logprobs $BF16KL \
    --linear-backend triton
  echo "=== [$(date +%F\ %T)] seed ${SEED}: confirm-only exit=$? ==="
}

run_seed 20260915
run_seed 20260916
echo "=== [$(date +%F\ %T)] campaign complete ==="
