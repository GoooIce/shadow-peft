#!/bin/bash
# N dose-response with the per-coordinate step held INVARIANT across N.
# First attempts (alpha=12) halted at gen301 with moved=0 and u_abs_max=0.4615
# < 0.5: the mirror aggregation normalises by total rank weight (~M+1), so a
# bigger population shrinks the step below the half-grid projection threshold.
# Fix: alpha = 12 * (M+1)/26  -> alpha(50)=23.54, alpha(80)=37.38. Everything
# else identical to the N=25 arm (same gen300 start, same pools, same seed).
set -euo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PATH="$HOME/venvs/minicpm5-vllm/bin:$PATH"
export CPATH="$HOME/.cache/minicpm5/python-dev/usr/include:$HOME/.cache/minicpm5/python-dev/usr/include/python3.12:$HOME/.cache/minicpm5/python-dev/usr/include/aarch64-linux-gnu/python3.12${CPATH:+:$CPATH}"
rm -f ~/.cache/flashinfer/*/121a/cached_ops/tmp/*.lock 2>/dev/null || true
cd ~/minicpm5-w8a8qes/scripts/minicpm5

MODEL=~/minicpm5-payload/out/MiniCPM5-2B-INT8-W8A8
FITNESS=~/minicpm5-w8a8qes/prime/taskset_prime_fitness.jsonl
KB=~/minicpm5-w8a8qes/out/kernel_base.safetensors
KBS=~/minicpm5-w8a8qes/scripts/minicpm5/build_kernel_payload.py
PROBES=~/minicpm5-payload/out/refit/probes.json
BF16KL=~/minicpm5-search/out/bf16_kl.json
DEV=~/minicpm5-search/artifacts/taskset_dev.jsonl
TEST=~/minicpm5-w8a8qes/prime/taskset_prime_test.jsonl

run_one() {
  local SEED=$1 CANDS=$2 ALPHA=$3 TAG=$4
  local OUT=~/minicpm5-w8a8qes/out/${TAG}
  echo "=== [$(date +%F\ %T)] ${TAG}: start (candidates=${CANDS}, alpha=${ALPHA}) ==="
  python3 pilot_search.py \
    --model $MODEL --taskset $FITNESS --out $OUT \
    --kernel-base $KB --kernel-base-script $KBS \
    --resume ~/minicpm5-w8a8qes/out/seed${SEED}/state.json \
    --generations 400 --candidates ${CANDS} --edits 32 --alpha ${ALPHA} \
    --seed ${SEED} --tau 0.02 --verify-live --gate-halt \
    --probe-prompts $PROBES --bf16-logprobs $BF16KL \
    --stop-policy --alpha-ladder --kl-baseline 1.1040 --trend-materiality 1.5 \
    --linear-backend triton --fitness-metric gold_prob
  echo "=== [$(date +%F\ %T)] ${TAG}: search done (exit $?), eval next ==="
  python3 ~/minicpm5-w8a8qes/scripts/eval_centre.py \
    --model $MODEL --taskset $FITNESS --state $OUT/state.json \
    --kernel-base $KB --probe-prompts $PROBES --bf16-logprobs $BF16KL \
    --linear-backend triton \
    --eval prime_test=$TEST --eval old_dev=$DEV \
    --out $OUT/eval || echo "=== [$(date +%F\ %T)] ${TAG}: eval exited nonzero (check report) ==="
  echo "=== [$(date +%F\ %T)] ${TAG}: eval step finished ==="
}

run_one 20260915 100 23.54 seed20260915_primeN50scaled
run_one 20260916 100 23.54 seed20260916_primeN50scaled
run_one 20260915 160 37.38 seed20260915_primeN80scaled
run_one 20260916 160 37.38 seed20260916_primeN80scaled
echo "=== [$(date +%F\ %T)] scaled N campaign complete ==="
