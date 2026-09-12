#!/bin/bash
# Paper-literal population: N=80 mirror pairs (160 candidates), everything else
# identical to the N=25 prime continuation. Same gen300 starting centres ->
# single-variable comparison against the N=25/N=50 arms.
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

run_seed() {
  local SEED=$1
  local OUT=~/minicpm5-w8a8qes/out/seed${SEED}_primeN80
  echo "=== [$(date +%F\ %T)] seed ${SEED}: N=80-pairs prime continuation start ==="
  python3 pilot_search.py \
    --model $MODEL --taskset $FITNESS --out $OUT \
    --kernel-base $KB --kernel-base-script $KBS \
    --resume ~/minicpm5-w8a8qes/out/seed${SEED}/state.json \
    --generations 400 --candidates 160 --edits 32 --alpha 12 \
    --seed ${SEED} --tau 0.02 --verify-live --gate-halt \
    --probe-prompts $PROBES --bf16-logprobs $BF16KL \
    --stop-policy --alpha-ladder --kl-baseline 1.1040 --trend-materiality 1.5 \
    --linear-backend triton --fitness-metric gold_prob
  echo "=== [$(date +%F\ %T)] seed ${SEED}: search done, eval next ==="
  python3 ~/minicpm5-w8a8qes/scripts/eval_centre.py \
    --model $MODEL --taskset $FITNESS --state $OUT/state.json \
    --kernel-base $KB --probe-prompts $PROBES --bf16-logprobs $BF16KL \
    --linear-backend triton \
    --eval prime_test=$TEST --eval old_dev=$DEV \
    --out $OUT/eval
  echo "=== [$(date +%F\ %T)] seed ${SEED}: eval done ==="
}

run_seed 20260915
run_seed 20260916
echo "=== [$(date +%F\ %T)] N=80-pairs prime continuation complete ==="
