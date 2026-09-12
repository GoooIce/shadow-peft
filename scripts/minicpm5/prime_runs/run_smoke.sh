#!/bin/bash
set -euo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PATH="$HOME/venvs/minicpm5-vllm/bin:$PATH"
export CPATH="$HOME/.cache/minicpm5/python-dev/usr/include:$HOME/.cache/minicpm5/python-dev/usr/include/python3.12:$HOME/.cache/minicpm5/python-dev/usr/include/aarch64-linux-gnu/python3.12${CPATH:+:$CPATH}"
rm -f ~/.cache/flashinfer/*/121a/cached_ops/tmp/*.lock 2>/dev/null || true
cd ~/minicpm5-w8a8qes/scripts/minicpm5
python3 pilot_search.py \
  --model ~/minicpm5-payload/out/MiniCPM5-2B-INT8-W8A8 \
  --taskset ~/minicpm5-search/artifacts/taskset_dev.jsonl \
  --out ~/minicpm5-w8a8qes/out/smoke_seed20260914 \
  --kernel-base ~/minicpm5-w8a8qes/out/kernel_base.safetensors \
  --kernel-base-script ~/minicpm5-w8a8qes/scripts/minicpm5/build_kernel_payload.py \
  --generations 2 --candidates 10 --edits 32 --alpha 12 \
  --seed 20260914 --tau 0.02 --verify-live --gate-halt \
  --probe-prompts ~/minicpm5-payload/out/refit/probes.json \
  --bf16-logprobs ~/minicpm5-search/out/bf16_kl.json \
  --linear-backend triton
