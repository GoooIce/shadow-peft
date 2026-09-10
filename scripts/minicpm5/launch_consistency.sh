#!/bin/bash
# Ticket 15: delivery-time train/inference consistency check (engine vs shipped payload).
set -uo pipefail
V2=$HOME/venvs/minicpm5-vllm
export PATH="$V2/bin:$PATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
rm -f "$HOME"/.cache/flashinfer/*/121a/cached_ops/tmp/*.lock 2>/dev/null

exec setsid "$V2/bin/python" "$HOME/minicpm5-search/final/final_consistency_check.py" \
  --model "$HOME/minicpm5-payload/out/MiniCPM5-2B-NVFP4-W4A4" \
  --taskset "$HOME/minicpm5-search/artifacts/taskset_dev.jsonl" \
  --state "$HOME/minicpm5-search/out/formal/state.json" \
  --payload-dir "$HOME/minicpm5-search/out/formal/payload_gen17" \
  --out "$HOME/minicpm5-search/final/out/consistency" \
  --kernel-base "$HOME/minicpm5-search/out/kernel_base.safetensors" \
  --linear-backend cutlass \
  --probe-prompts "$HOME/minicpm5-payload/out/refit/probes.json" \
  --bf16-logprobs "$HOME/minicpm5-search/out/bf16_kl.json" \
  > "$HOME/minicpm5-search/final/logs/consistency.log" 2>&1 < /dev/null
