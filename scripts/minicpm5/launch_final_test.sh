#!/bin/bash
# Ticket 15: sealed final-pool test — ONE open, both arms, one process.
# Frozen shape per ticket 13 §4 / 18 §6.4 (same process, same pool, q0 + centre).
set -uo pipefail
V2=$HOME/venvs/minicpm5-vllm
SC=$HOME/minicpm5-search/scripts/minicpm5
FIN=$HOME/minicpm5-search/final

# ticket 06/14 environment traps
export PATH="$V2/bin:$PATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# killed processes leave these behind and they block later runs
rm -f "$HOME"/.cache/flashinfer/*/121a/cached_ops/tmp/*.lock 2>/dev/null

exec setsid "$V2/bin/python" "$SC/pilot_search.py" \
  --model "$HOME/minicpm5-payload/out/MiniCPM5-2B-NVFP4-W4A4" \
  --taskset "$HOME/minicpm5-search/artifacts/taskset_dev.jsonl" \
  --confirm "$FIN/artifacts/taskset_final.jsonl" \
  --out "$FIN/out/final" \
  --kernel-base "$HOME/minicpm5-search/out/kernel_base.safetensors" \
  --resume "$HOME/minicpm5-search/out/formal/state.json" \
  --generations 200 --candidates 50 --edits 32 --alpha 12.0 --tau 0.02 \
  --linear-backend cutlass \
  --verify-live --gate-halt \
  --confirm-only --reference-check \
  --probe-prompts "$HOME/minicpm5-payload/out/refit/probes.json" \
  --bf16-logprobs "$HOME/minicpm5-search/out/bf16_kl.json" \
  > "$FIN/logs/final_test.log" 2>&1 < /dev/null
