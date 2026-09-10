#!/bin/bash
# Launch `vllm serve` for MiniCPM5-2B on GB10 (map ticket 06).
# Discipline: exec setsid + triple redirect; kill with a separate ssh call.
# Extra flags via VLLM_EXTRA_ARGS (word-split on purpose).
set -uo pipefail
V2=$HOME/venvs/minicpm5-vllm
MODEL=$HOME/models/MiniCPM5-2B
LOG=${VLLM_SERVE_LOG:-$HOME/logs/minicpm5-vllm-serve-20260909.log}
PORT=${VLLM_PORT:-8123}

# vLLM's kernel warmup shells out to `ninja`; it ships in the venv bin and is
# NOT on PATH unless we add it (first attempt died with
# FileNotFoundError: 'ninja' during compile_or_warm_up_model).
export PATH="$V2/bin:$PATH"

# shellcheck disable=SC2086
exec setsid "$V2/bin/vllm" serve "$MODEL" \
  --served-model-name minicpm5-2b \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.30 \
  ${VLLM_EXTRA_ARGS:-} \
  > "$LOG" 2>&1 < /dev/null
