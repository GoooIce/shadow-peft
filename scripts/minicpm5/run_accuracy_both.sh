#!/bin/bash
# Three accuracy arms on the frozen v3 dev pool, identical scorer
# (md5 ca2b0a1519d4f0490bd6b7ddb1e23fd7, the copy that produced the archived
# pool numbers) and the frozen protocol settings (tf arm, enable_thinking=False,
# max_model_len=1024, gpu_mem=0.30, prefix caching off).
#
#   bf16 : unquantized reference
#   w4a4 : the existing NVFP4 payload served natively by CutlassNvFp4LinearKernel
#   w8a8 : the INT8 payload; --linear-backend must be triton on SM121 because
#          vLLM refuses Cutlass INT8 there (see kernel_evidence_int8-cutlass.json)
#
# Logs and summaries land in ~/minicpm5-w8a8/{logs,out}.
set -uo pipefail

W=$HOME/minicpm5-w8a8
PY=$HOME/venvs/minicpm5-vllm/bin/python
ITEMS=$W/data/taskset_dev.jsonl
export PATH="$HOME/venvs/minicpm5-vllm/bin:$PATH"
export CPATH="$HOME/.cache/minicpm5/python-dev/usr/include/python3.12:$HOME/.cache/minicpm5/python-dev/usr/include"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

run_arm() {
  local tag=$1 model=$2 quant=$3 backend=$4
  local log=$W/logs/both_score_${tag}.log
  echo "=== arm $tag start $(date -Is) ==="
  "$PY" "$W/scripts/score_taskset.py" \
    --model "$model" \
    --items "$ITEMS" \
    --out "$W/out_both/scores_${tag}_dev3.jsonl" \
    --summary "$W/out_both/scores_${tag}_dev3_summary.json" \
    --arm both \
    --tag "$tag" \
    --quantization "$quant" \
    --linear-backend "$backend" \
    --max-model-len 1024 \
    --gpu-mem 0.30 \
    > "$log" 2>&1
  echo "=== arm $tag exit=$? $(date -Is) ==="
}

run_arm bf16 "$HOME/models/MiniCPM5-2B" "" cutlass
run_arm w4a4 "$HOME/minicpm5-payload/out/MiniCPM5-2B-NVFP4-W4A4" "compressed-tensors" cutlass
run_arm w8a8 "$HOME/minicpm5-payload/out/MiniCPM5-2B-INT8-W8A8" "compressed-tensors" triton
echo "=== all arms done $(date -Is) ==="
