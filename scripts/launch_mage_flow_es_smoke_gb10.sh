#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi

output_dir=$1
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mage_venv=${MAGE_FLOW_VENV:-"$HOME/Code/Mage/mage_flow/.venv"}
model_path=${MAGE_FLOW_MODEL_PATH:-"$HOME/models/Mage-Flow-Turbo"}
pool_root=${MAGE_FLOW_POOL_ROOT:-"$HOME/data/mage-flow-eval/pool-500-v2-20260904"}
hps_checkpoint=${HPS_V21_CHECKPOINT:-"$HOME/models/HPSv2/HPS_v2.1_compressed.pt"}
clip_checkpoint=${CLIP_H14_CHECKPOINT:-"$HOME/models/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_model.safetensors"}
run_name=$(basename "$output_dir")
log_path=${MAGE_FLOW_LOG_PATH:-"$HOME/logs/${run_name}.log"}

for required in \
  "$mage_venv/bin/python" \
  "$model_path/model_index.json" \
  "$pool_root/es_train.jsonl" \
  "$pool_root/bf16-reward-calibration.json" \
  "$hps_checkpoint" \
  "$clip_checkpoint"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required file: $required" >&2
    exit 1
  fi
done
if [[ -e "$output_dir" ]]; then
  echo "refusing existing output path: $output_dir" >&2
  exit 1
fi

mkdir -p "$(dirname "$output_dir")" "$(dirname "$log_path")"
echo "launching Phase-1 ES NVFP4 smoke output=$output_dir log=$log_path"
exec setsid "$mage_venv/bin/python" \
  "$repo_root/scripts/run_mage_flow_es.py" \
  --model-path "$model_path" \
  --prompts-jsonl "$pool_root/es_train.jsonl" \
  --calibration "$pool_root/bf16-reward-calibration.json" \
  --hps-checkpoint "$hps_checkpoint" \
  --clip-checkpoint "$clip_checkpoint" \
  --generations 3 \
  --population-size 4 \
  --episode-size 4 \
  --sigma 0.03 \
  --learning-rate 0.015 \
  --shaping zscore \
  --reward-batch-size 4 \
  --height 512 \
  --width 512 \
  --steps 4 \
  --cfg 1.0 \
  --audit-first-candidate \
  --out "$output_dir" \
  >"$log_path" 2>&1 < /dev/null
