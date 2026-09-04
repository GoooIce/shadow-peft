#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi

output_root=$1
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mage_venv=${MAGE_FLOW_VENV:-"$HOME/Code/Mage/mage_flow/.venv"}
model_path=${MAGE_FLOW_MODEL_PATH:-"$HOME/models/Mage-Flow-Turbo"}
pool_root=${MAGE_FLOW_POOL_ROOT:-"$HOME/data/mage-flow-eval/pool-500-v2-20260904"}
hps_checkpoint=${HPS_V21_CHECKPOINT:-"$HOME/models/HPSv2/HPS_v2.1_compressed.pt"}
clip_checkpoint=${CLIP_H14_CHECKPOINT:-"$HOME/models/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_model.safetensors"}
initialization_root=${MAGE_ES_INITIALIZATION_ROOT:-"$HOME/runs/mage-flow/es-nvfp4-smoke-n4-p4-g3-20260904"}

for required in \
  "$mage_venv/bin/python" \
  "$model_path/model_index.json" \
  "$pool_root/es_train.jsonl" \
  "$pool_root/bf16-reward-calibration.json" \
  "$hps_checkpoint" \
  "$clip_checkpoint" \
  "$initialization_root/initial-clips.pt" \
  "$initialization_root/initialization.json"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required file: $required" >&2
    exit 1
  fi
done

mkdir -p "$output_root"
for sigma in 0.003 0.01 0.03 0.1; do
  tag=${sigma/./p}
  arm_dir="$output_root/sigma-$tag"
  resume_args=()
  if [[ -f "$arm_dir/run.json" ]]; then
    "$mage_venv/bin/python" \
      "$repo_root/scripts/validate_mage_flow_es_run.py" \
      --require-replay-audit \
      "$arm_dir"
    echo "validated completed sigma=$sigma; skipping"
    continue
  fi
  if [[ -d "$arm_dir" ]]; then
    resume_args=(--resume)
  fi
  echo "starting sigma=$sigma arm=$arm_dir"
  "$mage_venv/bin/python" \
    "$repo_root/scripts/run_mage_flow_es.py" \
    --model-path "$model_path" \
    --prompts-jsonl "$pool_root/es_train.jsonl" \
    --calibration "$pool_root/bf16-reward-calibration.json" \
    --hps-checkpoint "$hps_checkpoint" \
    --clip-checkpoint "$clip_checkpoint" \
    --initial-clips "$initialization_root/initial-clips.pt" \
    --initialization-report "$initialization_root/initialization.json" \
    --generations 1 \
    --population-size 8 \
    --episode-size 8 \
    --sigma "$sigma" \
    --learning-rate 0.001 \
    --shaping zscore \
    --reward-batch-size 8 \
    --height 512 \
    --width 512 \
    --steps 4 \
    --cfg 1.0 \
    --audit-first-candidate \
    "${resume_args[@]}" \
    --out "$arm_dir"
  "$mage_venv/bin/python" \
    "$repo_root/scripts/validate_mage_flow_es_run.py" \
    --require-replay-audit \
    "$arm_dir"
done
