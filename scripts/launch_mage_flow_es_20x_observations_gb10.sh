#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 OUTPUT_ROOT EPISODE [EPISODE ...]" >&2
  exit 2
fi

output_root=$1
shift
episodes=("$@")
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

declare -A seen_episodes=()
for episode in "${episodes[@]}"; do
  if [[ ! "$episode" =~ ^([0-9]|1[0-9]|2[0-3])$ ]]; then
    echo "observation episode must be an integer from 0 through 23: $episode" >&2
    exit 2
  fi
  if [[ -n "${seen_episodes[$episode]:-}" ]]; then
    echo "duplicate observation episode: $episode" >&2
    exit 2
  fi
  seen_episodes[$episode]=1
done

mkdir -p "$output_root"
for episode in "${episodes[@]}"; do
  arm_dir="$output_root/gradient-episode-$episode"
  if [[ ! -f "$arm_dir/run.json" ]]; then
    resume_args=()
    if [[ -d "$arm_dir" ]]; then
      resume_args=(--resume)
    fi
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
      --population-size 80 \
      --episode-size 8 \
      --episode-offset "$episode" \
      --sigma 0.003 \
      --learning-rate 0.0001 \
      --shaping zscore \
      --es-seed 20260907 \
      --schedule-seed 20260904 \
      --reward-batch-size 8 \
      --height 512 \
      --width 512 \
      --steps 4 \
      --cfg 1.0 \
      --audit-first-candidate \
      "${resume_args[@]}" \
      --out "$arm_dir"
  fi
  validation_path="$output_root/gradient-episode-$episode-validation.json"
  validation_args=()
  if [[ ! -f "$validation_path" ]]; then
    validation_args=(--out "$validation_path")
  fi
  "$mage_venv/bin/python" \
    "$repo_root/scripts/validate_mage_flow_es_run.py" \
    --require-replay-audit \
    "${validation_args[@]}" \
    "$arm_dir"
done
