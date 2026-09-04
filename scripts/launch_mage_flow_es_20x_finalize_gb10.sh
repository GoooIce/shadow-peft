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

source_run_args=()
additional_source_args=()
source_runs=()
episode_offsets=()
for episode in $(seq 0 23); do
  arm_dir="$output_root/gradient-episode-$episode"
  if [[ ! -f "$arm_dir/run.json" ]]; then
    echo "missing completed gradient source: $arm_dir" >&2
    exit 1
  fi
  source_runs+=("$arm_dir")
  source_run_args+=(--source-run "$arm_dir")
  episode_offsets+=("$episode")
  if [[ "$episode" -gt 0 ]]; then
    additional_source_args+=(--additional-source-run "$arm_dir")
  fi
done

gradient_report="$output_root/gradient-report.json"
centers_dir="$output_root/average-gradient-centers"
if [[ ! -f "$gradient_report" ]]; then
  "$mage_venv/bin/python" \
    "$repo_root/scripts/aggregate_mage_flow_es_gradients.py" \
    "${source_run_args[@]}" \
    --expected-episode-offsets "${episode_offsets[@]}" \
    --expected-population-size 80 \
    --expected-episode-size 8 \
    --expected-sigma 0.003 \
    --expected-es-seed 20260907 \
    --expected-schedule-seed 20260904 \
    --alphas 0.00003 0.0001 \
    --clip-step-rms-max 0.02 \
    --centers-dir "$centers_dir" \
    --out "$gradient_report"
fi

initial_center="$centers_dir/initial-effective-clips.pt"
small_center="$centers_dir/alpha-3em05-clips.pt"
large_center="$centers_dir/alpha-0p0001-clips.pt"
prior_evaluation_args=()
selection_args=()
for episode in $(seq 24 31); do
  evaluation="$output_root/selection-episode-$episode.json"
  if [[ ! -f "$evaluation" ]]; then
    "$mage_venv/bin/python" \
      "$repo_root/scripts/evaluate_mage_flow_es_centers.py" \
      --model-path "$model_path" \
      --prompts-jsonl "$pool_root/es_train.jsonl" \
      --calibration "$pool_root/bf16-reward-calibration.json" \
      --hps-checkpoint "$hps_checkpoint" \
      --clip-checkpoint "$clip_checkpoint" \
      --source-run "${source_runs[0]}" \
      "${additional_source_args[@]}" \
      "${prior_evaluation_args[@]}" \
      --candidate "initial=$initial_center" \
      --candidate "small=$small_center" \
      --candidate "large=$large_center" \
      --episode-index "$episode" \
      --episode-size 8 \
      --schedule-seed 20260904 \
      --reward-batch-size 8 \
      --height 512 \
      --width 512 \
      --steps 4 \
      --cfg 1.0 \
      --audit-first-candidate \
      --out "$evaluation"
  fi
  selection_args+=(--selection-evaluation "$evaluation")
  prior_evaluation_args+=(--prior-evaluation "$evaluation")
done

selection_decision="$output_root/selection-decision.json"
if [[ ! -f "$selection_decision" ]]; then
  "$mage_venv/bin/python" \
    "$repo_root/scripts/finalize_mage_flow_es_cross_prompt_gate_20x.py" \
    --gradient-report "$gradient_report" \
    "${selection_args[@]}" \
    --out "$selection_decision"
fi

selected_center=$(
  "$mage_venv/bin/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_candidate"]["center_clip_path"])' \
    "$selection_decision"
)
selected_alpha=$(
  "$mage_venv/bin/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_candidate"]["alpha"])' \
    "$selection_decision"
)

confirmation_args=()
for episode in $(seq 32 39); do
  evaluation="$output_root/confirmation-episode-$episode.json"
  if [[ ! -f "$evaluation" ]]; then
    "$mage_venv/bin/python" \
      "$repo_root/scripts/evaluate_mage_flow_es_centers.py" \
      --model-path "$model_path" \
      --prompts-jsonl "$pool_root/es_train.jsonl" \
      --calibration "$pool_root/bf16-reward-calibration.json" \
      --hps-checkpoint "$hps_checkpoint" \
      --clip-checkpoint "$clip_checkpoint" \
      --source-run "${source_runs[0]}" \
      "${additional_source_args[@]}" \
      "${prior_evaluation_args[@]}" \
      --candidate "initial=$initial_center" \
      --candidate "selected-alpha-$selected_alpha=$selected_center" \
      --episode-index "$episode" \
      --episode-size 8 \
      --schedule-seed 20260904 \
      --reward-batch-size 8 \
      --height 512 \
      --width 512 \
      --steps 4 \
      --cfg 1.0 \
      --audit-first-candidate \
      --out "$evaluation"
  fi
  confirmation_args+=(--confirmation-evaluation "$evaluation")
  prior_evaluation_args+=(--prior-evaluation "$evaluation")
done

result="$output_root/result.json"
if [[ ! -f "$result" ]]; then
  "$mage_venv/bin/python" \
    "$repo_root/scripts/finalize_mage_flow_es_cross_prompt_gate_20x.py" \
    --gradient-report "$gradient_report" \
    "${selection_args[@]}" \
    "${confirmation_args[@]}" \
    --out "$result"
fi
"$mage_venv/bin/python" -m json.tool "$result"
