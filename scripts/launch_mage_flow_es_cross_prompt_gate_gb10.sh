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
source_runs=()
for episode in 4 5 6; do
  arm_dir="$output_root/gradient-episode-$episode"
  source_runs+=("$arm_dir")
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
      --population-size 30 \
      --episode-size 8 \
      --episode-offset "$episode" \
      --sigma 0.003 \
      --learning-rate 0.0001 \
      --shaping zscore \
      --es-seed 20260906 \
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

gradient_report="$output_root/gradient-report.json"
centers_dir="$output_root/average-gradient-centers"
if [[ ! -f "$gradient_report" ]]; then
  "$mage_venv/bin/python" \
    "$repo_root/scripts/aggregate_mage_flow_es_gradients.py" \
    --source-run "${source_runs[0]}" \
    --source-run "${source_runs[1]}" \
    --source-run "${source_runs[2]}" \
    --expected-episode-offsets 4 5 6 \
    --alphas 0.00003 0.0001 \
    --centers-dir "$centers_dir" \
    --out "$gradient_report"
fi

initial_center="$centers_dir/initial-effective-clips.pt"
small_center="$centers_dir/alpha-3em05-clips.pt"
large_center="$centers_dir/alpha-0p0001-clips.pt"
gate_evaluation="$output_root/center-gate-episode7.json"
if [[ ! -f "$gate_evaluation" ]]; then
  "$mage_venv/bin/python" \
    "$repo_root/scripts/evaluate_mage_flow_es_centers.py" \
    --model-path "$model_path" \
    --prompts-jsonl "$pool_root/es_train.jsonl" \
    --calibration "$pool_root/bf16-reward-calibration.json" \
    --hps-checkpoint "$hps_checkpoint" \
    --clip-checkpoint "$clip_checkpoint" \
    --source-run "${source_runs[0]}" \
    --additional-source-run "${source_runs[1]}" \
    --additional-source-run "${source_runs[2]}" \
    --candidate "initial=$initial_center" \
    --candidate "alpha-3em05=$small_center" \
    --candidate "alpha-0p0001=$large_center" \
    --episode-index 7 \
    --episode-size 8 \
    --schedule-seed 20260904 \
    --reward-batch-size 8 \
    --height 512 \
    --width 512 \
    --steps 4 \
    --cfg 1.0 \
    --audit-first-candidate \
    --out "$gate_evaluation"
fi

gate_decision="$output_root/gate-decision.json"
if [[ ! -f "$gate_decision" ]]; then
  "$mage_venv/bin/python" \
    "$repo_root/scripts/finalize_mage_flow_es_cross_prompt_gate.py" \
    --gradient-report "$gradient_report" \
    --gate-evaluation "$gate_evaluation" \
    --out "$gate_decision"
fi

status=$("$mage_venv/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' \
  "$gate_decision")
if [[ "$status" == "stopped-gate" ]]; then
  echo "EXP-006 stopped at episode 7; no confirmation or main study is permitted"
  exit 0
fi
if [[ "$status" != "awaiting-confirmation" ]]; then
  echo "unexpected gate decision status: $status" >&2
  exit 1
fi

selected_center=$("$mage_venv/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_candidate"]["center_clip_path"])' \
  "$gate_decision")
selected_alpha=$("$mage_venv/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_candidate"]["alpha"])' \
  "$gate_decision")
confirmation="$output_root/center-confirmation-episode8.json"
if [[ ! -f "$confirmation" ]]; then
  "$mage_venv/bin/python" \
    "$repo_root/scripts/evaluate_mage_flow_es_centers.py" \
    --model-path "$model_path" \
    --prompts-jsonl "$pool_root/es_train.jsonl" \
    --calibration "$pool_root/bf16-reward-calibration.json" \
    --hps-checkpoint "$hps_checkpoint" \
    --clip-checkpoint "$clip_checkpoint" \
    --source-run "${source_runs[0]}" \
    --additional-source-run "${source_runs[1]}" \
    --additional-source-run "${source_runs[2]}" \
    --prior-evaluation "$gate_evaluation" \
    --candidate "initial=$initial_center" \
    --candidate "selected-alpha-$selected_alpha=$selected_center" \
    --episode-index 8 \
    --episode-size 8 \
    --schedule-seed 20260904 \
    --reward-batch-size 8 \
    --height 512 \
    --width 512 \
    --steps 4 \
    --cfg 1.0 \
    --audit-first-candidate \
    --out "$confirmation"
fi

result="$output_root/result.json"
if [[ ! -f "$result" ]]; then
  "$mage_venv/bin/python" \
    "$repo_root/scripts/finalize_mage_flow_es_cross_prompt_gate.py" \
    --gradient-report "$gradient_report" \
    --gate-evaluation "$gate_evaluation" \
    --confirmation "$confirmation" \
    --out "$result"
fi
