#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 {rtn|quasar-grid} OUTPUT_DIR [PROMPT_INDEX]" >&2
  exit 2
fi

scheme=$1
output_dir=$2
prompt_index=${3:-0}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mage_venv=${MAGE_FLOW_VENV:-"$HOME/Code/Mage/mage_flow/.venv"}
model_path=${MAGE_FLOW_MODEL_PATH:-"$HOME/models/Mage-Flow-Turbo"}
prompts_path=${MAGE_FLOW_PROMPTS_PATH:-"$HOME/runs/mage-flow/prompt-filter-hpsv2-32-20260904/accepted.jsonl"}
reference_dir=${MAGE_FLOW_REFERENCE_DIR:-"$HOME/runs/mage-flow/bf16-quality-golden-32-20260904"}
seed=$((525200 + prompt_index))
reference_image="$reference_dir/gen_$(printf '%04d' "$prompt_index")_seed_${seed}.png"
log_path=${MAGE_FLOW_LOG_PATH:-"$HOME/logs/mage-flow-nvfp4-${scheme}-prompt-${prompt_index}.log"}

case "$scheme" in
  rtn | quasar-grid) ;;
  *)
    echo "scheme must be rtn or quasar-grid" >&2
    exit 2
    ;;
esac

for required in \
  "$mage_venv/bin/python" \
  "$model_path/model_index.json" \
  "$prompts_path" \
  "$reference_image"; do
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
echo "launching $scheme prompt=$prompt_index output=$output_dir log=$log_path"
exec setsid "$mage_venv/bin/python" \
  "$repo_root/scripts/run_mage_flow_nvfp4_baseline.py" \
  --model-path "$model_path" \
  --scheme "$scheme" \
  --prompts-jsonl "$prompts_path" \
  --prompt-index "$prompt_index" \
  --seed "$seed" \
  --height 512 \
  --width 512 \
  --steps 4 \
  --cfg 1.0 \
  --reference-image "$reference_image" \
  --out "$output_dir" \
  >"$log_path" 2>&1 < /dev/null
