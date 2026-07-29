#!/bin/bash
# Run all the deep-research-report.md "not-yet-done" analyses end to end.
#
# Produces three report directories under experiment/:
#   analysis_container/      - safetensors header / shape / bpw / codebook (report §布局)
#   analysis_error_layers/   - weight MAE/max-abs + hidden cos + logits KL/top-k (report §评估指标)
#   analysis_short_gen/      - greedy first-token consistency (report §验证计划 last row)
#
# Usage:
#   bash experiment/run_analysis_experiments.sh
#   MODEL=/path/to/Qwen3-1.7B bash experiment/run_analysis_experiments.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PY="${PY:-.venv/bin/python}"
MODEL="${MODEL:-/root/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master}"

echo "=== [1/3] container format (header / shape / bpw / codebook) ==="
"$PY" experiment/analyze_container_format.py --model "$MODEL"

echo
echo "=== [2/3] three-layer error (weight MAE/max + hidden + logits KL/topk) ==="
"$PY" experiment/analyze_error_layers.py --model "$MODEL"

echo
echo "=== [3/3] short generation (greedy first-token consistency) ==="
"$PY" experiment/analyze_short_gen.py --model "$MODEL"

echo
echo "=== Done. Reports: ==="
for d in analysis_container analysis_error_layers analysis_short_gen; do
  echo "  experiment/$d/REPORT.md"
done
