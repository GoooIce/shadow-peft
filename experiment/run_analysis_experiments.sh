#!/bin/bash
# Run all the deep-research-report.md "not-yet-done" analyses end to end.
#
# Produces four report directories under experiment/:
#   analysis_container/               - safetensors header / shape / bpw / codebook (report §布局)
#   analysis_error_layers/            - weight MAE/max-abs + hidden cos + logits KL/top-k (report §评估指标)
#   analysis_short_gen/               - greedy first-token consistency (report §验证计划 last row)
#   analysis_bonsai_vs_unpacked/      - real Bonsai packed vs unpacked FP16 (report §改进建议)
#
# Usage:
#   bash experiment/run_analysis_experiments.sh
#   MODEL=/path/to/Qwen3-1.7B bash experiment/run_analysis_experiments.sh
#
#   # The [4/4] step needs the real Bonsai checkpoints (~63 GB). Point at local
#   # dirs or let it auto-download from ModelScope with BONSAI_DOWNLOAD=1.
#   BONSAI_DOWNLOAD=1 bash experiment/run_analysis_experiments.sh
#   PACKED_DIR=/data/Ternary-Bonsai-27B-mlx-2bit \
#     UNPACKED_DIR=/data/Ternary-Bonsai-27B-unpacked \
#     bash experiment/run_analysis_experiments.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PY="${PY:-.venv/bin/python}"
MODEL="${MODEL:-/root/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master}"

echo "=== [1/4] container format (header / shape / bpw / codebook) ==="
"$PY" experiment/analyze_container_format.py --model "$MODEL"

echo
echo "=== [2/4] three-layer error (weight MAE/max + hidden + logits KL/topk) ==="
"$PY" experiment/analyze_error_layers.py --model "$MODEL"

echo
echo "=== [3/4] short generation (greedy first-token consistency) ==="
"$PY" experiment/analyze_short_gen.py --model "$MODEL"

echo
echo "=== [4/4] real Bonsai packed vs unpacked FP16 (needs ~63 GB download) ==="
PACKED_DIR="${PACKED_DIR:-}"
UNPACKED_DIR="${UNPACKED_DIR:-}"
BONSAI_ARGS=()
if [[ "${BONSAI_DOWNLOAD:-0}" == "1" ]]; then
  BONSAI_ARGS+=(--download)
else
  [[ -n "$PACKED_DIR" ]] && BONSAI_ARGS+=(--packed-dir "$PACKED_DIR")
  [[ -n "$UNPACKED_DIR" ]] && BONSAI_ARGS+=(--unpacked-dir "$UNPACKED_DIR")
fi
"$PY" experiment/analyze_bonsai_vs_unpacked.py "${BONSAI_ARGS[@]}"

echo
echo "=== Done. Reports: ==="
for d in analysis_container analysis_error_layers analysis_short_gen analysis_bonsai_vs_unpacked; do
  echo "  experiment/$d/REPORT.md"
done
