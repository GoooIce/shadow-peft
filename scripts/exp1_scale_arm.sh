#!/bin/bash
# One scale arm: teacher data (10M tokens) then rung B (10M-token distill).
# Usage: exp1_scale_arm.sh <model_path> <tag> [teacher_batch]
set -e
MODEL="$1"; TAG="$2"; TB="${3:-128}"
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "=== [$TAG] teacher data (10M tokens) ==="
$PY -m shadow_lab.teacher_data --model "$MODEL" \
    --out "runs/exp1/teacher_${TAG}.jsonl" \
    --num-seq 20000 --seq-len 512 --batch "$TB"

echo "=== [$TAG] rung B (bits=2, 10M tokens) ==="
$PY scripts/exp1_rung.py --bits 2 --model "$MODEL" \
    --data "runs/exp1/teacher_${TAG}.jsonl" \
    --out-dir "runs/exp1/rungB_${TAG}" --max-tokens 1e7

echo "=== [$TAG] done ==="
cat "runs/exp1/rungB_${TAG}/verdict.json"
