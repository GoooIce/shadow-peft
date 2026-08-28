#!/bin/bash
# Exp2 chain: chat teacher data (5M) -> v1 draft distillation with tau trajectory.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "[chain] teacher chat data (5M tokens)"
$PY -m shadow_lab.teacher_data --model ~/models/LFM2.5-1.2B-Instruct \
    --chat --prompts-file runs/exp2/shared/prompts.jsonl \
    --out runs/exp2/shared/teacher_chat_1p2b.jsonl \
    --num-seq 17000 --seq-len 512

echo "[chain] v1 distillation"
$PY scripts/exp2_v1.py --max-tokens 5e6

echo "[chain] exp2 v1 done"
cat runs/exp2/v1/verdict.json
