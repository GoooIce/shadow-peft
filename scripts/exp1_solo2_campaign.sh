#!/bin/bash
# exp1-solo2: explicit pretrained shadows, one arm per GPU (spec: exp1-solo2).
#
#   GPU0: E35 — LFM2.5-350M-Base as the explicit shadow (rl pass, ce_kl +
#         shadow-CE x1.0, 10M tokens)
#   GPU1: E12 — LFM2.5-1.2B-Base ditto
#   after both: axis2 re-eval (recovery role, primary axis) on GPU0
#
# References reused from exp1-solo (no re-run): recovery grace-rl-axis2_B0
# 4.4075 / S10 4.2718; solo solo-rl_ref_grace-rl_B0 799.5 / S10 733.6.
#
#     cd /root/Code/shadow-peft && nohup bash scripts/exp1_solo2_campaign.sh \
#       > runs/logs/solo2.log 2>&1 < /dev/null &
set -e
cd "$(dirname "$0")/.."
PY=/root/miniconda3/bin/python

MODEL="${MODEL:-/root/models/LFM2.5-2.6B}"
DATA="${DATA:-/root/data/teacher_2p6b_rl_chat.jsonl}"
HELDOUT="${HELDOUT:-/root/data/teacher_2p6b_rl_chat_heldout.jsonl}"
BASEFROM="${BASEFROM:-/root/data/refs/grace-rl_B0.verdict.json}"

run_arm () {  # run_arm <gpu> <suffix> <shadow-model-path>
  local GPU="$1" SUFFIX="$2" SHADOW="$3"
  echo "[solo2] === arm $SUFFIX (shadow: $SHADOW) on GPU$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/exp1_solo_arm.py --solo-weight 1.0 \
    --suffix "$SUFFIX" --tag rl --model "$MODEL" --data "$DATA" --heldout "$HELDOUT" \
    --shadow-model "$SHADOW" --max-tokens 1e7 --baselines-from "$BASEFROM" \
    > "runs/logs/solo2-$SUFFIX.log" 2>&1
}

mkdir -p runs/logs
run_arm 0 E35 /root/models/LFM2.5-350M-Base &
PID0=$!
run_arm 1 E12 /root/models/LFM2.5-1.2B-Base &
PID1=$!
wait $PID0 $PID1
echo "[solo2] both arms done — verdicts:"
ls runs/exp1/solo-rl_E35/verdict.json runs/exp1/solo-rl_E12/verdict.json

echo "[solo2] === axis2 re-eval (recovery role, primary) ==="
CUDA_VISIBLE_DEVICES=0 $PY scripts/exp1_grace_axis2.py --tag rl --in-prefix solo-rl \
  --arms E35,E12 --model "$MODEL" --data "$HELDOUT" --out-prefix solo-rl-axis2

echo "[solo2] complete"
