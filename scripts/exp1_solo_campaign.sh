#!/bin/bash
# exp1-solo campaign, one pass per machine (spec: exp1-solo).
#
#   base pass (gooo): solo eval-only of the existing 0.05 refs -> S3 -> S10
#   rl pass  (dc4a):  refs -> S3 -> S10 -> axis2 re-eval (recovery role)
#
# Zero trainer changes: --solo-weight maps to TrainConfig.shadow_loss_weight.
#
#     ssh gooo 'cd ~/Code/shadow-peft && PASS=base nohup bash \
#       scripts/exp1_solo_campaign.sh > runs/logs/$(date +%m%d)-solo-base.log 2>&1 < /dev/null &'
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python

PASS="${PASS:-base}"

case "$PASS" in
  base)
    MODEL="${MODEL:-$HOME/models/LFM2.5-2.6B-Base}"
    DATA="${DATA:-runs/exp1/teacher_2p6b_base.jsonl}"
    BASEFROM=runs/exp1/grace_B0/verdict.json
    REFARMS="grace_B0 grace_Q0 n4b_B0_30m n4b_Q0_30m"
    TAGARGS=()
    ;;
  rl)
    MODEL="${MODEL:-$HOME/models/LFM2.5-2.6B}"
    DATA="${DATA:-runs/exp1/teacher_2p6b_rl_chat.jsonl}"
    HELDOUT="${HELDOUT:-runs/exp1/teacher_2p6b_rl_chat_heldout.jsonl}"
    BASEFROM=runs/exp1/grace-rl_B0/verdict.json
    REFARMS="grace-rl_B0 grace-rl_Q0 n4b-rl_B0_30m n4b-rl_Q0_30m"
    TAGARGS=(--tag rl --heldout "$HELDOUT")
    ;;
  *) echo "[solo] unknown PASS: $PASS"; exit 1;;
esac

for SRC in $REFARMS; do
  echo ""
  echo "[solo] === ref solo eval: $SRC (lambda=0.05) ==="
  $PY scripts/exp1_solo_arm.py --ckpt "runs/exp1/$SRC" \
    --src-verdict "runs/exp1/$SRC/verdict.json" --model "$MODEL" "${TAGARGS[@]}"
done

for W in 0.3 1.0; do
  echo ""
  echo "[solo] === arm S (shadow-CE x$W) @1e7 ==="
  $PY scripts/exp1_solo_arm.py --solo-weight "$W" --model "$MODEL" --data "$DATA" \
    --max-tokens 1e7 --baselines-from "$BASEFROM" "${TAGARGS[@]}"
done

if [ "$PASS" = "rl" ]; then
  echo ""
  echo "[solo] === axis2 re-eval of S3/S10 (recovery role, primary axis) ==="
  $PY scripts/exp1_grace_axis2.py --tag rl --in-prefix solo-rl \
    --arms S3,S10 --model "$MODEL" --data "$HELDOUT" --out-prefix solo-rl-axis2
fi

echo ""
echo "[solo] $PASS pass complete"
