#!/bin/bash
# exp1-native4b single-machine pass (spec: exp1-native4b).
#
#   base pass (gooo): B0_30m -> Q0_30m -> PTQ-swap@30M -> N0 -> N1  (wiki axis)
#   rl pass  (dc4a):  B0_30m -> Q0_30m -> N0 -> axis2 re-eval (PRIMARY) + on-axis PTQ-swap
#
# The 10M references (grace_* / grace-rl-* / grace-rl-axis2_* verdicts) are
# REUSED, never re-run. Structure frozen by exp1-grace (L1_5x = 53760); no
# probe. ppl never compares across the two passes (each B0 is its own anchor).
#
# Env overrides (defaults in parentheses):
#   PASS (base)   MODEL (per pass)   LR (1e-5)   TOKENS_10M (1e7)   TOKENS_30M (3e7)
#   BASE_DATA (runs/exp1/teacher_2p6b_base.jsonl)
#   RL_DATA (runs/exp1/teacher_2p6b_rl_chat.jsonl)
#   RL_HELDOUT (runs/exp1/teacher_2p6b_rl_chat_heldout.jsonl)
#
#     ssh gooo 'cd ~/Code/shadow-peft && PASS=base nohup bash \
#       scripts/exp1_native4b_campaign.sh > runs/logs/$(date +%m%d)-n4b-base.log 2>&1 < /dev/null &'
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python

PASS="${PASS:-base}"
LR="${LR:-1e-5}"
T10="${TOKENS_10M:-1e7}"
T30="${TOKENS_30M:-3e7}"
INTER=53760

run_arm () {  # run_arm <arm> <suffix> <tokens> <model> <data> <baselines-from>
  local ARM="$1" SUFFIX="$2" TOKENS="$3" MODEL="$4" DATA="$5" BASEFROM="$6"
  echo ""
  echo "[n4b] === arm ${ARM}${SUFFIX:+_$SUFFIX} @${TOKENS} tokens ==="
  $PY scripts/exp1_native4b_arm.py --arm "$ARM" ${SUFFIX:+--suffix "$SUFFIX"} \
    --tag "$TAG" --model "$MODEL" --data "$DATA" --lr "$LR" --shadow-intermediate "$INTER" \
    --max-tokens "$TOKENS" --baselines-from "$BASEFROM"
}

case "$PASS" in
  base)
    TAG=""
    MODEL="${MODEL:-$HOME/models/LFM2.5-2.6B-Base}"
    DATA="${BASE_DATA:-runs/exp1/teacher_2p6b_base.jsonl}"
    BASEFROM=runs/exp1/grace_B0/verdict.json
    run_arm B0 30m "$T30" "$MODEL" "$DATA" "$BASEFROM"
    run_arm Q0 30m "$T30" "$MODEL" "$DATA" "$BASEFROM"
    echo ""
    echo "[n4b] === PTQ-swap nvfp4 anchor on B0_30m (wiki axis) ==="
    $PY scripts/eval_shadow_precision.py --model "$MODEL" \
      --ckpt runs/exp1/n4b_B0_30m/train/final --modes nvfp4 \
      --out runs/exp1/n4b_shadow_precision_30m.json
    run_arm N0 "" "$T10" "$MODEL" "$DATA" "$BASEFROM"
    run_arm N1 "" "$T10" "$MODEL" "$DATA" "$BASEFROM"
    ;;
  rl)
    TAG="rl"
    MODEL="${MODEL:-$HOME/models/LFM2.5-2.6B}"
    DATA="${RL_DATA:-runs/exp1/teacher_2p6b_rl_chat.jsonl}"
    HELDOUT="${RL_HELDOUT:-runs/exp1/teacher_2p6b_rl_chat_heldout.jsonl}"
    BASEFROM=runs/exp1/grace-rl_B0/verdict.json
    run_arm B0 30m "$T30" "$MODEL" "$DATA" "$BASEFROM"
    run_arm Q0 30m "$T30" "$MODEL" "$DATA" "$BASEFROM"
    run_arm N0 "" "$T10" "$MODEL" "$DATA" "$BASEFROM"
    echo ""
    echo "[n4b] === axis2 re-eval (PRIMARY axis for the rl pass) + on-axis PTQ-swap ==="
    $PY scripts/exp1_grace_axis2.py --tag rl --in-prefix n4b-rl \
      --arms B0_30m,Q0_30m,N0 --ptq-arms B0_30m \
      --model "$MODEL" --data "$HELDOUT" --out-prefix n4b-rl-axis2
    ;;
  *)
    echo "[n4b] unknown PASS: $PASS (expected 'base' or 'rl')"; exit 1;;
esac

echo ""
echo "[n4b] $PASS pass complete — verdicts under runs/exp1/n4b*_*/verdict.json"
