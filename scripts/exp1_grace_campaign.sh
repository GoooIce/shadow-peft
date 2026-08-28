#!/bin/bash
# exp1-grace two-pass campaign queue: Base first, then the RL (instruct) variant.
#
# Each pass is a self-contained campaign in its own run namespace (grace_ for
# base, grace-rl_ for the RL variant): B0-B3 ladder -> adjudication ->
# PTQ-swap anchor on B0 -> Q0/Q1 native-nvfp4 cross -> final adjudication.
# The RL pass reuses the structure frozen by the base probe (same 2.6B
# architecture) and generates its own chat-protocol teacher pool
# (instruct models need --chat sampling; a BOS+random pool is off-distribution).
# ppl values are NOT comparable across the two passes — each pass's B0 is its
# own anchor; compare within a pass, contrast conclusions across passes.
#
# Env overrides (defaults in parentheses):
#   BASE_MODEL   (~/models/LFM2.5-2.6B-Base)  — pre-trained base
#   RL_MODEL     (~/models/LFM2.5-2.6B)       — post-trained (SFT + agentic RL) variant
#   BASE_DATA    (runs/exp1/teacher_2p6b_base.jsonl) — floor2v anchor pool (a+b halves)
#   RL_PROMPTS   (runs/exp2/shared/prompts.jsonl)
#   MAX_TOKENS   (1e7)   LR (1e-5)   PASSES (base rl)   RUN_30M (0)
#
#     ssh <host> 'cd ~/Code/shadow-peft && PASSES="base" nohup bash \
#       scripts/exp1_grace_campaign.sh > runs/logs/$(date +%m%d)-grace-base.log 2>&1 < /dev/null &'
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python

BASE_MODEL="${BASE_MODEL:-$HOME/models/LFM2.5-2.6B-Base}"
RL_MODEL="${RL_MODEL:-$HOME/models/LFM2.5-2.6B}"
BASE_DATA="${BASE_DATA:-runs/exp1/teacher_2p6b_base.jsonl}"
RL_DATA="${RL_DATA:-runs/exp1/teacher_2p6b_rl_chat.jsonl}"
RL_PROMPTS="${RL_PROMPTS:-runs/exp2/shared/prompts.jsonl}"
MAX_TOKENS="${MAX_TOKENS:-1e7}"
LR="${LR:-1e-5}"
PASSES="${PASSES:-base rl}"
RUN_30M="${RUN_30M:-0}"

echo "[queue] exp1-grace campaign: passes=($PASSES) max_tokens=$MAX_TOKENS lr=$LR"

# --- structure freeze (once; both passes share the 2.6B architecture) ---
echo "[queue] === structure probe (ticket 01, >=300M trainable ladder) ==="
$PY scripts/exp1_grace_probe.py --model "$BASE_MODEL" --out-dir runs/exp1/grace_probe
INTER=$($PY -c "import json; print(json.load(open('runs/exp1/grace_probe/structure.json'))['shadow_intermediate'])")
echo "[queue] frozen structure: shadow_layers=1 shadow_intermediate=$INTER"

run_pass () {  # run_pass <tag> <model> <data>
  local TAG="$1" MODEL="$2" DATA="$3" PREFIX ARGS=()
  PREFIX="grace"; [ -n "$TAG" ] && PREFIX="grace-$TAG"
  echo ""
  echo "[queue] ################ pass: ${TAG:-base} ($MODEL) ################"

  for ARM in B0 B1 B2 B3; do
    echo "[queue] === $PREFIX $ARM @${MAX_TOKENS} tokens ==="
    $PY scripts/exp1_grace_arm.py --arm "$ARM" --tag "$TAG" --model "$MODEL" \
      --data "$DATA" --lr "$LR" --shadow-intermediate "$INTER" --max-tokens "$MAX_TOKENS"
  done

  echo "[queue] === $PREFIX ladder adjudication ==="
  $PY scripts/exp1_grace_adjudicate.py --tag "$TAG" --runs-root runs/exp1 \
    --out "runs/exp1/${PREFIX}_adjudication/ladder_verdict.json"
  RECIPE=$($PY -c "import json; print(json.load(open('runs/exp1/${PREFIX}_adjudication/ladder_verdict.json'))['recipe_valid'])")
  if [ "$RECIPE" != "True" ]; then
    echo "[queue] $PREFIX: recipe line FAILED (see ladder_verdict) — Q0 still runs for precision attribution; Q1 skipped"
  fi
  WINNER_MODE=$($PY -c "import json; print(json.load(open('runs/exp1/${PREFIX}_adjudication/ladder_verdict.json'))['winner_loss_mode'] or '')")

  echo "[queue] === $PREFIX PTQ-swap nvfp4 anchor (B0 checkpoint) ==="
  $PY scripts/eval_shadow_precision.py --model "$MODEL" \
    --ckpt "runs/exp1/${PREFIX}_B0/train/final" --modes nvfp4 \
    --out "runs/exp1/${PREFIX}_shadow_precision.json"
  PTQ_PPL=$($PY -c "import json; print([r['ppl'] for r in json.load(open('runs/exp1/${PREFIX}_shadow_precision.json')) if r['mode']=='nvfp4'][0])")
  echo "[queue] $PREFIX PTQ-swap nvfp4 ppl = $PTQ_PPL"

  echo "[queue] === $PREFIX Q0 (ce_kl x nvfp4) ==="
  $PY scripts/exp1_grace_arm.py --arm Q0 --tag "$TAG" --model "$MODEL" \
    --data "$DATA" --lr "$LR" --shadow-intermediate "$INTER" --max-tokens "$MAX_TOKENS"

  if [ "$RECIPE" = "True" ]; then
    echo "[queue] === $PREFIX Q1 (winner $WINNER_MODE x nvfp4) ==="
    $PY scripts/exp1_grace_arm.py --arm Q1 --tag "$TAG" --model "$MODEL" \
      --data "$DATA" --lr "$LR" --shadow-intermediate "$INTER" --max-tokens "$MAX_TOKENS" \
      --loss-mode "$WINNER_MODE"
  else
    echo "[queue] $PREFIX Q1 skipped (no qualifying recipe)"
  fi

  echo "[queue] === $PREFIX final adjudication (criteria summary) ==="
  $PY scripts/exp1_grace_adjudicate.py --tag "$TAG" --runs-root runs/exp1 \
    --ptq-swap "runs/exp1/${PREFIX}_shadow_precision.json"

  if [ "$RUN_30M" = "1" ] && [ "$RECIPE" = "True" ]; then
    echo "[queue] === $PREFIX 30M confirmation rerun (winner combo) ==="
    $PY scripts/exp1_grace_arm.py --arm Q1 --tag "$TAG" --model "$MODEL" \
      --data "$DATA" --lr "$LR" --shadow-intermediate "$INTER" --max-tokens 3e7 \
      --loss-mode "$WINNER_MODE" --out-dir "runs/exp1/${PREFIX}_Q1_30m"
    $PY scripts/exp1_grace_adjudicate.py --tag "$TAG" --runs-root runs/exp1 \
      --ptq-swap "runs/exp1/${PREFIX}_shadow_precision.json" \
      --rerun "runs/exp1/${PREFIX}_Q1_30m/verdict.json"
  fi
}

for PASS in $PASSES; do
  case "$PASS" in
    base)
      if [ ! -s "$BASE_DATA" ]; then
        echo "[queue] base teacher pool not found: $BASE_DATA"
        echo "[queue] set BASE_DATA to the floor2v 2.6B pool path on this machine"
        exit 1
      fi
      run_pass "" "$BASE_MODEL" "$BASE_DATA"
      ;;
    rl)
      if [ ! -s "$RL_DATA" ]; then
        # LFM2.5-2.6B is a pure-reasoning post-trained model: the chat
        # template opens a <think> block, so pool continuations carry
        # reasoning traces — on-distribution for self-distillation, and
        # EOS (<|im_end|>) truncation applies as usual.
        echo "[queue] === RL teacher pool generation (chat protocol, ~10M tokens) ==="
        $PY -m shadow_lab.teacher_data --model "$RL_MODEL" --chat \
          --prompts-file "$RL_PROMPTS" --out "$RL_DATA" \
          --num-seq 20000 --seq-len 512
      fi
      run_pass "rl" "$RL_MODEL" "$RL_DATA"
      ;;
    *)
      echo "[queue] unknown pass: $PASS (expected 'base' or 'rl')"; exit 1
      ;;
  esac
done

echo "[queue] all passes complete — verdicts under runs/exp1/grace*_*/verdict.json"
