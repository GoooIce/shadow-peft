#!/bin/bash
# exp1-rl1b phased campaign (spec: exp1-rl1b) — one machine, one phase at a
# time; the monitoring cron drives the sequence across BOTH machines.
#
#   PHASE=genhalf  HALF=a|b     10 chunks x 100k seqs (seed 303xx/304xx),
#                                chunk-level resume (wc -l check), ~52h/machine
#   PHASE=mergepool             dc4a only: pull b-half, cat halves, md5, push
#                                merged pool back to gooo, write pool_manifest.json
#   PHASE=ab100    FAMILY=B0|Q0 two 100M arms serially (lr 1e-5 then 3e-5)
#   PHASE=axis2    ARMS=<csv> [OUT=rl1b-axis2] [PTQARMS=<csv>]
#                                re-eval on the held-out pool (primary axis)
#   PHASE=final1b  FAMILY=B0|Q0 LR=<x>
#                                single 1B arm at the gate-selected lr
#
#   Env: MODEL (~/models/LFM2.5-2.6B)  PROMPTS (runs/exp2/shared/prompts.jsonl)
#        HELDOUT (runs/exp1/teacher_2p6b_rl_chat_heldout.jsonl)
#        BASEFROM (runs/exp1/grace-rl_B0/verdict.json)
#
#     ssh <host> 'cd ~/Code/shadow-peft && PHASE=genhalf HALF=a nohup bash \
#       scripts/exp1_rl1b_campaign.sh > runs/logs/$(date +%m%d)-rl1b-gen-a.log 2>&1 < /dev/null &'
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python

PHASE="${PHASE:?PHASE required (genhalf|mergepool|ab100|axis2|final1b)}"
MODEL="${MODEL:-$HOME/models/LFM2.5-2.6B}"
PROMPTS="${PROMPTS:-runs/exp2/shared/prompts.jsonl}"
HELDOUT="${HELDOUT:-runs/exp1/teacher_2p6b_rl_chat_heldout.jsonl}"
BASEFROM="${BASEFROM:-runs/exp1/grace-rl_B0/verdict.json}"
POOL=runs/exp1/teacher_2p6b_rl_chat_1b.jsonl
INTER=53760
N_CHUNKS=10
SEQ_PER_CHUNK=100000
T100M=1e8
T1B=1e9

run_arm () {  # run_arm <arm> <suffix> <lr> <tokens> [extra flags]
  local ARM="$1" SUFFIX="$2" LR="$3" TOKENS="$4"; shift 4
  echo ""
  echo "[rl1b] === arm ${ARM}_${SUFFIX} @${TOKENS} lr=${LR} ==="
  $PY scripts/exp1_native4b_arm.py --arm "$ARM" --suffix "$SUFFIX" --tag rl \
    --model "$MODEL" --data "$POOL" --lr "$LR" --shadow-intermediate "$INTER" \
    --max-tokens "$TOKENS" --baselines-from "$BASEFROM" \
    --out-dir "runs/exp1/rl1b_${ARM}_${SUFFIX}" "$@"
}

case "$PHASE" in
  genhalf)
    HALF="${HALF:?HALF required (a=dc4a seed 303xx | b=gooo seed 304xx)}"
    SEEDBASE=30300; [ "$HALF" = "b" ] && SEEDBASE=30400
    for i in $(seq 0 $((N_CHUNKS - 1))); do
      N=$(printf "%02d" "$i")
      F="runs/exp1/teacher_2p6b_rl_chat_1b_${HALF}_${N}.jsonl"
      LINES=$(wc -l < "$F" 2>/dev/null || echo 0)
      if [ "$LINES" -eq "$SEQ_PER_CHUNK" ]; then
        echo "[rl1b] chunk ${HALF}_${N} complete ($LINES seqs) — skip"
        continue
      fi
      echo "[rl1b] === chunk ${HALF}_${N} (seed $((SEEDBASE + i)), partial=$LINES) ==="
      rm -f "$F"
      $PY -m shadow_lab.teacher_data --model "$MODEL" --chat \
        --prompts-file "$PROMPTS" --out "$F" \
        --num-seq "$SEQ_PER_CHUNK" --seq-len 512 --seed $((SEEDBASE + i))
    done
    echo "[rl1b] half $HALF generation complete"
    ;;

  mergepool)
    # dc4a owns half a; pull half b from gooo over the direct link
    rsync -a gooo:Code/shadow-peft/runs/exp1/teacher_2p6b_rl_chat_1b_b_*.jsonl runs/exp1/
    cat $(for i in $(seq 0 $((N_CHUNKS - 1))); do printf "runs/exp1/teacher_2p6b_rl_chat_1b_a_%02d.jsonl " "$i"; done) \
        $(for i in $(seq 0 $((N_CHUNKS - 1))); do printf "runs/exp1/teacher_2p6b_rl_chat_1b_b_%02d.jsonl " "$i"; done) > "$POOL"
    MD5=$(md5sum "$POOL" | cut -d" " -f1)
    ROWS=$(wc -l < "$POOL")
    TOKENS=$($PY -c "import sys, json; t=0
for line in open('$POOL'): t += len(json.loads(line)['ids'])
print(t)")
    echo "[rl1b] merged pool: $ROWS rows, $TOKENS tokens, md5 $MD5"
    rsync -a "$POOL" gooo:Code/shadow-peft/runs/exp1/
    echo "{\"rows\": $ROWS, \"tokens\": $TOKENS, \"md5\": \"$MD5\", \"halves\": {\"a\": \"seed 303xx (dc4a)\", \"b\": \"seed 304xx (gooo)\"}, \"chunks_per_half\": $N_CHUNKS, \"seq_per_chunk\": $SEQ_PER_CHUNK}" > runs/exp1/rl1b_pool_manifest.json
    rsync -a runs/exp1/rl1b_pool_manifest.json gooo:Code/shadow-peft/runs/exp1/
    goooMD5=$(ssh gooo "md5sum Code/shadow-peft/$POOL" | cut -d" " -f1)
    [ "$goooMD5" = "$MD5" ] || { echo "[rl1b] md5 MISMATCH on gooo ($goooMD5 vs $MD5)"; exit 1; }
    echo "[rl1b] pool merged + md5 verified on both machines"
    ;;

  ab100)
    FAMILY="${FAMILY:?FAMILY required (B0|Q0)}"
    run_arm "$FAMILY" 100m 1e-5 "$T100M"
    run_arm "$FAMILY" 100m_lr3 3e-5 "$T100M"
    ;;

  axis2)
    ARMS="${ARMS:?ARMS required (comma-separated, e.g. B0_100m,Q0_100m)}"
    OUT="${OUT:-rl1b-axis2}"
    PTQFLAG=""
    [ -n "${PTQARMS:-}" ] && PTQFLAG="--ptq-arms $PTQARMS"
    $PY scripts/exp1_grace_axis2.py --tag rl --in-prefix rl1b \
      --arms "$ARMS" $PTQFLAG --model "$MODEL" --data "$HELDOUT" --out-prefix "$OUT"
    ;;

  final1b)
    FAMILY="${FAMILY:?FAMILY required (B0|Q0)}"
    LR="${LR:?LR required (gate100-selected)}"
    run_arm "$FAMILY" 1b "$LR" "$T1B" --eval-every 1000
    ;;

  *)
    echo "[rl1b] unknown PHASE: $PHASE"; exit 1;;
esac

echo "[rl1b] $PHASE done"
