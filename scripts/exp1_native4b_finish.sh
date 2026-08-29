#!/bin/bash
# exp1-native4b finisher — runs ON THE MAC once both passes have all verdicts.
# Pulls verdicts/summaries/logs (checkpoints stay remote until the archive
# decision) including the reused grace references, then adjudicates both
# passes locally (pure JSON, no GPU).
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOCAL=runs-local/exp1-native4b
R=Code/shadow-peft/runs/exp1
mkdir -p "$LOCAL/logs"

pull_arm () {  # pull_arm <host> <arm-dir>
  local HOST="$1" D="$2"
  mkdir -p "$LOCAL/$D/train"
  rsync -a "$HOST:$R/$D/verdict.json" "$LOCAL/$D/"
  rsync -a "$HOST:$R/$D/train/summary.json" \
            "$HOST:$R/$D/train/train_log.jsonl" "$LOCAL/$D/train/" \
    || echo "[finish] warn: train summary/log missing for $D"
}

echo "[finish] === base pass (gooo) ==="
for D in n4b_B0_30m n4b_Q0_30m n4b_N0 n4b_N1 grace_B0 grace_Q0; do
  pull_arm gooo "$D"
done
rsync -a gooo:"$R/n4b_shadow_precision_30m.json" gooo:"$R/grace_shadow_precision.json" "$LOCAL/"
rsync -a gooo:Code/shadow-peft/runs/logs/*n4b*.log "$LOCAL/logs/" 2>/dev/null || true

echo "[finish] === rl pass (dc4a) ==="
for D in n4b-rl_B0_30m n4b-rl_Q0_30m n4b-rl_N0 \
         n4b-rl-axis2_B0_30m n4b-rl-axis2_Q0_30m n4b-rl-axis2_N0 n4b-rl-axis2_B0_30m_ptq \
         grace-rl-axis2_B0 grace-rl-axis2_Q0; do
  pull_arm goooinext "$D"
done
rsync -a goooinext:Code/shadow-peft/runs/logs/*n4b*.log "$LOCAL/logs/" 2>/dev/null || true

echo "[finish] === adjudicate (base wiki / rl axis2) ==="
$PY scripts/exp1_native4b_adjudicate.py --pass base --runs-root "$LOCAL"
$PY scripts/exp1_native4b_adjudicate.py --pass rl --runs-root "$LOCAL"

echo "[finish] done — verdicts under $LOCAL"
