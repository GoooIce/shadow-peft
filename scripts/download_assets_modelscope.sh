#!/bin/bash
# Asset fetcher for a fresh rented GPU box (exp-line prep, 2026-08-30).
#
# Everything verified on ModelScope at write time:
#   models  : LiquidAI/LFM2.5-2.6B-Base, LiquidAI/LFM2.5-2.6B (RL/post-trained),
#             LiquidAI/LFM2.5-350M-Base (same-family pretrained shadow candidate);
#             optional --with-1p2b adds LiquidAI/LFM2.5-1.2B-Base
#             (LFM2.5-700M-Base is NOT on ModelScope — use hf-mirror if needed)
#   corpora : Datawhale/fineweb-edu-100b-shuffle   (English, the paper's FineWeb-Edu role)
#             opencsg/chinese-fineweb-edu-v2        (Chinese, substitutes the paper's Wudao)
#   eval    : wikitext-2-raw-v1 test parquet via HF_ENDPOINT=hf-mirror.com
#             (the repo's eval axis loader reads it through huggingface_hub)
#
# NOT fetched here (by design):
#   - teacher pools: SELF-DISTILLED, generated on the box after repo+venv are up
#     (python -m shadow_lab.teacher_data --model <m> --out <p> --num-seq ... )
#   - the repo itself: git clone + uv sync (or pip install -e ".[lab]")
#   - no credentials ever go in scripts
#
# Usage:
#   pip install -U modelscope huggingface_hub
#   bash scripts/download_assets_modelscope.sh all          # models + corpora + eval axis
#   bash scripts/download_assets_modelscope.sh models
#   bash scripts/download_assets_modelscope.sh corpora      # sample 100K docs/language
#   bash scripts/download_assets_modelscope.sh eval-axis
#
# Env:  DEST (default ~/models)   CORPUS_DIR (default ~/data)
#       CORPUS_DOCS (default 100000)   SKIP_ZH=1   WITH_1P2B=1
# AutoDL: point DEST/CORPUS_DIR at the data disk (/root/autodl-tmp/...) — the
# system disk is small. After everything, keep HF_ENDPOINT exported for evals.
set -e

DEST="${DEST:-$HOME/models}"
CORPUS_DIR="${CORPUS_DIR:-$HOME/data}"
CORPUS_DOCS="${CORPUS_DOCS:-100000}"
STAGE="${1:-all}"

fetch_model () {  # fetch_model <repo-id> <dir-name>
  if [ -s "$DEST/$2/config.json" ]; then
    echo "[fetch] $2 already present — skip"
  else
    echo "[fetch] model $1 -> $DEST/$2"
    modelscope download --model "$1" --local_dir "$DEST/$2"
  fi
}

case "$STAGE" in
  models|all)
    mkdir -p "$DEST"
    fetch_model LiquidAI/LFM2.5-2.6B-Base LFM2.5-2.6B-Base    # base testbed
    fetch_model LiquidAI/LFM2.5-2.6B      LFM2.5-2.6B         # rl/post-trained testbed
    fetch_model LiquidAI/LFM2.5-350M-Base LFM2.5-350M-Base    # pretrained shadow candidate
    [ "${WITH_1P2B:-0}" = "1" ] && fetch_model LiquidAI/LFM2.5-1.2B-Base LFM2.5-1.2B-Base
    ;;
esac

case "$STAGE" in
  corpora|all)
    mkdir -p "$CORPUS_DIR"
    python - <<EOF
# Stream-sample the shadow-pretraining corpora (no 100B download): fixed doc
# count per language into plain jsonl {"text": ...}, reproducible via seed.
import json, os
from modelscope.msdatasets import MsDataset

DOCS = int("${CORPUS_DOCS}")
OUT = os.path.expanduser("${CORPUS_DIR}")

def sample(repo, out_name, n, text_key="text"):
    path = os.path.join(OUT, out_name)
    if os.path.exists(path) and sum(1 for _ in open(path)) >= n:
        print(f"[corpus] {out_name} already complete — skip")
        return
    print(f"[corpus] streaming {repo} -> {path} ({n} docs)")
    ds = MsDataset.load(repo, split="train", streaming=True)
    with open(path, "w") as f:
        for i, row in enumerate(ds):
            if i >= n:
                break
            text = row.get(text_key) or row.get("content") or ""
            if text.strip():
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"[corpus] wrote {min(i + 1, n)} docs")

sample("Datawhale/fineweb-edu-100b-shuffle", "shadow_pretrain_en.jsonl", DOCS)
if "${SKIP_ZH:-0}" != "1":
    sample("opencsg/chinese-fineweb-edu-v2", "shadow_pretrain_zh.jsonl", DOCS)
EOF
    ;;
esac

case "$STAGE" in
  eval-axis|all)
    mkdir -p "$DEST"
    HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" python - <<'EOF'
# Warm the wikitext-2 test parquet exactly the way the repo's eval axis reads
# it (shadow_lab.atoms.load_wikitext2_test_text), through the HF mirror.
import os
from huggingface_hub import hf_hub_download

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
p = hf_hub_download("wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                    repo_type="dataset")
print(f"[eval-axis] wikitext-2 test parquet cached at {p}")
EOF
    echo "[eval-axis] NOTE: export HF_ENDPOINT=https://hf-mirror.com in the box shell"
    echo "             profile — the repo's eval loader resolves through huggingface_hub"
    ;;
esac

echo ""
echo "[fetch] done. inventory:"
du -sh "$DEST"/* 2>/dev/null || true
du -sh "$CORPUS_DIR"/* 2>/dev/null || true
echo ""
echo "[fetch] next steps on this box: git clone the repo, uv sync (or pip install -e \".[lab]\"),"
echo "        then generate the self-distilled teacher pools with shadow_lab.teacher_data."
