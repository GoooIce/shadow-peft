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
#   bash scripts/download_assets_modelscope.sh mage         # Mage-Flow-Turbo only
#   bash scripts/download_assets_modelscope.sh mage-prompts # HPS-v2 prompt source only
#   bash scripts/download_assets_modelscope.sh mage-rewards # HPS/CLIP + text prompts
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAGE_CHECKSUMS="$SCRIPT_DIR/../assets/manifests/mage-flow-turbo-modelscope.sha256"
MAGE_PROMPT_CHECKSUMS="$SCRIPT_DIR/../assets/manifests/t2i-prompts-hpsv2-modelscope.sha256"
HPS_CHECKSUMS="$SCRIPT_DIR/../assets/manifests/hpsv21-modelscope.sha256"
CLIP_CHECKSUMS="$SCRIPT_DIR/../assets/manifests/clip-h14-modelscope.sha256"
QWEN_IMAGE_PROMPT_CHECKSUMS="$SCRIPT_DIR/../assets/manifests/qwen-image-bench-prompts-modelscope.sha256"
GENEVAL_DETECTOR_CHECKSUMS="$SCRIPT_DIR/../assets/manifests/mask2former-coco-modelscope.sha256"
MODELSCOPE_BIN="${MODELSCOPE_BIN:-modelscope}"

fetch_model () {  # fetch_model <repo-id> <dir-name>
  if [ -s "$DEST/$2/config.json" ]; then
    echo "[fetch] $2 already present — skip"
  else
    echo "[fetch] model $1 -> $DEST/$2"
    modelscope download --model "$1" --local_dir "$DEST/$2"
  fi
}

fetch_mage_flow () {
  local out="$DEST/Mage-Flow-Turbo"
  if [ -s "$out/model_index.json" ]; then
    echo "[fetch] Mage-Flow-Turbo already present — skip"
  else
    echo "[fetch] model microsoft/Mage-Flow-Turbo@master -> $out"
    modelscope download microsoft/Mage-Flow-Turbo \
      --revision master --local-dir "$out"
  fi
  if [ ! -s "$MAGE_CHECKSUMS" ]; then
    echo "Mage-Flow checksum inventory not found at $MAGE_CHECKSUMS" >&2
    exit 3
  fi
  (cd "$out" && sha256sum -c "$MAGE_CHECKSUMS")
}

fetch_mage_prompts () {
  local out="$CORPUS_DIR/t2i-prompts-hpsv2"
  if [ -s "$out/data/test-00000-of-00001.parquet" ]; then
    echo "[fetch] t2i-prompts-hpsv2 already present — skip"
  else
    echo "[fetch] dataset Lakonik/t2i-prompts-hpsv2@master -> $out"
    modelscope download --dataset Lakonik/t2i-prompts-hpsv2 \
      --revision master --local-dir "$out"
  fi
  if [ ! -s "$MAGE_PROMPT_CHECKSUMS" ]; then
    echo "Prompt checksum inventory not found at $MAGE_PROMPT_CHECKSUMS" >&2
    exit 3
  fi
  (cd "$out" && sha256sum -c "$MAGE_PROMPT_CHECKSUMS")
}

fetch_selected_files () {  # fetch_selected_files <repo-type> <repo-id> <out> <checksums> <files...>
  local repo_type="$1"
  local repo_id="$2"
  local out="$3"
  local checksums="$4"
  shift 4
  local local_flag="--local-dir"
  if ! "$MODELSCOPE_BIN" download --help 2>&1 | grep -q -- '--local-dir'; then
    local_flag="--local_dir"
  fi
  mkdir -p "$out"
  for file in "$@"; do
    if [ ! -s "$out/$file" ]; then
      echo "[fetch] $repo_id:$file -> $out/$file"
      "$MODELSCOPE_BIN" download --repo-type "$repo_type" --revision master \
        "$local_flag" "$out" "$repo_id" "$file"
    fi
  done
  if [ ! -s "$checksums" ]; then
    echo "checksum inventory not found at $checksums" >&2
    exit 3
  fi
  (cd "$out" && sha256sum -c "$checksums")
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
  mage)
    mkdir -p "$DEST"
    fetch_mage_flow
    ;;
esac

case "$STAGE" in
  mage-prompts)
    mkdir -p "$CORPUS_DIR"
    fetch_mage_prompts
    ;;
esac

case "$STAGE" in
  mage-rewards)
    fetch_selected_files model AI-ModelScope/HPSv2 "$DEST/HPSv2" "$HPS_CHECKSUMS" \
      HPS_v2.1_compressed.pt configuration.json README.md
    fetch_selected_files model laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
      "$DEST/CLIP-ViT-H-14-laion2B-s32B-b79K" "$CLIP_CHECKSUMS" \
      config.json merges.txt open_clip_config.json open_clip_model.safetensors \
      preprocessor_config.json README.md special_tokens_map.json tokenizer.json \
      tokenizer_config.json vocab.json
    fetch_selected_files dataset Qwen/Qwen-Image-Bench \
      "$CORPUS_DIR/qwen-image-bench" "$QWEN_IMAGE_PROMPT_CHECKSUMS" \
      metadata/bench_metadata.json qwen_image_bench_hf_v0518.jsonl
    fetch_selected_files model facebook/mask2former-swin-small-coco-instance \
      "$DEST/mask2former-swin-small-coco-instance" "$GENEVAL_DETECTOR_CHECKSUMS" \
      config.json configuration.json model.safetensors preprocessor_config.json README.md
    ;;
esac

case "$STAGE" in
  corpora|all)
    mkdir -p "$CORPUS_DIR"
    python - <<EOF
# Sample the shadow-pretraining corpora WITHOUT MsDataset (its streaming
# wrapper breaks against datasets>=4: load_dataset_with_ctx() signature
# clash). Instead: ModelScope dataset file-tree API -> bounded parquet
# shard download -> pyarrow local sampling into plain jsonl {"text": ...}.
import json, os, urllib.parse, urllib.request
import pyarrow.parquet as pq

DOCS = int("${CORPUS_DOCS}")
OUT = os.path.expanduser("${CORPUS_DIR}")
API = "https://modelscope.cn/api/v1/datasets"

def list_parquet(repo):
    url = f"{API}/{repo}/repo/tree?Revision=master&Recursive=true"
    with urllib.request.urlopen(url, timeout=60) as r:
        files = json.load(r)["Data"]["Files"]
    paths = sorted(f["Path"] for f in files if f["Path"].endswith(".parquet"))
    if not paths:
        raise SystemExit(f"no parquet files listed for {repo}")
    return paths

def sample(repo, out_name, n):
    path = os.path.join(OUT, out_name)
    if os.path.exists(path) and sum(1 for _ in open(path)) >= n:
        print(f"[corpus] {out_name} already complete — skip")
        return
    written = 0
    shard_i = 0
    with open(path, "w") as f:
        for p in list_parquet(repo):
            if written >= n:
                break
            local = os.path.join(OUT, f"_shard_{repo.replace('/', '_')}_{shard_i}.parquet")
            if not os.path.exists(local):
                url = f"{API}/{repo}/repo?Revision=master&FilePath={urllib.parse.quote(p)}"
                print(f"[corpus] {repo}: shard {shard_i} <- {p}")
                urllib.request.urlretrieve(url, local)
            tbl = pq.ParquetFile(local)
            names = tbl.schema_arrow.names
            col = "text" if "text" in names else names[0]
            for rg in range(tbl.num_row_groups):
                if written >= n:
                    break
                # row-group at a time: the 2GB no-GPU boot mode OOM-kills a
                # whole-file read_table on ~1GB shards
                chunk = tbl.read_row_group(rg, columns=[col])
                for text in chunk.column(col).to_pylist():
                    if written >= n:
                        break
                    text = text or ""
                    if text.strip():
                        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                        written += 1
            shard_i += 1
    # keep only the sampled jsonl; shards are scratch
    for i in range(shard_i):
        scratch = os.path.join(OUT, f"_shard_{repo.replace('/', '_')}_{i}.parquet")
        if os.path.exists(scratch):
            os.remove(scratch)
    print(f"[corpus] wrote {written} docs -> {path}")

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
