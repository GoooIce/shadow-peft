#!/bin/bash
# 16 票：UltraData 去污染基线下载（只做重叠检查，不作题集/评测集）。
# 目标机：goooinext（1.9 TB 空闲）。运行方式（远端）：
#   bash ~/runs/minicpm5-taskset/download_ultradata.sh
# 预算：约 5 GB / 15 min 网络。全量下 Math+Knowledge+Long_Context；Code/SFT 层做
# 确定性抽样（记录覆盖度），因为 Code 层 170 GB、SFT-2605 需 gated token。
set -uo pipefail
export PATH="$HOME/venvs/minicpm5-vllm/bin:$PATH"
ROOT="$HOME/data/ultradata"
LOG="$ROOT/download.log"
mkdir -p "$ROOT"
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---- 0) token（从 gooo 复制，仅本地使用，不入库） ----
mkdir -p "$HOME/.cache/huggingface"
if [ ! -s "$HOME/.cache/huggingface/token" ]; then
  ssh -o BatchMode=yes gooo 'cat ~/.cache/huggingface/token' > "$HOME/.cache/huggingface/token" 2>/dev/null || true
fi
say "token present: $([ -s $HOME/.cache/huggingface/token ] && echo yes || echo no)"

say "=== 1/4 RL-2609 Math + Knowledge + Long_Context (full) ==="
hf download openbmb/UltraData-RL-2609 --repo-type dataset \
  --include "data/Math/*" "data/Knowledge/*" "data/Long_Context/*" \
  --local-dir "$ROOT/rl2609" >>"$LOG" 2>&1
say "rl2609 full done: $(du -sh $ROOT/rl2609 2>/dev/null | cut -f1)"

say "=== 2/4 RL-2609 Code: byte-range 抽样（12 shard x 3 offset x 3 MB） ==="
python3 - <<'PY' >>"$LOG" 2>&1
import json, os, subprocess, urllib.request
tok = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
out = os.path.expanduser("~/data/ultradata/rl2609_code_samples")
os.makedirs(out, exist_ok=True)
api = "https://huggingface.co/api/datasets/openbmb/UltraData-RL-2609/tree/main/data/Code?recursive=true"
files = json.load(urllib.request.urlopen(api))
for f in files:
    if f["type"] != "file":
        continue
    p, sz = f["path"], f["size"]
    name = os.path.basename(p)
    for frac in (0.10, 0.50, 0.90):
        off = int(sz * frac); end = min(off + 3_000_000, sz - 1)
        url = f"https://huggingface.co/datasets/openbmb/UltraData-RL-2609/resolve/main/{p}"
        dest = f"{out}/{name}.f{int(frac*100)}"
        r = subprocess.run(["curl", "-sL", "-H", f"Authorization: Bearer {tok}",
                            "-r", f"{off}-{end}", "-o", dest, url])
        print(name, frac, os.path.getsize(dest) if os.path.exists(dest) else -1, "rc", r.returncode, flush=True)
PY
say "code samples: $(du -sh $ROOT/rl2609_code_samples 2>/dev/null | cut -f1)"

say "=== 3/4 SFT-2605 (gated) 确定性抽样 ==="
hf download openbmb/UltraData-SFT-2605 --repo-type dataset \
  --include "data/no_think/Math/Math_no_think_part-001-of-300.jsonl" \
            "data/no_think/Math/Math_no_think_part-051-of-300.jsonl" \
            "data/no_think/Math/Math_no_think_part-101-of-300.jsonl" \
            "data/no_think/Math/Math_no_think_part-151-of-300.jsonl" \
            "data/no_think/Math/Math_no_think_part-201-of-300.jsonl" \
            "data/no_think/Math/Math_no_think_part-251-of-300.jsonl" \
            "data/no_think/Knowledge/Knowledge_no_think_part-01-of-80.jsonl" \
            "data/no_think/Knowledge/Knowledge_no_think_part-21-of-80.jsonl" \
            "data/no_think/Knowledge/Knowledge_no_think_part-41-of-80.jsonl" \
            "data/no_think/Knowledge/Knowledge_no_think_part-61-of-80.jsonl" \
  --local-dir "$ROOT/sft2605" >>"$LOG" 2>&1
say "sft2605 done: $(du -sh $ROOT/sft2605 2>/dev/null | cut -f1)"

say "=== 4/4 SFT-Agent-2609 确定性抽样 ==="
hf download openbmb/UltraData-SFT-Agent-2609 --repo-type dataset \
  --include "data/Tool_Use/Tool_Use_part-4-of-9.jsonl" \
            "data/Tool_Use/Tool_Use_part-5-of-9.jsonl" \
            "data/Search_Agent/Search_Agent_part-2-of-2.jsonl" \
            "data/General_Agent/General_Agent_part-16-of-32.jsonl" \
  --local-dir "$ROOT/sft_agent2609" >>"$LOG" 2>&1
say "sft_agent done: $(du -sh $ROOT/sft_agent2609 2>/dev/null | cut -f1)"

say "=== manifest ==="
find "$ROOT" -type f ! -name "download.log" -printf "%s\t%p\n" | sort -k2 > "$ROOT/manifest.tsv"
wc -l < "$ROOT/manifest.tsv" | xargs say "files:"
du -sh "$ROOT" | cut -f1 | xargs say "total:"
say "ALL DONE"
