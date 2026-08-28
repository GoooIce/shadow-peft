"""Post-hoc 20-chunk eval of a floor-run BEST adapter (spec: exp1-ppl-floor).

The 160M DDP run's rank-0 crashed between loop end and final save (rank-local
exit-condition desync, since fixed), so the final model was lost — but the
best adapter (8-window checkpoint) survives. This re-mounts it on the exact
2-bit base and evaluates the 20-chunk axis, giving the honest "best model"
ladder point.

    ssh <host> 'cd ~/Code/shadow-peft && .venv/bin/python \
      scripts/exp1_floor_best_eval.py --adapter runs/exp1/floor_L2/train/best --tag L2_40m_best'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_lab.atoms import despace_wikitext, eval_ppl, load_wikitext2_test_text
from shadow_lab.trainer import TrainConfig, _quantize_base
from shadow_peft import ShadowPeftModel

AXIS_CTX = 512
AXIS_CHUNKS = 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default=str(Path.home() / "models" / "LFM2.5-350M-Base"))
    ap.add_argument("--out", default="runs/exp1/floor_best_evals.jsonl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    axis_text = despace_wikitext(load_wikitext2_test_text())
    cfg = TrainConfig(bits=2, loss_mode="ce_kl", model=args.model, data="unused")

    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=cfg.profile.dtype).to("cuda").eval()
    _quantize_base(base, cfg, len(tok))
    task = ShadowPeftModel.from_pretrained(base, args.adapter)
    task = task.to("cuda").eval()

    ppl = eval_ppl(task, tok, axis_text, chunks=AXIS_CHUNKS, device="cuda", context_len=AXIS_CTX)
    rec = {"tag": args.tag, "adapter": args.adapter, "ppl_20chunks": round(ppl, 4),
           "axis": "despaced wikitext, ctx=512, chunks=20"}
    print(json.dumps(rec), flush=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
