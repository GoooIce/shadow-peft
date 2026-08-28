"""Exp3 B2: recirculation on the SHADOW hidden-state stream (original idea).

Anchor = exp1 rung A products (8-bit near-lossless base, zero-damage shadow,
ppl_shadow 52.52 on the frozen axis). Rebuild the exact eval stack (quantized
base + distilled shadow), then recirculate *within the shadow model's own
layer stream*: the shadow mini-LM's last-layer output at token t-1
(norm-matched) mixes into its first layer's input at token t — the same
recipe as B1, applied to the ShadowUpdate stream instead of the backbone.

Judgment (spec): ppl improvement >= 2% vs the same-machinery no-recirc
baseline -> original idea holds; 0-2% -> "composable, no gain"; negative ->
terminate and record.

Run on the first GB10 (dc4a, holds the rung A artifacts):
    ssh goooinext 'cd ~/Code/shadow-peft && .venv/bin/python scripts/exp3_b2.py'
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_lab.atoms import despace_wikitext, load_wikitext2_test_text
from shadow_lab.specdec import _cached_kwargs
from shadow_lab.trainer import TrainConfig, _quantize_base
from shadow_peft import ShadowForCausalLM, ShadowPeftModel
from exp3_b1 import Recirculator, serial_ppl


def build_rungA(cfg_bits8: TrainConfig, tok_len: int, adapter_dir: Path, device: str):
    base = AutoModelForCausalLM.from_pretrained(cfg_bits8.model, dtype=cfg_bits8.profile.dtype)
    _quantize_base(base, cfg_bits8, tok_len)
    base = base.to(device).eval()
    # no shadow_model arg: from_pretrained builds the implicit shadow on the
    # quantized base — the exact structure the adapter was saved from.
    peft = ShadowPeftModel.from_pretrained(base, adapter_dir)
    task = ShadowForCausalLM(peft, inference_mode="base_shadow")
    return task.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "models" / "LFM2.5-350M-Base"))
    ap.add_argument("--adapter", default="runs/exp1/rungA/train/final")
    ap.add_argument("--out", default="runs/exp3/b2")
    ap.add_argument("--context-len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=16000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    text = despace_wikitext(load_wikitext2_test_text())
    ids = tok.encode(text)
    b = getattr(tok, "bos_token_id", None)
    if b is not None and ids and ids[0] != b:
        ids = [b] + ids

    cfg = TrainConfig(bits=8, loss_mode="ce_kl", model=args.model,
                      data="runs/exp1/teacher_lfm25_350m_base.jsonl",
                      max_tokens=1, seq_len=512, batch_seq=8, lr=1e-5,
                      shadow_layers=2, thr_ratio=0.5, ppl_context_len=512, pad_id=pad_id)
    task = build_rungA(cfg, len(tok), Path(args.adapter), args.device)

    shadow_lm = task.peft_model.shadow_model  # bare backbone (Lfm2Model)
    shadow_layers = shadow_lm.layers
    n_layers = len(shadow_layers)
    print(f"shadow stream: {n_layers} layers; recirc s={n_layers - 1} -> d=0", flush=True)

    def run(alpha, beta, r, limit):
        recirc = Recirculator(shadow_layers, n_layers - 1, 0, alpha, beta, r) if alpha > 0 else None
        t0 = time.time()
        ppl = serial_ppl(task, ids, context_len=args.context_len, device=args.device,
                         limit=limit)
        if recirc:
            recirc.close()
        return ppl, round(time.time() - t0, 1)

    base_ppl, sec = run(0.0, 1.0, 1, args.limit)
    base_rec = {"config": "rungA_no_recirc", "ppl": round(base_ppl, 4), "s_sec": sec,
                "anchor_axis_note": "exp1 windowed ppl_shadow was 52.52; serial-stream axis differs slightly"}
    print(json.dumps(base_rec), flush=True)
    with open(out_dir / "sweep.jsonl", "a") as f:
        f.write(json.dumps(base_rec) + "\n")

    rows = []
    for alpha in (0.07, 0.15):
        for beta_v in (round(1.0 - alpha, 4), 1.0):
            for r in (1, 2):
                ppl, sec = run(alpha, beta_v, r, args.limit)
                rec = {"s": n_layers - 1, "d": 0, "alpha": alpha, "beta": beta_v, "r": r,
                       "ppl": round(ppl, 4),
                       "delta_pct": round((ppl / base_ppl - 1) * 100, 3), "s_sec": sec}
                rows.append(rec)
                print(json.dumps(rec), flush=True)
                with open(out_dir / "sweep.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")

    best = min(rows, key=lambda x: x["ppl"])
    verdict = {
        "axis": "despaced wikitext + BOS, continuous serial stream, bf16, quant(8bit)+shadow stack",
        "baseline_ppl": round(base_ppl, 4),
        "best_config": best,
        "judgment": "PASS_shadow_stream_injectable" if best["delta_pct"] <= -2.0 else
                    ("COMPOSABLE_NO_GAIN" if best["delta_pct"] < 0 else "NEGATIVE_terminate"),
        "success_line": "ppl improvement >= 2% vs no-recirc same-machinery baseline",
    }
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
