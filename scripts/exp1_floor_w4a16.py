"""exp1-floor P3 diagnostic: w4a16 backbone + L2 shadow @10M (quant-noise probe).

Separates "2-bit noise floor" from "shadow capacity floor": if the same
shadow recipe on a 4-bit backbone lands far lower, the 2-bit line's floor is
quantization noise; if it plateaus similarly, the shadow recipe itself binds.

    ssh gooo 'cd ~/Code/shadow-peft && nohup .venv/bin/python \
      scripts/exp1_floor_w4a16.py > runs/logs/<date>-floor-w4a16.log 2>&1 < /dev/null &'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_lab import tiers
from shadow_lab.atoms import despace_wikitext, eval_ppl, load_wikitext2_test_text
from shadow_lab.trainer import TrainConfig, train_shadow

AXIS_CTX = 512
AXIS_CHUNKS = 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "models" / "LFM2.5-350M-Base"))
    ap.add_argument("--data", default="runs/exp1/teacher_floor_pool.jsonl")
    ap.add_argument("--out-dir", default="runs/exp1/floor_w4a16_L2_10m")
    ap.add_argument("--max-tokens", type=float, default=1e7)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--shadow-layers", type=int, default=2)
    ap.add_argument("--shadow-intermediate", type=int, default=None)
    ap.add_argument("--shadow-heads", type=int, default=None)
    ap.add_argument("--shadow-kv-heads", type=int, default=None)
    ap.add_argument("--shadow-quant", default=None, choices=["nvfp8", "nvfp4"],
                    help="native shadow-weight precision (projected-gradient training)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    axis_text = despace_wikitext(load_wikitext2_test_text())

    def ppl(model) -> float:
        return eval_ppl(model, tok, axis_text, chunks=AXIS_CHUNKS, device="cuda",
                        context_len=AXIS_CTX)

    print("=== 1/3 bf16 baseline ===", flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    ppl_bf16 = ppl(base)
    print(f"ppl_bf16 = {ppl_bf16:.4f}", flush=True)
    del base
    torch.cuda.empty_cache()

    print("=== 2/3 w4a16 quant-only baseline ===", flush=True)
    quant_copy = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    tiers.apply_training_conversion(quant_copy, "w4a16", torch.device("cuda"), fake_quant=True)
    ppl_quant = ppl(quant_copy)
    print(f"ppl_quant(w4a16 fake) = {ppl_quant:.4f}", flush=True)
    del quant_copy
    torch.cuda.empty_cache()

    print("=== 3/3 shadow distillation on w4a16 backbone ===", flush=True)
    cfg = TrainConfig(
        bits=0,
        backbone_tier="w4a16",
        tier_fake_quant=True,
        loss_mode="ce_kl",
        model=args.model,
        data=args.data,
        max_tokens=args.max_tokens,
        seq_len=512,
        batch_seq=8,
        lr=args.lr,
        shadow_layers=args.shadow_layers,
        shadow_intermediate_size=args.shadow_intermediate,
        shadow_num_attention_heads=args.shadow_heads,
        shadow_num_key_value_heads=args.shadow_kv_heads,
        shadow_quant=args.shadow_quant,
        ppl_context_len=AXIS_CTX,
        pad_id=pad_id,
        save_dir=str(out_dir / "train"),
    )
    report = train_shadow(cfg)
    ppl_shadow = report.final_ppl_20chunks

    recovery = (ppl_quant - ppl_shadow) / (ppl_quant - ppl_bf16)
    verdict = {
        "tier": "w4a16",
        "shadow_layers": args.shadow_layers,
        "shadow_intermediate": args.shadow_intermediate,
        "shadow_heads": args.shadow_heads,
        "shadow_kv_heads": args.shadow_kv_heads,
        "shadow_quant": args.shadow_quant,
        "ppl_bf16": round(ppl_bf16, 4),
        "ppl_quant": round(ppl_quant, 4),
        "ppl_shadow": round(ppl_shadow, 4),
        "recovery": round(recovery, 4),
        "axis": "despaced wikitext, ctx=512, chunks=20",
        "tokens": report.tokens,
        "train_seconds": round(report.wall_time_s, 1),
    }
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
