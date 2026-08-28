"""exp1-grace ticket 01: B0 anchor param check + structure freeze record.

Measures the trainable-parameter count of candidate shadow structures in the
trainer's own accounting (ShadowForCausalLM.parameters() & requires_grad:
shadow trunk + injection + update modules) on the 2.6B testbed, walks the
intermediate ladder upward from L1_4x until the >=300M constraint is met, and
writes the structure-freeze record (structure.json) into the run directory.

The backbone tier does not enter the count (only shadow-side modules are
trainable), so the probe skips tier conversion entirely.

    ssh <host> 'cd ~/Code/shadow-peft && .venv/bin/python \
      scripts/exp1_grace_probe.py --model ~/models/LFM2.5-2.6B-Base \
      --out-dir runs/exp1/grace_B0 --base-intermediate 10752'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from shadow_peft import ShadowConfig, ShadowForCausalLM, get_shadow_model

MIN_TRAINABLE = 300_000_000


def probe(base, intermediate: int, shadow_layers: int) -> int:
    sc = ShadowConfig(
        num_shadow_layers=shadow_layers,
        injection_hidden_size=16,
        gate_hidden_size=10,
        alpha=0.1,
        dropout=0.0,
        shadow_intermediate_size=intermediate,
    )
    task = ShadowForCausalLM(get_shadow_model(base, sc))
    n = sum(p.numel() for p in task.parameters() if p.requires_grad)
    del task
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "models" / "LFM2.5-2.6B-Base"))
    ap.add_argument("--base-intermediate", type=int, default=10752,
                    help="base FFN intermediate width; ladder steps are multiples of it")
    ap.add_argument("--anchor-multiple", type=int, default=4, help="L1_4x anchor = 4x base intermediate")
    ap.add_argument("--max-multiple", type=int, default=8, help="ladder ceiling")
    ap.add_argument("--shadow-layers", type=int, default=1)
    ap.add_argument("--min-trainable", type=int, default=MIN_TRAINABLE)
    ap.add_argument("--out-dir", default="runs/exp1/grace_probe")
    args = ap.parse_args()

    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    base.eval()

    ladder = []
    frozen = None
    for k in range(args.anchor_multiple, args.max_multiple + 1):
        intermediate = args.base_intermediate * k
        n = probe(base, intermediate, args.shadow_layers)
        ladder.append({"multiple": k, "shadow_intermediate": intermediate, "trainable_params": n})
        print(f"L1_{k}x (intermediate={intermediate:,}): {n:,} trainable", flush=True)
        if n >= args.min_trainable:
            frozen = ladder[-1]
            decision = (
                f"L1_{k}x frozen (first ladder rung >= {args.min_trainable:,})"
                if k != args.anchor_multiple
                else "L1_4x anchor meets the constraint directly — no upward walk needed"
            )
            break
    if frozen is None:
        raise SystemExit(
            f"no ladder rung up to {args.max_multiple}x reaches {args.min_trainable:,} trainable params — "
            "widen the ladder (--max-multiple) or revisit the constraint"
        )

    record = {
        "structure": "L1",
        "shadow_layers": args.shadow_layers,
        "shadow_intermediate": frozen["shadow_intermediate"],
        "intermediate_multiple": frozen["multiple"],
        "trainable_params": frozen["trainable_params"],
        "min_trainable": args.min_trainable,
        "meets_min": True,
        "decision": decision,
        "ladder": ladder,
        "model": args.model,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "structure.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
