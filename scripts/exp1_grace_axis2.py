"""exp1-grace axis-2 re-evaluation: a campaign's arms on a held-out
self-distillation pool (raw token-id stream — no decode/re-encode round-trip).

Motivation: despaced-wikitext is deeply OOD for the RL (post-trained) model —
the rl pass showed recovery>1 (axis origin distorted) and a KLD/ppl split.
This second axis scores the recovery question on-distribution: ppl and
KLD(bf16 teacher || quant+shadow) over held-out continuations from the same
teacher (same prompts, disjoint seed), full-row stream.

Outputs verdicts in the campaign shape under
``runs/exp1/grace-{tag}-axis2_{arm}/verdict.json`` — the standard adjudicator
then works unchanged: ``exp1_grace_adjudicate.py --tag <tag>-axis2``.

    ssh <host> 'cd ~/Code/shadow-peft && .venv/bin/python \
      scripts/exp1_grace_axis2.py --tag rl \
      --model ~/models/LFM2.5-2.6B \
      --data runs/exp1/teacher_2p6b_rl_chat_heldout.jsonl'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from shadow_lab import tiers
from shadow_lab.atoms import eval_kld_ids, eval_ppl_ids, load_teacher_data
from shadow_peft import ShadowPeftModel

AXIS_CTX = 512
AXIS_CHUNKS = 20
ARMS = ("B0", "B1", "B2", "B3", "Q0", "Q1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rl",
                    help="campaign whose arms to re-evaluate ('' = base)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="held-out teacher JSONL pool (disjoint seed)")
    ap.add_argument("--runs-root", default="runs/exp1")
    ap.add_argument("--axis-tag", default=None,
                    help="output namespace (default <tag>-axis2 -> grace-<tag>-axis2_{arm})")
    ap.add_argument("--chunks", type=int, default=AXIS_CHUNKS)
    args = ap.parse_args()

    prefix = "grace" if not args.tag else f"grace-{args.tag}"
    axis_prefix = "grace-" + (args.axis_tag or f"{args.tag or 'base'}-axis2")
    runs_root = Path(args.runs_root)
    axis_label = f"held-out self-distilled pool ({Path(args.data).name}), full-row stream, ctx={AXIS_CTX}, chunks={args.chunks}"

    stream: list[int] = []
    for row in load_teacher_data(args.data, min_len=16):
        stream.extend(row)
    print(f"held-out stream: {len(stream):,} tokens from {args.data}", flush=True)

    device = "cuda"

    def ppl_ids(model) -> float:
        return eval_ppl_ids(model, stream, chunks=args.chunks, device=device, context_len=AXIS_CTX)

    def kld_ids(ref, model) -> float:
        return eval_kld_ids(ref, model, stream, chunks=args.chunks, device=device, context_len=AXIS_CTX)

    print("=== bf16 teacher baseline ===", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    ppl_bf16 = round(ppl_ids(teacher), 4)
    print(f"ppl_bf16(ax2) = {ppl_bf16}", flush=True)

    print("=== w4a16 quant-only baseline (+KLD) ===", flush=True)
    quant = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    tiers.apply_training_conversion(quant, "w4a16", torch.device(device), fake_quant=True)
    quant = quant.to(device).eval()
    ppl_quant = round(ppl_ids(quant), 4)
    kld_quant = round(kld_ids(teacher, quant), 4)
    print(f"ppl_quant(ax2) = {ppl_quant} | kld_quant(ax2) = {kld_quant}", flush=True)
    del quant
    torch.cuda.empty_cache()

    for arm in ARMS:
        src = runs_root / f"{prefix}_{arm}"
        vpath = src / "verdict.json"
        if not vpath.exists():
            continue
        orig = json.loads(vpath.read_text())
        ckpt = src / "train" / "final"
        if not ckpt.exists():
            print(f"skip {arm}: no final checkpoint under {ckpt}", flush=True)
            continue
        print(f"=== {axis_prefix}_{arm} ===", flush=True)
        base_q = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        tiers.apply_training_conversion(base_q, "w4a16", torch.device(device), fake_quant=True)
        student = ShadowPeftModel.from_pretrained(base_q.to(device), str(ckpt)).eval()
        ppl_shadow = round(ppl_ids(student), 4)
        kld_shadow = round(kld_ids(teacher, student), 4)
        recovery = (ppl_quant - ppl_shadow) / (ppl_quant - ppl_bf16)
        verdict = {
            "arm": arm,
            "campaign": f"{args.tag or 'base'}-axis2",
            "tier": "w4a16",
            "loss_mode": orig.get("loss_mode"),
            "model": args.model,
            "shadow_quant": orig.get("shadow_quant"),
            "ppl_bf16": ppl_bf16,
            "ppl_quant": ppl_quant,
            "ppl_shadow": ppl_shadow,
            "recovery": round(recovery, 4),
            "kld_quant": kld_quant,
            "kld_shadow": kld_shadow,
            # carried over for table context; measured on the PRIMARY axis
            "gate_degraded": orig.get("gate_degraded"),
            "gate": orig.get("gate"),
            "beta": orig.get("beta"),
            "axis": axis_label,
            "primary_axis_verdict": str(vpath),
        }
        out = runs_root / f"{axis_prefix}_{arm}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
        print(json.dumps(verdict, indent=2), flush=True)
        del student, base_q
        torch.cuda.empty_cache()

    print(f"axis2 verdicts written under {runs_root}/{axis_prefix}_*/verdict.json", flush=True)
    axis_tag_out = args.axis_tag or f"{args.tag or 'base'}-axis2"
    print(f"adjudicate with: exp1_grace_adjudicate.py --tag {axis_tag_out} --runs-root {args.runs_root}", flush=True)


if __name__ == "__main__":
    main()
