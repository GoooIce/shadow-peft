"""exp1-grace axis-2 re-evaluation: a campaign's arms on a held-out
self-distillation pool (raw token-id stream — no decode/re-encode round-trip).

Motivation: despaced-wikitext is deeply OOD for the RL (post-trained) model —
the rl pass showed recovery>1 (axis origin distorted) and a KLD/ppl split.
This second axis scores the recovery question on-distribution: ppl and
KLD(bf16 teacher || quant+shadow) over held-out continuations from the same
teacher (same prompts, disjoint seed), full-row stream. Since exp1-native4b
this is the DEFAULT judgment axis for RL/instruct-class models (p0-common).

Outputs verdicts in the campaign shape under
``runs/exp1/<out-prefix>_<arm>/verdict.json`` — the standard adjudicator
then works unchanged. ``--ptq-arms`` additionally evaluates those arms'
checkpoints in PTQ-swap form (shadow linears projected onto the nvfp4 grid
after training) on the same axis, writing ``<out-prefix>_<arm>_ptq``.

    ssh <host> 'cd ~/Code/shadow-peft && .venv/bin/python \
      scripts/exp1_grace_axis2.py --tag rl \
      --model ~/models/LFM2.5-2.6B \
      --data runs/exp1/teacher_2p6b_rl_chat_heldout.jsonl'

    # a non-grace namespace + arm subset + on-axis PTQ-swap anchors:
    scripts/exp1_grace_axis2.py --tag rl --in-prefix n4b-rl \
      --arms B0_30m,Q0_30m,N0 --ptq-arms B0_30m \
      --model ~/models/LFM2.5-2.6B \
      --data runs/exp1/teacher_2p6b_rl_chat_heldout.jsonl \
      --out-prefix n4b-rl-axis2
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
DEFAULT_ARMS = ("B0", "B1", "B2", "B3", "Q0", "Q1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rl",
                    help="campaign whose arms to re-evaluate ('' = base)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="held-out teacher JSONL pool (disjoint seed)")
    ap.add_argument("--runs-root", default="runs/exp1")
    ap.add_argument("--in-prefix", default=None,
                    help="source run prefix (default grace[-<tag>] -> grace[-<tag>]_<arm>)")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                    help="comma-separated arm names under the in-prefix")
    ap.add_argument("--axis-tag", default=None,
                    help="output namespace (default <tag>-axis2 -> grace-<tag>-axis2_<arm>)")
    ap.add_argument("--out-prefix", default=None,
                    help="full output run prefix (overrides --axis-tag, e.g. n4b-rl-axis2)")
    ap.add_argument("--ptq-arms", default="",
                    help="comma-separated arms to additionally evaluate in PTQ-swap nvfp4 form")
    ap.add_argument("--shadow-models", default="",
                    help="comma-separated arm=path entries: explicit shadow models to reload "
                         "those arms' checkpoints onto (e.g. E35=/root/models/LFM2.5-350M-Base)")
    ap.add_argument("--chunks", type=int, default=AXIS_CHUNKS)
    args = ap.parse_args()

    prefix = args.in_prefix or ("grace" if not args.tag else f"grace-{args.tag}")
    axis_prefix = args.out_prefix or ("grace-" + (args.axis_tag or f"{args.tag or 'base'}-axis2"))
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    ptq_arms = [a.strip() for a in args.ptq_arms.split(",") if a.strip()]
    unknown = [a for a in ptq_arms if a not in arms]
    if unknown:
        ap.error(f"--ptq-arms must be a subset of --arms (unknown: {unknown})")
    shadow_models = {}
    for entry in args.shadow_models.split(","):
        if entry.strip():
            arm, _, path = entry.partition("=")
            shadow_models[arm.strip()] = path.strip()

    def load_explicit_shadow(arm: str):
        """Explicit-shadow arms save a trunk whose arch only the source model
        knows — reload the instance (with base-vocab resize) they were built on."""
        path = shadow_models.get(arm)
        if path is None:
            return None
        from transformers import AutoConfig

        inst = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
        target = AutoConfig.from_pretrained(args.model).vocab_size
        if inst.get_input_embeddings().weight.shape[0] < target:
            inst.resize_token_embeddings(target)
        return inst
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

    def eval_arm(arm: str, ptq_swap: bool) -> None:
        src = runs_root / f"{prefix}_{arm}"
        vpath = src / "verdict.json"
        if not vpath.exists():
            if not ptq_swap:
                return
            raise SystemExit(f"--ptq-arms entry missing its arm verdict: {vpath}")
        orig = json.loads(vpath.read_text())
        ckpt = src / "train" / "final"
        if not ckpt.exists():
            print(f"skip {arm}: no final checkpoint under {ckpt}", flush=True)
            return
        out_arm = f"{arm}_ptq" if ptq_swap else arm
        print(f"=== {axis_prefix}_{out_arm} ===", flush=True)
        base_q = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        tiers.apply_training_conversion(base_q, "w4a16", torch.device(device), fake_quant=True)
        student = ShadowPeftModel.from_pretrained(
            base_q.to(device), str(ckpt), shadow_model=load_explicit_shadow(arm)
        ).eval()
        if ptq_swap:
            from shadow_lab import shadow_quant

            linears = shadow_quant.collect_shadow_linears(student)
            n = shadow_quant.project_linears(linears, "nvfp4")
            print(f"PTQ-swap: {n} shadow linears projected onto the nvfp4 grid", flush=True)
        ppl_shadow = round(ppl_ids(student), 4)
        kld_shadow = round(kld_ids(teacher, student), 4)
        recovery = (ppl_quant - ppl_shadow) / (ppl_quant - ppl_bf16)
        verdict = {
            "arm": out_arm,
            "campaign": axis_prefix,
            "tier": "w4a16",
            "loss_mode": orig.get("loss_mode"),
            "model": args.model,
            "shadow_quant": "nvfp4_ptq_swap" if ptq_swap else orig.get("shadow_quant"),
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
        out = runs_root / f"{axis_prefix}_{out_arm}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
        print(json.dumps(verdict, indent=2), flush=True)
        del student, base_q
        torch.cuda.empty_cache()

    for arm in arms:
        eval_arm(arm, ptq_swap=False)
    for arm in ptq_arms:
        eval_arm(arm, ptq_swap=True)

    print(f"axis2 verdicts written under {runs_root}/{axis_prefix}_*/verdict.json", flush=True)
    adj_tag = axis_prefix.removeprefix("grace-") if axis_prefix.startswith("grace-") else axis_prefix
    print(f"adjudicate with: exp1_grace_adjudicate.py --tag {adj_tag} --runs-root {args.runs_root}",
          flush=True)


if __name__ == "__main__":
    main()
