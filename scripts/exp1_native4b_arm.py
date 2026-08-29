"""exp1-native4b arm runner: Q0-line confirmation arms (spec: exp1-native4b).

On the structure frozen by exp1-grace (L1_5x, shadow_intermediate=53760,
345,073,792 trainable) on the 2.6B x w4a16 testbed:

    B0_30m  ce_kl x bf16        @30M  — the 30M anchor
    Q0_30m  ce_kl x native-nvfp4 @30M — the 30M confirm (saturation read)
    N0      ce_kl x bf16 + residual-matched noise x1.0 @10M — mechanism arm
    N1      ce_kl x bf16 + residual-matched noise x2.0 @10M — dose arm (base only)

The 10M B0/Q0 references are REUSED from the exp1-grace campaign (grace_* /
grace-rl_* verdicts) — never re-run here. Baselines (bf16 / quant-only ppl +
KLD) are axis+model measurements shared across arms: pass --baselines-from
<verdict.json> to reuse a sibling campaign's instead of re-measuring.

Verdicts are floor-series-isomorphic plus the noise fields. 30M arms are the
same command with --suffix 30m --max-tokens 3e7.

    ssh <host> 'cd ~/Code/shadow-peft && nohup .venv/bin/python \
      scripts/exp1_native4b_arm.py --arm Q0 --suffix 30m --max-tokens 3e7 \
      --data runs/exp1/teacher_2p6b_base.jsonl \
      --baselines-from runs/exp1/grace_B0/verdict.json \
      > runs/logs/<date>-n4b-Q0-30m.log 2>&1 < /dev/null &'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_lab import tiers
from shadow_lab.atoms import despace_wikitext, eval_kld, eval_ppl, load_wikitext2_test_text
from shadow_lab.trainer import TrainConfig, train_shadow
from shadow_peft import ShadowPeftModel

AXIS_CTX = 512
AXIS_CHUNKS = 20

ARMS = {
    "B0": {"loss_mode": "ce_kl"},
    "Q0": {"loss_mode": "ce_kl", "shadow_quant": "nvfp4"},
    "N0": {"loss_mode": "ce_kl", "shadow_noise": "nvfp4", "shadow_noise_mult": 1.0},
    "N1": {"loss_mode": "ce_kl", "shadow_noise": "nvfp4", "shadow_noise_mult": 2.0},
}
BASELINE_FIELDS = ("ppl_bf16", "ppl_quant", "kld_quant")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--model", default=str(Path.home() / "models" / "LFM2.5-2.6B-Base"))
    ap.add_argument("--data", required=True, help="teacher JSONL pool (exp1 generator output)")
    ap.add_argument("--tag", default="",
                    help="campaign tag ('' = base -> n4b_{arm}; 'rl' -> n4b-rl_{arm}). Each "
                         "pass's B0 is its own anchor; ppl never compares across passes.")
    ap.add_argument("--suffix", default="",
                    help="run-name suffix (e.g. '30m' -> n4b_B0_30m)")
    ap.add_argument("--out-dir", default=None,
                    help="default runs/exp1/{n4b[-tag]}_{arm}[_{suffix}]")
    ap.add_argument("--max-tokens", type=float, default=1e7)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-seq", type=int, default=8)
    ap.add_argument("--shadow-layers", type=int, default=1)
    ap.add_argument("--shadow-intermediate", type=int, default=53760,
                    help="exp1-grace frozen structure (L1_5x)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--baselines-from", default=None,
                    help="reuse {ppl_bf16, ppl_quant, kld_quant} from a sibling verdict "
                         "(same model + axis) instead of re-measuring")
    args = ap.parse_args()

    arm = ARMS[args.arm]
    prefix = "n4b" if not args.tag else f"n4b-{args.tag}"
    arm_name = f"{args.arm}_{args.suffix}" if args.suffix else args.arm
    out_dir = Path(args.out_dir or f"runs/exp1/{prefix}_{arm_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    axis_text = despace_wikitext(load_wikitext2_test_text())

    def ppl(model) -> float:
        return eval_ppl(model, tok, axis_text, chunks=AXIS_CHUNKS, device="cuda",
                        context_len=AXIS_CTX)

    def kld(ref, model) -> float:
        return eval_kld(ref, model, tok, axis_text, chunks=AXIS_CHUNKS, device="cuda",
                        context_len=AXIS_CTX)

    baselines_from = None
    if args.baselines_from:
        prior = json.loads(Path(args.baselines_from).read_text())
        if not set(BASELINE_FIELDS) <= set(prior):
            raise SystemExit(f"--baselines-from verdict lacks fields: {args.baselines_from}")
        ppl_bf16, ppl_quant, kld_quant = (prior[f] for f in BASELINE_FIELDS)
        baselines_from = args.baselines_from
        print(f"reusing baselines from {args.baselines_from}: "
              f"bf16={ppl_bf16} quant={ppl_quant} kld_quant={kld_quant}", flush=True)
    else:
        print("=== 1/4 bf16 baseline ===", flush=True)
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
        ppl_bf16 = round(ppl(base), 4)
        print(f"ppl_bf16 = {ppl_bf16}", flush=True)

        print("=== 2/4 w4a16 quant-only baseline (+KLD axis) ===", flush=True)
        quant_copy = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
        tiers.apply_training_conversion(quant_copy, "w4a16", torch.device("cuda"), fake_quant=True)
        ppl_quant = round(ppl(quant_copy), 4)
        kld_quant = round(kld(base, quant_copy), 4)
        print(f"ppl_quant(w4a16 fake) = {ppl_quant} | kld_quant = {kld_quant}", flush=True)
        del base, quant_copy
        torch.cuda.empty_cache()

    quant_cfg = ", ".join(
        f"{k}={v}" for k, v in (("shadow_quant", arm.get("shadow_quant")),
                                ("shadow_noise", arm.get("shadow_noise")),
                                ("noise_mult", arm.get("shadow_noise_mult"))) if v is not None
    )
    print(f"=== 3/4 arm {arm_name} ({arm['loss_mode']}{', ' + quant_cfg if quant_cfg else ''}) ===",
          flush=True)
    cfg = TrainConfig(
        bits=0,
        backbone_tier="w4a16",
        tier_fake_quant=True,
        loss_mode=arm["loss_mode"],
        model=args.model,
        data=args.data,
        max_tokens=args.max_tokens,
        seq_len=512,
        batch_seq=args.batch_seq,
        lr=args.lr,
        shadow_layers=args.shadow_layers,
        shadow_intermediate_size=args.shadow_intermediate,
        shadow_quant=arm.get("shadow_quant"),
        shadow_noise=arm.get("shadow_noise"),
        shadow_noise_mult=arm.get("shadow_noise_mult", 1.0),
        ppl_context_len=AXIS_CTX,
        eval_every=args.eval_every,
        pad_id=pad_id,
        save_dir=str(out_dir / "train"),
    )
    report = train_shadow(cfg)
    ppl_shadow = report.final_ppl_20chunks

    print("=== 4/4 KLD(bf16 || quant+shadow) ===", flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    eval_base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    tiers.apply_training_conversion(eval_base, "w4a16", torch.device("cuda"), fake_quant=True)
    student = ShadowPeftModel.from_pretrained(eval_base.to("cuda"), str(out_dir / "train" / "final")).eval()
    kld_shadow = round(kld(base, student), 4)
    print(f"kld_shadow = {kld_shadow}", flush=True)
    del base, eval_base, student
    torch.cuda.empty_cache()

    recovery = (ppl_quant - ppl_shadow) / (ppl_quant - ppl_bf16)
    verdict = {
        "arm": arm_name,
        "campaign": args.tag or "base",
        "tier": "w4a16",
        "loss_mode": arm["loss_mode"],
        "model": args.model,
        "shadow_layers": args.shadow_layers,
        "shadow_intermediate": args.shadow_intermediate,
        "shadow_quant": arm.get("shadow_quant"),
        "shadow_noise": arm.get("shadow_noise"),
        "shadow_noise_mult": arm.get("shadow_noise_mult") if arm.get("shadow_noise") else None,
        "trainable_params": report.trainable_params,
        "ppl_bf16": ppl_bf16,
        "ppl_quant": ppl_quant,
        "ppl_shadow": round(ppl_shadow, 4),
        "recovery": round(recovery, 4),
        "kld_quant": kld_quant,
        "kld_shadow": kld_shadow,
        "gate_degraded": report.grace["gate_degraded"] if report.grace else None,
        "gate": ({"final_mean": report.grace["gate_final_mean"],
                  "final_cv": report.grace["gate_final_cv"],
                  "max_degrade_streak": report.grace["gate_max_degrade_streak"]}
                 if report.grace else None),
        "beta": None,
        "axis": "despaced wikitext, ctx=512, chunks=20",
        "tokens": report.tokens,
        "train_seconds": round(report.wall_time_s, 1),
        "data": args.data,
        "lr": args.lr,
        "baselines_from": baselines_from,
    }
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
