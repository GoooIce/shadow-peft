"""exp1-grace arm runner: one loss-recipe arm end-to-end (spec: exp1-grace).

Ladder over the frozen L1 structure (>=300M trainable, ticket 01) on the
2.6B x w4a16 testbed:

    B0  baseline ce_kl (anchor arm)
    B1  dkd_gate        (DKD + confidence soft gate)
    B2  dkd_gate_ib     (B1 + adaptive IB controller on the distill weight)
    B3  dkd_gate_ib_cka (B2 + span-CKA on the penultimate hidden layer)
    Q0  ce_kl        x native-nvfp4 shadow (precision-effect attribution)
    Q1  winner recipe x native-nvfp4 (--loss-mode required: from the ladder
        adjudication in ticket 05)

Each arm writes a floor-series-isomorphic verdict.json (ppl_bf16 / ppl_quant /
ppl_shadow / recovery / axis / tokens / train_seconds) plus the GRACE
monitoring fields (gate_degraded, gate/beta summaries, KLD axis) and the
structure-freeze record (structure.json). 30M confirmation reruns are the
same command with --max-tokens 3e7.

    ssh <host> 'cd ~/Code/shadow-peft && nohup .venv/bin/python \
      scripts/exp1_grace_arm.py --arm B1 --data runs/exp1/teacher_2p6b.jsonl \
      > runs/logs/<date>-grace-B1.log 2>&1 < /dev/null &'
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
MIN_TRAINABLE = 300_000_000

ARMS = {
    "B0": {"loss_mode": "ce_kl"},
    "B1": {"loss_mode": "dkd_gate"},
    "B2": {"loss_mode": "dkd_gate_ib"},
    "B3": {"loss_mode": "dkd_gate_ib_cka"},
    "Q0": {"loss_mode": "ce_kl", "shadow_quant": "nvfp4"},
    # Q1's recipe comes from the B0-B3 adjudication — pass --loss-mode.
    "Q1": {"shadow_quant": "nvfp4"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--loss-mode", default=None,
                    help="override the arm's loss mode (required for Q1)")
    ap.add_argument("--model", default=str(Path.home() / "models" / "LFM2.5-2.6B-Base"))
    ap.add_argument("--data", required=True, help="teacher JSONL pool (exp1 generator output)")
    ap.add_argument("--out-dir", default=None, help="default runs/exp1/grace_{arm}")
    ap.add_argument("--max-tokens", type=float, default=1e7)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-seq", type=int, default=8)
    ap.add_argument("--shadow-layers", type=int, default=1)
    ap.add_argument("--shadow-intermediate", type=int, default=43008,
                    help="L1_4x anchor; the probe's structure.json overrides if bumped")
    ap.add_argument("--shadow-quant", default=None, choices=["nvfp8", "nvfp4"])
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--skip-baselines", action="store_true",
                    help="reuse an existing verdict's baselines (fills from verdict if present)")
    args = ap.parse_args()

    arm = ARMS[args.arm]
    loss_mode = args.loss_mode or arm.get("loss_mode")
    if loss_mode is None:
        ap.error("--loss-mode is required for arm Q1 (winner recipe from the B0-B3 adjudication)")
    shadow_quant = args.shadow_quant or arm.get("shadow_quant")
    out_dir = Path(args.out_dir or f"runs/exp1/grace_{args.arm}")
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

    verdict_path = out_dir / "verdict.json"
    prior = json.loads(verdict_path.read_text()) if verdict_path.exists() else {}

    if args.skip_baselines and {"ppl_bf16", "ppl_quant", "kld_quant"} <= set(prior):
        ppl_bf16 = prior["ppl_bf16"]
        ppl_quant = prior["ppl_quant"]
        kld_quant = prior["kld_quant"]
        print(f"reusing baselines: bf16={ppl_bf16} quant={ppl_quant} kld_quant={kld_quant}", flush=True)
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

    print(f"=== 3/4 arm {args.arm} ({loss_mode}{', ' + shadow_quant if shadow_quant else ''}) ===", flush=True)
    cfg = TrainConfig(
        bits=0,
        backbone_tier="w4a16",
        tier_fake_quant=True,
        loss_mode=loss_mode,
        model=args.model,
        data=args.data,
        max_tokens=args.max_tokens,
        seq_len=512,
        batch_seq=args.batch_seq,
        lr=args.lr,
        shadow_layers=args.shadow_layers,
        shadow_intermediate_size=args.shadow_intermediate,
        shadow_quant=shadow_quant,
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
        "arm": args.arm,
        "tier": "w4a16",
        "loss_mode": loss_mode,
        "shadow_layers": args.shadow_layers,
        "shadow_intermediate": args.shadow_intermediate,
        "shadow_heads": None,
        "shadow_kv_heads": None,
        "shadow_quant": shadow_quant,
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
        "beta": ({"init": report.grace["beta_init"], "final": report.grace["beta_final"],
                  "min": report.grace["beta_min"], "max": report.grace["beta_max"]}
                 if report.grace else None),
        "axis": "despaced wikitext, ctx=512, chunks=20",
        "tokens": report.tokens,
        "train_seconds": round(report.wall_time_s, 1),
        "data": args.data,
        "lr": args.lr,
    }
    verdict_path.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)

    structure = {
        "structure": f"L{args.shadow_layers}",
        "shadow_layers": args.shadow_layers,
        "shadow_intermediate": args.shadow_intermediate,
        "trainable_params": report.trainable_params,
        "min_trainable": MIN_TRAINABLE,
        "meets_min": report.trainable_params >= MIN_TRAINABLE,
        "model": args.model,
    }
    (out_dir / "structure.json").write_text(json.dumps(structure, indent=2) + "\n")


if __name__ == "__main__":
    main()
