"""exp6 layer-schedule arm runner (spec: docs/exp6-layer-group-recovery.md).

Equal-budget adjudication of the greedy layer-group schedule against the
end-to-end joint loop on the 2.6B x w4a16 testbed (same L1_4x shadow anchor
as the exp1-grace series, so the E2E anchor is comparable):

    E2E  layer_schedule=end2end  (anchor arm — the current protocol)
    G1   layer_schedule=per_layer (greedy per-layer + polish)
    G4   layer_schedule=groups, group_size=4 (greedy 4-layer groups + polish)

Each arm writes a floor-series-isomorphic verdict.json (ppl_bf16 / ppl_quant /
ppl_shadow / recovery / kld axes / tokens / train_seconds) plus the exp6 axes:
peak CUDA training memory, per-phase wall-clock split, and the per-layer
attribution table (init vs final normalized hidden MSE vs teacher).

A replay-parity preflight gates the greedy arms: the capture/replay frames
must reproduce the student's own forward on the real mounted model before any
layer schedule is allowed to train.

    ssh <host> 'cd ~/Code/shadow-peft && nohup .venv/bin/python \
      scripts/exp6_layer_arm.py --arm G4 --data runs/exp1/teacher_2p6b.jsonl \
      > runs/logs/<date>-exp6-G4.log 2>&1 < /dev/null &'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_lab import tiers
from shadow_lab.atoms import despace_wikitext, eval_kld, eval_ppl, load_wikitext2_test_text
from shadow_lab.layerwise import replay_parity
from shadow_lab.trainer import TrainConfig, train_shadow
from shadow_peft import ShadowConfig, ShadowPeftModel, get_shadow_model

AXIS_CTX = 512
AXIS_CHUNKS = 20
PARITY_TOL = 1e-3

ARMS = {
    "E2E": {"layer_schedule": "end2end"},
    "G1": {"layer_schedule": "per_layer"},
    "G4": {"layer_schedule": "groups", "group_size": 4},
}


def preflight(args, arm: dict) -> float:
    """Replay-parity gate on the real testbed before any training.

    Builds the exact student the trainer would build (w4a16 fake-quant base +
    the arm's shadow anchor), then checks that driving the wrapped layers from
    the captured frames reproduces the model's own forward. The greedy
    schedule's correctness rests entirely on this contract.
    """
    with open(args.data) as f:
        first_row = json.loads(f.readline())["ids"]
    ids = torch.tensor([first_row[:64]], device="cuda")
    attn = torch.ones_like(ids)

    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    tiers.apply_training_conversion(base, "w4a16", torch.device("cuda"), fake_quant=True)
    base = base.to("cuda")
    peft = get_shadow_model(
        base,
        ShadowConfig(
            num_shadow_layers=args.shadow_layers,
            injection_hidden_size=16,
            gate_hidden_size=10,
            alpha=0.1,
            dropout=0.0,
            shadow_intermediate_size=args.shadow_intermediate,
        ),
    )
    peft.eval()
    with torch.no_grad():
        report = replay_parity(peft, input_ids=ids, attention_mask=attn)
    diff = report["max_abs_diff"]
    print(f"preflight replay parity: max_abs_diff={diff:.3e} (tol {PARITY_TOL})", flush=True)
    del peft, base
    torch.cuda.empty_cache()
    if diff > PARITY_TOL:
        raise SystemExit(
            f"replay parity FAILED ({diff:.3e} > {PARITY_TOL}): the capture frames do "
            "not carry this architecture's forward; refusing to train a greedy schedule"
        )
    return diff


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--model", default=str(Path.home() / "models" / "LFM2.5-2.6B-Base"))
    ap.add_argument("--data", required=True, help="teacher JSONL pool (exp1 generator output)")
    ap.add_argument("--out-dir", default=None, help="default runs/exp6/{arm}")
    ap.add_argument("--max-tokens", type=float, default=1e7)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-seq", type=int, default=8)
    ap.add_argument("--shadow-layers", type=int, default=1)
    ap.add_argument("--shadow-intermediate", type=int, default=43008,
                    help="L1_4x anchor, same as the exp1-grace series")
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--polish-frac", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--skip-baselines", action="store_true",
                    help="reuse an existing verdict's baselines (fills from verdict if present)")
    args = ap.parse_args()

    arm = ARMS[args.arm]
    out_dir = Path(args.out_dir or f"runs/exp6/{args.arm}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (Path("runs") / "logs").mkdir(parents=True, exist_ok=True)
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

    print(f"=== 3/4 arm {args.arm} (schedule={arm['layer_schedule']}) ===", flush=True)
    parity = preflight(args, arm) if arm["layer_schedule"] != "end2end" else None
    cfg = TrainConfig(
        bits=0,
        backbone_tier="w4a16",
        tier_fake_quant=True,
        loss_mode="ce_kl",
        model=args.model,
        data=args.data,
        max_tokens=args.max_tokens,
        seq_len=512,
        batch_seq=args.batch_seq,
        lr=args.lr,
        shadow_layers=args.shadow_layers,
        shadow_intermediate_size=args.shadow_intermediate,
        layer_schedule=arm["layer_schedule"],
        group_size=args.group_size,
        polish_token_frac=args.polish_frac,
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
        "campaign": "exp6-layer-schedule",
        "tier": "w4a16",
        "loss_mode": "ce_kl",
        "model": args.model,
        "shadow_layers": args.shadow_layers,
        "shadow_intermediate": args.shadow_intermediate,
        "layer_schedule": arm["layer_schedule"],
        "group_size": args.group_size if arm["layer_schedule"] == "groups" else None,
        "polish_token_frac": args.polish_frac,
        "replay_parity_max_abs_diff": parity,
        "trainable_params": report.trainable_params,
        "ppl_bf16": ppl_bf16,
        "ppl_quant": ppl_quant,
        "ppl_shadow": round(ppl_shadow, 4),
        "recovery": round(recovery, 4),
        "kld_quant": kld_quant,
        "kld_shadow": kld_shadow,
        "peak_mem_gb": report.peak_mem_gb,
        "phase_seconds": report.phase_seconds,
        "layer_attribution": report.layer_attribution,
        "axis": "despaced wikitext, ctx=512, chunks=20",
        "tokens": report.tokens,
        "train_seconds": round(report.wall_time_s, 1),
        "data": args.data,
        "lr": args.lr,
    }
    verdict_path.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
