"""Direction-① per-tier distillation training runner (tickets 06/07).

One tier per invocation; identical data / hyperparameters across tiers (the
only difference is the backbone tier flag), teacher data generated once and
shared. Iron rules baked in: every shadow is freshly trained (legacy adapters
are never loaded), auto power-off + WeChat notify after the run, budget
ledger appended, overspend => stop and report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dir1.ops import BudgetLedger, StageTimer, power_off_now, wechat_notify  # noqa: E402
from lab.trainer import TrainConfig, train_shadow  # noqa: E402

TIERS = ("bf16", "w8a8_fp8", "w4a16")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=TIERS)
    ap.add_argument("--model", default="google/gemma-4-12b-it")
    ap.add_argument("--data", default="experiment/data/teacher_gemma4_12b.jsonl")
    ap.add_argument("--out-root", default="experiment/runs/dir1")
    ap.add_argument("--max-tokens", type=float, default=5e6)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-seq", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--shadow-layers", type=int, default=2)
    ap.add_argument("--calib-steps", type=int, default=8)
    ap.add_argument("--no-fake-quant", action="store_true",
                    help="train against the real kernel path (autograd unsupported; eval-only)")
    ap.add_argument("--budget-file", default="experiment/runs/dir1/budget.jsonl")
    ap.add_argument("--no-shutdown", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    cfg = TrainConfig(
        bits=0,
        backbone_tier=args.tier,
        loss_mode="ce_kl_hmse",
        model=args.model,
        data=args.data,
        max_tokens=args.max_tokens,
        seq_len=args.seq_len,
        batch_seq=args.batch_seq,
        lr=args.lr,
        shadow_layers=args.shadow_layers,
        calib_steps=args.calib_steps,
        tier_fake_quant=not args.no_fake_quant,
        pad_id=pad_id,
        save_dir=str(Path(args.out_root) / args.tier),
    )

    ledger = BudgetLedger(args.budget_file)
    summary = {"tier": args.tier, "save_dir": str(cfg.resolved_save_dir())}
    try:
        with StageTimer(ledger, f"train_{args.tier}", note=f"max_tokens={args.max_tokens}"):
            report = train_shadow(cfg)
        summary["report"] = report.__dict__
        summary["status"] = "ok"
    except Exception as exc:
        summary["status"] = f"failed: {exc!r}"
        raise
    finally:
        budget = ledger.summary()
        summary["budget"] = budget
        wechat_notify(
            f"dir1 {args.tier} 训练{'完成' if summary['status'] == 'ok' else '失败'}",
            f"budget: {budget['total_hours']:.2f}h ≈ ¥{budget['total_cny']} / ¥230",
        )
        if not args.no_shutdown and not budget["over_budget"]:
            power_off_now()
        elif budget["over_budget"]:
            # 铁律: 超预算线先停下汇报 — 不再启动后续阶段,也不静默关机。
            wechat_notify("dir1 超预算停机", str(budget))


if __name__ == "__main__":
    main()
