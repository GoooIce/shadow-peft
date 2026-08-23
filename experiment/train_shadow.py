"""Train a Shadow adapter on a quantized Qwen3 base with distillation (E1/E2).

Bit-width is an argument, not a script: --bits 1 runs the binary affine
student (E1), --bits 2 the ternary student (E2). Per-bits defaults
(dtype, refine iters) come from the lab's BitProfile.

Run: python experiment/train_shadow.py --bits 2 --loss-mode ce_kl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lab import LOSS_MODES, TrainConfig, train_shadow


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, choices=[1, 2], default=1)
    ap.add_argument("--loss-mode", choices=LOSS_MODES, required=True)
    ap.add_argument("--data", default="experiment/data/teacher_qwen3_1p7b.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B", help="HF model id or local path")
    ap.add_argument("--max-tokens", type=float, default=5e6)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-seq", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta2", type=float, default=0.98)
    ap.add_argument("--distill-weight", type=float, default=1.0)
    ap.add_argument("--distill-temperature", type=float, default=1.0)
    ap.add_argument("--distill-hidden-weight", type=float, default=1.0)
    ap.add_argument("--shadow-loss-weight", type=float, default=0.05)
    ap.add_argument("--shadow-layers", type=int, default=2)
    ap.add_argument("--injection-hidden", type=int, default=16)
    ap.add_argument("--gate-hidden", type=int, default=10)
    ap.add_argument("--refine-iters", type=int, default=None)
    ap.add_argument("--thr-ratio", type=float, default=0.5)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    cfg = TrainConfig(
        bits=args.bits,
        loss_mode=args.loss_mode,
        model=args.model,
        data=args.data,
        max_tokens=args.max_tokens,
        seq_len=args.seq_len,
        batch_seq=args.batch_seq,
        lr=args.lr,
        beta2=args.beta2,
        distill_weight=args.distill_weight,
        distill_temperature=args.distill_temperature,
        distill_hidden_weight=args.distill_hidden_weight,
        shadow_loss_weight=args.shadow_loss_weight,
        shadow_layers=args.shadow_layers,
        injection_hidden=args.injection_hidden,
        gate_hidden=args.gate_hidden,
        refine_iters=args.refine_iters,
        thr_ratio=args.thr_ratio,
        group_size=args.group_size,
        log_every=args.log_every,
        eval_every=args.eval_every,
        seed=args.seed,
        save_dir=args.save_dir,
    )
    train_shadow(cfg)


if __name__ == "__main__":
    main()
