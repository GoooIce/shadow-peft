"""Run the first-phase QUASAR QAD arm: NVFP4 W4A4 only.

The input is the same teacher-token JSONL consumed by ``train_shadow`` (one
``{"ids": [...]}`` record per line).  The output is a packed
compressed-tensors NVFP4 checkpoint plus ``quasar_manifest.json``.

Example:

    .venv/bin/python scripts/quasar_qad_arm.py \
      --model Qwen/Qwen3-8B --data experiment/data/teacher_qwen3_8b.jsonl \
      --out runs/quasar/qwen3_8b_nvfp4
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from shadow_lab.quasar_qad import QuasarQADConfig, run_quasar_qad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="student base model path or repo id")
    parser.add_argument("--teacher", default=None, help="teacher path (default: --model)")
    parser.add_argument("--data", required=True, help='teacher-token JSONL: one {"ids": [...]} per line')
    parser.add_argument("--out", required=True, help="packed checkpoint output directory")
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-seq", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="cosine")
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    report = run_quasar_qad(
        QuasarQADConfig(
            model=args.model,
            teacher_model=args.teacher,
            data=args.data,
            save_dir=args.out,
            steps=args.steps,
            seq_len=args.seq_len,
            batch_seq=args.batch_seq,
            gradient_accumulation_steps=args.grad_accum,
            lr=args.lr,
            lr_schedule=args.lr_schedule,
            beta2=args.beta2,
            log_every=args.log_every,
            seed=args.seed,
            device=args.device,
            gradient_checkpointing=args.gradient_checkpointing,
        )
    )
    print(f"[quasar-nvfp4] done: {json.dumps(asdict(report))}")


if __name__ == "__main__":
    main()
