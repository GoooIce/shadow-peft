"""slim QAD arm: the no-shadow recovery baseline on the same backbone tier.

Produces the comparison curve for the AngelSlim-replacement verdict (step 2
gates on this data): quantized-backbone recovery via AngelSlim's QAD
(quantization-aware distillation, no shadow adapter) against the shadow
arms' recovery in the same recovery-rate coordinate system.

Grid alignment matters — two modes:

    --from-ptq-ckpt <dir>   Student starts from a packed GPTQ checkpoint
                            produced by the shadow arm's own backbone tier
                            (slim.ptq.convert_backbone(..., fake_quant=False,
                            save_dir=...)): same grid, directly comparable.
    (default)               QAD's own w4a8_fp8 student (learnable scales,
                            its own grid) — the "another toolkit" row.

Example (GB10):

    ssh <host> 'cd ~/Code/shadow-peft && nohup .venv/bin/python \
      scripts/slim_qad_arm.py --model Qwen/Qwen3-1.7B \
      --data experiment/data/teacher_qwen3_1p7b.jsonl \
      --out runs/slim/qad_rl1b_w4a8 > runs/logs/slim-qad-rl1b.log 2>&1 \
      < /dev/null &'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shadow_lab.slim import run_qad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="student base model (path or repo id)")
    parser.add_argument(
        "--data", required=True, help="calibration/distillation JSONL or HF dataset spec"
    )
    parser.add_argument("--out", required=True, help="output checkpoint dir")
    parser.add_argument(
        "--from-ptq-ckpt",
        default=None,
        help="packed GPTQ checkpoint dir (same grid as the shadow arm); "
        "omit for QAD's own w4a8_fp8 student",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--num-samples", type=int, default=128)
    args = parser.parse_args()

    manifest = run_qad(
        args.model,
        teacher_path=args.model,  # self-distillation: teacher = full-precision base
        data_path=args.data,
        out_dir=args.out,
        from_ptq_ckpt=args.from_ptq_ckpt,
        hf_args={
            "bf16": True,
            "logging_steps": 1,
            "per_device_train_batch_size": 1,
            "learning_rate": args.lr,
            "lr_scheduler_type": "cosine",
            "num_train_epochs": args.epochs,
            "save_strategy": "no",
        },
        max_seq_length=args.max_seq_length,
        num_samples=args.num_samples,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "slim_qad_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[slim-qad-arm] done: {json.dumps(manifest)}")


if __name__ == "__main__":
    main()
