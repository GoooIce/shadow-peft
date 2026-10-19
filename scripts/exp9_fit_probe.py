#!/usr/bin/env python3
"""Fit an offline handoff probe from exp9 candidate rows (离线校准 probe).

Example::

    PYTHONPATH=src python scripts/exp9_fit_probe.py \
        runs/exp9/candidates.jsonl --out runs/exp9/probe.json --threshold 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from edge_cloud.handoff_eval import fit_candidates, load_candidates
from edge_cloud.probe import write_probe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="读取 exp9 probe 候选行，拟合并写出 handoff probe JSON"
    )
    parser.add_argument("candidates", help="exp9_handoff_eval.py --export-candidates 产物")
    parser.add_argument("--out", default="runs/exp9/probe.json", help="probe JSON 输出路径")
    parser.add_argument("--threshold", type=float, default=0.5, help="写入产物的操作阈值")
    parser.add_argument(
        "--calibration-bins", type=int, default=10, help="校准分箱数（至少 1）"
    )
    parser.add_argument(
        "--trained-at",
        help="覆盖 trained_at 时间戳，便于复现逐位一致的产物",
    )
    args = parser.parse_args(argv)

    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold 必须在 [0, 1] 内")
    if args.calibration_bins < 1:
        raise SystemExit("--calibration-bins 必须至少为 1")

    try:
        rows = load_candidates(args.candidates)
        if not rows:
            raise ValueError("候选文件为空")
        payload = fit_candidates(
            rows,
            threshold=args.threshold,
            calibration_bins=args.calibration_bins,
            trained_at=args.trained_at,
        )
        target = Path(args.out).expanduser()
        write_probe(payload, target)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"probe 校准失败: {exc}") from exc

    positives = sum(row["label"] for row in rows)
    prompts = len({row["prompt_id"] for row in rows})
    print(f"samples    : {len(rows)} rows / {prompts} prompts")
    print(f"positives  : {positives} ({positives / len(rows):.2%})")
    print(f"features   : {len(payload['feature_names'])}")
    print(f"threshold  : {payload['threshold']:.3f}")
    print(f"fingerprint: {payload['fingerprint']}")
    print(f"probe      : {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
