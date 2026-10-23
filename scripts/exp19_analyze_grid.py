#!/usr/bin/env python3
"""exp19: paired analysis of the fixed-vs-dynamic GKD mass grid (issue #54).

Reads ``<root>/arms/<arm>/gkd-200/report.json`` (exp17 GKD output) and the
matching ``train_stats.json``, aligns examples by id, and reports:

- per-arm task accuracy / joint ppl / support width;
- ``fixed - dynamic`` paired accuracy differences for each seed;
- the per-example seed-averaged difference for each mass, with a 95% bootstrap
  CI.

Positive differences mean the fixed top-k arm is more accurate. The script
never trains or evaluates; it only re-reads reports.

Example::

    python scripts/exp19_analyze_grid.py --root runs-local/exp19 \
        --out runs-local/exp19/grid_summary.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

EXPERIMENT = "exp19-dynamic-topk-gkd"
SCHEMA_VERSION = "exp19-grid/v1"
DEFAULT_BOOT = 20000
ARM_RE = re.compile(
    r"^(fixed-k(?P<fixed_k>\d+)|dynamic-m(?P<mass>\d+)-k32)"
    r"(?:-seed(?P<seed>\d+))?$"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired fixed-vs-dynamic analysis for the exp19 mass grid."
    )
    parser.add_argument("--root", required=True, help="exp19 output root (contains arms/).")
    parser.add_argument("--out", required=True, help="JSON summary path.")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_BOOT)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _parse_arm(name: str) -> dict[str, Any] | None:
    match = ARM_RE.match(name)
    if not match:
        return None
    seed = int(match.group("seed") or 0)
    if match.group("fixed_k") is not None:
        return {"arm": name, "kind": "fixed", "k": int(match.group("fixed_k")), "seed": seed}
    digits = match.group("mass")
    mass = float(f"0.{digits[1:]}") if digits.startswith("0") else float(digits)
    return {"arm": name, "kind": "dynamic", "mass": mass, "seed": seed}


def _load_arm(arm_dir: Path, info: dict[str, Any]) -> dict[str, Any] | None:
    report_path = arm_dir / "gkd-200" / "report.json"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    suite = report["suites"]["gkd"]
    metrics = suite["metrics"]
    per_example = {
        str(row["id"]): int(bool(row.get("correct")))
        for row in suite.get("per_example", [])
    }
    stats_path = arm_dir / "train_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    return {
        **info,
        "accuracy": metrics.get("task_accuracy"),
        "accuracy_ci": metrics.get("task_accuracy_ci"),
        "joint_ppl": metrics.get("joint_ppl"),
        "joint_ppl_ci": metrics.get("joint_ppl_ci"),
        "support_stats": stats.get("support_stats"),
        "topk": stats.get("topk"),
        "topk_mass": stats.get("topk_mass"),
        "per_example": per_example,
    }


def _bootstrap_ci(diffs: list[int], n_boot: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def _paired(fixed: dict[str, Any], dynamic: dict[str, Any]) -> dict[str, Any] | None:
    ids = sorted(set(fixed["per_example"]) & set(dynamic["per_example"]))
    if not ids:
        return None
    diffs = [fixed["per_example"][i] - dynamic["per_example"][i] for i in ids]
    return {
        "n": len(ids),
        "mean_fixed_minus_dynamic": sum(diffs) / len(diffs),
        "fixed_wins": sum(1 for d in diffs if d > 0),
        "dynamic_wins": sum(1 for d in diffs if d < 0),
        "ties": sum(1 for d in diffs if d == 0),
        "diffs": diffs,
        "ids": ids,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).expanduser()
    arms_dir = root / "arms"
    if not arms_dir.is_dir():
        raise SystemExit(f"arms directory not found: {arms_dir}")

    arms: dict[str, dict[str, Any]] = {}
    for arm_dir in sorted(arms_dir.iterdir()):
        if not arm_dir.is_dir():
            continue
        info = _parse_arm(arm_dir.name)
        if info is None:
            continue
        loaded = _load_arm(arm_dir, info)
        if loaded is not None:
            arms[arm_dir.name] = loaded

    fixed_by_seed = {
        arm["seed"]: arm for arm in arms.values() if arm["kind"] == "fixed"
    }
    per_seed: list[dict[str, Any]] = []
    mass_groups: dict[float, list[dict[str, Any]]] = {}
    for arm in arms.values():
        if arm["kind"] != "dynamic":
            continue
        fixed = fixed_by_seed.get(arm["seed"])
        if fixed is None:
            continue
        paired = _paired(fixed, arm)
        if paired is None:
            continue
        entry = {
            "mass": arm["mass"],
            "seed": arm["seed"],
            "arm": arm["arm"],
            "fixed_arm": fixed["arm"],
            "dynamic_accuracy": arm["accuracy"],
            "fixed_accuracy": fixed["accuracy"],
            "dynamic_joint_ppl": arm["joint_ppl"],
            "fixed_joint_ppl": fixed["joint_ppl"],
            "support_stats": arm["support_stats"],
            "paired": {k: v for k, v in paired.items() if k not in ("diffs", "ids")},
        }
        per_seed.append(entry)
        mass_groups.setdefault(arm["mass"], []).append(entry)

    masses: list[dict[str, Any]] = []
    for mass, entries in sorted(mass_groups.items()):
        # Seed-averaged per-example difference: fixed - dynamic. Rebuild the
        # aligned per-seed diffs from the arm records (the summary view above
        # intentionally drops the ids).
        diffs_by_seed = []
        for entry in entries:
            arm = arms[entry["arm"]]
            fixed = arms[entry["fixed_arm"]]
            ids = sorted(set(arm["per_example"]) & set(fixed["per_example"]))
            diffs_by_seed.append(
                (ids, [fixed["per_example"][i] - arm["per_example"][i] for i in ids])
            )
        common_ids = sorted(set.intersection(*(set(ids) for ids, _ in diffs_by_seed)))
        averaged = []
        for example_id in common_ids:
            values = []
            for ids, diffs in diffs_by_seed:
                values.append(diffs[ids.index(example_id)])
            averaged.append(sum(values) / len(values))
        lo, hi = _bootstrap_ci([int(round(value * 1000)) for value in averaged], args.n_boot, args.seed)
        masses.append(
            {
                "mass": mass,
                "n_seeds": len(entries),
                "n_examples": len(common_ids),
                "mean_fixed_minus_dynamic": sum(averaged) / len(averaged),
                "ci95": [lo / 1000, hi / 1000],
                "fixed_wins": sum(1 for value in averaged if value > 0),
                "dynamic_wins": sum(1 for value in averaged if value < 0),
                "seeds": [
                    {
                        "seed": entry["seed"],
                        "paired": entry["paired"],
                        "support_stats": entry["support_stats"],
                        "dynamic_accuracy": entry["dynamic_accuracy"],
                        "fixed_accuracy": entry["fixed_accuracy"],
                        "dynamic_joint_ppl": entry["dynamic_joint_ppl"],
                        "fixed_joint_ppl": entry["fixed_joint_ppl"],
                    }
                    for entry in sorted(entries, key=lambda item: item["seed"])
                ],
            }
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "root": str(root),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "arms": [
            {
                key: value
                for key, value in arm.items()
                if key != "per_example"
            }
            for arm in sorted(arms.values(), key=lambda item: item["arm"])
        ],
        "per_seed": per_seed,
        "masses": masses,
    }
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out": str(out),
                "n_arms": len(arms),
                "masses": [
                    {
                        "mass": item["mass"],
                        "n_seeds": item["n_seeds"],
                        "mean_fixed_minus_dynamic": item["mean_fixed_minus_dynamic"],
                        "ci95": item["ci95"],
                    }
                    for item in masses
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
