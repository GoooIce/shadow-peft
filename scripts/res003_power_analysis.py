#!/usr/bin/env python3
"""Run the frozen RES-003 power and null-calibration grid on idle CPU (0 GPUh).

This script consumes only the frozen statistical plan
(``configs/mage-healed-es/res003-statistical-plan.json``); every grid, seed and
resample budget lives in the plan so the simulation is pre-registered before any
new outcome data exists. It never reads campaign scores and never touches GPUs.

Modes:
  --dump-plan   print the frozen plan JSON (used once to create the freeze file)
  --smoke       feasibility run with reduced simulation counts (not evidence)
  default       full pre-registered run; writes one JSON artifact
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import subprocess
import time
from pathlib import Path

import torch

from shadow_lab.es_inference import (
    SIGN_MODEL_PROBABILITY_6_OF_8,
    StatisticalPlan,
    budget_verdict,
    consistency_checkpoint_power,
    cosine_episode_flip_null,
    delta_support_table,
    episode_pair_cosines,
    joint_gate_power,
    preference_power,
    sequential_checkpoint_power,
)


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{int(time.time())}-{time.monotonic_ns()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def naive_z_type_one(episodes: int, dimension: int, n_sims: int, seed: int) -> dict:
    """Realized type-I error of the naive independent-pair z for mean cosine at null.

    Under independent random directions (true mean cosine 0) this records how often
    the naive z, which treats all C(k,2) cosines as independent, would reject at the
    nominal two-sided 5% level — the calibration F02 asked for alongside the
    episode-flip null.
    """

    generator = torch.Generator()
    generator.manual_seed(seed)
    directions = torch.randn(n_sims, episodes, dimension, generator=generator)
    cosines = episode_pair_cosines(directions)
    pair_sd = cosines.std(dim=1, unbiased=False)
    naive_sd = pair_sd / math.sqrt(cosines.shape[1])
    naive_z = cosines.mean(dim=1) / naive_sd
    rejects = (naive_z.abs() >= 1.959964).float().mean().item()
    return {
        "episodes": episodes,
        "dimension": dimension,
        "pair_count": int(cosines.shape[1]),
        "n_sims": n_sims,
        "seed": seed,
        "nominal_two_sided_5pct": 0.05,
        "naive_z_reject_rate": rejects,
        "mean_naive_z": float(naive_z.mean().item()),
        "p99_naive_z": float(naive_z.quantile(0.99).item()),
    }


def flip_null_reference(episodes: int, dimension: int, seed: int, resamples: int) -> dict:
    """One synthetic null gradient set analysed with the episode-flip null."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    vectors = [torch.randn(dimension, generator=generator) for _ in range(episodes)]
    return cosine_episode_flip_null(vectors, resamples=resamples)


def run_plan(plan: StatisticalPlan, *, smoke: bool) -> tuple[dict, dict, list[str]]:
    scale = 0.1 if smoke else 1.0
    n_sims = max(200, int(plan.power_sims_per_cell * scale))
    resamples = max(200, int(plan.bootstrap_resamples * scale))
    sections: dict = {}
    notes: list[str] = []

    started = time.time()
    gate_rules = {}
    for episodes in plan.gate_episodes:
        gate_rules[str(episodes)] = joint_gate_power(
            episodes=episodes,
            # The registered rule is 6-of-8; larger k columns scale the same
            # proportion and are design calibration, not a re-registered rule.
            min_non_degrading=round(
                episodes * plan.min_non_degrading_batches / plan.selection_episodes
            ),
            effects=plan.gate_effect_grid,
            n_sims=n_sims,
            seed=plan.bootstrap_seed,
        )
    sections["joint_gate"] = gate_rules
    gate_seconds = time.time() - started

    started = time.time()
    support = delta_support_table(
        episode_counts=plan.gate_episodes,
        sigma_over_delta=plan.sigma_over_delta_grid,
        effects=plan.support_effect_grid,
        n_sims=n_sims,
        bootstrap_resamples=resamples,
        seed=plan.bootstrap_seed,
        confidence=plan.confidence,
        delta=plan.delta_primary,
    )
    budget = {
        f"k{row['episodes']}/sigma{row['sigma_over_delta']:g}": budget_verdict(
            support,
            episodes=row["episodes"],
            sigma_over_delta=row["sigma_over_delta"],
            min_power_support=plan.min_power_support,
            min_power_interval=plan.min_power_interval,
        )
        for row in support
        if row["effect"] == 0.0
    }
    sections["delta_support"] = {"rows": support, "budget_verdicts": budget}
    support_seconds = time.time() - started

    started = time.time()
    sections["consistency_checkpoint"] = consistency_checkpoint_power(
        episode_counts=plan.checkpoint_episode_counts,
        rho_grid=plan.checkpoint_rho_grid,
        dimension=plan.checkpoint_dimension,
        n_sims=n_sims,
        seed=plan.bootstrap_seed,
        mean_threshold=plan.checkpoint_mean_threshold,
        positive_fraction_threshold=plan.checkpoint_positive_fraction_threshold,
    )
    sections["sequential_checkpoint_rule"] = sequential_checkpoint_power(
        gradient_episodes=plan.gradient_episodes,
        checkpoints=[
            (
                count,
                plan.checkpoint_mean_threshold,
                plan.checkpoint_positive_fraction_threshold,
            )
            for count in plan.checkpoint_episode_counts
            if count <= 16
        ],
        rho_grid=plan.checkpoint_rho_grid,
        dimension=plan.checkpoint_dimension,
        n_sims=n_sims,
        seed=plan.bootstrap_seed,
    )
    checkpoint_seconds = time.time() - started

    started = time.time()
    preference = {}
    for decisive in plan.preference_decisive_rate_grid:
        preference[f"decisive_{decisive:g}"] = preference_power(
            prompts=plan.preference_prompts,
            judgments_per_prompt=plan.preference_judgments_per_prompt,
            q_grid=plan.preference_q_grid,
            decisive_rate=decisive,
            n_sims=n_sims,
            bootstrap_resamples=resamples,
            seed=plan.bootstrap_seed,
            delta=plan.preference_delta,
        )
    sections["preference"] = preference
    preference_seconds = time.time() - started

    started = time.time()
    calibration_n = max(200, int(2_000 * scale))
    sections["cosine_hierarchy"] = {
        "sign_model_probability_6_of_8": SIGN_MODEL_PROBABILITY_6_OF_8,
        "naive_z_type_one": [
            naive_z_type_one(episodes, plan.checkpoint_dimension, calibration_n, plan.bootstrap_seed)
            for episodes in (8, 24)
        ],
        "flip_null_reference_k8": flip_null_reference(
            8, plan.checkpoint_dimension, plan.bootstrap_seed, plan.flip_resamples
        ),
    }
    cosine_seconds = time.time() - started

    # Determinism self-check: identical seed must reproduce a cell bit-for-bit.
    repeat = joint_gate_power(
        episodes=plan.selection_episodes,
        min_non_degrading=plan.min_non_degrading_batches,
        effects=[0.0],
        n_sims=n_sims,
        seed=plan.bootstrap_seed,
    )
    first = sections["joint_gate"][str(plan.selection_episodes)][0]
    if repeat[0]["pass_rate"] != first["pass_rate"]:
        notes.append("determinism self-check failed for the null gate cell")
    sections_timing = {
        "joint_gate_seconds": gate_seconds,
        "delta_support_seconds": support_seconds,
        "consistency_checkpoint_seconds": checkpoint_seconds,
        "preference_seconds": preference_seconds,
        "cosine_hierarchy_seconds": cosine_seconds,
    }
    return sections, sections_timing, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/mage-healed-es/res003-statistical-plan.json"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="reduced counts, not evidence")
    parser.add_argument("--dump-plan", action="store_true", help="print frozen plan JSON")
    parser.add_argument("--source-commit", default=None, help="git commit of the synced source")
    args = parser.parse_args()

    if args.dump_plan:
        print(StatisticalPlan.default().canonical_json())
        return 0

    plan_record = json.loads(args.plan.read_text(encoding="utf-8"))
    plan = StatisticalPlan.from_json(json.dumps(plan_record))
    plan_sha = sha256_file(args.plan)

    started = time.time()
    sections, timings, notes = run_plan(plan, smoke=args.smoke)
    elapsed = time.time() - started

    # ru_maxrss is bytes on macOS but KiB on Linux; normalize to MiB.
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != "Darwin":
        peak_rss *= 1024
    payload = {
        "schema_version": 1,
        "ticket": "RES-003",
        "mode": "smoke" if args.smoke else "full",
        "plan_path": str(args.plan),
        "plan_sha256": plan_sha,
        "plan": plan_record,
        "source_commit": args.source_commit or git_commit(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
        },
        "wall_seconds": elapsed,
        "section_seconds": timings,
        "peak_rss_mib": peak_rss / (1024 * 1024),
        "stability_notes": notes,
        "results": sections,
    }
    output = args.output
    if output is None:
        suffix = "smoke" if args.smoke else "full"
        output = Path(f"runs-local/mage-flow/res003-power-analysis-{suffix}.json")
    atomic_write_json(payload, output)
    print(f"plan sha256: {plan_sha}")
    print(f"artifact: {output} ({sha256_file(output)})")
    print(f"wall: {elapsed:.1f}s, peak rss: {payload['peak_rss_mib']:.0f} MiB")
    if notes:
        print("NOTES:", *notes, sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
