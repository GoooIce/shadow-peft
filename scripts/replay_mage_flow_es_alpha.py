#!/usr/bin/env python3
"""Replay one ES observation batch across alpha values without new generations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path

import torch

from shadow_lab.es import (
    CandidateObservation,
    CandidateSpec,
    ESConfig,
    candidate_specs,
    initialize_es,
    latent_to_clips,
    load_es_checkpoint,
    update_es,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=(0.0003, 0.001, 0.003),
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--centers-dir",
        type=Path,
        help="Optionally materialize the initial and alpha-updated effective clip tensors.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    payload = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def alpha_candidates(
    initial_clips: torch.Tensor,
    base_config: ESConfig,
    observations: list[CandidateObservation],
    alphas: list[float],
) -> list[tuple[dict, torch.Tensor]]:
    if not alphas or len(set(alphas)) != len(alphas):
        raise ValueError("alpha values must be non-empty and unique")
    if any(not math.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("alpha values must be finite and positive")
    candidates = []
    for alpha in alphas:
        config = replace(base_config, learning_rate=alpha)
        state = initialize_es(initial_clips, config)
        if tuple(item.spec for item in observations) != candidate_specs(state):
            raise ValueError("observations are not the generation-zero replay population")
        initial_effective = latent_to_clips(state.center, config)
        update = update_es(state, observations)
        updated_effective = latent_to_clips(update.state.center, config)
        candidates.append(
            (
                {
                    "alpha": alpha,
                    "gradient_norm": update.gradient_norm,
                    "gradient_rms": update.gradient_rms,
                    "latent_step_rms": update.latent_step_rms,
                    "clip_step_rms": float(
                        torch.sqrt(
                            torch.mean((updated_effective - initial_effective).square())
                        )
                    ),
                    "center_clip_sha256": sha256_tensor(updated_effective),
                    "center_clip_min": float(updated_effective.min()),
                    "center_clip_max": float(updated_effective.max()),
                    "center_clip_mean": float(updated_effective.mean()),
                    "center_clip_std": float(updated_effective.std()),
                },
                updated_effective,
            )
        )
    return candidates


def alpha_records(
    initial_clips: torch.Tensor,
    base_config: ESConfig,
    observations: list[CandidateObservation],
    alphas: list[float],
) -> list[dict]:
    return [
        record
        for record, _ in alpha_candidates(
            initial_clips,
            base_config,
            observations,
            alphas,
        )
    ]


def alpha_slug(alpha: float) -> str:
    return format(alpha, ".12g").replace("-", "m").replace(".", "p")


def atomic_save_tensor(path: Path, clips: torch.Tensor, metadata: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save({"clips": clips, "metadata": metadata}, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    source_run = args.source_run.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing existing alpha replay output: {out_path}")
    centers_dir = (
        args.centers_dir.expanduser().resolve() if args.centers_dir is not None else None
    )
    if centers_dir is not None and centers_dir.exists():
        raise FileExistsError(f"refusing existing alpha centers directory: {centers_dir}")

    final_state = load_es_checkpoint(source_run / "es-state.pt")
    initial_clips = torch.load(
        source_run / "initial-clips.pt", map_location="cpu", weights_only=True
    )["clips"]
    candidate_path = source_run / "generation-0000-candidates.jsonl"
    candidate_rows = [
        json.loads(line)
        for line in candidate_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    observations = [
        CandidateObservation(
            CandidateSpec(**row["spec"]),
            None if row.get("reward") is None else float(row["reward"]),
            row.get("failure_reason"),
        )
        for row in candidate_rows
    ]
    if not observations or any(item.reward is None for item in observations):
        raise ValueError("alpha replay requires one complete failure-free generation")
    candidates = alpha_candidates(
        initial_clips,
        final_state.config,
        observations,
        list(args.alphas),
    )
    records = [record for record, _ in candidates]
    materialized = None
    if centers_dir is not None:
        centers_dir.mkdir(parents=True)
        initial_state = initialize_es(initial_clips, final_state.config)
        effective_initial = latent_to_clips(
            initial_state.center,
            initial_state.config,
        )
        initial_path = centers_dir / "initial-effective-clips.pt"
        atomic_save_tensor(
            initial_path,
            effective_initial,
            {"kind": "initial-effective-center", "source_run": str(source_run)},
        )
        materialized = {
            "initial": {
                "path": str(initial_path),
                "sha256": sha256_file(initial_path),
                "clip_sha256": sha256_tensor(effective_initial),
            },
            "alphas": {},
        }
        for record, clips in candidates:
            alpha = float(record["alpha"])
            center_path = centers_dir / f"alpha-{alpha_slug(alpha)}-clips.pt"
            atomic_save_tensor(
                center_path,
                clips,
                {
                    "kind": "alpha-updated-center",
                    "alpha": alpha,
                    "source_run": str(source_run),
                },
            )
            record["center_clip_path"] = str(center_path)
            record["center_file_sha256"] = sha256_file(center_path)
            materialized["alphas"][str(alpha)] = {
                "path": str(center_path),
                "sha256": record["center_file_sha256"],
                "clip_sha256": record["center_clip_sha256"],
            }
    report = {
        "schema_version": 1,
        "source": {
            "run_dir": str(source_run),
            "run_sha256": sha256_file(source_run / "run.json"),
            "initial_clips_sha256": sha256_file(source_run / "initial-clips.pt"),
            "generation_0_candidates_sha256": sha256_file(candidate_path),
        },
        "population_size": final_state.config.population_size,
        "sigma": final_state.config.sigma,
        "observation_count": len(observations),
        "reward_mean": sum(
            float(item.reward) for item in observations if item.reward is not None
        )
        / len(observations),
        "records": records,
    }
    if materialized is not None:
        report["materialized_centers"] = materialized
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(f".{out_path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, out_path)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
