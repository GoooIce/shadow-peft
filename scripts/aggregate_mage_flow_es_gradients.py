#!/usr/bin/env python3
"""Aggregate replayed ES gradients across disjoint Mage prompt episodes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

import torch

from shadow_lab.es import (
    CandidateObservation,
    CandidateSpec,
    ESConfig,
    candidate_specs,
    estimate_es_gradient,
    initialize_es,
    latent_to_clips,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    payload = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not values or not all(isinstance(value, dict) for value in values):
        raise ValueError(f"expected non-empty JSONL objects: {path}")
    return values


def comparable_identity(experiment: dict) -> dict:
    identity = copy.deepcopy(experiment["identity"])
    identity.get("search", {}).pop("episode_offset", None)
    return identity


def load_gradient_sources(
    source_runs: list[Path],
    *,
    expected_episode_offsets: list[int],
    expected_population_size: int = 30,
    expected_episode_size: int = 8,
    expected_sigma: float = 0.003,
    expected_es_seed: int = 20260906,
    expected_schedule_seed: int = 20260904,
) -> list[dict]:
    if len(source_runs) != 3 or len(expected_episode_offsets) != 3:
        raise ValueError("cross-prompt aggregation requires exactly three source runs")
    if len(set(expected_episode_offsets)) != 3 or any(
        episode < 0 for episode in expected_episode_offsets
    ):
        raise ValueError("expected episode offsets must be unique and non-negative")

    loaded = []
    reference_identity = None
    reference_clips = None
    reference_specs = None
    all_prompt_ids: set[str] = set()
    for source_path, expected_episode in zip(
        source_runs, expected_episode_offsets, strict=True
    ):
        source = source_path.expanduser().resolve()
        run = load_json(source / "run.json")
        experiment = load_json(source / "experiment.json")
        initialization = load_json(source / "initialization.json")
        summary = load_json(source / "generation-0000-summary.json")
        rows = load_jsonl(source / "generation-0000-candidates.jsonl")
        if run.get("completed_generations") != 1:
            raise ValueError(f"gradient source must contain one generation: {source}")
        config = ESConfig(**run["es_config"])
        if not (
            config.population_size == expected_population_size
            and math.isclose(config.sigma, expected_sigma, rel_tol=0.0, abs_tol=1e-15)
            and config.seed == expected_es_seed
            and config.shaping == "zscore"
        ):
            raise ValueError(f"gradient source ES protocol differs: {source}")
        generation = run["generation"]
        if not (
            generation["episode_size"] == expected_episode_size
            and generation["schedule_seed"] == expected_schedule_seed
            and generation.get("episode_offset", 0) == expected_episode
            and summary.get("prompt_episode_index", 0) == expected_episode
        ):
            raise ValueError(f"gradient source prompt protocol differs: {source}")

        clips = torch.load(
            source / "initial-clips.pt", map_location="cpu", weights_only=True
        )["clips"]
        if sha256_tensor(clips) != initialization["clip_sha256"]:
            raise ValueError(f"gradient source initialization differs from tensor: {source}")
        state = initialize_es(clips, config)
        expected_specs = candidate_specs(state)
        observations = []
        for index, row in enumerate(rows):
            if index >= len(expected_specs):
                raise ValueError(f"gradient source has too many candidates: {source}")
            spec = CandidateSpec(**row["spec"])
            if spec != expected_specs[index]:
                raise ValueError(f"gradient source candidate spec differs: {source}")
            observations.append(
                CandidateObservation(
                    spec,
                    None if row.get("reward") is None else float(row["reward"]),
                    row.get("failure_reason"),
                )
            )
        if len(observations) != expected_population_size or any(
            observation.reward is None for observation in observations
        ):
            raise ValueError(f"gradient source must be complete and failure-free: {source}")
        audits = [row["replay_audit"] for row in rows if "replay_audit" in row]
        replay_exact = bool(audits) and all(
            audit.get("exact") is True and all(audit.get("matches", {}).values())
            for audit in audits
        )
        if not replay_exact:
            raise ValueError(f"gradient source has no exact replay audit: {source}")
        estimate = estimate_es_gradient(state, observations)
        if estimate.reward_std < 0.02:
            raise ValueError(f"gradient source reward standard deviation is below 0.02: {source}")

        prompt_ids = [str(value) for value in summary["prompt_sample_ids"]]
        prompt_seeds = [int(value) for value in summary["prompt_seeds"]]
        if not (
            len(prompt_ids) == len(set(prompt_ids)) == expected_episode_size
            and len(prompt_seeds) == len(set(prompt_seeds)) == expected_episode_size
        ):
            raise ValueError(f"gradient source prompt batch is invalid: {source}")
        overlap = all_prompt_ids.intersection(prompt_ids)
        if overlap:
            raise ValueError(f"gradient source prompt batches overlap: {sorted(overlap)}")
        all_prompt_ids.update(prompt_ids)

        identity = comparable_identity(experiment)
        if reference_identity is None:
            reference_identity = identity
            reference_clips = clips
            reference_specs = expected_specs
        elif not (
            identity == reference_identity
            and torch.equal(clips, reference_clips)
            and expected_specs == reference_specs
        ):
            raise ValueError("gradient sources do not share identity, initialization and noise")
        loaded.append(
            {
                "path": source,
                "episode_index": expected_episode,
                "prompt_sample_ids": prompt_ids,
                "prompt_seeds": prompt_seeds,
                "specs": expected_specs,
                "estimate": estimate,
                "replay_exact": replay_exact,
                "initial_clips": clips,
                "config": config,
            }
        )
    return loaded


def gradient_consistency(
    labels: list[str], gradients: list[torch.Tensor]
) -> tuple[dict, torch.Tensor]:
    if len(labels) != len(gradients) or len(gradients) < 2:
        raise ValueError("gradient consistency requires matching labels and gradients")
    if len(set(labels)) != len(labels):
        raise ValueError("gradient labels must be unique")
    shape = gradients[0].shape
    if any(
        gradient.device.type != "cpu"
        or gradient.dtype != torch.float32
        or gradient.shape != shape
        or gradient.ndim != 1
        or not torch.isfinite(gradient).all()
        for gradient in gradients
    ):
        raise ValueError("gradients must be finite CPU float32 vectors of one shape")
    signs = [torch.sign(gradient) for gradient in gradients]
    pairwise = []
    for left_index, right_index in itertools.combinations(range(len(gradients)), 2):
        left = gradients[left_index]
        right = gradients[right_index]
        denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
        if denominator == 0:
            raise ValueError("gradient consistency is undefined for a zero vector")
        pairwise.append(
            {
                "left": labels[left_index],
                "right": labels[right_index],
                "cosine": float(torch.dot(left, right) / denominator),
                "sign_agreement_fraction": float(
                    torch.mean((signs[left_index] == signs[right_index]).float())
                ),
            }
        )
    stacked_signs = torch.stack(signs)
    unanimous = torch.all(stacked_signs == stacked_signs[0], dim=0)
    all_zero = torch.all(stacked_signs == 0, dim=0)
    mean = torch.stack(gradients).mean(dim=0).contiguous()
    return (
        {
            "pairwise": pairwise,
            "unanimous_sign_agreement_fraction": float(unanimous.float().mean()),
            "unanimous_nonzero_sign_fraction": float(
                (unanimous & ~all_zero).float().mean()
            ),
            "all_zero_fraction": float(all_zero.float().mean()),
        },
        mean,
    )


def average_gradient_candidates(
    initial_clips: torch.Tensor,
    config: ESConfig,
    mean_gradient: torch.Tensor,
    alphas: list[float],
) -> list[tuple[dict, torch.Tensor]]:
    if not alphas or len(set(alphas)) != len(alphas):
        raise ValueError("alpha values must be non-empty and unique")
    if any(not math.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("alpha values must be finite and positive")
    state = initialize_es(initial_clips, config)
    if (
        mean_gradient.device.type != "cpu"
        or mean_gradient.dtype != torch.float32
        or mean_gradient.shape != state.center.shape
        or not torch.isfinite(mean_gradient).all()
    ):
        raise ValueError("mean gradient differs from the initial ES center")
    initial_effective = latent_to_clips(state.center, config)
    candidates = []
    for alpha in alphas:
        clips = latent_to_clips(state.center + alpha * mean_gradient, config)
        candidates.append(
            (
                {
                    "alpha": alpha,
                    "latent_step_rms": alpha
                    * float(torch.sqrt(torch.mean(mean_gradient.square()))),
                    "clip_step_rms": float(
                        torch.sqrt(torch.mean((clips - initial_effective).square()))
                    ),
                    "center_clip_sha256": sha256_tensor(clips),
                    "center_clip_min": float(clips.min()),
                    "center_clip_max": float(clips.max()),
                    "center_clip_mean": float(clips.mean()),
                    "center_clip_std": float(clips.std()),
                },
                clips,
            )
        )
    return candidates


def aggregate_gradient_sources(
    sources: list[dict], alphas: list[float]
) -> tuple[dict, list[tuple[dict, torch.Tensor]]]:
    if len(sources) != 3:
        raise ValueError("cross-prompt aggregation requires exactly three loaded sources")
    labels = [f"episode-{source['episode_index']}" for source in sources]
    gradients = [source["estimate"].vector for source in sources]
    consistency, mean_gradient = gradient_consistency(labels, gradients)
    candidates = average_gradient_candidates(
        sources[0]["initial_clips"],
        sources[0]["config"],
        mean_gradient,
        alphas,
    )
    source_records = []
    for source, label in zip(sources, labels, strict=True):
        estimate = source["estimate"]
        source_path = source["path"]
        source_records.append(
            {
                "label": label,
                "path": str(source_path),
                "run_sha256": sha256_file(source_path / "run.json"),
                "experiment_sha256": sha256_file(source_path / "experiment.json"),
                "initialization_sha256": sha256_file(
                    source_path / "initialization.json"
                ),
                "candidates_sha256": sha256_file(
                    source_path / "generation-0000-candidates.jsonl"
                ),
                "summary_sha256": sha256_file(
                    source_path / "generation-0000-summary.json"
                ),
                "episode_index": source["episode_index"],
                "prompt_sample_ids": source["prompt_sample_ids"],
                "prompt_seeds": source["prompt_seeds"],
                "candidate_count": len(source["specs"]),
                "failure_count": estimate.failure_count,
                "reward_mean": estimate.reward_mean,
                "reward_std": estimate.reward_std,
                "gradient_sha256": sha256_tensor(estimate.vector),
                "gradient_norm": estimate.norm,
                "gradient_rms": estimate.rms,
                "replay_exact": source["replay_exact"],
            }
        )
    config = sources[0]["config"]
    report = {
        "schema_version": 1,
        "protocol": {
            "observation_episodes": [
                source["episode_index"] for source in sources
            ],
            "population_size": config.population_size,
            "episode_size": len(sources[0]["prompt_sample_ids"]),
            "sigma": config.sigma,
            "es_seed": config.seed,
            "schedule_seed": 20260904,
            "shaping": config.shaping,
            "aggregation": "elementwise arithmetic mean of three z-score ES gradients",
            "alphas": alphas,
            "clip_step_rms_max": 0.02,
        },
        "initial_clip_sha256": sha256_tensor(sources[0]["initial_clips"]),
        "sources": source_records,
        "consistency": consistency,
        "mean_gradient": {
            "sha256": sha256_tensor(mean_gradient),
            "norm": float(torch.linalg.vector_norm(mean_gradient)),
            "rms": float(torch.sqrt(torch.mean(mean_gradient.square()))),
        },
        "candidates": [record for record, _ in candidates],
    }
    return report, candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", action="append", required=True, type=Path)
    parser.add_argument(
        "--expected-episode-offsets",
        nargs=3,
        type=int,
        default=(4, 5, 6),
    )
    parser.add_argument(
        "--alphas",
        nargs=2,
        type=float,
        default=(0.00003, 0.0001),
    )
    parser.add_argument("--centers-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def alpha_slug(alpha: float) -> str:
    return format(alpha, ".12g").replace("-", "m").replace(".", "p")


def atomic_save_tensor(path: Path, clips: torch.Tensor, metadata: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save({"clips": clips, "metadata": metadata}, temporary)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    out_path = args.out.expanduser().resolve()
    centers_dir = args.centers_dir.expanduser().resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing existing gradient report: {out_path}")
    if centers_dir.exists():
        raise FileExistsError(f"refusing existing centers directory: {centers_dir}")
    sources = load_gradient_sources(
        args.source_run,
        expected_episode_offsets=list(args.expected_episode_offsets),
    )
    report, candidates = aggregate_gradient_sources(sources, list(args.alphas))

    centers_dir.mkdir(parents=True)
    state = initialize_es(sources[0]["initial_clips"], sources[0]["config"])
    initial_effective = latent_to_clips(state.center, state.config)
    initial_path = centers_dir / "initial-effective-clips.pt"
    atomic_save_tensor(
        initial_path,
        initial_effective,
        {
            "kind": "initial-effective-center",
            "gradient_report": str(out_path),
        },
    )
    report["initial_effective_center"] = {
        "path": str(initial_path),
        "file_sha256": sha256_file(initial_path),
        "clip_sha256": sha256_tensor(initial_effective),
    }
    for record, clips in candidates:
        path = centers_dir / f"alpha-{alpha_slug(float(record['alpha']))}-clips.pt"
        atomic_save_tensor(
            path,
            clips,
            {
                "kind": "cross-prompt-mean-gradient-center",
                "alpha": record["alpha"],
                "mean_gradient_sha256": report["mean_gradient"]["sha256"],
                "gradient_report": str(out_path),
            },
        )
        record["center_clip_path"] = str(path)
        record["center_file_sha256"] = sha256_file(path)
    report["implementation"] = {
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "es_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "src/shadow_lab/es.py"
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
