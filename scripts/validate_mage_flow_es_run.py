#!/usr/bin/env python3
"""Independently validate a completed Mage Phase-1 ES artifact directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from shadow_lab.es import (
    CandidateObservation,
    CandidateSpec,
    candidate_clips,
    candidate_specs,
    initialize_es,
    latent_to_clips,
    load_es_checkpoint,
    update_es,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--require-replay-audit", action="store_true")
    parser.add_argument("--out", type=Path)
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
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"expected JSONL objects: {path}")
    return values


def require_close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} differs: {actual} != {expected}")


def validate_run(run_dir: Path, *, require_replay_audit: bool = False) -> dict:
    run_dir = run_dir.expanduser().resolve()
    run = load_json(run_dir / "run.json")
    if torch.__version__ != run["versions"]["torch"]:
        raise RuntimeError(
            "seed replay requires the recorded Torch version: "
            f"running={torch.__version__}, recorded={run['versions']['torch']}"
        )
    experiment = load_json(run_dir / "experiment.json")
    initialization = load_json(run_dir / "initialization.json")
    layout = load_json(run_dir / "layout.json")
    final_state = load_es_checkpoint(run_dir / "es-state.pt")
    initial_payload = torch.load(
        run_dir / "initial-clips.pt", map_location="cpu", weights_only=True
    )
    initial_clips = initial_payload["clips"]

    for field in ("experiment", "layout", "initialization", "checkpoint"):
        reference = run[field]
        path = run_dir / reference["path"]
        if sha256_file(path) != reference["sha256"]:
            raise ValueError(f"run manifest hash differs for {field}")
    if sha256_tensor(initial_clips) != initialization["clip_sha256"]:
        raise ValueError("initial clip tensor differs from initialization manifest")

    dimension = int(layout["dimension"])
    if not (
        dimension
        == initial_clips.numel()
        == final_state.config.dimension
        == initialization["quasar"]["dimension"]
    ):
        raise ValueError("ES dimensions disagree across artifacts")
    if sum(layer["stop"] - layer["start"] for layer in layout["layers"]) != dimension:
        raise ValueError("layout layer slices do not cover the ES dimension")

    configured_generations = int(experiment["identity"]["search"]["generations"])
    completed_generations = int(run["completed_generations"])
    if not (
        configured_generations == completed_generations == final_state.generation
    ):
        raise ValueError("generation cursors disagree")

    state = initialize_es(initial_clips, final_state.config)
    if "effective_clip_sha256" in initialization and sha256_tensor(
        latent_to_clips(state.center, state.config)
    ) != initialization["effective_clip_sha256"]:
        raise ValueError("effective initial clips differ from initialization manifest")
    committed_summaries = load_jsonl(run_dir / "generations.jsonl")
    if len(committed_summaries) != completed_generations:
        raise ValueError("generation rollup length differs from checkpoint")
    all_prompt_ids: list[str] = []
    reward_stds = []
    failure_count = 0
    replay_audits = []
    episode_offset = int(run["generation"].get("episode_offset", 0))

    for generation in range(completed_generations):
        center_clips = latent_to_clips(state.center, state.config)
        rows = load_jsonl(
            run_dir / f"generation-{generation:04d}-candidates.jsonl"
        )
        expected_specs = candidate_specs(state)
        if len(rows) != len(expected_specs):
            raise ValueError(f"generation {generation} candidate count differs")
        observations = []
        for index, (row, expected_spec) in enumerate(
            zip(rows, expected_specs, strict=True)
        ):
            spec = CandidateSpec(**row["spec"])
            if spec != expected_spec:
                raise ValueError(f"generation {generation} candidate {index} spec differs")
            reward = None if row.get("reward") is None else float(row["reward"])
            observation = CandidateObservation(spec, reward, row.get("failure_reason"))
            observations.append(observation)
            if reward is not None:
                replayed_clips = candidate_clips(state, spec)
                if sha256_tensor(replayed_clips) != row["clip_sha256"]:
                    raise ValueError(
                        f"generation {generation} candidate {index} clip replay differs"
                    )
                if "clip_statistics" in row:
                    clip_statistics = row["clip_statistics"]
                    expected_statistics = {
                        "min": float(replayed_clips.min()),
                        "max": float(replayed_clips.max()),
                        "mean": float(replayed_clips.mean()),
                        "std": float(replayed_clips.std()),
                        "perturbation_rms": float(
                            torch.sqrt(
                                torch.mean((replayed_clips - center_clips).square())
                            )
                        ),
                    }
                    if clip_statistics != expected_statistics:
                        raise ValueError("candidate clip statistics differ from replay")
                    if row.get("trainable_parameters_after") != 0:
                        raise ValueError("candidate retained trainable model parameters")
                    if row.get("quantized_layers") != len(layout["layers"]):
                        raise ValueError("candidate quantized layer count differs")
                    if row.get("clip_group_count") != dimension:
                        raise ValueError("candidate clip group count differs")
                if not math.isfinite(float(row["weighted_reconstruction_error"])):
                    raise ValueError("non-finite reconstruction error")
                if len(row["image_sha256"]) != experiment["identity"]["search"][
                    "episode_size"
                ]:
                    raise ValueError("candidate image count differs from episode size")
            audit = row.get("replay_audit")
            if audit is not None:
                replay_audits.append(audit)
                if audit.get("exact") is not True or not all(
                    audit.get("matches", {}).values()
                ):
                    raise ValueError("candidate replay audit is not exact")

        update = update_es(state, observations)
        summary_path = run_dir / f"generation-{generation:04d}-summary.json"
        summary = load_json(summary_path)
        if summary.get("prompt_episode_index", generation) != episode_offset + generation:
            raise ValueError(f"generation {generation} prompt episode index differs")
        if committed_summaries[generation] != summary:
            raise ValueError(f"generation {generation} rollup differs from summary")
        require_close(update.reward_mean, float(summary["reward_mean"]), "reward mean")
        require_close(update.reward_std, float(summary["reward_std"]), "reward std")
        require_close(max(update.raw_rewards), float(summary["reward_max"]), "reward max")
        require_close(
            update.gradient_norm,
            float(summary["gradient_norm"]),
            "gradient norm",
        )
        next_center_clips = latent_to_clips(update.state.center, update.state.config)
        if "gradient_rms" in summary:
            require_close(update.gradient_rms, summary["gradient_rms"], "gradient RMS")
            require_close(
                update.latent_step_rms,
                summary["latent_step_rms"],
                "latent step RMS",
            )
            require_close(
                float(torch.sqrt(torch.mean((next_center_clips - center_clips).square()))),
                summary["clip_step_rms"],
                "clip step RMS",
            )
        if sha256_tensor(next_center_clips) != summary["center_clip_sha256"]:
            raise ValueError(f"generation {generation} center clip replay differs")
        if update.failure_count != int(summary["failure_count"]):
            raise ValueError(f"generation {generation} failure count differs")
        prompt_ids = [str(value) for value in summary["prompt_sample_ids"]]
        if len(prompt_ids) != len(set(prompt_ids)):
            raise ValueError(f"generation {generation} repeats prompts")
        all_prompt_ids.extend(prompt_ids)
        reward_stds.append(update.reward_std)
        failure_count += update.failure_count
        state = update.state

    if not torch.equal(state.center, final_state.center):
        raise ValueError("replayed final ES center differs from checkpoint")
    if len(all_prompt_ids) != len(set(all_prompt_ids)):
        raise ValueError("smoke generations reused a prompt before epoch exhaustion")
    if require_replay_audit and not replay_audits:
        raise ValueError("run has no candidate replay audit")
    return {
        "valid": True,
        "run_dir": str(run_dir),
        "dimension": dimension,
        "completed_generations": completed_generations,
        "population_size": final_state.config.population_size,
        "candidate_count": completed_generations * final_state.config.population_size,
        "failure_count": failure_count,
        "reward_stds": reward_stds,
        "all_generations_have_reward_signal": all(value > 0 for value in reward_stds),
        "replay_audit_count": len(replay_audits),
        "process_peak_rss_bytes": int(run["process_peak_rss_bytes"]),
        "final_checkpoint_sha256": sha256_file(run_dir / "es-state.pt"),
    }


def main() -> None:
    args = parse_args()
    report = validate_run(
        args.run_dir,
        require_replay_audit=args.require_replay_audit,
    )
    encoded = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    if args.out is not None:
        out_path = args.out.expanduser().resolve()
        if out_path.exists():
            raise FileExistsError(f"refusing existing validation output: {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = out_path.with_name(f".{out_path.name}.tmp-{os.getpid()}")
        temporary.write_text(encoded + "\n", encoding="utf-8")
        os.replace(temporary, out_path)
    print(encoded)


if __name__ == "__main__":
    main()
