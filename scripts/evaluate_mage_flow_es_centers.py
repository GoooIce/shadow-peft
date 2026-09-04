#!/usr/bin/env python3
"""Evaluate fixed Mage NVFP4 clip centers on one held-out ES prompt batch."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import socket
import statistics
import time
from pathlib import Path

from shadow_lab.es_quant import ClipGenomeLayout, FrozenLinearWeights, materialize_es_candidate
from shadow_lab.image_eval import (
    RewardCalibration,
    RewardInput,
    aggregate_episode_reward,
    episode_prompt_batch,
    load_frozen_eval_samples,
    score_reward_batch,
)
from shadow_lab.image_reward_models import HPSv21Scorer, OpenCLIPH14Scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--prompts-jsonl", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--hps-checkpoint", required=True, type=Path)
    parser.add_argument("--clip-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--source-run",
        required=True,
        type=Path,
        help="Completed one-generation sigma arm from which alpha centers were replayed.",
    )
    parser.add_argument(
        "--additional-source-run",
        action="append",
        default=[],
        type=Path,
        help="Additional gradient source whose prompt episodes must remain excluded.",
    )
    parser.add_argument(
        "--prior-evaluation",
        action="append",
        default=[],
        type=Path,
        help="Prior center-evaluation report whose prompt batch must not be reused.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="A unique LABEL=PATH clip tensor; first candidate is the comparison baseline.",
    )
    parser.add_argument("--episode-index", type=int, default=1)
    parser.add_argument("--episode-size", type=int, default=8)
    parser.add_argument("--schedule-seed", type=int, default=20260904)
    parser.add_argument("--reward-batch-size", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--audit-first-candidate", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(value) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def model_directory_identity(path: Path) -> dict:
    files = []
    for item in sorted(item for item in path.rglob("*") if item.is_file()):
        record = {
            "path": item.relative_to(path).as_posix(),
            "size": item.stat().st_size,
        }
        if item.suffix.lower() == ".json":
            record["sha256"] = sha256_file(item)
        files.append(record)
    if not files:
        raise ValueError(f"model directory is empty: {path}")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "path": str(path),
        "file_count": len(files),
        "metadata_and_size_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
    }


def parse_candidates(values: list[str]) -> list[tuple[str, Path]]:
    candidates = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"candidate must use LABEL=PATH: {value}")
        label, raw_path = value.split("=", 1)
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", label):
            raise ValueError(f"invalid candidate label: {label}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"candidate tensor does not exist: {path}")
        candidates.append((label, path))
    labels = [label for label, _ in candidates]
    if len(set(labels)) != len(labels):
        raise ValueError("candidate labels must be unique")
    return candidates


def png_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def generate_inputs(pipeline, samples, args) -> list[RewardInput]:
    inputs = []
    for sample in samples:
        image = pipeline.generate(
            [sample.prompt],
            seeds=[sample.seed],
            heights=[args.height],
            widths=[args.width],
            steps=args.steps,
            cfg=args.cfg,
        )[0]
        inputs.append(RewardInput(png_bytes(image), sample))
    return inputs


def evaluate_clips(
    *,
    torch,
    clips,
    transformer,
    master,
    layout,
    pipeline,
    prompt_batch,
    scorers,
    calibration,
    args,
) -> dict:
    quantize_started = time.perf_counter()
    quantization = materialize_es_candidate(
        transformer,
        master,
        layout,
        clips,
        device="cuda",
    )
    torch.cuda.synchronize()
    quantize_seconds = time.perf_counter() - quantize_started
    if quantization["trainable_parameters_after"] != 0:
        raise RuntimeError("center materialization left trainable weights")

    generate_started = time.perf_counter()
    reward_inputs = generate_inputs(pipeline, prompt_batch, args)
    torch.cuda.synchronize()
    generate_seconds = time.perf_counter() - generate_started
    score_started = time.perf_counter()
    scores = score_reward_batch(reward_inputs, scorers)
    episode = aggregate_episode_reward(scores, calibration)
    torch.cuda.synchronize()
    score_seconds = time.perf_counter() - score_started
    if not episode.valid or episode.value is None:
        raise RuntimeError(episode.failure_reason)
    return {
        "reward": episode.value,
        "clip_sha256": sha256_tensor(clips),
        "clip_statistics": {
            "min": float(clips.min()),
            "max": float(clips.max()),
            "mean": float(clips.mean()),
            "std": float(clips.std()),
        },
        "weighted_reconstruction_error": quantization[
            "weighted_reconstruction_error"
        ],
        "quantized_layers": quantization["quantized_layers"],
        "clip_group_count": quantization["clip_group_count"],
        "trainable_parameters_after": quantization["trainable_parameters_after"],
        "axis_means": {
            axis: statistics.fmean(row["scores"][axis] for row in scores)
            for axis in ("hps_v2_1", "clip_h14")
        },
        "image_sha256": [item.image_sha256 for item in reward_inputs],
        "timing_seconds": {
            "quantize": quantize_seconds,
            "generate": generate_seconds,
            "score": score_seconds,
        },
    }


def replay_comparison(first: dict, replay: dict) -> dict:
    fields = (
        "reward",
        "clip_sha256",
        "clip_statistics",
        "weighted_reconstruction_error",
        "quantized_layers",
        "clip_group_count",
        "trainable_parameters_after",
        "axis_means",
        "image_sha256",
    )
    matches = {field: first[field] == replay[field] for field in fields}
    return {"exact": all(matches.values()), "matches": matches}


def reward_deltas(records: list[dict]) -> list[dict]:
    if not records:
        raise ValueError("center evaluation has no records")
    baseline = float(records[0]["reward"])
    return [
        {
            "label": record["label"],
            "reward_delta_vs_baseline": float(record["reward"]) - baseline,
            "non_degrading": float(record["reward"]) >= baseline,
        }
        for record in records
    ]


def validate_source_run(
    source_run: Path,
    *,
    assets: dict,
    model_identity: dict,
    episode_index: int,
    episode_sample_ids: list[str],
    schedule_seed: int,
    excluded_sample_ids: set[str] | None = None,
) -> dict:
    run_path = source_run / "run.json"
    experiment_path = source_run / "experiment.json"
    if not run_path.is_file() or not experiment_path.is_file():
        raise FileNotFoundError("source sigma arm must be complete")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    identity = experiment["identity"]
    if identity["model"]["metadata_and_size_sha256"] != model_identity[
        "metadata_and_size_sha256"
    ]:
        raise ValueError("center evaluation model differs from source sigma arm")
    for key in ("prompts", "calibration", "hps_v2_1", "clip_h14"):
        if identity["assets"][key]["sha256"] != assets[key]["sha256"]:
            raise ValueError(f"center evaluation {key} differs from source sigma arm")
    if run["generation"]["schedule_seed"] != schedule_seed:
        raise ValueError("center evaluation schedule seed differs from source sigma arm")
    episode_offset = int(run["generation"].get("episode_offset", 0))
    if episode_index < episode_offset + run["completed_generations"]:
        raise ValueError("center evaluation must use an unseen episode index")
    seen_ids = set()
    for generation in range(run["completed_generations"]):
        path = source_run / f"generation-{generation:04d}-summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("prompt_episode_index", episode_offset + generation) != (
            episode_offset + generation
        ):
            raise ValueError("source run prompt episode index differs from its manifest")
        seen_ids.update(summary["prompt_sample_ids"])
    seen_ids.update(excluded_sample_ids or ())
    overlap = seen_ids.intersection(episode_sample_ids)
    if overlap:
        raise ValueError(f"center evaluation episode reuses source prompts: {sorted(overlap)}")
    return {
        "run_path": str(run_path),
        "run_sha256": sha256_file(run_path),
        "experiment_path": str(experiment_path),
        "experiment_sha256": sha256_file(experiment_path),
        "source_completed_generations": run["completed_generations"],
        "episode_offset": episode_offset,
    }


def validate_source_runs(
    source_runs: list[Path],
    *,
    assets: dict,
    model_identity: dict,
    episode_index: int,
    episode_sample_ids: list[str],
    schedule_seed: int,
    excluded_sample_ids: set[str] | None = None,
) -> list[dict]:
    if not source_runs or len(set(source_runs)) != len(source_runs):
        raise ValueError("source runs must be non-empty and unique")
    prior_ids = set(excluded_sample_ids or ())
    source_ids: set[str] = set()
    identities = []
    for source_run in source_runs:
        run = json.loads((source_run / "run.json").read_text(encoding="utf-8"))
        current_ids = set()
        for generation in range(int(run["completed_generations"])):
            summary = json.loads(
                (
                    source_run / f"generation-{generation:04d}-summary.json"
                ).read_text(encoding="utf-8")
            )
            current_ids.update(str(value) for value in summary["prompt_sample_ids"])
        overlap = source_ids.intersection(current_ids)
        if overlap:
            raise ValueError(f"source run prompt batches overlap: {sorted(overlap)}")
        identities.append(
            validate_source_run(
                source_run,
                assets=assets,
                model_identity=model_identity,
                episode_index=episode_index,
                episode_sample_ids=episode_sample_ids,
                schedule_seed=schedule_seed,
                excluded_sample_ids=prior_ids | source_ids,
            )
        )
        source_ids.update(current_ids)
    return identities


def atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.episode_index < 0 or args.episode_size <= 0 or args.reward_batch_size <= 0:
        raise ValueError("episode index must be non-negative and sizes must be positive")
    if args.height % 16 or args.width % 16:
        raise ValueError("Mage image dimensions must be multiples of 16")
    out_path = args.out.expanduser().resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing existing center evaluation: {out_path}")
    candidates = parse_candidates(args.candidate)
    model_path = args.model_path.expanduser().resolve()
    prompts_path = args.prompts_jsonl.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    hps_checkpoint = args.hps_checkpoint.expanduser().resolve()
    clip_checkpoint = args.clip_checkpoint.expanduser().resolve()
    source_run = args.source_run.expanduser().resolve()
    additional_source_runs = [
        path.expanduser().resolve() for path in args.additional_source_run
    ]
    prior_paths = [path.expanduser().resolve() for path in args.prior_evaluation]

    samples = load_frozen_eval_samples(prompts_path)
    if any(sample.split != "es_train" for sample in samples):
        raise ValueError("Mage ES prompts file must contain only the es_train split")
    calibration = RewardCalibration.from_record(
        json.loads(calibration_path.read_text(encoding="utf-8"))
    )
    prompt_batch = episode_prompt_batch(
        samples,
        args.episode_index,
        batch_size=args.episode_size,
        schedule_seed=args.schedule_seed,
    )
    assets = {
        "prompts": {"path": str(prompts_path), "sha256": sha256_file(prompts_path)},
        "calibration": {
            "path": str(calibration_path),
            "sha256": sha256_file(calibration_path),
        },
        "hps_v2_1": {
            "path": str(hps_checkpoint),
            "sha256": sha256_file(hps_checkpoint),
        },
        "clip_h14": {
            "path": str(clip_checkpoint),
            "sha256": sha256_file(clip_checkpoint),
        },
    }
    model_identity = model_directory_identity(model_path)
    excluded_sample_ids = set()
    prior_evaluations = []
    for prior_path in prior_paths:
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if prior["assets"]["prompts"]["sha256"] != assets["prompts"]["sha256"]:
            raise ValueError("prior evaluation used a different prompt asset")
        excluded_sample_ids.update(prior["episode"]["sample_ids"])
        prior_evaluations.append(
            {"path": str(prior_path), "sha256": sha256_file(prior_path)}
        )
    source_identities = validate_source_runs(
        [source_run, *additional_source_runs],
        assets=assets,
        model_identity=model_identity,
        episode_index=args.episode_index,
        episode_sample_ids=[sample.sample_id for sample in prompt_batch],
        schedule_seed=args.schedule_seed,
        excluded_sample_ids=excluded_sample_ids,
    )
    source_identity = source_identities[0]
    source_identity["additional_source_runs"] = source_identities[1:]
    source_identity["prior_evaluations"] = prior_evaluations

    import torch
    from mage_flow import MageFlowPipeline

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("Mage center evaluation must run on a CUDA GB10")
    torch.backends.cuda.matmul.allow_tf32 = False
    started_at = time.time()
    pipeline = MageFlowPipeline.from_pretrained(str(model_path), "cuda")
    pipeline.model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in pipeline.model.parameters()):
        raise RuntimeError("Mage pipeline contains trainable parameters")
    transformer = pipeline.model.transformer
    layout = ClipGenomeLayout.from_model(transformer)
    master = FrozenLinearWeights.capture(transformer, layout)
    scorers = [
        HPSv21Scorer(hps_checkpoint, batch_size=args.reward_batch_size),
        OpenCLIPH14Scorer(clip_checkpoint, batch_size=args.reward_batch_size),
    ]

    records = []
    for index, (label, path) in enumerate(candidates):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        clips = payload["clips"]
        if clips.numel() != layout.dimension or not torch.isfinite(clips).all():
            raise ValueError(f"candidate {label} clip shape/value mismatch")
        result = evaluate_clips(
            torch=torch,
            clips=clips,
            transformer=transformer,
            master=master,
            layout=layout,
            pipeline=pipeline,
            prompt_batch=prompt_batch,
            scorers=scorers,
            calibration=calibration,
            args=args,
        )
        record = {
            "label": label,
            "path": str(path),
            "file_sha256": sha256_file(path),
            **result,
        }
        if args.audit_first_candidate and index == 0:
            replay = evaluate_clips(
                torch=torch,
                clips=clips,
                transformer=transformer,
                master=master,
                layout=layout,
                pipeline=pipeline,
                prompt_batch=prompt_batch,
                scorers=scorers,
                calibration=calibration,
                args=args,
            )
            record["replay_audit"] = replay_comparison(result, replay)
            if not record["replay_audit"]["exact"]:
                raise RuntimeError("first center replay audit was not exact")
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        gc.collect()

    report = {
        "schema_version": 1,
        "started_at_unix": started_at,
        "finished_at_unix": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "mage-flow": package_version("mage-flow"),
            "torchao": package_version("torchao"),
            "hpsv2": package_version("hpsv2"),
            "open_clip_torch": package_version("open_clip_torch"),
        },
        "assets": {
            "model": model_identity,
            **assets,
        },
        "source": source_identity,
        "implementation": {
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "es_quant_sha256": sha256_file(
                Path(__file__).resolve().parents[1] / "src/shadow_lab/es_quant.py"
            ),
            "image_eval_sha256": sha256_file(
                Path(__file__).resolve().parents[1] / "src/shadow_lab/image_eval.py"
            ),
            "image_reward_models_sha256": sha256_file(
                Path(__file__).resolve().parents[1]
                / "src/shadow_lab/image_reward_models.py"
            ),
            "nvfp4_runtime_sha256": sha256_file(
                Path(__file__).resolve().parents[1] / "src/shadow_lab/nvfp4_runtime.py"
            ),
            "quasar_sha256": sha256_file(
                Path(__file__).resolve().parents[1] / "src/shadow_lab/quasar.py"
            ),
        },
        "episode": {
            "index": args.episode_index,
            "size": args.episode_size,
            "schedule_seed": args.schedule_seed,
            "sample_ids": [sample.sample_id for sample in prompt_batch],
            "seeds": [sample.seed for sample in prompt_batch],
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "cfg": args.cfg,
        },
        "dimension": layout.dimension,
        "records": records,
        "comparisons": reward_deltas(records),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
