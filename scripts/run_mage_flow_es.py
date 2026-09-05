#!/usr/bin/env python3
"""Run replayable Phase-1 ES over Mage MMDiT group-128 NVFP4 clips."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import resource
import socket
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from shadow_lab.es import (
    CandidateObservation,
    CandidateSpec,
    ESConfig,
    candidate_clips,
    candidate_specs,
    initialization_projection_report,
    initialize_es,
    latent_to_clips,
    load_es_checkpoint,
    save_es_checkpoint,
    update_es,
)
from shadow_lab.es_quant import (
    ClipGenomeLayout,
    FrozenLinearWeights,
    quasar_initial_clips,
)
from shadow_lab.image_eval import (
    RewardCalibration,
    episode_prompt_batch,
    load_frozen_eval_samples,
)
from shadow_lab.image_reward_models import HPSv21Scorer, OpenCLIPH14Scorer
from shadow_lab.mage_es_eval import evaluate_clips as evaluate_mage_es_clips


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--prompts-jsonl", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--hps-checkpoint", required=True, type=Path)
    parser.add_argument("--clip-checkpoint", required=True, type=Path)
    parser.add_argument("--initial-clips", type=Path)
    parser.add_argument("--initialization-report", type=Path)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--episode-size", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--learning-rate", type=float, default=0.015)
    parser.add_argument("--shaping", choices=("zscore", "rank"), default="zscore")
    parser.add_argument("--es-seed", type=int, default=20260904)
    parser.add_argument("--schedule-seed", type=int, default=20260904)
    parser.add_argument(
        "--episode-offset",
        type=int,
        default=0,
        help="Prompt-schedule episode used by ES generation zero.",
    )
    parser.add_argument("--transform-temperature", type=float, default=0.10)
    parser.add_argument("--boundary-margin", type=float, default=0.001)
    parser.add_argument("--reward-batch-size", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--audit-first-candidate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_episode_index(generation: int, *, offset: int) -> int:
    if generation < 0 or offset < 0:
        raise ValueError("generation and episode offset must be non-negative")
    return offset + generation


def atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_torch_save(torch, value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def tensor_sha256(value) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def model_directory_identity(path: Path) -> dict:
    """Fingerprint small metadata and the size/layout of all model files."""

    if not path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {path}")
    files = []
    for item in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = item.relative_to(path).as_posix()
        record = {"path": relative, "size": item.stat().st_size}
        if item.suffix.lower() == ".json":
            record["sha256"] = file_sha256(item)
        files.append(record)
    if not files:
        raise ValueError(f"model directory is empty: {path}")
    encoded = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "path": str(path),
        "file_count": len(files),
        "metadata_and_size_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
    }


def source_tree_identity(path: Path) -> dict:
    records = []
    for item in sorted(item for item in path.rglob("*.py") if item.is_file()):
        relative = item.relative_to(path)
        if any(part in {".venv", "__pycache__"} for part in relative.parts):
            continue
        records.append(
            {
                "path": relative.as_posix(),
                "sha256": file_sha256(item),
            }
        )
    if not records:
        raise ValueError(f"source tree has no Python files: {path}")
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return {
        "file_count": len(records),
        "python_source_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
    }


def package_source_identity(name: str) -> dict:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"cannot locate runtime package source: {name}")
    return source_tree_identity(Path(spec.origin).resolve().parent)


def implementation_identity() -> dict:
    import torch

    repo_root = Path(__file__).resolve().parents[1]
    relative_paths = (
        "scripts/run_mage_flow_es.py",
        "src/shadow_lab/es.py",
        "src/shadow_lab/es_quant.py",
        "src/shadow_lab/image_eval.py",
        "src/shadow_lab/image_reward_models.py",
        "src/shadow_lab/mage_es_eval.py",
        "src/shadow_lab/nvfp4_runtime.py",
        "src/shadow_lab/quasar.py",
    )
    project_files = {
        relative: file_sha256(repo_root / relative) for relative in relative_paths
    }
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "packages": {
            name: package_version(name)
            for name in ("torchao", "hpsv2", "open_clip_torch", "mage-flow")
        },
        "project_files": project_files,
        "runtime_source": {
            name: package_source_identity(name)
            for name in ("mage_flow", "hpsv2", "open_clip")
        },
    }


def experiment_identity(
    args,
    *,
    model_path: Path,
    prompts_path: Path,
    calibration_path: Path,
    hps_checkpoint: Path,
    clip_checkpoint: Path,
    initial_clips_path: Path | None,
    initialization_report_path: Path | None,
) -> dict:
    identity = {
        "schema_version": 1,
        "implementation": implementation_identity(),
        "model": model_directory_identity(model_path),
        "assets": {
            "prompts": {"path": str(prompts_path), "sha256": file_sha256(prompts_path)},
            "calibration": {
                "path": str(calibration_path),
                "sha256": file_sha256(calibration_path),
            },
            "hps_v2_1": {
                "path": str(hps_checkpoint),
                "sha256": file_sha256(hps_checkpoint),
            },
            "clip_h14": {
                "path": str(clip_checkpoint),
                "sha256": file_sha256(clip_checkpoint),
            },
        },
        "search": {
            "generations": args.generations,
            "population_size": args.population_size,
            "episode_size": args.episode_size,
            "sigma": args.sigma,
            "learning_rate": args.learning_rate,
            "shaping": args.shaping,
            "es_seed": args.es_seed,
            "schedule_seed": args.schedule_seed,
            "episode_offset": args.episode_offset,
            "transform_temperature": args.transform_temperature,
            "boundary_margin": args.boundary_margin,
            "audit_first_candidate": args.audit_first_candidate,
        },
        "generation": {
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "cfg": args.cfg,
        },
        "reward_batch_size": args.reward_batch_size,
    }
    if initial_clips_path is not None and initialization_report_path is not None:
        identity["initialization_source"] = {
            "clips": {
                "path": str(initial_clips_path),
                "sha256": file_sha256(initial_clips_path),
            },
            "report": {
                "path": str(initialization_report_path),
                "sha256": file_sha256(initialization_report_path),
            },
        }
    return identity


def evaluate_candidate(
    *,
    torch,
    state,
    spec,
    transformer,
    master,
    layout,
    pipeline,
    prompt_batch,
    scorers,
    calibration,
    args,
    center_clips,
) -> tuple[float, dict]:
    clips = candidate_clips(state, spec)
    record = evaluate_mage_es_clips(
        torch_runtime=torch,
        clips=clips,
        transformer=transformer,
        master=master,
        layout=layout,
        pipeline=pipeline,
        prompt_batch=prompt_batch,
        scorers=scorers,
        calibration=calibration,
        height=args.height,
        width=args.width,
        steps=args.steps,
        cfg=args.cfg,
    )
    reward = float(record.pop("reward"))
    record["clip_statistics"]["perturbation_rms"] = float(
        torch.sqrt(torch.mean((clips - center_clips).square()))
    )
    return reward, record


def replay_audit(first_reward: float, first: dict, second_reward: float, second: dict) -> dict:
    compared_fields = (
        "clip_sha256",
        "clip_statistics",
        "weighted_reconstruction_error",
        "quantized_layers",
        "clip_group_count",
        "trainable_parameters_after",
        "axis_means",
        "image_sha256",
    )
    matches = {
        "reward": first_reward == second_reward,
        **{field: first[field] == second[field] for field in compared_fields},
    }
    return {"exact": all(matches.values()), "matches": matches}


def load_observation_prefix(path: Path, expected: tuple[CandidateSpec, ...]):
    if not path.is_file():
        return [], []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) > len(expected):
        raise ValueError(f"observation log has too many rows: {path}")
    observations = []
    for index, row in enumerate(rows):
        spec = CandidateSpec(**row["spec"])
        if spec != expected[index]:
            raise ValueError(f"observation {index} does not match replayed candidate spec")
        observations.append(
            CandidateObservation(
                spec,
                float(row["reward"]) if row.get("reward") is not None else None,
                row.get("failure_reason"),
            )
        )
    return rows, observations


def layout_record(layout: ClipGenomeLayout) -> dict:
    return {
        "dimension": layout.dimension,
        "layers": [
            {
                "name": layer.name,
                "shape": list(layer.shape),
                "start": layer.start,
                "stop": layer.stop,
            }
            for layer in layout.layers
        ],
    }


def append_jsonl(path: Path, record: dict) -> None:
    prefix = path.read_text(encoding="utf-8") if path.is_file() else ""
    atomic_write_text(
        path,
        prefix + json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n",
    )


def generation_summary_path(out_dir: Path, generation: int) -> Path:
    return out_dir / f"generation-{generation:04d}-summary.json"


def validate_committed_generations(
    out_dir: Path, *, completed: int, population_size: int
) -> None:
    for generation in range(completed):
        summary_path = generation_summary_path(out_dir, generation)
        if not summary_path.is_file():
            raise ValueError(
                f"checkpoint generation {completed} is missing committed {summary_path.name}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("generation") != generation:
            raise ValueError(f"invalid committed generation summary: {summary_path}")
        observations = out_dir / f"generation-{generation:04d}-candidates.jsonl"
        row_count = sum(
            bool(line) for line in observations.read_text(encoding="utf-8").splitlines()
        )
        if row_count != population_size:
            raise ValueError(
                f"committed generation {generation} has {row_count} candidate rows"
            )


def main() -> None:
    args = parse_args()
    if args.generations <= 0 or args.episode_size <= 0 or args.reward_batch_size <= 0:
        raise ValueError("generation, episode and reward batch sizes must be positive")
    prompt_episode_index(0, offset=args.episode_offset)
    if args.height % 16 or args.width % 16:
        raise ValueError("Mage image dimensions must be multiples of 16")
    model_path = args.model_path.expanduser().resolve()
    prompts_path = args.prompts_jsonl.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    hps_checkpoint = args.hps_checkpoint.expanduser().resolve()
    clip_checkpoint = args.clip_checkpoint.expanduser().resolve()
    if (args.initial_clips is None) != (args.initialization_report is None):
        raise ValueError("--initial-clips and --initialization-report must be provided together")
    initial_clips_source = (
        args.initial_clips.expanduser().resolve() if args.initial_clips is not None else None
    )
    initialization_report_source = (
        args.initialization_report.expanduser().resolve()
        if args.initialization_report is not None
        else None
    )
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing non-empty ES output directory: {out_dir}")
    checkpoint_path = out_dir / "es-state.pt"
    experiment_path = out_dir / "experiment.json"
    run_path = out_dir / "run.json"
    if args.resume and run_path.is_file():
        raise FileExistsError(f"ES run is already complete: {run_path}")

    identity = experiment_identity(
        args,
        model_path=model_path,
        prompts_path=prompts_path,
        calibration_path=calibration_path,
        hps_checkpoint=hps_checkpoint,
        clip_checkpoint=clip_checkpoint,
        initial_clips_path=initial_clips_source,
        initialization_report_path=initialization_report_source,
    )
    if args.resume:
        if not experiment_path.is_file():
            raise FileNotFoundError(f"resume requires {experiment_path}")
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        if experiment.get("identity") != identity:
            raise ValueError("resume experiment identity differs from the original run")
        started_at = float(experiment["started_at_unix"])
    else:
        started_at = time.time()
        experiment = {
            "schema_version": 1,
            "started_at_unix": started_at,
            "identity": identity,
        }
        atomic_write_json(experiment_path, experiment)

    samples = load_frozen_eval_samples(prompts_path)
    if any(sample.split != "es_train" for sample in samples):
        raise ValueError("Mage ES prompts file must contain only the es_train split")
    if len(samples) % args.episode_size:
        raise ValueError("es_train sample count must be divisible by episode size")
    calibration = RewardCalibration.from_record(
        json.loads(calibration_path.read_text(encoding="utf-8"))
    )

    import torch
    from mage_flow import MageFlowPipeline

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("Mage ES must run on a CUDA GB10 with compute capability 12.1")
    torch.backends.cuda.matmul.allow_tf32 = False
    invocation_started_at = time.time()
    load_started = time.perf_counter()
    pipeline = MageFlowPipeline.from_pretrained(str(model_path), "cuda")
    pipeline.model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in pipeline.model.parameters()):
        raise RuntimeError("Mage pipeline contains trainable parameters")
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started

    transformer = pipeline.model.transformer
    layout = ClipGenomeLayout.from_model(transformer)
    config = ESConfig(
        dimension=layout.dimension,
        population_size=args.population_size,
        sigma=args.sigma,
        learning_rate=args.learning_rate,
        shaping=args.shaping,
        seed=args.es_seed,
        transform_temperature=args.transform_temperature,
        boundary_margin=args.boundary_margin,
    )
    expected_layout = layout_record(layout)
    layout_path = out_dir / "layout.json"
    initial_path = out_dir / "initial-clips.pt"
    initial_report_path = out_dir / "initialization.json"

    master_started = time.perf_counter()
    master = FrozenLinearWeights.capture(transformer, layout)
    master_seconds = time.perf_counter() - master_started
    if args.resume and checkpoint_path.is_file():
        if json.loads(layout_path.read_text(encoding="utf-8")) != expected_layout:
            raise ValueError("resume layout differs from the loaded Mage transformer")
        state = load_es_checkpoint(checkpoint_path)
        if state.config != config:
            raise ValueError("resume ES config differs from command-line configuration")
        initial_payload = torch.load(initial_path, map_location="cpu", weights_only=True)
        initial_clips = initial_payload["clips"]
        initial_report = json.loads(initial_report_path.read_text(encoding="utf-8"))
    else:
        atomic_write_json(layout_path, expected_layout)
        initialization_started = time.perf_counter()
        if initial_clips_source is None:
            initial_clips, quasar_report = quasar_initial_clips(transformer, layout)
            source_record = None
        else:
            initial_clips = torch.load(
                initial_clips_source,
                map_location="cpu",
                weights_only=True,
            )["clips"]
            source_report = json.loads(
                initialization_report_source.read_text(encoding="utf-8")
            )
            if tensor_sha256(initial_clips) != source_report.get("clip_sha256"):
                raise ValueError("shared initial clip tensor differs from its report")
            quasar_report = source_report["quasar"]
            source_record = identity["initialization_source"]
        initialization_seconds = time.perf_counter() - initialization_started
        projection = initialization_projection_report(initial_clips, config)
        state = initialize_es(initial_clips, config)
        initial_report = {
            "quasar": quasar_report,
            "projection": projection,
            "seconds": initialization_seconds,
            "clip_sha256": tensor_sha256(initial_clips),
            "effective_clip_sha256": tensor_sha256(
                latent_to_clips(state.center, state.config)
            ),
            "source": source_record,
        }
        atomic_torch_save(torch, {"clips": initial_clips}, initial_path)
        atomic_write_json(initial_report_path, initial_report)
        save_es_checkpoint(state, checkpoint_path)
    if initial_clips.numel() != layout.dimension:
        raise ValueError("saved QUASAR initialization dimension differs from layout")
    validate_committed_generations(
        out_dir,
        completed=state.generation,
        population_size=config.population_size,
    )

    reward_load_started = time.perf_counter()
    scorers = [
        HPSv21Scorer(hps_checkpoint, batch_size=args.reward_batch_size),
        OpenCLIPH14Scorer(clip_checkpoint, batch_size=args.reward_batch_size),
    ]
    torch.cuda.synchronize()
    reward_load_seconds = time.perf_counter() - reward_load_started

    while state.generation < args.generations:
        generation = state.generation
        episode_index = prompt_episode_index(generation, offset=args.episode_offset)
        center_clips = latent_to_clips(state.center, state.config)
        prompt_batch = episode_prompt_batch(
            samples,
            episode_index,
            batch_size=args.episode_size,
            schedule_seed=args.schedule_seed,
        )
        specs = candidate_specs(state)
        observation_path = out_dir / f"generation-{generation:04d}-candidates.jsonl"
        _, observations = load_observation_prefix(observation_path, specs)
        for spec in specs[len(observations) :]:
            candidate_started = time.perf_counter()
            reward = None
            failure_reason = None
            record = {"spec": asdict(spec)}
            try:
                reward, metrics = evaluate_candidate(
                    torch=torch,
                    state=state,
                    spec=spec,
                    transformer=transformer,
                    master=master,
                    layout=layout,
                    pipeline=pipeline,
                    prompt_batch=prompt_batch,
                    scorers=scorers,
                    calibration=calibration,
                    args=args,
                    center_clips=center_clips,
                )
                if (
                    args.audit_first_candidate
                    and generation == 0
                    and spec.population_index == 0
                ):
                    replay_reward, replay_metrics = evaluate_candidate(
                        torch=torch,
                        state=state,
                        spec=spec,
                        transformer=transformer,
                        master=master,
                        layout=layout,
                        pipeline=pipeline,
                        prompt_batch=prompt_batch,
                        scorers=scorers,
                        calibration=calibration,
                        args=args,
                        center_clips=center_clips,
                    )
                    audit = replay_audit(
                        reward,
                        metrics,
                        replay_reward,
                        replay_metrics,
                    )
                    if not audit["exact"]:
                        raise RuntimeError(f"candidate replay audit differed: {audit}")
                    metrics["replay_audit"] = audit
                record.update({"reward": reward, "failure_reason": None, **metrics})
            except Exception as error:
                reward = None
                failure_reason = f"{type(error).__name__}: {error}"
                record.update({"reward": None, "failure_reason": failure_reason})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            record["elapsed_seconds"] = time.perf_counter() - candidate_started
            append_jsonl(observation_path, record)
            observations.append(CandidateObservation(spec, reward, failure_reason))
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if "metrics" in locals():
                del metrics
            gc.collect()

        update = update_es(state, observations)
        state = update.state
        next_center_clips = latent_to_clips(state.center, state.config)
        summary = {
            "generation": generation,
            "prompt_episode_index": episode_index,
            "prompt_sample_ids": [sample.sample_id for sample in prompt_batch],
            "prompt_seeds": [sample.seed for sample in prompt_batch],
            "population_size": len(observations),
            "failure_count": update.failure_count,
            "reward_mean": update.reward_mean,
            "reward_std": update.reward_std,
            "reward_max": max(update.raw_rewards),
            "gradient_norm": update.gradient_norm,
            "gradient_rms": update.gradient_rms,
            "latent_step_rms": update.latent_step_rms,
            "clip_step_rms": float(
                torch.sqrt(torch.mean((next_center_clips - center_clips).square()))
            ),
            "center_clip_sha256": tensor_sha256(
                next_center_clips
            ),
        }
        summary_path = generation_summary_path(out_dir, generation)
        if summary_path.is_file():
            committed = json.loads(summary_path.read_text(encoding="utf-8"))
            if committed != summary:
                raise ValueError(
                    f"recomputed generation differs from committed {summary_path.name}"
                )
        else:
            atomic_write_json(summary_path, summary)
        save_es_checkpoint(state, checkpoint_path)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        del center_clips, next_center_clips

    summaries = [
        json.loads(generation_summary_path(out_dir, generation).read_text(encoding="utf-8"))
        for generation in range(state.generation)
    ]
    atomic_write_text(
        out_dir / "generations.jsonl",
        "".join(
            json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n"
            for summary in summaries
        ),
    )

    run = {
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
        "model_path": str(model_path),
        "prompts": {"path": str(prompts_path), "sha256": file_sha256(prompts_path)},
        "calibration": {
            "path": str(calibration_path),
            "sha256": file_sha256(calibration_path),
        },
        "es_config": asdict(config),
        "generation": {
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "cfg": args.cfg,
            "episode_size": args.episode_size,
            "schedule_seed": args.schedule_seed,
            "episode_offset": args.episode_offset,
        },
        "layout": {"path": layout_path.name, "sha256": file_sha256(layout_path)},
        "experiment": {
            "path": experiment_path.name,
            "sha256": file_sha256(experiment_path),
        },
        "initialization": {
            "path": initial_report_path.name,
            "sha256": file_sha256(initial_report_path),
            **initial_report,
        },
        "timing_seconds": {
            "model_load": model_load_seconds,
            "cpu_master_capture": master_seconds,
            "reward_model_load": reward_load_seconds,
            "invocation": time.time() - invocation_started_at,
            "total_since_first_start": time.time() - started_at,
        },
        "cpu_master_parameters": master.parameter_count,
        "completed_generations": state.generation,
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": file_sha256(checkpoint_path),
        },
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    atomic_write_json(run_path, run)
    print(json.dumps({"run": str(run_path), "generations": state.generation}))


if __name__ == "__main__":
    main()
