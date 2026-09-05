#!/usr/bin/env python3
"""EXP-008: isolated Mage QAD, one-shot gates, explicit human decisions, bounded ES."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from dataclasses import replace
from pathlib import Path

import torch

from shadow_lab.image_eval import (
    EVAL_SPLIT_COUNTS,
    RewardCalibration,
    load_frozen_eval_samples,
    validate_frozen_eval_pool,
)
from shadow_lab.mage_healed_es import run_healed_es
from shadow_lab.mage_healed_master import HealedMaster
from shadow_lab.mage_healing import HealingWorkflow, quality_gate
from shadow_lab.mage_healing_blind import build_blind_packet
from shadow_lab.mage_healing_runtime import MageHealingRuntime, require_gb10
from shadow_lab.mage_qad import MageTrajectorySource, prepare_mage_qad
from shadow_lab.mage_qad_campaign import MageQADConfig, atomic_json, run_mage_qad
from shadow_lab.mage_qad_checkpoint import content_hash, read_checkpoint
from shadow_lab.mage_qad_data import build_qad_pool, data_hash, validate_qad_manifest


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_identity(model_path: Path) -> dict:
    import mage_flow  # ty: ignore[unresolved-import]  # Installed in the GB10 Mage venv.

    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), *sorted((root / "src" / "shadow_lab").glob("*.py"))]
    mage_root = Path(mage_flow.__file__).resolve().parent
    mage_paths = [
        p for p in sorted(mage_root.rglob("*.py")) if ".venv" not in p.relative_to(mage_root).parts
    ]
    model_files = {
        str(p.relative_to(model_path)): file_hash(p)
        for p in sorted(model_path.rglob("*"))
        if p.is_file()
    }
    if not model_files:
        raise ValueError("model snapshot is empty")
    return {
        "model": content_hash(model_files),
        "model_files": model_files,
        "code": content_hash({str(p.relative_to(root)): file_hash(p) for p in paths}),
        "mage_source": content_hash(
            {str(p.relative_to(mage_root)): file_hash(p) for p in mage_paths}
        ),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchao", "diffusers", "transformers", "mage-flow")
        },
    }


def selected_campaign(directory: Path) -> tuple[dict, dict]:
    report = json.loads((directory / "campaign.json").read_text())
    if (
        report.get("status") != "selected"
        or report.get("selection_split") != "qad_val"
        or report.get("failures") != 0
    ):
        raise RuntimeError(
            "training must finish and select a healthy checkpoint using only qad_val"
        )
    progress = report["progress"]
    if len(progress["stages"]) != 4 or not all(s["passed"] for s in progress["stages"]):
        raise RuntimeError("all four QAD stages must pass")
    best = min(progress["history"], key=lambda row: row["qad_val_loss"])
    if (
        best["step"] != progress["best_step"]
        or report["selected_checkpoint"] != f"best-{best['step']}.pt"
    ):
        raise ValueError("selected checkpoint is not the minimum qad_val loss")
    saved = read_checkpoint(directory / report["selected_checkpoint"])
    if saved["identity"] != report["identity"]:
        raise ValueError("selected checkpoint identity differs from campaign")
    return report, saved


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("build-data")
    data.add_argument("--sources", required=True, type=Path)
    data.add_argument("--eval-pool", required=True, type=Path)
    data.add_argument("--out", required=True, type=Path)
    train = sub.add_parser("train")
    train.add_argument("--model-path", required=True, type=Path)
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--campaign", required=True, type=Path)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--stop-after", type=int)
    train.add_argument(
        "--checkpoint-interval",
        type=int,
        default=500,
        help="Periodic saves in cumulative optimizer steps; best/stage/stop also save",
    )
    for command in (
        "checkpoint-gate",
        "human-dev",
        "es-sigma",
        "es-confirmation",
        "final-gate",
        "human-test",
    ):
        action = sub.add_parser(command)
        action.add_argument("--campaign", required=True, type=Path)
        action.add_argument("--model-path", required=True, type=Path)
        action.add_argument("--eval-pool", required=True, type=Path)
        action.add_argument("--calibration", required=True, type=Path)
        action.add_argument("--hps-checkpoint", required=True, type=Path)
        action.add_argument("--clip-checkpoint", required=True, type=Path)
        if command.startswith("es-"):
            action.add_argument("--protocol", required=True, type=Path)
            action.add_argument("--batch-offset", required=True, type=int)
            action.add_argument("--batch-size", type=int, default=32)
    human = sub.add_parser("approve-human")
    human.add_argument("--campaign", required=True, type=Path)
    human.add_argument("--split", choices=("human_dev", "human_test"), required=True)
    human.add_argument("--ballots", required=True, type=Path)
    human.add_argument("--maintainer", required=True)
    human.add_argument("--decision", choices=("approve", "reject"), required=True)
    status = sub.add_parser("status")
    status.add_argument("--campaign", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build-data":
        # Data preparation is an experiment operation too; use the GB10 environment.
        require_gb10()
        samples = load_frozen_eval_samples(args.eval_pool)
        validate_frozen_eval_pool(samples)
        rows, manifest = build_qad_pool(read_jsonl(args.sources), read_jsonl(args.eval_pool))
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "qad.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
        atomic_json(args.out / "manifest.json", manifest)
        print(json.dumps(manifest))
        return
    if args.command == "train":
        require_gb10()
        rows = read_jsonl(args.data / "qad.jsonl")
        manifest = json.loads((args.data / "manifest.json").read_text())
        validate_qad_manifest(rows, manifest)
        config = MageQADConfig(**json.loads(args.config.read_text()))
        identity = runtime_identity(args.model_path)
        from mage_flow import MageFlowPipeline  # ty: ignore[unresolved-import]

        pipeline = MageFlowPipeline.from_pretrained(str(args.model_path), "cuda")
        pipeline.model.requires_grad_(False).eval()
        teacher = copy.deepcopy(pipeline.model.transformer).requires_grad_(False).eval()
        controller = prepare_mage_qad(pipeline.model.transformer)
        source = MageTrajectorySource(pipeline, height=config.height, width=config.width)
        report = run_mage_qad(
            controller,
            teacher,
            source,
            rows,
            manifest,
            identity,
            config,
            args.campaign,
            resume=args.resume,
            stop_after=args.stop_after,
            checkpoint_interval=args.checkpoint_interval,
        )
        print(json.dumps(report))
        if report["status"] == "failed":
            raise SystemExit(1)
        return
    if args.command == "status":
        campaign_report = json.loads((args.campaign / "campaign.json").read_text())
        if campaign_report.get("status") != "selected":
            print(json.dumps(campaign_report, indent=2))
            return
    report, checkpoint = selected_campaign(args.campaign)
    workflow = HealingWorkflow(
        args.campaign / "healing",
        selected_sha256=checkpoint["sha256"],
        evaluation_sha256=report["identity"]["data"]["evaluation_sha256"],
    )
    if args.command == "status":
        print(json.dumps(workflow.read(), indent=2))
        return
    if args.command == "approve-human":
        result = workflow.approve_human(
            args.split,
            json.loads(args.ballots.read_text()),
            maintainer=args.maintainer,
            approved=args.decision == "approve",
        )
        print(json.dumps(result))
        return
    require_gb10()
    identity = runtime_identity(args.model_path)
    if any(report["identity"][k] != v for k, v in identity.items()):
        raise ValueError("model/runtime implementation differs from QAD training")
    if data_hash(read_jsonl(args.eval_pool)) != workflow.identity["evaluation_sha256"]:
        raise ValueError("evaluation pool identity differs from QAD exclusions")
    samples = load_frozen_eval_samples(args.eval_pool)
    validate_frozen_eval_pool(samples)
    master = HealedMaster(checkpoint, selected_sha256=checkpoint["sha256"])
    calibration = RewardCalibration.from_record(json.loads(args.calibration.read_text()))
    # Freeze scorer/calibration identity at the first gate, then require exact reuse.
    scoring_identity = {
        "calibration": file_hash(args.calibration),
        "hps": file_hash(args.hps_checkpoint),
        "clip": file_hash(args.clip_checkpoint),
    }
    scoring_path = workflow.directory / "scorers.json"
    if scoring_path.exists():
        if json.loads(scoring_path.read_text()) != scoring_identity:
            raise ValueError("scorer or calibration identity changed")
    else:
        atomic_json(scoring_path, scoring_identity)
    from mage_flow import MageFlowPipeline  # ty: ignore[unresolved-import]

    from shadow_lab.image_reward_models import HPSv21Scorer, OpenCLIPH14Scorer

    pipeline = MageFlowPipeline.from_pretrained(str(args.model_path), "cuda")
    teacher = copy.deepcopy(pipeline.model.transformer).requires_grad_(False).eval()
    config = MageQADConfig(**report["identity"]["config"])
    runtime = MageHealingRuntime(
        pipeline,
        teacher,
        master,
        [HPSv21Scorer(args.hps_checkpoint), OpenCLIPH14Scorer(args.clip_checkpoint)],
        height=config.height,
        width=config.width,
    )

    def paired(samples, directory):
        bf16 = runtime.evaluate(
            master.initial_clips, samples, directory, candidate_sha256="bf16", arm="bf16"
        )
        if bf16["failures"] or len(bf16["rows"]) != len(samples):
            raise RuntimeError("BF16 paired reference generation failed")
        for row in bf16["rows"]:
            if row["image_sha256"] != row["replay_sha256"] or row["scores"] != row["replay_scores"]:
                raise RuntimeError("BF16 reference replay differs")
        return replace(
            calibration, sample_baselines={r["sample_id"]: r["scores"] for r in bf16["rows"]}
        )

    def confirmed():
        confirmation = workflow.read()["es"].get("confirmation", {})
        if not confirmation.get("passed"):
            raise RuntimeError("N=30 confirmation has not passed")
        artifact = torch.load(confirmation["clips_path"], map_location="cpu", weights_only=True)
        if (
            content_hash(artifact) != confirmation["clips_sha256"]
            or artifact["selected_sha256"] != master.selected_sha256
        ):
            raise ValueError("confirmed clips/master identity mismatch")
        return artifact["clips"], confirmation["candidate_sha256"]

    if args.command.startswith("es-"):
        workflow.es_permit()
        protocol = json.loads(args.protocol.read_text())
        phase = "sigma" if args.command == "es-sigma" else "confirmation"
        if protocol["phase"] != phase:
            raise ValueError("protocol phase differs from command")
        pool = [s for s in samples if s.split == "es_train"]
        if (
            args.batch_offset < 0
            or args.batch_size <= 0
            or args.batch_offset + args.batch_size > len(pool)
        ):
            raise ValueError("invalid ES batch bounds")
        batch = pool[args.batch_offset : args.batch_offset + args.batch_size]
        calibration = paired(batch, workflow.directory / f"{phase}-bf16")
        result = run_healed_es(
            workflow,
            master,
            protocol,
            batch,
            calibration,
            runtime.evaluate,
            workflow.directory / f"es-{phase}",
        )
    else:
        split = {
            "checkpoint-gate": "checkpoint_val",
            "human-dev": "human_dev",
            "final-gate": "auto_test",
            "human-test": "human_test",
        }[args.command]
        batch = [s for s in samples if s.split == split]
        if len(batch) != EVAL_SPLIT_COUNTS[split]:
            raise ValueError("incorrect frozen split size")
        clips, sha = (
            confirmed()
            if split in {"auto_test", "human_test"}
            else (master.initial_clips, master.selected_sha256)
        )
        directory = workflow.directory / split

        def evaluate():
            if split.startswith("human_"):
                arms = {}
                for arm in ("bf16", "rtn", "quasar-grid", "qad"):
                    result = runtime.evaluate(
                        clips, batch, directory / arm, candidate_sha256=sha, arm=arm
                    )
                    if result["failures"] or len(result["rows"]) != 30:
                        raise RuntimeError("blind image generation failed")
                    arms[arm] = {
                        row["sample_id"]: Path(row["image_path"]) for row in result["rows"]
                    }
                return build_blind_packet(
                    directory / "blind",
                    arms,
                    candidate_sha256=sha,
                    prompts={s.sample_id: s.prompt for s in batch},
                )
            paired_calibration = paired(batch, directory / "bf16")
            limits = comparison = None
            if split == "auto_test":
                healed = runtime.evaluate(
                    master.initial_clips,
                    batch,
                    directory / "healed",
                    candidate_sha256=master.selected_sha256,
                )
                healed_gate = quality_gate(
                    healed,
                    paired_calibration,
                    [s.sample_id for s in batch],
                    candidate_sha256=master.selected_sha256,
                )
                if not healed_gate["passed"]:
                    return dict(healed_gate, candidate_sha256=sha)
                limits = {r["sample_id"]: r["teacher_loss"] for r in healed["rows"]}
                comparison = {r["sample_id"]: r["scores"] for r in healed["rows"]}
            evaluation = runtime.evaluate(
                clips, batch, directory / "candidate", candidate_sha256=sha
            )
            return quality_gate(
                evaluation,
                paired_calibration,
                [s.sample_id for s in batch],
                candidate_sha256=sha,
                final=split == "auto_test",
                trajectory_limits=limits,
                comparison_scores=comparison,
            )

        result = workflow.once(split, sha, evaluate)
    print(json.dumps(result, indent=2))
    if not result.get("passed") and result.get("status") != "awaiting_human":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
