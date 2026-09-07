"""Staged QAD runner. This module accepts only training/trajectory-validation data."""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

import torch

from .mage_qad import trajectory_loss, trajectory_objective
from .mage_qad_checkpoint import link_checkpoint, load_checkpoint, save_checkpoint
from .mage_qad_data import validate_qad_manifest


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".report-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@dataclass(frozen=True)
class MageQADConfig:
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    formal_steps: int = 1024
    validation_interval: int = 16
    seed: int = 8008
    height: int = 512
    width: int = 512
    text_initial_fraction: float = 0.0
    adamw_foreach: bool | None = None

    def __post_init__(self):
        if min(self.height, self.width) <= 0 or self.height % 16 or self.width % 16:
            raise ValueError("Mage dimensions must be positive multiples of 16")
        if not 1 <= self.formal_steps <= 1024:
            raise ValueError("formal training budget must be 1..1024")
        if not 1 <= self.validation_interval <= 128 or 128 % self.validation_interval:
            raise ValueError("validation interval must divide 128")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight decay must be nonnegative and finite")
        if (
            not math.isfinite(self.text_initial_fraction)
            or not 0 <= self.text_initial_fraction <= 1
        ):
            raise ValueError("initial text loss fraction must be finite and within 0..1")


class QADProgress(TypedDict):
    stage_index: int
    stage_step: int
    step: int
    best_loss: float | None
    best_step: int
    best_checkpoint: str
    stages: list[dict]
    history: list[dict]
    stage_initial_loss: float | None
    stage_best_step: int
    text_supervision: dict | None


def run_mage_qad(
    controller,
    teacher,
    source,
    rows: list[dict],
    manifest: dict,
    implementation: dict,
    config: MageQADConfig,
    out: Path,
    *,
    resume: bool = False,
    stop_after: int | None = None,
    checkpoint_interval: int = 500,
) -> dict:
    """Run smoke -> 32 preflight -> 256 pilot -> <=1024 formal steps.

    Periodic, best, stage-end and voluntary-stop checkpoints are resumable.
    Failed stages are terminal. A voluntary stop returns a paused report;
    it cannot be used to unlock evaluation gates.
    Checkpoint choice depends only on finite state and minimum qad_val MSE.
    """
    if type(checkpoint_interval) is not int or checkpoint_interval < 1:
        raise ValueError("checkpoint interval must be a positive integer")
    validate_qad_manifest(rows, manifest)
    if not all(implementation.get(k) for k in ("model", "code")):
        raise ValueError("model and implementation identities are required")
    identity = dict(implementation, data=manifest, config=asdict(config))
    # JSON normalization makes the on-disk report and torch checkpoint agree.
    identity = json.loads(json.dumps(identity))
    report_path = out / "campaign.json"
    latest_path = out / "latest.pt"
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError("QAD output directory is not empty")
    out.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        foreach=config.adamw_foreach,
    )
    controller.bind_optimizer(optimizer)
    progress: QADProgress
    if resume:
        report = json.loads(report_path.read_text())
        if report["status"] in {"failed", "selected"}:
            raise RuntimeError("QAD campaign is terminal")
        saved = load_checkpoint(latest_path, controller, optimizer, identity)
        progress = saved["progress"]
        # Model/optimizer/RNG are restored. Do not retain the full CPU artifact
        # (including deployment weights) throughout training and the next save.
        del saved
    else:
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        progress = {
            "stage_index": 0,
            "stage_step": 0,
            "step": 0,
            "best_loss": None,
            "best_step": 0,
            "best_checkpoint": "",
            "stages": [],
            "history": [],
            "stage_initial_loss": None,
            "stage_best_step": 0,
            "text_supervision": None,
        }
    train = [r for r in rows if r["split"] == "qad_train"]
    validation = [r for r in rows if r["split"] == "qad_val"]
    stages = (("smoke", 1), ("preflight", 32), ("pilot", 256), ("formal", config.formal_steps))
    controller.model.eval()
    teacher.requires_grad_(False).eval()
    invocation_steps = 0
    report = {
        "status": "running",
        "identity": identity,
        "stage": stages[min(progress["stage_index"], 3)][0],
        "checkpoint_interval": checkpoint_interval,
        "last_checkpoint_step": progress["step"],
    }
    atomic_json(report_path, report)
    invocation_started = time.time()

    def publish(phase: str, **details):
        live = {
            "updated_at": time.time(),
            "elapsed_seconds": time.time() - invocation_started,
            "stage": report["stage"],
            "phase": phase,
            "completed_optimizer_steps": progress["step"],
            **details,
        }
        if torch.cuda.is_available():
            live["cuda_allocated_gib"] = torch.cuda.memory_allocated() / 2**30
            live["cuda_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        atomic_json(out / "live.json", live)

    try:
        if config.text_initial_fraction:
            if not resume:
                publish("calibrating_text_supervision", sample_id=train[0]["sample_id"])
                with torch.random.fork_rng():
                    python_rng = random.getstate()
                    try:
                        calibration = trajectory_objective(
                            controller.model, teacher, source(train[0])
                        )
                    finally:
                        random.setstate(python_rng)
                image_mse, text_mse = calibration["image_mse"], calibration["text_mse"]
                if min(image_mse, text_mse) <= 0:
                    raise ValueError(
                        "text supervision calibration requires positive initial errors"
                    )
                progress["text_supervision"] = dict(
                    coefficient=config.text_initial_fraction * image_mse / text_mse,
                    initial_fraction=config.text_initial_fraction,
                    calibration_sample_id=train[0]["sample_id"],
                    calibration_split="qad_train",
                    image_mse=image_mse,
                    text_mse=text_mse,
                )
            supervision = progress.get("text_supervision")
            if (
                not supervision
                or supervision["calibration_sample_id"] != train[0]["sample_id"]
                or supervision["initial_fraction"] != config.text_initial_fraction
                or not math.isfinite(supervision["coefficient"])
                or supervision["coefficient"] <= 0
            ):
                raise ValueError("missing or invalid frozen text supervision coefficient")
            report["text_supervision"] = supervision
            atomic_json(out / "text-supervision.json", supervision)
            atomic_json(report_path, report)
        while progress["stage_index"] < len(stages):
            stage, budget = stages[progress["stage_index"]]
            report["stage"] = stage
            optimizer.zero_grad(set_to_none=True)
            publish("preparing_trajectory", next_step=progress["step"] + 1)
            batch = source(train[progress["step"] % len(train)])
            publish("forward_backward", next_step=progress["step"] + 1)
            if config.text_initial_fraction:
                assert progress["text_supervision"] is not None
                metrics = trajectory_objective(
                    controller.model,
                    teacher,
                    batch,
                    coefficient=progress["text_supervision"]["coefficient"],
                    backward=True,
                )
                training_loss = metrics["objective"]
            else:
                training_loss = trajectory_loss(controller.model, teacher, batch, backward=True)
                metrics = dict(objective=training_loss, image_mse=training_loss, text_mse=None)
            parameters = controller.parameters()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError("missing or non-finite all-layer gradient")
            publish("optimizer_step")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in parameters):
                raise FloatingPointError("non-finite latent master")
            for state in optimizer.state.values():
                if any(
                    isinstance(v, torch.Tensor) and not torch.isfinite(v).all()
                    for v in state.values()
                ):
                    raise FloatingPointError("non-finite optimizer state")
            if controller.saliency_layer_count != len(controller.layers):
                raise RuntimeError("incomplete AdamW saliency coverage")
            progress["step"] += 1
            progress["stage_step"] += 1
            invocation_steps += 1
            atomic_json(
                out / "last-step.json",
                dict(
                    step=progress["step"],
                    stage=stage,
                    metrics=metrics,
                    gradient_layers=len(parameters),
                    saliency_layers=controller.saliency_layer_count,
                    finite_state=True,
                    updated_at=time.time(),
                ),
            )
            at_end = progress["stage_step"] == budget
            early_stop = False
            improved = False
            if at_end or progress["stage_step"] % config.validation_interval == 0:
                # Validation must not perturb resumable training RNG state.
                with torch.random.fork_rng():
                    python_rng = random.getstate()
                    try:
                        losses = []
                        publish("qad_val", completed=0, total=len(validation))
                        for index, row in enumerate(validation):
                            losses.append(trajectory_loss(controller.model, teacher, source(row)))
                            if (index + 1) % 8 == 0 or index + 1 == len(validation):
                                publish(
                                    "qad_val",
                                    completed=index + 1,
                                    total=len(validation),
                                    mean_loss_so_far=statistics.fmean(losses),
                                )
                        val_loss = statistics.fmean(losses)
                    finally:
                        random.setstate(python_rng)
                if not math.isfinite(val_loss):
                    raise FloatingPointError("non-finite qad_val loss")
                if progress["stage_initial_loss"] is None:
                    progress["stage_initial_loss"] = val_loss
                    progress["stage_best_step"] = progress["step"]
                best_loss = progress["best_loss"]
                improved = best_loss is None or val_loss < best_loss
                if improved:
                    progress.update(
                        best_loss=val_loss,
                        best_step=progress["step"],
                        stage_best_step=progress["step"],
                        best_checkpoint=f"best-{progress['step']}.pt",
                    )
                progress["history"].append(
                    {
                        "step": progress["step"],
                        "stage": stage,
                        "qad_val_loss": val_loss,
                        "training_loss": training_loss,
                        "training_metrics": metrics,
                    }
                )
                early_stop = (
                    stage == "formal" and progress["step"] - progress["stage_best_step"] >= 128
                )
                if at_end or early_stop:
                    initial_loss = progress["stage_initial_loss"]
                    passed = stage == "smoke" or (
                        initial_loss is not None and val_loss <= initial_loss
                    )
                    stage_report = {
                        "stage": stage,
                        "steps": progress["stage_step"],
                        "passed": passed,
                        "qad_val_loss": val_loss,
                        "early_stop": early_stop,
                        "failures": 0,
                        "finite_state": True,
                    }
                    atomic_json(out / stage / "report.json", stage_report)
                    if not passed:
                        raise RuntimeError(f"{stage} validation worsened; next stage prohibited")
                    progress["stages"].append(stage_report)
                    progress["stage_index"] += 1
                    progress["stage_step"] = 0
                    progress["stage_initial_loss"] = val_loss
                    progress["stage_best_step"] = progress["step"]
            pausing = stop_after is not None and invocation_steps >= stop_after
            if (
                improved
                or at_end
                or early_stop
                or pausing
                or progress["step"] % checkpoint_interval == 0
            ):
                publish("saving_checkpoint", training_loss=training_loss)
                if improved:
                    best_path = out / progress["best_checkpoint"]
                    save_checkpoint(best_path, controller, optimizer, identity, progress)
                    link_checkpoint(best_path, latest_path)
                else:
                    save_checkpoint(latest_path, controller, optimizer, identity, progress)
                report["last_checkpoint_step"] = progress["step"]
                # Old best files are obsolete only after latest commits the new selection.
                for old in out.glob("best-*.pt"):
                    if old.name != progress["best_checkpoint"]:
                        old.unlink()
            report.update(progress=progress)
            atomic_json(report_path, report)
            if pausing and progress["stage_index"] < 4:
                report["status"] = "paused"
                atomic_json(report_path, report)
                return report
        report.update(
            status="selected",
            selected_checkpoint=progress["best_checkpoint"],
            selection_split="qad_val",
            failures=0,
            es_authorized=False,
        )
    except Exception as error:
        report.update(
            status="failed",
            failures=1,
            error=f"{type(error).__name__}: {error}",
            es_authorized=False,
            progress=progress,
        )
    finally:
        controller.close()
    atomic_json(report_path, report)
    return report
