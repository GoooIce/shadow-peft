#!/usr/bin/env python3
"""Run one GB10 arm of the fixed-budget EXP-008 text-supervision pilot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch

from shadow_lab.mage_healing_runtime import require_gb10
from shadow_lab.mage_qad import MageTrajectorySource, prepare_mage_qad
from shadow_lab.mage_qad_campaign import atomic_json
from shadow_lab.mage_qad_text_pilot import inactive_text_layers, text_pilot_loss


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=("image", "auxiliary"), required=True)
    args = parser.parse_args()
    require_gb10()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = json.loads(args.data.read_text())
    assert len(rows["train"]) == len(rows["validation"]) == 8
    started = time.time()
    code_paths = [
        Path(__file__),
        *[
            Path(__file__).resolve().parents[1] / "src/shadow_lab" / name
            for name in ("mage_qad_text_pilot.py", "mage_qad.py", "quasar.py")
        ],
    ]
    report: dict = dict(
        status="running",
        arm=args.arm,
        data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(b"".join(p.read_bytes() for p in code_paths)).hexdigest(),
        protocol=dict(
            steps=8,
            validation_prompts=8,
            seed=88008,
            learning_rate=1e-5,
            weight_decay=0.01,
            adamw_foreach=False,
            height=512,
            width=512,
            text_initial_fraction=0.1,
            checkpoint_selection=False,
            required_relative_improvement=0.02,
            required_paired_wins=6,
        ),
        train=[],
        baseline=[],
        validation=[],
    )

    def publish(phase, **updates):
        report.update(
            phase=phase, updated_at=time.time(), elapsed_seconds=time.time() - started, **updates
        )
        report["peak_cuda_gib"] = torch.cuda.max_memory_allocated() / 2**30
        atomic_json(args.out / "report.json", report)
        print(
            json.dumps(
                dict(
                    phase=phase,
                    elapsed_seconds=report["elapsed_seconds"],
                    train_steps=len(report["train"]),
                    validation_done=len(report["validation"]),
                    **updates,
                )
            ),
            flush=True,
        )

    try:
        from mage_flow import MageFlowPipeline  # ty: ignore[unresolved-import]
        from mage_flow.pipeline import _decode_one  # ty: ignore[unresolved-import]

        torch.manual_seed(88008)
        publish("loading")
        pipeline = MageFlowPipeline.from_pretrained(str(args.model_path), "cuda")
        pipeline.model.requires_grad_(False).eval()
        student = pipeline.model.transformer
        report["initial_state_sha256"] = state_hash(student)
        teacher = copy.deepcopy(student).requires_grad_(False).eval()
        controller = prepare_mage_qad(student)
        inactive = inactive_text_layers(student)
        if not set(inactive).issubset(controller.layers):
            raise ValueError("inactive layer identity mismatch")
        if args.arm == "image":
            for name in inactive:
                controller.layers[name].weight.requires_grad_(False)
        report["frozen_layers"] = list(inactive) if args.arm == "image" else []
        report["trainable_layers"] = sum(p.requires_grad for p in controller.parameters())
        # Keep all layer identities in the optimizer binding. AdamW skips frozen
        # parameters with grad=None, so no decay, moments, or fake saliency arise.
        optimizer = torch.optim.AdamW(
            controller.parameters(), lr=1e-5, weight_decay=0.01, foreach=False
        )
        controller.bind_optimizer(optimizer)
        source = MageTrajectorySource(pipeline, height=512, width=512)

        def save_image(phase, index):
            def save(latent):
                with torch.no_grad():
                    image = _decode_one(pipeline.model, latent, 512, 512, latent.device)
                directory = args.out / "images" / phase
                directory.mkdir(parents=True, exist_ok=True)
                image.save(directory / f"{index:02d}.png")

            return save

        for index, row in enumerate(rows["validation"]):
            publish("baseline", sample=index)
            report["baseline"].append(
                text_pilot_loss(
                    student, teacher, source(row), terminal_callback=save_image("baseline", index)
                )
            )
        publish("calibration")
        first = source(rows["train"][0])
        calibration = text_pilot_loss(student, teacher, first)
        if min(calibration["image_mse"], calibration["text_mse"]) <= 0:
            raise ValueError("cannot calibrate auxiliary loss from zero initial error")
        calibrated = 0.1 * calibration["image_mse"] / calibration["text_mse"]
        coefficient = calibrated if args.arm == "auxiliary" else 0.0
        report.update(
            calibration=calibration, calibrated_coefficient=calibrated, used_coefficient=coefficient
        )
        for index, row in enumerate(rows["train"]):
            optimizer.zero_grad(set_to_none=True)
            publish("training", sample=index)
            step_started = time.time()
            trajectory = first if index == 0 else source(row)
            metrics = text_pilot_loss(
                student, teacher, trajectory, coefficient=coefficient, backward=True
            )
            missing, nonfinite = [], []
            for name, layer in controller.layers.items():
                grad = layer.weight.grad
                if layer.weight.requires_grad and grad is None:
                    missing.append(name)
                if grad is not None and not torch.isfinite(grad).all().item():
                    nonfinite.append(name)
                if not layer.weight.requires_grad and grad is not None:
                    raise RuntimeError("frozen parameter received a gradient")
            if missing or nonfinite:
                raise FloatingPointError(
                    f"gradient health: missing={missing}, nonfinite={nonfinite}"
                )
            publish("optimizer", sample=index)
            optimizer.step()
            if any(not torch.isfinite(p).all().item() for p in controller.parameters()):
                raise FloatingPointError("non-finite updated weight")
            if any(
                isinstance(v, torch.Tensor) and not torch.isfinite(v).all().item()
                for state in optimizer.state.values()
                for v in state.values()
            ):
                raise FloatingPointError("non-finite optimizer state")
            if controller.saliency_layer_count != report["trainable_layers"]:
                raise RuntimeError("saliency coverage differs from active parameters")
            report["train"].append(
                dict(step=index + 1, seconds=time.time() - step_started, **metrics)
            )
            publish("step_completed")
        optimizer.zero_grad(set_to_none=True)
        for index, row in enumerate(rows["validation"]):
            publish("validation", sample=index)
            report["validation"].append(
                text_pilot_loss(
                    student, teacher, source(row), terminal_callback=save_image("validation", index)
                )
            )
        if args.arm == "image":
            teacher_layers = dict(teacher.named_modules())
            if not all(
                torch.equal(controller.layers[n].weight, teacher_layers[n].weight) for n in inactive
            ):
                raise RuntimeError("frozen text weights changed")
        publish("saving")
        torch.save(
            dict(
                latent={n: layer.weight.detach() for n, layer in controller.layers.items()},
                saliency={n: layer._saliency for n, layer in controller.layers.items()},
                report=report,
            ),
            args.out / "final.pt",
        )
        controller.close()
        publish("completed", status="completed")
    except Exception as error:
        publish("failed", status="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
