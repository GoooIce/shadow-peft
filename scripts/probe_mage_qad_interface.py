#!/usr/bin/env python3
"""Bounded GB10 four-position backward probe; no latent updates or checkpoint selection."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from shadow_lab.mage_healing_runtime import require_gb10
from shadow_lab.mage_qad import MageTrajectorySource, prepare_mage_qad, trajectory_loss
from shadow_lab.mage_qad_campaign import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require_gb10()
    from mage_flow import MageFlowPipeline  # ty: ignore[unresolved-import]

    sample = json.loads(args.sample.read_text())
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "loading",
        "started_at": started,
        "sample_id": sample["source_id"],
        "latent_updates": 0,
        "checkpoint_selection": False,
    }
    atomic_json(args.out / "probe.json", report)
    try:
        pipeline = MageFlowPipeline.from_pretrained(str(args.model_path), "cuda")
        pipeline.model.requires_grad_(False).eval()
        teacher = copy.deepcopy(pipeline.model.transformer).requires_grad_(False).eval()
        controller = prepare_mage_qad(pipeline.model.transformer)
        report.update(
            status="capture",
            linear_layers=len(controller.layers),
            load_seconds=time.time() - started,
        )
        atomic_json(args.out / "probe.json", report)
        trajectory = MageTrajectorySource(pipeline, height=512, width=512)(sample)
        report.update(status="backward", sigmas=trajectory.sigmas)
        atomic_json(args.out / "probe.json", report)
        backward_started = time.time()
        loss = trajectory_loss(pipeline.model.transformer, teacher, trajectory, backward=True)
        missing = [name for name, layer in controller.layers.items() if layer.weight.grad is None]
        nonfinite = [
            name
            for name, layer in controller.layers.items()
            if layer.weight.grad is not None and not torch.isfinite(layer.weight.grad).all().item()
        ]
        healthy = not missing and not nonfinite
        report.update(missing_gradients=missing, nonfinite_gradients=nonfinite)
        report.update(
            status="passed" if healthy else "failed",
            loss=loss,
            all_layer_gradients_finite=healthy,
            backward_seconds=time.time() - backward_started,
            peak_cuda_gib=torch.cuda.max_memory_allocated() / 2**30,
        )
        controller.close()
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    report["elapsed_seconds"] = time.time() - started
    atomic_json(args.out / "probe.json", report)
    print(json.dumps(report), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
