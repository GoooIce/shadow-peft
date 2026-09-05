"""Mage online teacher distillation, independent of image reward and held-out gates."""

from __future__ import annotations

import math
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .nvfp4_runtime import census_linear_layers
from .quasar import QuasarNVFP4Config, QuasarNVFP4Controller, prepare_quasar_nvfp4


@dataclass(frozen=True)
class MageTrajectory:
    latent: torch.Tensor
    conditioning: dict[str, Any]
    sigmas: tuple[float, ...]

    def __post_init__(self):
        if (
            len(self.sigmas) != 5
            or self.sigmas[-1] != 0
            or not all(math.isfinite(x) for x in self.sigmas)
            or not all(a > b for a, b in zip(self.sigmas, self.sigmas[1:], strict=False))
        ):
            raise ValueError("Mage QAD requires four decreasing denoising positions ending at zero")
        if {"img", "timesteps"} & self.conditioning.keys():
            raise ValueError("conditioning cannot override student latent or timestep")


def prepare_mage_qad(
    transformer: nn.Module, *, expected_layers: int = 174
) -> QuasarNVFP4Controller:
    rows = census_linear_layers(transformer)
    if len(rows) != expected_layers or not all(row.eligible for row in rows):
        raise ValueError(
            f"Mage QAD requires exactly {expected_layers} eligible MMDiT Linear layers"
        )
    return prepare_quasar_nvfp4(
        transformer,
        QuasarNVFP4Config(
            reconstruction_group_size=128,
            exclude_modules=(),
            quantize_activations=True,
            dynamic_activation_scale=True,
        ),
    )


def trajectory_loss(
    student: nn.Module, teacher: nn.Module, trajectory: MageTrajectory, *, backward: bool = False
) -> float:
    """Local output MSE at each student state; no backprop through the rollout.

    Backward per position releases activation graphs before the next MMDiT call.
    Teacher never rolls out its own trajectory. Both calls share all input values.
    """
    teacher.requires_grad_(False).eval()
    latent = trajectory.latent.detach().clone()
    losses = []
    for sigma, next_sigma in zip(trajectory.sigmas, trajectory.sigmas[1:], strict=False):
        inputs = dict(
            trajectory.conditioning,
            img=latent,
            timesteps=torch.full((1,), sigma, dtype=latent.dtype, device=latent.device),
        )
        with torch.no_grad():
            target = teacher(**inputs)
        with torch.set_grad_enabled(backward):
            prediction = student(**inputs)
            loss = F.mse_loss(prediction.float(), target.float()) / 4
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite teacher trajectory loss")
            if backward:
                loss.backward()
        losses.append(float(loss.detach()))
        # FlowMatchEulerDiscreteScheduler performs Euler in fp32, then restores dtype.
        latent = (latent.float() + (next_sigma - sigma) * prediction.detach().float()).to(
            latent.dtype
        )
    return sum(losses)


def trajectory_objective(
    student,
    teacher,
    trajectory: MageTrajectory,
    *,
    coefficient=0.0,
    backward=False,
    terminal_callback=None,
) -> dict:
    """Match final-block text and image outputs at the same student states.

    A zero coefficient leaves the unused text graph out of backward entirely.
    This is important: multiplying it by zero would manufacture zero gradients.
    """
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("text coefficient must be finite and nonnegative")
    teacher.requires_grad_(False).eval()
    captured = {}

    def capture(key):
        def hook(module, args, output):
            if not isinstance(output, tuple) or len(output) != 2:
                raise ValueError("Mage final block must return (text, image)")
            captured[key] = output[0]

        return hook

    handles = [
        student.transformer_blocks[-1].register_forward_hook(capture("student")),
        teacher.transformer_blocks[-1].register_forward_hook(capture("teacher")),
    ]
    latent = trajectory.latent.detach().clone()
    totals = dict(image_mse=0.0, text_mse=0.0, objective=0.0)
    try:
        for sigma, next_sigma in zip(trajectory.sigmas, trajectory.sigmas[1:], strict=False):
            captured.clear()
            inputs = dict(
                trajectory.conditioning,
                img=latent,
                timesteps=torch.full((1,), sigma, dtype=latent.dtype, device=latent.device),
            )
            with torch.no_grad():
                target = teacher(**inputs)
            with torch.set_grad_enabled(backward):
                prediction = student(**inputs)
                image_loss = F.mse_loss(prediction.float(), target.float()) / 4
                text_loss = F.mse_loss(captured["student"].float(), captured["teacher"].float()) / 4
                objective = image_loss + coefficient * text_loss if coefficient else image_loss
                if not all(torch.isfinite(x).item() for x in (image_loss, text_loss, objective)):
                    raise FloatingPointError("non-finite text pilot loss")
                if backward:
                    objective.backward()
            for key, value in zip(totals, (image_loss, text_loss, objective), strict=True):
                totals[key] += float(value.detach())
            latent = (latent.float() + (next_sigma - sigma) * prediction.detach().float()).to(
                latent.dtype
            )
        if terminal_callback is not None:
            terminal_callback(latent.detach())
        return totals
    finally:
        for handle in handles:
            handle.remove()


class MageTrajectorySource:
    """Capture upstream's first denoiser inputs, including conditioning and noise.

    Calling the real pipeline preserves its screening, watermark, text template,
    packing and scheduler. Stop before its first denoiser call; QAD owns the four
    differentiable forwards. cfg=1 is the frozen Turbo protocol.
    """

    def __init__(self, pipeline: Any, *, height: int = 1024, width: int = 1024):
        self.pipeline = pipeline
        self.height, self.width = height, width

    def __call__(self, sample: dict) -> MageTrajectory:
        class Captured(Exception):
            pass

        captured = {}

        def capture(module, args, kwargs):
            if args:
                raise ValueError("Mage transformer input signature changed")
            captured.update(kwargs)
            raise Captured

        handle = self.pipeline.model.transformer.register_forward_pre_hook(
            capture, with_kwargs=True
        )
        try:
            with suppress(Captured):
                self.pipeline.generate(
                    [sample["prompt"]],
                    seeds=[sample["seed"]],
                    heights=[self.height],
                    widths=[self.width],
                    steps=4,
                    cfg=1.0,
                )
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("Mage sample produced no denoising trajectory (including refusal)")
        sigmas = tuple(float(x) for x in self.pipeline.model.scheduler.sigmas)
        latent = captured.pop("img").detach().clone()
        captured.pop("timesteps")
        return MageTrajectory(latent, captured, sigmas)
