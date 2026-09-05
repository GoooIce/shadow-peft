"""Isolated protocol-development loss; does not change the formal QAD objective."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .mage_qad import MageTrajectory


def inactive_text_layers(model) -> tuple[str, ...]:
    prefix = f"transformer_blocks.{len(model.transformer_blocks) - 1}"
    return tuple(
        f"{prefix}.{suffix}"
        for suffix in ("attn.to_add_out", "txt_mlp.net.0.proj", "txt_mlp.net.2")
    )


def text_pilot_loss(
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


def compare_pilot_reports(image: dict, auxiliary: dict) -> dict:
    """Predeclared paired screen, not an image-quality or deployment gate."""
    if image["status"] != "completed" or auxiliary["status"] != "completed":
        raise ValueError("both arms must complete")
    for key in ("data_sha256", "initial_state_sha256", "code_sha256", "protocol"):
        if image[key] != auxiliary[key]:
            raise ValueError(f"arms differ in {key}")
    baseline_equal = image["baseline"] == auxiliary["baseline"]
    a = [r["image_mse"] for r in image["validation"]]
    b = [r["image_mse"] for r in auxiliary["validation"]]
    if len(a) != 8 or len(b) != 8:
        raise ValueError("pilot requires eight paired validation prompts")
    mean_a, mean_b = sum(a) / 8, sum(b) / 8
    relative_improvement = (mean_a - mean_b) / max(mean_a, 1e-12)
    wins = sum(y < x for x, y in zip(a, b, strict=True))
    baseline_mean = sum(r["image_mse"] for r in image["baseline"]) / 8
    promising = (
        baseline_equal and relative_improvement >= 0.02 and wins >= 6 and mean_b < baseline_mean
    )
    return dict(
        status="completed" if baseline_equal else "invalid_baseline_mismatch",
        baseline_exactly_equal=baseline_equal,
        initial_image_mse=baseline_mean,
        image_only_mse=mean_a,
        auxiliary_mse=mean_b,
        auxiliary_relative_improvement=relative_improvement,
        auxiliary_paired_wins=wins,
        per_prompt_delta_b_minus_a=[y - x for x, y in zip(a, b, strict=True)],
        recommendation=(
            "auxiliary_promising_requires_confirmation"
            if promising
            else "retain_image_only_no_clear_short_run_benefit"
        ),
        limitation="One seed, eight updates, eight prompts; no perceptual quality claim or formal gate.",
    )
