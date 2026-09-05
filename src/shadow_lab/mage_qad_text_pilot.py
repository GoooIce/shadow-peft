"""Isolated protocol-development loss; does not change the formal QAD objective."""

from __future__ import annotations

from .mage_qad import trajectory_objective as text_pilot_loss

__all__ = ["text_pilot_loss", "inactive_text_layers", "compare_pilot_reports"]


def inactive_text_layers(model) -> tuple[str, ...]:
    prefix = f"transformer_blocks.{len(model.transformer_blocks) - 1}"
    return tuple(
        f"{prefix}.{suffix}"
        for suffix in ("attn.to_add_out", "txt_mlp.net.0.proj", "txt_mlp.net.2")
    )


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
