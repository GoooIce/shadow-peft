"""Shared candidate evaluation path for Mage NVFP4 evolution strategies."""

from __future__ import annotations

import hashlib
import io
import statistics
import time
from collections.abc import Sequence
from typing import Any

from shadow_lab.es_quant import materialize_es_candidate
from shadow_lab.image_eval import (
    RewardInput,
    aggregate_episode_reward,
    score_reward_batch,
)


def _png_bytes(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _generate_inputs(
    pipeline: Any,
    samples: Sequence[Any],
    *,
    height: int,
    width: int,
    steps: int,
    cfg: float,
) -> list[RewardInput]:
    inputs = []
    for sample in samples:
        image = pipeline.generate(
            [sample.prompt],
            seeds=[sample.seed],
            heights=[height],
            widths=[width],
            steps=steps,
            cfg=cfg,
        )[0]
        inputs.append(RewardInput(_png_bytes(image), sample))
    return inputs


def evaluate_clips(
    *,
    torch_runtime: Any,
    clips: Any,
    transformer: Any,
    master: Any,
    layout: Any,
    pipeline: Any,
    prompt_batch: Sequence[Any],
    scorers: Sequence[Any],
    calibration: Any,
    height: int,
    width: int,
    steps: int,
    cfg: float,
) -> dict:
    """Materialize, generate, and score one clip genome through one shared path."""
    quantize_started = time.perf_counter()
    quantization = materialize_es_candidate(
        transformer,
        master,
        layout,
        clips,
        device="cuda",
    )
    torch_runtime.cuda.synchronize()
    quantize_seconds = time.perf_counter() - quantize_started
    if quantization["trainable_parameters_after"] != 0:
        raise RuntimeError("clip materialization left trainable weights")

    generate_started = time.perf_counter()
    reward_inputs = _generate_inputs(
        pipeline,
        prompt_batch,
        height=height,
        width=width,
        steps=steps,
        cfg=cfg,
    )
    torch_runtime.cuda.synchronize()
    generate_seconds = time.perf_counter() - generate_started

    score_started = time.perf_counter()
    scores = score_reward_batch(reward_inputs, scorers)
    episode = aggregate_episode_reward(scores, calibration)
    torch_runtime.cuda.synchronize()
    score_seconds = time.perf_counter() - score_started
    if not episode.valid or episode.value is None:
        raise RuntimeError(episode.failure_reason)

    clip_payload = clips.detach().cpu().contiguous().numpy().tobytes()
    return {
        "reward": episode.value,
        "clip_sha256": hashlib.sha256(clip_payload).hexdigest(),
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
