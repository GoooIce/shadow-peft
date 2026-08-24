"""Backbone tier seam for direction ①: ``{bf16, w8a8_fp8, w4a16}``.

Two iron rules from the spec live here, pinned by ``tests/test_lab_tiers.py``:

1. **Conversion before mounting.** torchao's ``quantize_`` swaps modules
   in place and prepare/convert-style APIs strip already-mounted hooks, so the
   torchao conversion MUST run before the shadow wraps the decoder layers.
   ``prepare_tiered_student`` is the only sanctioned order and records it in
   its manifest ``sequence``.
2. **Joint calibration.** W8A8 activation scales must be observed on
   activations that already carry the shadow's injections: calibration
   forwards run through the *mounted* ``ShadowPeftModel`` (see
   ``joint_calibrate``), never through the bare quantized base.

Device notes (spec: 5090 sm_120 satisfies FP8):

- ``w8a8_fp8`` uses ``Float8DynamicActivationFloat8WeightConfig`` on CUDA
  (sm >= 8.9). Float8 dynamic quant refuses to run on CPU, so local logic
  validation substitutes the *same dynamic activation/weight protocol* in
  INT8 (``Int8DynamicActivationInt8WeightConfig``) and the manifest records
  ``format=int8_dynamic_local``. The 5090 runs real FP8.
- ``w4a16`` uses ``Int4WeightOnlyConfig`` on CUDA. torchao's Int4 workflow
  hard-requires the ``mslk`` CUDA kernel package (cu130 index, no CPU/macOS
  wheels), so local logic validation substitutes the same *weight-only*
  protocol in INT8 (``Int8WeightOnlyConfig``, activations stay bf16) and the
  manifest records ``format=int8_weight_only_local``. The 5090 runs real
  Int4 (instance-side mslk availability is checked by ticket 04's ops pass).
- torchao tiers never trim the vocabulary (Gemma keeps the full 262,144).
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedModel

from shadow_peft import ShadowConfig, get_shadow_model
from shadow_peft.peft_model import ShadowPeftModel

TIERS = ("bf16", "w8a8_fp8", "w4a16")


def tier_config(tier: str, device: torch.device):
    """Return the torchao config for ``tier`` on ``device`` (None for bf16)."""
    if tier not in TIERS:
        raise ValueError(f"unsupported backbone tier {tier!r} (expected one of {TIERS})")
    if tier == "bf16":
        return None

    from torchao.quantization import quantize_

    _ = quantize_  # import probe: torchao must be installed for quant tiers
    if tier == "w4a16":
        if device.type == "cuda":
            from torchao.quantization import Int4WeightOnlyConfig

            return Int4WeightOnlyConfig()
        from torchao.quantization import Int8WeightOnlyConfig

        return Int8WeightOnlyConfig()
    assert tier == "w8a8_fp8"
    if device.type == "cuda":
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig

        return Float8DynamicActivationFloat8WeightConfig()
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig

    return Int8DynamicActivationInt8WeightConfig()


def apply_tier_conversion(base: PreTrainedModel, tier: str, device: torch.device) -> dict[str, Any]:
    """Convert ``base`` in place under ``tier``; returns the tier manifest.

    This is the seam torchao touches. Call it BEFORE mounting the shadow
    (iron rule 1) — ``prepare_tiered_student`` enforces the order.
    """
    config = tier_config(tier, device)
    if config is None:
        return {"tier": "bf16", "format": "bf16"}

    from torchao.quantization import quantize_

    quantize_(base, config)

    if tier == "w8a8_fp8":
        fmt = "fp8_dynamic" if device.type == "cuda" else "int8_dynamic_local"
    else:
        fmt = "int4_weight_only" if device.type == "cuda" else "int8_weight_only_local"
    return {"tier": tier, "format": fmt}


def joint_calibrate(peft_model: ShadowPeftModel, batches: list[dict[str, torch.Tensor]]) -> int:
    """Run calibration forwards **with the shadow attached** (iron rule 2).

    Each batch goes through ``peft_model.forward`` — the wrapper that
    initializes the shadow state and injects it into every decoder layer — so
    any activation-scale observer (dynamic warmup now, static collection
    later) sees injection-carrying activations. Returns the number of
    calibration forwards run.
    """
    peft_model.eval()
    n = 0
    with torch.no_grad():
        for batch in batches:
            peft_model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            n += 1
    return n


def prepare_tiered_student(
    base: PreTrainedModel,
    shadow_config: ShadowConfig,
    tier: str,
    *,
    device: torch.device,
    calib_batches: list[dict[str, torch.Tensor]] | None = None,
) -> tuple[ShadowPeftModel, dict[str, Any]]:
    """Build one direction-① student: tiered backbone + fresh shadow.

    Sanctioned order: (1) torchao conversion, (2) shadow mounting, (3) joint
    calibration forwards with the shadow live (W8A8 tiers). The manifest's
    ``sequence`` records what actually ran; ``tests/test_lab_tiers.py`` pins
    the order so a refactor cannot silently swap it.
    """
    manifest: dict[str, Any] = apply_tier_conversion(base, tier, device)
    sequence: list[str] = ["convert"]

    base.eval()
    peft_model = get_shadow_model(base, shadow_config)
    sequence.append("mount")

    if calib_batches:
        n = joint_calibrate(peft_model, calib_batches)
        manifest["calib_forwards"] = n
        sequence.append(f"calibrate:{n}")

    manifest["sequence"] = sequence
    return peft_model, manifest
