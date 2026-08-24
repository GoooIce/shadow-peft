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

- ``w8a8_fp8`` runs rowwise FP8 through ``torch._scaled_mm`` on CUDA
  (sm >= 8.9; 5090-verified eager configuration). Float8 dynamic quant
  refuses to run on CPU, so local logic validation substitutes the *same
  dynamic activation/weight protocol* in INT8
  (``Int8DynamicActivationInt8WeightConfig``) and the manifest records
  ``format=int8_dynamic_local``.
- ``w4a16`` runs the tinygemm tile-packed Int4 format on CUDA (native
  torch kernels, cc >= 8.0). Off-CUDA the mslk/torch kernels are
  unavailable, so local logic validation substitutes the same *weight-only*
  protocol in INT8 (``Int8WeightOnlyConfig``, activations stay bf16) and
  the manifest records ``format=int8_weight_only_local``.
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
            # tinygemm tile-packed format: native torch._weight_int4pack_mm
            # kernels (cc >= 8.0). The default PLAIN format needs the mslk
            # CUTLASS kernel package, whose sm_120 build fails to initialize
            # (torchao 0.17/0.18, mslk cu130 1.2.0) — 5090-verified choice.
            from torchao.quantization import Int4WeightOnlyConfig
            from torchao.quantization.quantize_.workflows.int4.int4_packing_format import (
                Int4PackingFormat,
            )

            return Int4WeightOnlyConfig(int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D)
        from torchao.quantization import Int8WeightOnlyConfig

        return Int8WeightOnlyConfig()
    assert tier == "w8a8_fp8"
    if device.type == "cuda":
        # Rowwise FP8 through torch._scaled_mm (eager, no mslk): the default
        # AUTO granularity lands on mismatched 128x128-block weights whose
        # forward asserts "input must be 1x128 scaled", and the mslk/rowwise
        # path dies in CUTLASS init — PerRow + TORCH is the 5090-verified
        # eager configuration.
        from torchao.quantization.quant_api import (
            Float8DynamicActivationFloat8WeightConfig,
            PerRow,
        )
        from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

        return Float8DynamicActivationFloat8WeightConfig(
            granularity=PerRow(), kernel_preference=KernelPreference.TORCH
        )
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


def to_fake_quant(module: torch.nn.Module) -> int:
    """Swap torchao-quantized weight tensors for dequantized plain tensors.

    Training seam: the real kernels (``_scaled_mm`` / ``_weight_int4pack_mm``)
    have no autograd derivatives, so a shadow trained against a tiered base
    runs the base in fake-quant numerics — dequantized weights carrying the
    quantization error, plain bf16 matmuls, straight-through-equivalent
    backward (the same numeric class the E1/E2 packed modules train under).
    The kernel path stays an eval-time measurement (spec's dual-path rule);
    the fake-quant eval cell is the train-consistent deployment point.

    Returns the number of swapped weights.
    """
    swapped = 0
    for _, mod in module.named_modules():
        w = getattr(mod, "weight", None)
        if (
            w is not None
            and type(w).__module__.split(".")[0] == "torchao"
            and hasattr(type(w), "dequantize")
        ):
            # torchao tensor-impl (Int8Tensor/Int4Tensor/Float8Tensor...);
            # plain tensors also expose a no-op `dequantize` that only raises,
            # so the module check is what selects the real implementations.
            deq = w.dequantize()  # back to the original high-precision dtype
            mod.weight = torch.nn.Parameter(deq, requires_grad=False)
            swapped += 1
    return swapped


def int4_fake_quant_weight(w: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Materialize an Int4 weight-only (W4A16) fake-quant of ``w`` in high precision.

    Uses torchao's public affine primitives (4-bit asymmetric, per-group-128,
    float scale/zero) — the self-consistent numerics shared by the *training*
    path and the eval grid's fake-quant cells. The kernel path (tinygemm
    tile-packed tensors) has no ``dequantize`` implementation and its packing
    convention is private, so the kernel-vs-fake delta measured at eval time
    includes the codebook-convention difference; both are reported per the
    spec's dual-path rule.
    """
    from torchao.quantization.quant_primitives import (
        MappingType,
        choose_qparams_affine,
        dequantize_affine,
        quantize_affine,
    )

    out_f, in_f = w.shape
    pad = (-in_f) % group_size
    wp = torch.nn.functional.pad(w.detach().float(), (0, pad)) if pad else w.detach().float()
    block_size = (1, group_size)
    scale, zero = choose_qparams_affine(
        wp,
        MappingType.ASYMMETRIC,
        block_size,
        torch.int8,
        quant_min=0,
        quant_max=15,
        scale_dtype=torch.float32,
        zero_point_dtype=torch.float32,
    )
    q = quantize_affine(wp, block_size, scale, zero, torch.int8, quant_min=0, quant_max=15)
    deq = dequantize_affine(
        q, block_size, scale, zero, torch.int8,
        quant_min=0, quant_max=15, output_dtype=torch.float32,
    )
    return (deq[:, :in_f] if pad else deq).to(w.dtype)


def fake_quant_int4_(module: torch.nn.Module, group_size: int = 128) -> int:
    """Replace eligible Linear weights with their Int4 fake-quant values."""
    swapped = 0
    for _, mod in module.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        w = mod.weight
        if w is None or mod.in_features % group_size != 0:
            continue
        with torch.no_grad():
            if type(w).__module__.split(".")[0] == "torchao":
                # Normalize a tensor-impl weight (e.g. the local stand-in)
                # back to plain high precision before Int4-quantizing.
                w = w.dequantize()
            fake = int4_fake_quant_weight(w.detach(), group_size)
        mod.weight = torch.nn.Parameter(fake, requires_grad=False)
        swapped += 1
    return swapped


def apply_training_conversion(
    base: torch.nn.Module,
    tier: str,
    device: torch.device,
    *,
    fake_quant: bool = False,
) -> dict[str, Any]:
    """Training-facing conversion: kernel conversion, or its fake-quant form.

    ``fake_quant=True`` prepares training numerics (real kernels have no
    autograd). For w4a16 the fake form *is* the conversion — the Int4
    fake-quant is materialized from the original weights directly (the
    tile-packed kernel tensor has no dequantize). For w8a8_fp8 the kernel
    conversion runs first and its tensors are dequantized back.
    """
    if fake_quant and tier == "w4a16":
        n = fake_quant_int4_(base)
        return {"tier": tier, "format": "int4_weight_only_fake", "fake_quant_swaps": n}
    manifest = apply_tier_conversion(base, tier, device)
    if fake_quant and manifest.get("format") not in (None, "bf16"):
        try:
            manifest["fake_quant_swaps"] = to_fake_quant(base)
        except NotImplementedError as exc:
            raise RuntimeError(
                "w8a8_fp8 fake-quant needs a backend whose quantized tensors "
                "implement dequantize (CUDA Float8 does; the local Int8 "
                "stand-in does not)"
            ) from exc
    return manifest


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
    fake_quant: bool = False,
) -> tuple[ShadowPeftModel, dict[str, Any]]:
    """Build one direction-① student: tiered backbone + fresh shadow.

    Sanctioned order: (1) torchao conversion, (2) optional fake-quant swap
    (training numerics — real kernels have no autograd), (3) shadow mounting,
    (4) joint calibration forwards with the shadow live (W8A8 tiers). The
    manifest's ``sequence`` records what actually ran;
    ``tests/test_lab_tiers.py`` pins the order so a refactor cannot silently
    swap it.
    """
    manifest: dict[str, Any] = apply_training_conversion(
        base, tier, device, fake_quant=fake_quant
    )
    sequence: list[str] = ["convert"]
    if fake_quant and "fake_quant_swaps" in manifest and tier != "w4a16":
        sequence.append("fake_quant")

    base.eval()
    peft_model = get_shadow_model(base, shadow_config)
    sequence.append("mount")

    if calib_batches:
        n = joint_calibrate(peft_model, calib_batches)
        manifest["calib_forwards"] = n
        sequence.append(f"calibrate:{n}")

    manifest["sequence"] = sequence
    return peft_model, manifest
