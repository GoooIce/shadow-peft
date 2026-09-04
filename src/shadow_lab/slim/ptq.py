"""Backbone PTQ through AngelSlim — the calibration-based compression backend.

Second compression backend for the shadow_lab backbone seam (torchao lives in
``shadow_lab.tiers``; the two coexist until the replacement's step 2 deletes
the torchao path). Interface facts a caller must know:

* **Calibration is mandatory.** GPTQ/AWQ solve the weight grid against a
  Hessian accumulated on the bare base — ``calib_batches`` are consumed at
  conversion time (before shadow mounting, iron rule 1). Weight-only tiers
  carry no static activation scales, so the joint-calibration rule (torchao
  W8A8 iron rule 2) does not apply; the static-activation tiers
  (``fp8_static`` / ``w4a8_fp8``) calibrate on the bare base through
  AngelSlim's own hook loop and the manifest records
  ``joint_calibration=False``.
* **Training numerics.** ``fake_quant=True`` keeps the GPTQ fake-quantized
  weights as plain bf16 tensors in the original Linear modules
  (``dequant_to_bf16``: the fasterquant solve writes back the quantized
  weights in place and convert skips packing) — mount the shadow and train.
  Only the GPTQ family supports this: ``int4_awq`` always packs
  ``WQLinearGEMM`` modules and the fp8/int8 tiers install their own
  QDQ/dynamic modules, so ``fake_quant=True`` is rejected for them.
* **nvfp4 deployment caveat.** The nvfp4_gptq fake-quant training path is
  pure torch (no FP4 kernels needed). Deployment is a separate decision:
  GB10/SM121 lacks native FP4 instructions — sglang main carries verified
  DGX-Spark NVFP4 cells, but community measurements put FP8 ~32% faster on
  both vLLM and SGLang. Let the arm verdict decide, not the whitelist.
"""

from __future__ import annotations

import torch
from transformers import PreTrainedModel

from . import engine as _engine

ANGELSLIM_TIERS = (
    "bf16",
    "int4_gptq",
    "int4_awq",
    "int8_dynamic",
    "fp8_dynamic",
    "fp8_static",
    "w4a8_fp8",
    "nvfp4_gptq",
)

# dequant_to_bf16 is a GPTQ-family knob: int4_awq always packs WQLinearGEMM;
# fp8/int8 tiers install QDQ/dynamic modules instead.
FAKE_QUANT_TIERS = ("int4_gptq", "nvfp4_gptq")

# Activation scales for these tiers are calibrated on the bare base (inside
# AngelSlim's compressor); the torchao-era joint-calibration rule cannot be
# routed through this backend. Manifest says so via joint_calibration=False.
STATIC_ACTIVATION_TIERS = ("fp8_static", "w4a8_fp8")


def _tier_builder(tier: str):
    """Return the fresh-config factory for a whitelisted tier."""
    from angelslim.utils import default_compress_config as dcc

    builders = {
        "int4_gptq": dcc.default_int4_gptq_config,
        "int4_awq": dcc.default_int4_awq_config,
        "int8_dynamic": dcc.default_int8_dynamic_config,
        "fp8_dynamic": dcc.default_fp8_dynamic_config,
        "fp8_static": dcc.default_fp8_static_config,
        "w4a8_fp8": dcc.default_w4a8_fp8_static_config,
    }
    return builders[tier]


def _assemble(
    base: PreTrainedModel,
    calib_batches: list[dict[str, torch.Tensor]],
    tier: str,
    fake_quant: bool,
    overrides: dict | None,
    save_dir: str | None,
):
    """Build (GlobalConfig, CompressionConfig) for the in-memory conversion."""
    from angelslim.utils.config_parser import CompressionConfig, QuantizationConfig

    global_config = _engine.build_global_config(
        base, calib_batches, save_path=save_dir or "./output"
    )

    if tier == "nvfp4_gptq":
        # No upstream default-config entry: assembled here from the schema the
        # upstream nvfp4_gptq YAMLs use (micro-scaling block fixed at 16).
        quant_method: dict = {
            "weight": "per-group",
            "group_size": 16,
            "dequant_to_bf16": fake_quant,
            "actorder": True,
            "share_gate_up_weight_scale_2": True,
        }
        compress_config = CompressionConfig(
            name="PTQ",
            quantization=QuantizationConfig(
                name=tier,
                bits=4,
                quant_method=quant_method,
                ignore_layers=["lm_head"],
            ),
        )
    else:
        # Factory call → fresh dataclasses; the engine-level
        # DEFAULT_COMPRESSION_CONFIG dict holds shared instances, mutating
        # those would poison global state.
        compress_config = _tier_builder(tier)()["compress_config"]
        if tier in FAKE_QUANT_TIERS:
            quant_method = {
                **compress_config.quantization.quant_method,
                "dequant_to_bf16": fake_quant,
            }
            compress_config.quantization.quant_method = quant_method

    if overrides:
        compress_config.quantization.quant_method = {
            **compress_config.quantization.quant_method,
            **overrides,
        }
    return global_config, compress_config


def convert_backbone(
    base: PreTrainedModel,
    tokenizer,
    tier: str,
    calib_batches: list[dict[str, torch.Tensor]],
    *,
    fake_quant: bool = True,
    save_dir: str | None = None,
    model_name: str | None = None,
    compress_overrides: dict | None = None,
) -> dict:
    """Convert ``base`` in place under an AngelSlim ``tier``; returns the manifest.

    Call BEFORE mounting the shadow (iron rule 1: the GPTQ solve replaces
    decoder-block weights in place). ``fake_quant=True`` leaves GPTQ
    fake-quantized bf16 weights in the original Linears — the training
    numerics; ``fake_quant=False`` packs kernel modules (eval/deploy form)
    and ``save_dir`` writes the deployable HF checkpoint in whichever form
    was produced.
    """
    if tier not in ANGELSLIM_TIERS:
        raise ValueError(f"unsupported AngelSlim tier {tier!r} (expected one of {ANGELSLIM_TIERS})")
    if tier == "bf16":
        return {"tier": "bf16", "backend": "angelslim", "format": "bf16", "sequence": ["noop"]}
    if fake_quant and tier not in FAKE_QUANT_TIERS:
        raise ValueError(
            f"fake_quant=True requires one of {FAKE_QUANT_TIERS} "
            f"(dequant_to_bf16 is a GPTQ-family knob; {tier} packs or installs "
            "its own modules — use fake_quant=False for the eval/deploy form)"
        )
    if not calib_batches:
        raise ValueError("AngelSlim tiers are calibration-based: pass calib_batches")

    manifest: dict = {
        "tier": tier,
        "backend": "angelslim",
        "model_name": model_name or _engine.resolve_family(base),
        "calib_forwards": len(calib_batches),
    }

    global_config, compress_config = _assemble(
        base, calib_batches, tier, fake_quant, compress_overrides, save_dir
    )

    slim_engine = _engine.build_engine(base, tokenizer, model_name=model_name)
    slim_engine.prepare_data(custom_dataloader=_engine.CalibLoader(calib_batches))
    slim_engine.prepare_compressor(
        "PTQ", global_config=global_config, compress_config=compress_config
    )
    slim_engine.run()  # GPTQ/AWQ solve + fake-quant writeback
    slim_engine.convert()  # pack, or skip under dequant_to_bf16
    sequence = ["calibrate", "convert"]
    if save_dir:
        slim_engine.save(save_dir)
        sequence.append("save")

    manifest["format"] = f"{tier}_fake_bf16" if fake_quant else f"{tier}_packed"
    manifest["sequence"] = sequence
    manifest["static_activation_scales"] = tier in STATIC_ACTIVATION_TIERS
    manifest["joint_calibration"] = False if tier in STATIC_ACTIVATION_TIERS else None
    return manifest
