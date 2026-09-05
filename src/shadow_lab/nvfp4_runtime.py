"""Truthful TorchAO runtime bridge for native NVFP4 Linear execution.

QUASAR owns code/scale reconstruction. This module only turns that immutable
representation into TorchAO's ``NVFP4Tensor`` and classifies profiler evidence;
it never infers native execution from storage dtype alone.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .quasar import (
    NVFP4_BLOCK_SIZE,
    NVFP4Weight,
    QuasarNVFP4Config,
    reconstruct_nvfp4,
    reconstruct_nvfp4_with_clip,
)


@dataclass(frozen=True)
class LinearCensusRow:
    name: str
    in_features: int
    out_features: int
    bias: bool
    weight_parameters: int
    bias_parameters: int
    parameters: int
    reconstruction_groups: int
    eligible: bool
    reason: str | None


def _module_name_matches(name: str, patterns: Iterable[str]) -> bool:
    return any(name == pattern or name.endswith(f".{pattern}") for pattern in patterns)


def census_linear_layers(
    root: nn.Module,
    *,
    target_modules: tuple[str, ...] | None = None,
    exclude_modules: tuple[str, ...] = (),
) -> list[LinearCensusRow]:
    """List every Linear and state why it is or is not native-NVFP4 eligible."""

    rows: list[LinearCensusRow] = []
    for name, module in root.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        reason = None
        if _module_name_matches(name, exclude_modules):
            reason = "excluded by module rule"
        elif target_modules is not None and not _module_name_matches(name, target_modules):
            reason = "not selected by target module rule"
        elif module.in_features % NVFP4_BLOCK_SIZE or module.out_features % NVFP4_BLOCK_SIZE:
            reason = "weight dimensions are not both divisible by 16"
        rows.append(
            LinearCensusRow(
                name=name,
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                weight_parameters=module.weight.numel(),
                bias_parameters=module.bias.numel() if module.bias is not None else 0,
                parameters=module.weight.numel()
                + (module.bias.numel() if module.bias is not None else 0),
                reconstruction_groups=module.out_features
                * ((module.in_features + 127) // 128),
                eligible=reason is None,
                reason=reason,
            )
        )
    return rows


def census_manifest(rows: Iterable[LinearCensusRow]) -> dict:
    materialized = list(rows)
    eligible = [row for row in materialized if row.eligible]
    return {
        "linear_layers": len(materialized),
        "eligible_layers": len(eligible),
        "excluded_layers": len(materialized) - len(eligible),
        "linear_parameters": sum(row.parameters for row in materialized),
        "eligible_parameters": sum(row.parameters for row in eligible),
        "linear_weight_parameters": sum(row.weight_parameters for row in materialized),
        "eligible_weight_parameters": sum(row.weight_parameters for row in eligible),
        "eligible_bias_parameters": sum(row.bias_parameters for row in eligible),
        "es_clip_dimensions_group_128": sum(
            row.reconstruction_groups for row in eligible
        ),
        "layers": [asdict(row) for row in materialized],
    }


def _torchao_nvfp4_symbols():
    try:
        from torchao.prototype.mx_formats.nvfp4_tensor import (
            NVFP4Tensor,
            QuantizeTensorToNVFP4Kwargs,
            per_tensor_amax_to_scale,
        )
        from torchao.prototype.mx_formats.utils import (
            hp_data_dims_to_swizzled_scale_dims_nvfp4,
            to_blocked,
        )
    except ImportError as error:  # pragma: no cover - exercised in minimal installs
        raise RuntimeError(
            "native NVFP4 requires torchao==0.18.0 in the GB10 Mage environment"
        ) from error
    return (
        NVFP4Tensor,
        QuantizeTensorToNVFP4Kwargs,
        per_tensor_amax_to_scale,
        hp_data_dims_to_swizzled_scale_dims_nvfp4,
        to_blocked,
    )


def _activation_quant_kwargs(use_triton_kernel: bool):
    _, kwargs_cls, _, _, _ = _torchao_nvfp4_symbols()
    return kwargs_cls(
        block_size=NVFP4_BLOCK_SIZE,
        is_swizzled_scales=True,
        use_triton_kernel=use_triton_kernel,
        use_dynamic_per_tensor_scale=True,
    )


@torch.no_grad()
def install_rtn_nvfp4_linear(
    linear: nn.Linear,
    *,
    use_triton_kernel: bool = False,
) -> None:
    """Install TorchAO's standard max-scaled RTN NVFP4 representation."""

    if linear.weight.ndim != 2:
        raise ValueError("native NVFP4 Linear weight must be two-dimensional")
    out_features, in_features = linear.weight.shape
    if out_features % NVFP4_BLOCK_SIZE or in_features % NVFP4_BLOCK_SIZE:
        raise ValueError("native NVFP4 weight dimensions must both be divisible by 16")
    nvfp4_cls, _, per_tensor_amax_to_scale, _, _ = _torchao_nvfp4_symbols()
    per_tensor_scale = per_tensor_amax_to_scale(linear.weight.detach().abs().max())
    quantized = nvfp4_cls.to_nvfp4(
        linear.weight.detach(),
        block_size=NVFP4_BLOCK_SIZE,
        per_tensor_scale=per_tensor_scale,
        is_swizzled_scales=True,
        use_triton_kernel=False,
        act_quant_kwargs=_activation_quant_kwargs(use_triton_kernel),
    )
    quantized.use_triton_kernel = use_triton_kernel
    linear.weight = nn.Parameter(quantized, requires_grad=False)
    if linear.bias is not None:
        linear.bias.requires_grad_(False)


@torch.no_grad()
def install_reconstructed_nvfp4_linear(
    linear: nn.Linear,
    reconstruction: NVFP4Weight,
    *,
    use_triton_kernel: bool = False,
) -> None:
    """Install QUASAR/ES codes and scales as a native-capable TorchAO weight."""

    if tuple(linear.weight.shape) != reconstruction.original_shape:
        raise ValueError(
            f"reconstruction shape {reconstruction.original_shape} does not match "
            f"Linear weight {tuple(linear.weight.shape)}"
        )
    out_features, in_features = reconstruction.original_shape
    if out_features % NVFP4_BLOCK_SIZE or in_features % NVFP4_BLOCK_SIZE:
        raise ValueError("native NVFP4 weight dimensions must both be divisible by 16")

    nvfp4_cls, _, _, scale_dims, to_blocked = _torchao_nvfp4_symbols()
    qdata = reconstruction.weight_packed.contiguous()
    scales = reconstruction.weight_scale.contiguous().view(
        out_features, in_features // NVFP4_BLOCK_SIZE
    )
    swizzled_scales = to_blocked(scales).flatten()
    scale_m, scale_k = scale_dims(out_features, in_features)
    swizzled_scales = swizzled_scales.view(scale_m, scale_k)
    per_tensor_scale = reconstruction.tensor_scale.detach().float().reshape(())
    quantized = nvfp4_cls(
        qdata,
        swizzled_scales,
        NVFP4_BLOCK_SIZE,
        linear.weight.dtype,
        per_tensor_scale=per_tensor_scale,
        act_per_tensor_scale=None,
        is_swizzled_scales=True,
        use_triton_kernel=use_triton_kernel,
        act_quant_kwargs=_activation_quant_kwargs(use_triton_kernel),
    )
    linear.weight = nn.Parameter(quantized, requires_grad=False)
    if linear.bias is not None:
        linear.bias.requires_grad_(False)


def native_kernel_evidence(kernel_names: Iterable[str]) -> dict:
    """Classify profiler names using the strict GB10 native-compute gate."""

    names = sorted(set(kernel_names))
    matches = [
        name
        for name in names
        if "block_scaled" in name
        and name.count("ue4m3xe2m1") >= 2
        and ("sm120" in name or "sm121" in name)
    ]
    return {
        "quant_representation": "packed_e2m1_with_e4m3_block_scale",
        "compute_kernel": matches,
        "native_compute": bool(matches),
        "fallback_reason": None
        if matches
        else "no SM120/SM121 block-scaled E2M1×E2M1 kernel found in profiler events",
    }


@torch.no_grad()
def materialize_native_nvfp4(
    root: nn.Module,
    *,
    scheme: str,
    quasar_config: QuasarNVFP4Config | None = None,
    clip_by_layer: Mapping[str, torch.Tensor | float] | None = None,
    target_modules: tuple[str, ...] | None = None,
    exclude_modules: tuple[str, ...] = (),
    saliency_by_layer: Mapping[str, torch.Tensor] | None = None,
    use_triton_kernel: bool = False,
) -> dict:
    """Destructively materialize eligible Linear weights for native W4A4.

    ``scheme`` is one of ``rtn``, ``quasar-grid`` or ``external-clip``. The
    function reports compute as unverified until the caller supplies profiler
    evidence; an NVFP4Tensor representation alone never proves native compute.
    """

    if scheme not in {"rtn", "quasar-grid", "external-clip"}:
        raise ValueError("scheme must be rtn, quasar-grid, or external-clip")
    if saliency_by_layer is not None and scheme != "external-clip":
        raise ValueError("saliency_by_layer requires external-clip")
    if scheme != "external-clip" and clip_by_layer is not None:
        raise ValueError("clip_by_layer is only valid for external-clip")

    rows = census_linear_layers(
        root,
        target_modules=target_modules,
        exclude_modules=exclude_modules,
    )
    eligible = [row for row in rows if row.eligible]
    if not eligible:
        raise ValueError("no native-NVFP4 eligible Linear layers")
    eligible_names = {row.name for row in eligible}
    if saliency_by_layer is not None and set(saliency_by_layer) != eligible_names:
        raise ValueError("saliency must exactly match eligible layers")
    if scheme == "external-clip":
        if clip_by_layer is None:
            raise ValueError("external-clip requires clip_by_layer")
        missing = sorted(eligible_names - set(clip_by_layer))
        unknown = sorted(set(clip_by_layer) - eligible_names)
        if missing or unknown:
            raise ValueError(
                f"clip_by_layer must exactly match eligible layers; "
                f"missing={missing[:3]}, unknown={unknown[:3]}"
            )

    if quasar_config is None:
        quasar_config = QuasarNVFP4Config(
            reconstruction_group_size=128,
            quantize_activations=True,
        )
    if quasar_config.reconstruction_group_size != 128 and scheme != "rtn":
        raise ValueError("Mage Phase-1 reconstruction_group_size must be 128")
    if not quasar_config.quantize_activations and scheme != "rtn":
        raise ValueError("native Mage Phase-1 requires NVFP4 W4A4 activations")

    trainable_parameters_before = sum(
        parameter.numel() for parameter in root.parameters() if parameter.requires_grad
    )
    root.requires_grad_(False)
    weighted_error = 0.0
    clip_group_count = 0
    clip_factor_counts = [0] * len(quasar_config.clip_factors)
    layer_reports: list[dict] = []
    packed_hashes: dict[str, str] = {}
    for row in eligible:
        linear = root.get_submodule(row.name)
        if not isinstance(linear, nn.Linear):  # pragma: no cover - mutation guard
            raise TypeError(f"{row.name} stopped being a Linear during materialization")
        original = linear.weight.detach()
        if scheme == "rtn":
            install_rtn_nvfp4_linear(linear, use_triton_kernel=use_triton_kernel)
            reconstructed = linear.weight.dequantize(torch.float32)
            layer_error = (reconstructed - original.float()).square().sum().item()
            layer_clip_groups = 0
        else:
            if scheme == "quasar-grid":
                reconstruction = reconstruct_nvfp4(
                    original,
                    saliency=None,
                    config=quasar_config,
                )
                indices = reconstruction.clip_factor_index.reshape(-1).to(torch.long)
                counts = torch.bincount(
                    indices, minlength=len(clip_factor_counts)
                ).cpu()
                clip_factor_counts = [
                    old + int(new)
                    for old, new in zip(clip_factor_counts, counts, strict=True)
                ]
            else:
                reconstruction = reconstruct_nvfp4_with_clip(
                    original,
                    clip_by_layer[row.name],
                    saliency=(saliency_by_layer[row.name].to(original.device)
                              if saliency_by_layer is not None else None),
                    config=quasar_config,
                )
            layer_error = reconstruction.weighted_error.item()
            layer_clip_groups = reconstruction.clip_factor_value.numel()
            if saliency_by_layer is not None:
                digest = hashlib.sha256()
                for tensor in (reconstruction.weight_packed, reconstruction.weight_scale,
                               reconstruction.tensor_scale):
                    digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
                packed_hashes[row.name] = digest.hexdigest()
            install_reconstructed_nvfp4_linear(
                linear,
                reconstruction,
                use_triton_kernel=use_triton_kernel,
            )
        weighted_error += layer_error
        clip_group_count += layer_clip_groups
        layer_reports.append(
            {
                "name": row.name,
                "in_features": row.in_features,
                "out_features": row.out_features,
                "parameters": row.parameters,
                "weight_parameters": row.weight_parameters,
                "bias_parameters": row.bias_parameters,
                "weighted_reconstruction_error": layer_error,
                "clip_groups": layer_clip_groups,
            }
        )

    return {
        "scheme": scheme,
        "packed_weight_hashes": packed_hashes,
        "format": "nvfp4_w4a4",
        "quant_representation": "torchao.NVFP4Tensor",
        "compute_kernel": "unverified_pending_profiler",
        "native_compute": None,
        "fallback_reason": None,
        "nvfp4_block_size": NVFP4_BLOCK_SIZE,
        "reconstruction_group_size": 128 if scheme != "rtn" else None,
        "use_triton_activation_quantizer": use_triton_kernel,
        "trainable_parameters_before": trainable_parameters_before,
        "trainable_parameters_after": sum(
            parameter.numel() for parameter in root.parameters() if parameter.requires_grad
        ),
        "census": census_manifest(rows),
        "quantized_layers": len(eligible),
        "quantized_parameters": sum(row.parameters for row in eligible),
        "quantized_weight_parameters": sum(row.weight_parameters for row in eligible),
        "frozen_bias_parameters": sum(row.bias_parameters for row in eligible),
        "weighted_reconstruction_error": weighted_error,
        "clip_group_count": clip_group_count,
        "clip_candidates": list(quasar_config.clip_factors) if scheme == "quasar-grid" else None,
        "clip_factor_counts": clip_factor_counts if scheme == "quasar-grid" else None,
        "layers": layer_reports,
    }
