"""QUASAR loss-aware reconstruction for the NVFP4 QAD arm.

This module owns the complete quantization seam used during training:

* latent BF16/FP32 Linear weights remain AdamW parameters;
* each forward reconstructs them on the NVFP4 E2M1/E4M3 two-level grid;
* AdamW's existing ``exp_avg_sq`` tensor is read as the saliency ``h``;
* the reconstructed weight and activation use a straight-through estimator;
* materialization replaces training wrappers with a packed reference runtime.

The packed reference runtime dequantizes before ``F.linear``. Native W4A4
execution is supplied by the separate TorchAO adapter and must be verified with
profiler kernel evidence.

It is deliberately separate from ``shadow_quant``.  That module projects
*shadow-adapter* weights onto a grid after every optimizer step; QUASAR is a
no-shadow quantized-student training strategy with a latent master weight.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

NVFP4_BLOCK_SIZE = 16
NVFP4_E2M1_MAX = 6.0
NVFP4_E4M3_MAX = 448.0
ES_CLIP_MIN = 0.30
ES_CLIP_MAX = 1.00
E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
DEFAULT_CLIP_FACTORS = tuple(round(0.30 + 0.05 * index, 2) for index in range(15))
_FUSED_PROJECTIONS = {
    "qkv": ("q_proj", "k_proj", "v_proj"),
    "gate_up": ("gate_proj", "up_proj"),
}
_MAX_E4M3_RELATIVE_ROUNDING_ERROR = 0.063


@dataclass(frozen=True)
class QuasarNVFP4Config:
    """The fixed NVFP4 format plus QUASAR's code-assignment search space."""

    clip_factors: tuple[float, ...] = DEFAULT_CLIP_FACTORS
    reconstruction_group_size: int = NVFP4_BLOCK_SIZE
    target_modules: tuple[str, ...] | None = None
    exclude_modules: tuple[str, ...] = ("lm_head",)
    quantize_activations: bool = True
    saliency_eps: float = 1e-12

    def __post_init__(self) -> None:
        if not self.clip_factors:
            raise ValueError("clip_factors must contain at least one candidate")
        if any(not 0.0 < factor <= 1.0 for factor in self.clip_factors):
            raise ValueError("clip_factors must be in (0, 1]")
        if self.saliency_eps <= 0:
            raise ValueError("saliency_eps must be positive")
        if (
            self.reconstruction_group_size < NVFP4_BLOCK_SIZE
            or self.reconstruction_group_size % NVFP4_BLOCK_SIZE
        ):
            raise ValueError("reconstruction_group_size must be a positive multiple of 16")


def _device_constant(values: tuple[float, ...], tensor: torch.Tensor) -> torch.Tensor:
    return torch.tensor(values, device=tensor.device, dtype=torch.float32)


def e2m1_encode(value: torch.Tensor) -> torch.Tensor:
    """Encode pre-scaled values as NVFP4 uint4 codes, ties-to-even included."""

    value_f = value.float()
    magnitude = value_f.abs()
    bounds = _device_constant(E2M1_BOUNDS, value_f)
    ordinal = torch.searchsorted(bounds, magnitude, out_int32=True).to(torch.uint8)

    # At a midpoint, the adjacent code whose mantissa LSB is zero wins.
    # Midpoint indices 1/3/5 therefore round up; 0/2/4/6 round down.
    round_up_mask = torch.tensor(
        (0, 1, 0, 1, 0, 1, 0), device=value.device, dtype=torch.bool
    )
    round_up = ((magnitude.unsqueeze(-1) == bounds) & round_up_mask).any(dim=-1)
    sign = (value_f < 0).to(torch.uint8) << 3
    return (sign + ordinal + round_up.to(torch.uint8)).to(torch.uint8)


def e2m1_decode(code: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Decode NVFP4 uint4 codes to their signed E2M1 values."""

    values = _device_constant(E2M1_VALUES, code)
    return values[(code.to(torch.long) & 0x0F)].to(dtype)


def pack_e2m1(code: torch.Tensor) -> torch.Tensor:
    """Pack adjacent E2M1 codes using compressed-tensors' nibble order."""

    if code.dtype != torch.uint8:
        raise TypeError(f"E2M1 codes must be uint8, got {code.dtype}")
    if code.shape[-1] % 2:
        raise ValueError("the E2M1 code width must be even")
    return (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous()


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_e2m1`."""

    if packed.dtype != torch.uint8:
        raise TypeError(f"packed NVFP4 weights must be uint8, got {packed.dtype}")
    code = torch.empty(
        *packed.shape[:-1],
        packed.shape[-1] * 2,
        device=packed.device,
        dtype=torch.uint8,
    )
    code[..., 0::2] = packed & 0x0F
    code[..., 1::2] = packed >> 4
    return code


def _positive_tensor_scale(amax: torch.Tensor) -> torch.Tensor:
    scale = amax.float() / (NVFP4_E2M1_MAX * NVFP4_E4M3_MAX)
    return torch.where(scale > 0, scale, torch.ones_like(scale))


def _fp8_scale(value: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    return value.float().clamp(min=finfo.tiny, max=finfo.max).to(torch.float8_e4m3fn)


@dataclass(frozen=True)
class NVFP4Weight:
    """One deployable NVFP4 weight tensor plus QUASAR search diagnostics."""

    codes: torch.Tensor
    weight_scale: torch.Tensor
    tensor_scale: torch.Tensor
    original_shape: tuple[int, int]
    clip_factor_index: torch.Tensor
    clip_factor_value: torch.Tensor
    weighted_error: torch.Tensor

    @property
    def weight_packed(self) -> torch.Tensor:
        return pack_e2m1(self.codes.reshape(self.original_shape[0], -1))

    @property
    def weight_global_scale(self) -> torch.Tensor:
        """Deployment convention: reciprocal of the FP32 tensor scale."""

        return (1.0 / self.tensor_scale.float()).reshape(1)

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        out_features, in_features = self.original_shape
        values = e2m1_decode(self.codes)
        effective_scale = self.weight_scale.float() * self.tensor_scale.float()
        weight = (values * effective_scale.unsqueeze(-1)).reshape(out_features, -1)
        return weight[:, :in_features].to(dtype)


def nvfp4_clip_shape(
    weight_shape: tuple[int, ...] | torch.Size,
    reconstruction_group_size: int = 128,
) -> tuple[int, int]:
    """Return ``(output rows, clip groups per row)`` for an ES clip vector."""

    if len(weight_shape) != 2:
        raise ValueError(f"NVFP4 Linear weight must be 2D, got {tuple(weight_shape)}")
    if (
        reconstruction_group_size < NVFP4_BLOCK_SIZE
        or reconstruction_group_size % NVFP4_BLOCK_SIZE
    ):
        raise ValueError("reconstruction_group_size must be a positive multiple of 16")
    out_features, in_features = (int(value) for value in weight_shape)
    groups_per_row = (
        in_features + reconstruction_group_size - 1
    ) // reconstruction_group_size
    return out_features, groups_per_row


def _fit_nvfp4_candidate(
    blocks: torch.Tensor,
    importance: torch.Tensor,
    tensor_scale: torch.Tensor,
    block_amax: torch.Tensor,
    block_factors: torch.Tensor,
    saliency_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit E4M3 micro-scales after fixing one clip factor per NVFP4 block."""

    search_scale = block_amax * (block_factors.unsqueeze(-1) / NVFP4_E2M1_MAX)
    search_scale = torch.where(search_scale > 0, search_scale, tensor_scale)
    codes = e2m1_encode(blocks / search_scale)
    quant_values = e2m1_decode(codes)

    numerator = (importance * quant_values * blocks).sum(dim=-1)
    denominator = (importance * quant_values.square()).sum(dim=-1)
    fitted_effective_scale = torch.where(
        denominator > saliency_eps,
        numerator / denominator.clamp_min(saliency_eps),
        torch.ones_like(denominator) * tensor_scale,
    ).clamp_min_(0.0)
    stored_scale = _fp8_scale(fitted_effective_scale / tensor_scale)
    effective_scale = stored_scale.float() * tensor_scale
    reconstructed = quant_values * effective_scale.unsqueeze(-1)
    error = (importance * (reconstructed - blocks).square()).sum(dim=-1)
    return codes, stored_scale, error


def _reconstruction_inputs(
    weight: torch.Tensor,
    saliency: torch.Tensor | None,
    config: QuasarNVFP4Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if weight.ndim != 2:
        raise ValueError(f"NVFP4 Linear weight must be 2D, got {tuple(weight.shape)}")
    if saliency is not None and saliency.shape != weight.shape:
        raise ValueError(
            f"saliency shape {tuple(saliency.shape)} does not match weight {tuple(weight.shape)}"
        )

    out_features, in_features = weight.shape
    group_size = config.reconstruction_group_size
    padded_in = ((in_features + group_size - 1) // group_size) * group_size
    pad = padded_in - in_features
    weight_f = weight.detach().float()
    if pad:
        weight_f = F.pad(weight_f, (0, pad))
    blocks = weight_f.reshape(out_features, -1, NVFP4_BLOCK_SIZE)

    if saliency is None:
        importance = torch.ones_like(weight_f)
    else:
        importance = torch.nan_to_num(
            saliency.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min_(0.0)
        if pad:
            importance = F.pad(importance, (0, pad))
    importance = importance.reshape_as(blocks)
    empty_importance = importance.sum(dim=-1, keepdim=True) <= config.saliency_eps
    importance = torch.where(empty_importance, torch.ones_like(importance), importance)
    tensor_scale = _positive_tensor_scale(blocks.abs().amax())
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)
    return blocks, importance, tensor_scale, block_amax, padded_in


@torch.no_grad()
def reconstruct_nvfp4(
    weight: torch.Tensor,
    saliency: torch.Tensor | None = None,
    config: QuasarNVFP4Config | None = None,
) -> NVFP4Weight:
    """Run QUASAR's search + saliency-weighted scale fit for one Linear weight.

    The FP32 tensor scale is fixed from the tensor absolute maximum.  Each
    candidate clipping factor produces an E2M1 code assignment.  Holding that
    assignment fixed, the continuous least-squares optimum is solved and then
    rounded to the required E4M3 block-scale grid.  The candidate with minimum
    ``sum(h * (reconstruction - weight) ** 2)`` wins independently per block.
    """

    config = config or QuasarNVFP4Config()
    out_features, in_features = weight.shape
    blocks, importance, tensor_scale, block_amax, padded_in = _reconstruction_inputs(
        weight, saliency, config
    )
    blocks_per_group = config.reconstruction_group_size // NVFP4_BLOCK_SIZE
    groups_per_row = padded_in // config.reconstruction_group_size
    best_group_error = torch.full(
        (out_features, groups_per_row), float("inf"), device=weight.device
    )
    best_block_error = torch.full(
        block_amax.shape[:-1], float("inf"), device=weight.device
    )
    best_codes = torch.zeros_like(blocks, dtype=torch.uint8)
    best_scale = torch.ones(block_amax.shape[:-1], device=weight.device, dtype=torch.float32)
    best_factor = torch.zeros(
        (out_features, groups_per_row), device=weight.device, dtype=torch.int16
    )

    for factor_index, factor in enumerate(config.clip_factors):
        block_factors = torch.full(
            block_amax.shape[:-1], float(factor), device=weight.device
        )
        codes, stored_scale, block_error = _fit_nvfp4_candidate(
            blocks,
            importance,
            tensor_scale,
            block_amax,
            block_factors,
            config.saliency_eps,
        )
        group_error = block_error.reshape(
            out_features, groups_per_row, blocks_per_group
        ).sum(dim=-1)
        take_group = group_error < best_group_error
        take_block = take_group.repeat_interleave(blocks_per_group, dim=-1)
        best_group_error = torch.where(take_group, group_error, best_group_error)
        best_block_error = torch.where(take_block, block_error, best_block_error)
        best_codes = torch.where(take_block.unsqueeze(-1), codes, best_codes)
        best_scale = torch.where(take_block, stored_scale.float(), best_scale)
        best_factor = torch.where(
            take_group,
            torch.full_like(best_factor, factor_index),
            best_factor,
        )

    factor_values = _device_constant(config.clip_factors, weight)[best_factor.long()]

    return NVFP4Weight(
        codes=best_codes,
        weight_scale=best_scale.to(torch.float8_e4m3fn),
        tensor_scale=tensor_scale.reshape(1),
        original_shape=(out_features, in_features),
        clip_factor_index=best_factor,
        clip_factor_value=factor_values,
        weighted_error=best_block_error.sum(),
    )


@torch.no_grad()
def reconstruct_nvfp4_with_clip(
    weight: torch.Tensor,
    clip_factors: torch.Tensor | float,
    saliency: torch.Tensor | None = None,
    config: QuasarNVFP4Config | None = None,
) -> NVFP4Weight:
    """Reconstruct a weight using an externally supplied ES clip vector.

    One scalar controls ``config.reconstruction_group_size`` contiguous values
    within each output row. For the Phase-1 layout this is 128 values, broadcast
    to eight independently fitted NVFP4 block-16 E4M3 scales.
    """

    config = config or QuasarNVFP4Config(reconstruction_group_size=128)
    expected_shape = nvfp4_clip_shape(weight.shape, config.reconstruction_group_size)
    out_features, in_features = (int(value) for value in weight.shape)
    factors = torch.as_tensor(clip_factors, device=weight.device, dtype=torch.float32)
    if factors.numel() == 1:
        factors = factors.reshape(1).expand(expected_shape)
    elif factors.numel() == expected_shape[0] * expected_shape[1]:
        factors = factors.reshape(expected_shape)
    else:
        raise ValueError(
            f"clip vector has {factors.numel()} values; expected 1 or "
            f"{expected_shape[0] * expected_shape[1]} ({expected_shape})"
        )
    if not torch.isfinite(factors).all() or (factors < ES_CLIP_MIN).any() or (
        factors > ES_CLIP_MAX
    ).any():
        raise ValueError("external clip factors must be finite and in [0.30, 1.00]")

    blocks, importance, tensor_scale, block_amax, padded_in = _reconstruction_inputs(
        weight, saliency, config
    )
    blocks_per_group = config.reconstruction_group_size // NVFP4_BLOCK_SIZE
    groups_per_row = padded_in // config.reconstruction_group_size
    block_factors = factors.repeat_interleave(blocks_per_group, dim=-1)
    codes, stored_scale, block_error = _fit_nvfp4_candidate(
        blocks,
        importance,
        tensor_scale,
        block_amax,
        block_factors,
        config.saliency_eps,
    )
    return NVFP4Weight(
        codes=codes,
        weight_scale=stored_scale,
        tensor_scale=tensor_scale.reshape(1),
        original_shape=(out_features, in_features),
        clip_factor_index=torch.full(
            (out_features, groups_per_row), -1, device=weight.device, dtype=torch.int16
        ),
        clip_factor_value=factors.clone(),
        weighted_error=block_error.sum(),
    )


@torch.no_grad()
def quantize_nvfp4_activation(
    value: torch.Tensor,
    tensor_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard dynamic-block NVFP4 activation QDQ used around QUASAR weights."""

    if value.ndim < 1:
        raise ValueError("NVFP4 activation must have at least one dimension")
    in_features = value.shape[-1]
    padded_in = ((in_features + NVFP4_BLOCK_SIZE - 1) // NVFP4_BLOCK_SIZE) * NVFP4_BLOCK_SIZE
    value_f = value.detach().float()
    if padded_in != in_features:
        value_f = F.pad(value_f, (0, padded_in - in_features))
    blocks = value_f.reshape(-1, padded_in // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)

    if tensor_scale is None:
        tensor_scale = _positive_tensor_scale(blocks.abs().amax()).reshape(1)
    else:
        tensor_scale = tensor_scale.detach().float().reshape(1).to(value.device)
        if not torch.isfinite(tensor_scale).all() or not (tensor_scale > 0).all():
            raise ValueError("NVFP4 activation tensor_scale must be finite and positive")

    block_amax = blocks.abs().amax(dim=-1)
    local_scale = block_amax / NVFP4_E2M1_MAX / tensor_scale
    local_scale = torch.where(block_amax > 0, local_scale, torch.ones_like(local_scale))
    local_scale = _fp8_scale(local_scale).float()
    effective_scale = local_scale * tensor_scale
    codes = e2m1_encode(blocks / effective_scale.unsqueeze(-1))
    reconstructed = e2m1_decode(codes) * effective_scale.unsqueeze(-1)
    reconstructed = reconstructed.reshape(*value_f.shape)[..., :in_features]
    return reconstructed.to(value.dtype), tensor_scale


class QuasarNVFP4Linear(nn.Module):
    """Training Linear: QUASAR weight reconstruction with weight/activation STE."""

    def __init__(self, linear: nn.Linear, config: QuasarNVFP4Config):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.config = config
        self.register_buffer("input_tensor_scale", torch.zeros(1, dtype=torch.float32))
        self._input_scale_initialized = False
        self._saliency: torch.Tensor | None = None
        self._cached_weight_version = -1
        self._cached_reconstruction: NVFP4Weight | None = None

    def set_saliency(self, saliency: torch.Tensor | None) -> None:
        if saliency is not None and saliency.shape != self.weight.shape:
            raise ValueError("AdamW exp_avg_sq shape does not match the Linear weight")
        self._saliency = saliency
        self.clear_cache()

    def clear_cache(self) -> None:
        self._cached_weight_version = -1
        self._cached_reconstruction = None

    def reconstruction(self) -> NVFP4Weight:
        version = self.weight._version
        if self._cached_reconstruction is None or version != self._cached_weight_version:
            self._cached_reconstruction = reconstruct_nvfp4(
                self.weight, self._saliency, self.config
            )
            self._cached_weight_version = version
        return self._cached_reconstruction

    def _quantize_input(self, value: torch.Tensor) -> torch.Tensor:
        if not self.config.quantize_activations:
            return value
        scale = self.input_tensor_scale if self._input_scale_initialized else None
        reconstructed, observed_scale = quantize_nvfp4_activation(value, scale)
        if scale is None:
            self.input_tensor_scale.copy_(observed_scale)
            self._input_scale_initialized = True
        return value + (reconstructed - value).detach()

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        super()._load_from_state_dict(*args, **kwargs)
        self._input_scale_initialized = bool(self.input_tensor_scale.item() > 0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        quantized_input = self._quantize_input(value)
        reconstructed = self.reconstruction().dequantize(self.weight.dtype)
        ste_weight = self.weight + (reconstructed - self.weight).detach()
        return F.linear(quantized_input, ste_weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"activation_nvfp4={self.config.quantize_activations}"
        )


class PackedNVFP4Linear(nn.Module):
    """Reference runtime for a materialized compressed-tensors NVFP4 Linear."""

    def __init__(self, source: QuasarNVFP4Linear):
        super().__init__()
        reconstructed = source.reconstruction()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.activation_nvfp4 = source.config.quantize_activations
        if self.activation_nvfp4 and not (source.input_tensor_scale > 0).all():
            raise RuntimeError(
                "cannot materialize NVFP4 W4A4 before an activation calibration forward"
            )
        self.register_buffer("weight_packed", reconstructed.weight_packed)
        self.register_buffer("weight_scale", reconstructed.weight_scale)
        self.register_buffer("weight_global_scale", reconstructed.weight_global_scale)
        input_global_scale = (
            1.0 / source.input_tensor_scale.float()
            if self.activation_nvfp4
            else torch.ones(1, device=source.weight.device)
        )
        self.register_buffer("input_global_scale", input_global_scale.reshape(1))
        if source.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(source.bias.detach(), requires_grad=False)

    def dequantize_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        codes = unpack_e2m1(self.weight_packed)
        values = e2m1_decode(codes)
        blocks = values.reshape(self.out_features, -1, NVFP4_BLOCK_SIZE)
        effective_scale = self.weight_scale.float() / self.weight_global_scale.float()
        weight = (blocks * effective_scale.unsqueeze(-1)).reshape(self.out_features, -1)
        return weight[:, : self.in_features].to(dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.activation_nvfp4:
            value, _ = quantize_nvfp4_activation(value, 1.0 / self.input_global_scale)
        weight = self.dequantize_weight(value.dtype)
        return F.linear(value, weight, self.bias)


def _module_name_matches(name: str, patterns: Iterable[str]) -> bool:
    return any(name == pattern or name.endswith(f".{pattern}") for pattern in patterns)


class QuasarNVFP4Controller:
    """Own optimizer saliency binding and final materialization for a model."""

    def __init__(
        self,
        model: nn.Module,
        layers: dict[str, QuasarNVFP4Linear],
        config: QuasarNVFP4Config,
    ):
        self.model = model
        self.layers = layers
        self.config = config
        self._optimizer: torch.optim.Optimizer | None = None
        self._step_hook = None

    def parameters(self) -> list[nn.Parameter]:
        return [layer.weight for layer in self.layers.values()]

    def _refresh_saliency(self, optimizer: torch.optim.Optimizer) -> None:
        for layer in self.layers.values():
            state = optimizer.state.get(layer.weight, {})
            saliency = state.get("exp_avg_sq")
            layer.set_saliency(saliency if isinstance(saliency, torch.Tensor) else None)

    def bind_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Reuse AdamW's live second-moment tensors after every optimizer step."""

        optimizer_params = {id(param) for group in optimizer.param_groups for param in group["params"]}
        missing = [name for name, layer in self.layers.items() if id(layer.weight) not in optimizer_params]
        if missing:
            raise ValueError(f"optimizer is missing {len(missing)} QUASAR weights: {missing[:3]}")
        if self._step_hook is not None:
            self._step_hook.remove()
        self._optimizer = optimizer
        self._refresh_saliency(optimizer)

        def after_step(opt, _args, _kwargs):
            self._refresh_saliency(opt)

        self._step_hook = optimizer.register_step_post_hook(after_step)

    def clear_caches(self) -> None:
        for layer in self.layers.values():
            layer.clear_cache()

    @property
    def saliency_layer_count(self) -> int:
        return sum(layer._saliency is not None for layer in self.layers.values())

    def close(self) -> None:
        if self._step_hook is not None:
            self._step_hook.remove()
            self._step_hook = None


def prepare_quasar_nvfp4(
    model: nn.Module,
    config: QuasarNVFP4Config | None = None,
    *,
    freeze_non_quantized: bool = True,
) -> QuasarNVFP4Controller:
    """Wrap selected Linear modules and return the one training controller."""

    config = config or QuasarNVFP4Config()
    if freeze_non_quantized:
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    layers: dict[str, QuasarNVFP4Linear] = {}
    for name, module in list(model.named_modules()):
        if not name or not isinstance(module, nn.Linear):
            continue
        if isinstance(module, (QuasarNVFP4Linear, PackedNVFP4Linear)):
            continue
        if _module_name_matches(name, config.exclude_modules):
            continue
        if config.target_modules is not None and not _module_name_matches(
            name, config.target_modules
        ):
            continue

        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapped = QuasarNVFP4Linear(module, config)
        wrapped.weight.requires_grad_(True)
        if wrapped.bias is not None and freeze_non_quantized:
            wrapped.bias.requires_grad_(False)
        setattr(parent, child_name, wrapped)
        layers[name] = wrapped

    if not layers:
        raise ValueError("no Linear modules matched the QUASAR NVFP4 target/exclude rules")
    return QuasarNVFP4Controller(model, layers, config)


def build_nvfp4_quantization_config(
    ignored_layers: Iterable[str] = ("lm_head", "embed_tokens"),
) -> dict:
    """Return the vLLM compressed-tensors W4A4 checkpoint contract."""

    scheme = {
        "num_bits": 4,
        "group_size": NVFP4_BLOCK_SIZE,
        "strategy": "tensor_group",
        "symmetric": True,
        "type": "float",
    }
    return {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {**scheme, "dynamic": False},
                "input_activations": {**scheme, "dynamic": "local"},
                "output_activations": None,
            }
        },
        "ignore": list(ignored_layers),
        "kv_cache_scheme": None,
    }


def _harmonize_fused_scales(packed_layers: dict[str, PackedNVFP4Linear]) -> dict:
    """Share global scales required by fused qkv/gate-up deployment GEMMs."""

    grouped: dict[tuple[str, str], dict[str, tuple[str, PackedNVFP4Linear]]] = {}
    member_to_group = {
        member: group for group, members in _FUSED_PROJECTIONS.items() for member in members
    }
    for name, layer in packed_layers.items():
        member = name.rsplit(".", 1)[-1]
        group = member_to_group.get(member)
        if group is None:
            continue
        prefix = name[: -len(member)]
        grouped.setdefault((prefix, group), {})[member] = (name, layer)

    rebased_layers = 0
    max_relative_error = 0.0
    finfo = torch.finfo(torch.float8_e4m3fn)
    for (prefix, group), found in grouped.items():
        expected = _FUSED_PROJECTIONS[group]
        if set(found) != set(expected):
            missing = sorted(set(expected) - set(found))
            raise ValueError(f"incomplete NVFP4 {group} fusion group {prefix!r}: missing={missing}")

        layers = [found[member][1] for member in expected]
        input_scales = [layer.input_global_scale.float() for layer in layers]
        if any(not torch.equal(input_scales[0], scale) for scale in input_scales[1:]):
            raise ValueError(f"NVFP4 {group} fusion group {prefix!r} has mismatched input scales")

        tensor_scales = [1.0 / layer.weight_global_scale.float() for layer in layers]
        shared_tensor_scale = torch.stack(tensor_scales).max()
        shared_global_scale = (1.0 / shared_tensor_scale).reshape(1)
        for layer, tensor_scale in zip(layers, tensor_scales, strict=True):
            target = layer.weight_scale.float() * (tensor_scale / shared_tensor_scale)
            if (target < finfo.tiny).any() or (target > finfo.max).any():
                raise ValueError(
                    f"rebased NVFP4 {group} block scale falls outside the E4M3 range"
                )
            rebased = target.to(torch.float8_e4m3fn)
            relative_error = ((rebased.float() - target).abs() / target).max().item()
            if relative_error > _MAX_E4M3_RELATIVE_ROUNDING_ERROR:
                raise ValueError(
                    f"rebased NVFP4 {group} scale exceeds the E4M3 error bound: "
                    f"{relative_error:.6g}"
                )
            max_relative_error = max(max_relative_error, relative_error)
            layer.weight_scale.copy_(rebased)
            layer.weight_global_scale.copy_(shared_global_scale)
            rebased_layers += 1

    return {
        "fused_group_count": len(grouped),
        "fused_layer_count": rebased_layers,
        "max_local_scale_relative_error": max_relative_error,
    }


def materialize_quasar_nvfp4(controller: QuasarNVFP4Controller) -> dict:
    """Replace latent wrappers with packed modules that dequantize before BF16 GEMM."""

    if not controller.config.quantize_activations:
        raise ValueError("NVFP4 W4A4 packed materialization requires quantized activations")
    controller.close()
    layer_names = list(controller.layers)
    packed_layers: dict[str, PackedNVFP4Linear] = {}
    factor_counts = [0] * len(controller.config.clip_factors)
    external_clip_group_count = 0
    clip_value_count = 0
    clip_value_sum = 0.0
    clip_value_min = float("inf")
    clip_value_max = float("-inf")
    weighted_error = 0.0
    for name in layer_names:
        parent_name, _, child_name = name.rpartition(".")
        parent = controller.model.get_submodule(parent_name) if parent_name else controller.model
        reconstruction = controller.layers[name].reconstruction()
        factor_indices = reconstruction.clip_factor_index.reshape(-1).to(torch.long)
        grid_indices = factor_indices[factor_indices >= 0]
        counts = torch.bincount(grid_indices, minlength=len(factor_counts)).cpu()
        factor_counts = [old + int(new) for old, new in zip(factor_counts, counts, strict=True)]
        external_clip_group_count += int((factor_indices < 0).sum().item())
        clip_values = reconstruction.clip_factor_value.float()
        clip_value_count += clip_values.numel()
        clip_value_sum += clip_values.sum().item()
        clip_value_min = min(clip_value_min, clip_values.min().item())
        clip_value_max = max(clip_value_max, clip_values.max().item())
        weighted_error += reconstruction.weighted_error.item()
        packed = PackedNVFP4Linear(controller.layers[name])
        setattr(parent, child_name, packed)
        packed_layers[name] = packed

    fusion_summary = _harmonize_fused_scales(packed_layers)

    ignored_layers = tuple(dict.fromkeys((*controller.config.exclude_modules, "embed_tokens")))
    quantization_config = build_nvfp4_quantization_config(ignored_layers)
    model_config = getattr(controller.model, "config", None)
    if model_config is not None:
        if hasattr(model_config, "update"):
            model_config.update({"quantization_config": quantization_config, "use_cache": True})
        else:
            model_config.quantization_config = quantization_config

    return {
        "algorithm": "quasar",
        "format": "nvfp4-pack-quantized",
        "weight_format": "e2m1",
        "block_scale_format": "e4m3",
        "block_size": NVFP4_BLOCK_SIZE,
        "reconstruction_group_size": controller.config.reconstruction_group_size,
        "activation_format": "nvfp4" if controller.config.quantize_activations else "bf16",
        "quant_representation": "packed_e2m1_with_e4m3_block_scale",
        "compute_kernel": "bf16_dequantized_linear",
        "native_compute": False,
        "fallback_reason": "PackedNVFP4Linear dequantizes before torch.nn.functional.linear",
        "clip_factors": list(controller.config.clip_factors),
        "clip_factor_counts": factor_counts,
        "external_clip_group_count": external_clip_group_count,
        "clip_factor_value_summary": {
            "count": clip_value_count,
            "min": clip_value_min,
            "max": clip_value_max,
            "mean": clip_value_sum / clip_value_count,
        },
        "weighted_reconstruction_error": weighted_error,
        "fusion_scale_harmonization": fusion_summary,
        "layers": layer_names,
        "quantization_config": quantization_config,
    }
