from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .modules import QuantizedEmbedding as _QuantizedEmbedding
from .modules import QuantizedLinear as _QuantizedLinear
from .modules import pin_scheme_classes
from .scheme import QuantScheme, register_scheme, unpack_codes
from .walker import load_quantized_checkpoint as _load_quantized_checkpoint
from .walker import quantize_model as _quantize_model
from .walker import save_quantized_checkpoint  # noqa: F401  # re-exported for import compatibility

_EPS = 1e-8


def _check_dims(in_features: int, group_size: int) -> None:
    if group_size < 32 or group_size % 32 != 0:
        raise ValueError(f"group_size must be a positive multiple of 32, got {group_size}")
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        )


def quantize_1bit_affine(
    weight: torch.Tensor, group_size: int = 128, refine_iters: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a 2D weight `[out, in]` to packed 1-bit affine form.

    Per group of `group_size` values along the last dim:
      w_min = min(w), w_max = max(w), scale = max(w_max - w_min, eps), bias = w_min
      q = round((w - w_min) / scale) in {0, 1}; dequant: w_hat = scale * q + bias

    With `refine_iters > 0`, the min/max assignment is improved by Lloyd-style
    alternation (least-squares fit of the two levels, then rethreshold at their
    midpoint). Group MSE decreases monotonically; outliers are clipped onto the
    nearer level instead of stretching the grid. Output format is unchanged.

    Bits are packed 32 per word, LSB-first (weight j occupies bit j % 32), stored
    as int32 bit patterns. Returns `(packed_int32 [out, in/32], scales [out,
    in/group_size], biases [out, in/group_size])`; scales/biases keep the input
    dtype (non-fp16/bf16 inputs are cast to fp16 first).
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D [out, in], got shape {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    _check_dims(in_features, group_size)
    if refine_iters < 0:
        raise ValueError(f"refine_iters must be >= 0, got {refine_iters}")

    dtype = weight.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        weight = weight.to(torch.float16)
        dtype = weight.dtype

    # Group stats in fp32 so the eps clamp survives (1e-8 underflows fp16).
    groups = weight.reshape(out_features, in_features // group_size, group_size).float()
    w_min = groups.amin(dim=-1)
    w_max = groups.amax(dim=-1)
    scales = (w_max - w_min).clamp_min(_EPS)
    biases = w_min

    if refine_iters > 0:
        bits = groups > (biases.unsqueeze(-1) + 0.5 * scales.unsqueeze(-1))
        for _ in range(refine_iters):
            qf = bits.float()
            n1 = qf.sum(dim=-1)
            n0 = group_size - n1
            sum1 = (groups * qf).sum(dim=-1)
            sum0 = groups.sum(dim=-1) - sum1
            mean1 = sum1 / n1.clamp_min(1)
            mean0 = sum0 / n0.clamp_min(1)
            new_scale = mean1 - mean0
            valid = (n1 > 0) & (n0 > 0) & (new_scale > _EPS)
            scales = torch.where(valid, new_scale, scales)
            biases = torch.where(valid, mean0, biases)
            bits = groups > (biases.unsqueeze(-1) + 0.5 * scales.unsqueeze(-1))
        q = bits.to(torch.int64)
    else:
        q = torch.round((groups - w_min.unsqueeze(-1)) / scales.unsqueeze(-1)).to(torch.int64)

    q = q.reshape(out_features, in_features // 32, 32)
    powers = 2 ** torch.arange(32, dtype=torch.int64, device=weight.device)
    packed = (q * powers).sum(dim=-1)  # values in [0, 2^32)
    packed = (packed & 0xFFFFFFFF).to(torch.int32)  # wrap to int32 bit pattern
    return packed, scales.to(dtype), biases.to(dtype)


def dequantize_1bit_affine(
    packed: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Inverse of `quantize_1bit_affine`: unpack `packed [out, in/32]` (int32,
    LSB-first) and broadcast `scale * q + bias` per group. Output dtype follows
    `scales`.
    """
    out_features = packed.shape[0]
    in_features = packed.shape[1] * 32
    bits = unpack_codes(packed, 1)
    q = bits.reshape(out_features, in_features // group_size, group_size).to(scales.dtype)
    w = scales.unsqueeze(-1) * q + biases.unsqueeze(-1)
    return w.reshape(out_features, in_features)


def _quantize(weight, group_size, refine_iters=0, thr_ratio=0.5):
    # thr_ratio is absorbed for interface parity with ternary schemes.
    return quantize_1bit_affine(weight, group_size, refine_iters=refine_iters)


AFFINE_1BIT: QuantScheme = register_scheme(
    QuantScheme(
        bits=1,
        label="affine_1bit",
        levels=2,
        words_per_int=32,
        group_multiple=32,
        ternary=False,
        quantize=_quantize,
        dequantize=dequantize_1bit_affine,
    )
)


class QuantizedLinear1Bit(_QuantizedLinear):
    """Packed 1-bit affine Linear: the parametric module with the scheme pinned."""

    scheme = AFFINE_1BIT


class QuantizedEmbedding1Bit(_QuantizedEmbedding):
    """Packed 1-bit affine Embedding: the parametric module with the scheme pinned."""

    scheme = AFFINE_1BIT


pin_scheme_classes(1, QuantizedLinear1Bit, QuantizedEmbedding1Bit)


def quantize_model_1bit(
    model: nn.Module,
    group_size: int = 128,
    skip_patterns: tuple[str, ...] | list[str] = (),
    refine_iters: int = 0,
    trim_vocab_to: int | None = None,
) -> dict[str, Any]:
    """
    Replace every `nn.Linear`/`nn.Embedding` in `model` (in place) with its
    1-bit quantized counterpart. Returns a JSON-serializable manifest.

    See `shadow_peft.quantization.walker.quantize_model` for the full contract
    (skip patterns, tied weights, divisibility skips, Lloyd refinement, vocab
    trimming).
    """
    return _quantize_model(
        model,
        AFFINE_1BIT,
        group_size=group_size,
        skip_patterns=skip_patterns,
        refine_iters=refine_iters,
        trim_vocab_to=trim_vocab_to,
    )
