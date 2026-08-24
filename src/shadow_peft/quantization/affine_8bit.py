"""Affine 8-bit (INT8) scheme: the shadow-side post-training quantization row.

Direction ① trains every shadow in BF16 and PTQ's the shadow weights to INT8
at evaluation time (the grid's shadow-{BF16, INT8} columns share one training).
Like the 1-bit affine and 2-bit ternary rows, this is one table entry — the
parametric modules, the walker, and the checkpoint save/load pair are shared.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .modules import QuantizedEmbedding as _QuantizedEmbedding
from .modules import QuantizedLinear as _QuantizedLinear
from .modules import pin_scheme_classes
from .scheme import QuantScheme, register_scheme, unpack_codes, wrap_int32
from .walker import load_quantized_checkpoint as _load_quantized_checkpoint  # noqa: F401
from .walker import quantize_model as _quantize_model
from .walker import save_quantized_checkpoint  # noqa: F401  # re-exported for import compatibility

_EPS = 1e-8
_LEVELS = 255.0  # max code value: codes span [0, 255]


def _check_dims(in_features: int, group_size: int) -> None:
    if group_size < 32 or group_size % 32 != 0:
        raise ValueError(f"group_size must be a positive multiple of 32, got {group_size}")
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        )


def quantize_8bit_affine(
    weight: torch.Tensor, group_size: int = 128, refine_iters: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a 2D weight `[out, in]` to packed 8-bit affine form.

    Per group of `group_size` values along the last dim:
      w_min = min(w), w_max = max(w), scale = (w_max - w_min) / 255, bias = w_min
      q = round((w - w_min) / scale) in [0, 255]; dequant: w_hat = scale * q + bias

    Roundtrip error is bounded by the half step (scale / 2) before fp16
    scale/bias storage. `refine_iters` is accepted for interface parity with
    the low-bit schemes and ignored: at 256 levels the min/max grid is already
    within noise of the Lloyd solution, so refinement has nothing to recover.

    Codes are packed 4 per int32 word, LSB-first (value j occupies bits
    8*(j%4) .. 8*(j%4)+7). Returns `(packed_int32 [out, in/4], scales [out,
    in/group_size], biases [out, in/group_size])`; scales/biases keep the
    input dtype (non-fp16/bf16 inputs are cast to fp16 first).
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D [out, in], got shape {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    _check_dims(in_features, group_size)

    dtype = weight.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        weight = weight.to(torch.float16)
        dtype = weight.dtype

    groups = weight.reshape(out_features, in_features // group_size, group_size).float()
    w_min = groups.amin(dim=-1)
    w_max = groups.amax(dim=-1)
    scales = ((w_max - w_min) / _LEVELS).clamp_min(_EPS)
    biases = w_min

    q = torch.round((groups - biases.unsqueeze(-1)) / scales.unsqueeze(-1))
    q = q.clamp_(0, _LEVELS).to(torch.int64)
    q = q.reshape(out_features, in_features // 4, 4)
    powers = 256 ** torch.arange(4, dtype=torch.int64, device=weight.device)
    packed = (q * powers).sum(dim=-1)  # values in [0, 2^32)
    packed = wrap_int32(packed)  # wrap to int32 bit pattern
    return packed, scales.to(dtype), biases.to(dtype)


def dequantize_8bit_affine(
    packed: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Inverse of `quantize_8bit_affine`: unpack `packed [out, in/4]` (int32,
    LSB-first, 4 codes per word) and broadcast `scale * q + bias` per group.
    Output dtype follows `scales`.
    """
    out_features = packed.shape[0]
    in_features = packed.shape[1] * 4
    codes = unpack_codes(packed, 8)
    q = codes.reshape(out_features, in_features // group_size, group_size).to(scales.dtype)
    w = scales.unsqueeze(-1) * q + biases.unsqueeze(-1)
    return w.reshape(out_features, in_features)


def _quantize(weight, group_size, refine_iters=0, thr_ratio=0.5):
    # refine_iters/thr_ratio are absorbed for interface parity with the other rows.
    return quantize_8bit_affine(weight, group_size)


AFFINE_8BIT: QuantScheme = register_scheme(
    QuantScheme(
        bits=8,
        label="affine_8bit",
        levels=256,
        words_per_int=4,
        group_multiple=32,
        ternary=False,
        quantize=_quantize,
        dequantize=dequantize_8bit_affine,
    )
)


class QuantizedLinear8Bit(_QuantizedLinear):
    """Packed 8-bit affine Linear: the parametric module with the scheme pinned."""

    scheme = AFFINE_8BIT


class QuantizedEmbedding8Bit(_QuantizedEmbedding):
    """Packed 8-bit affine Embedding: the parametric module with the scheme pinned."""

    scheme = AFFINE_8BIT


pin_scheme_classes(8, QuantizedLinear8Bit, QuantizedEmbedding8Bit)


def quantize_model_8bit(
    model: nn.Module,
    group_size: int = 128,
    skip_patterns: tuple[str, ...] | list[str] = (),
    refine_iters: int = 0,
    trim_vocab_to: int | None = None,
) -> dict[str, Any]:
    """
    Replace every `nn.Linear`/`nn.Embedding` in `model` (in place) with its
    8-bit quantized counterpart. Returns a JSON-serializable manifest.

    See `shadow_peft.quantization.walker.quantize_model` for the full contract.
    """
    return _quantize_model(
        model,
        AFFINE_8BIT,
        group_size=group_size,
        skip_patterns=skip_patterns,
        refine_iters=refine_iters,
        trim_vocab_to=trim_vocab_to,
    )
