"""
Ternary 2-bit affine quantization: weights are projected to ``{-1, 0, +1}``
with a per-group scale, then packed into 2-bit slots.

Follows the BitNet b1.58 / Bonsai ternary g128 recipe (arXiv:2504.12285):

  - Per group of ``group_size`` weights: ``s_g = mean(|w|)``, then
    ``t_i = 0`` if ``|w_i| < thr_ratio * s_g`` else ``sign(w_i)``.
  - Least-squares refit of ``s_g = sum(w*t) / sum(t*t)`` (BitNet absmean
    init, then Lloyd-style alternation when ``refine_iters > 0``).

Code/affine mapping (the only valid one for 3 ternary levels in a 4-level
2-bit affine grid)::

    ternary  code   dequant (scale=s_g, bias=-s_g)
      -1  -->  0  --> -s_g
       0  -->  1  -->  0
      +1  -->  2  --> +s_g
               3  --> unused (safety-clamped to 2)

Bits are packed 16 per word, LSB-first (weight j occupies bits 2*(j%16)
and 2*(j%16)+1), stored as int32 bit patterns. Returns
``(packed_int32 [out, in/16], scales [out, in/group_size], biases [out,
in/group_size])``; scales/biases keep the input dtype (non-fp16/bf16 inputs
are cast to fp16 first).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .modules import QuantizedEmbedding as _QuantizedEmbedding
from .modules import QuantizedLinear as _QuantizedLinear
from .modules import pin_scheme_classes
from .scheme import QuantScheme, register_scheme, unpack_codes, wrap_int32
from .walker import quantize_model as _quantize_model

_EPS = 1e-8


def _check_dims(in_features: int, group_size: int) -> None:
    if group_size < 16 or group_size % 16 != 0:
        raise ValueError(f"group_size must be a positive multiple of 16, got {group_size}")
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        )


def quantize_2bit_ternary(
    weight: torch.Tensor,
    group_size: int = 128,
    refine_iters: int = 0,
    thr_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a 2D weight ``[out, in]`` to packed ternary 2-bit affine form.

    Per group of ``group_size`` values along the last dim:
      s_g = mean(|w|);  t = 0 if |w| < thr_ratio*s_g else sign(w)
      dequant: w_hat = s_g * t, stored as scale=s_g, bias=-s_g.

    With ``refine_iters > 0``, ``s_g`` is refined by Lloyd-style alternation
    (least-squares refit, then re-threshold), which reduces group MSE.

    Bits are packed 16 per word, LSB-first. Returns ``(packed_int32 [out,
    in/16], scales [out, in/group_size], biases [out, in/group_size])``.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D [out, in], got shape {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    _check_dims(in_features, group_size)
    if refine_iters < 0:
        raise ValueError(f"refine_iters must be >= 0, got {refine_iters}")
    if not 0.0 < thr_ratio <= 1.0:
        raise ValueError(f"thr_ratio must be in (0, 1], got {thr_ratio}")

    dtype = weight.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        weight = weight.to(torch.float16)
        dtype = weight.dtype

    # Group stats in fp32 so the eps clamp survives (1e-8 underflows fp16).
    groups = weight.reshape(out_features, in_features // group_size, group_size).float()

    # absmean init (BitNet b1.58 style).
    scales = groups.abs().mean(dim=-1).clamp_min(_EPS)

    # Ternary assignment: |w| < thr*s -> 0, else sign(w).
    thr = thr_ratio * scales.unsqueeze(-1)
    ternary = torch.zeros_like(groups)
    ternary[groups > thr] = 1.0
    ternary[groups < -thr] = -1.0

    if refine_iters > 0:
        for _ in range(refine_iters):
            denom = (ternary * ternary).sum(dim=-1).clamp_min(1.0)
            new_scales = (groups * ternary).sum(dim=-1) / denom
            new_scales = new_scales.abs().clamp_min(_EPS)
            scales = new_scales
            thr = thr_ratio * scales.unsqueeze(-1)
            ternary = torch.zeros_like(groups)
            ternary[groups > thr] = 1.0
            ternary[groups < -thr] = -1.0

    # Map ternary {-1,0,+1} -> code {0,1,2}. Code 3 is never produced.
    code = (ternary + 1.0).to(torch.int64)  # -1->0, 0->1, +1->2

    # Pack 16 2-bit codes per int32 word, LSB-first.
    n_words = in_features // 16
    code = code.reshape(out_features, in_features)
    code = code.reshape(out_features, n_words, 16)
    shifts = 2 ** (2 * torch.arange(16, dtype=torch.int64, device=weight.device))
    packed = (code * shifts).sum(dim=-1)  # values in [0, 2^32)
    packed = wrap_int32(packed)  # wrap to int32 bit pattern

    # Affine params: scale=s_g, bias=-s_g  =>  dequant = s_g * (code-1) = s_g * ternary
    biases = (-scales).to(dtype)

    return packed, scales.to(dtype), biases


def dequantize_2bit_ternary(
    packed: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Inverse of :func:`quantize_2bit_ternary`: unpack ``packed [out, in/16]``
    (int32, LSB-first, 16 2-bit codes per word) and broadcast
    ``scale * q + bias`` per group. Output dtype follows ``scales``.
    """
    out_features = packed.shape[0]
    in_features = packed.shape[1] * 16
    codes = unpack_codes(packed, 2)
    codes = codes.reshape(out_features, in_features // group_size, group_size).to(scales.dtype)
    w = scales.unsqueeze(-1) * codes + biases.unsqueeze(-1)
    return w.reshape(out_features, in_features)


TERNARY_2BIT: QuantScheme = register_scheme(
    QuantScheme(
        bits=2,
        label="ternary_2bit",
        levels=3,
        words_per_int=16,
        group_multiple=16,
        ternary=True,
        quantize=quantize_2bit_ternary,
        dequantize=dequantize_2bit_ternary,
    )
)


class QuantizedLinear2Bit(_QuantizedLinear):
    """Packed ternary 2-bit Linear: the parametric module with the scheme pinned."""

    scheme = TERNARY_2BIT


class QuantizedEmbedding2Bit(_QuantizedEmbedding):
    """Packed ternary 2-bit Embedding: the parametric module with the scheme pinned."""

    scheme = TERNARY_2BIT


pin_scheme_classes(2, QuantizedLinear2Bit, QuantizedEmbedding2Bit)


def quantize_model_2bit(
    model: nn.Module,
    group_size: int = 128,
    skip_patterns: tuple[str, ...] | list[str] = (),
    refine_iters: int = 0,
    trim_vocab_to: int | None = None,
    thr_ratio: float = 0.5,
) -> dict[str, Any]:
    """
    Replace every ``nn.Linear`` / ``nn.Embedding`` in ``model`` (in place) with
    its ternary 2-bit quantized counterpart. Returns a JSON-serializable manifest.

    See `shadow_peft.quantization.walker.quantize_model` for the full contract
    (skip patterns, tied weights, divisibility skips, Lloyd refinement, vocab
    trimming).
    """
    return _quantize_model(
        model,
        TERNARY_2BIT,
        group_size=group_size,
        skip_patterns=skip_patterns,
        refine_iters=refine_iters,
        trim_vocab_to=trim_vocab_to,
        thr_ratio=thr_ratio,
    )
