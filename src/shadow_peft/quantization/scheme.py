"""The quantization scheme table: one row per bit-width.

A scheme captures everything that varies between, e.g., the 1-bit binary
affine packing and the 2-bit ternary packing: level count, packing density,
group-size constraint, whether the grid is ternary, and the encode/decode
pair. Adding a 3-bit scheme means adding one row — not a third copy of the
module classes, the model walker, or the checkpoint save/load pair.

``unpack_codes`` is the single home for the shift-and-mask unpacking idiom;
tests and analysis scripts consume it instead of re-implementing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class QuantScheme:
    """Everything that differs between packing formats, in one place."""

    bits: int
    label: str  # manifest-facing name, e.g. "affine_1bit"
    levels: int  # 2 for binary, 3 for ternary
    words_per_int: int  # values packed per int32 word (32 / bits)
    group_multiple: int  # group_size must be a positive multiple of this
    ternary: bool
    quantize: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    dequantize: Callable[..., torch.Tensor]

    def check_dims(self, in_features: int, group_size: int) -> None:
        if group_size < self.group_multiple or group_size % self.group_multiple != 0:
            raise ValueError(
                f"group_size must be a positive multiple of {self.group_multiple}, got {group_size}"
            )
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by group_size ({group_size})"
            )


_SCHEMES: dict[int, QuantScheme] = {}


def register_scheme(scheme: QuantScheme) -> QuantScheme:
    if scheme.bits in _SCHEMES:
        raise ValueError(f"scheme for bits={scheme.bits} already registered")
    _SCHEMES[scheme.bits] = scheme
    return scheme


def get_scheme(bits: int) -> QuantScheme:
    try:
        return _SCHEMES[bits]
    except KeyError:
        raise ValueError(f"no quantization scheme registered for bits={bits}") from None


def unpack_codes(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Unpack int32 bit patterns into per-value codes, LSB-first.

    ``packed [out, words]`` -> ``[out, words * (32 // bits)]`` with codes in
    ``[0, 2**bits)``. The one implementation of the shift-and-mask idiom the
    whole repo (modules, tests, analysis scripts) should use.
    """
    if bits not in _SCHEMES and bits not in (1, 2, 3, 4, 8):
        raise ValueError(f"unsupported bits={bits}")
    words = 32 // bits
    shifts = torch.arange(words, dtype=torch.int32, device=packed.device) * bits
    return (packed.unsqueeze(-1) >> shifts) & (2**bits - 1)


def wrap_int32(values: torch.Tensor) -> torch.Tensor:
    """Reinterpret unsigned bit patterns in ``[0, 2**32)`` as int32 words.

    ``Tensor.to(torch.int32)`` raises on values >= 2^31, but a fully-packed
    word (e.g. an 8-bit code 250 sitting in the top byte) routinely exceeds
    it — the packing idiom needs the two's-complement bit pattern, not the
    value.
    """
    values = values & 0xFFFFFFFF
    return torch.where(values >= 0x80000000, values - 0x1_0000_0000, values).to(torch.int32)
