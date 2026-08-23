"""Parametric quantized modules: one implementation, scheme-pinned subclasses.

``QuantizedLinear(scheme=...)`` / ``QuantizedEmbedding(scheme=...)`` hide the
packing format behind the scheme table. The per-bit class names
(``QuantizedLinear1Bit`` etc., defined next to their schemes in
``affine_1bit`` / ``affine_2bit``) are subclasses that pin the scheme, so
``isinstance`` checks and import paths keep working.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scheme import QuantScheme

# Scheme-pinned subclasses register here so the walker constructs
# QuantizedLinear1Bit / 2Bit (not the anonymous parametric base), keeping
# isinstance checks on the per-bit class names working.
_PINNED_CLASSES: dict[int, tuple[type, type]] = {}


def pin_scheme_classes(bits: int, linear_cls: type, embedding_cls: type) -> None:
    _PINNED_CLASSES[bits] = (linear_cls, embedding_cls)


def classes_for_scheme(scheme: QuantScheme) -> tuple[type, type]:
    return _PINNED_CLASSES.get(scheme.bits, (QuantizedLinear, QuantizedEmbedding))


class QuantizedLinear(nn.Module):
    """
    Drop-in replacement for ``nn.Linear`` storing a packed quantized weight.

    The packed weight and group scales/biases are buffers (no gradient); the
    optional output bias is a frozen Parameter. In eval mode the dequantized
    weight is cached and reused; in train mode it is recomputed every forward
    (so packed weights stay bit-identical under training).
    """

    scheme: QuantScheme  # pinned by subclasses

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 128,
        *,
        scheme: QuantScheme | None = None,
    ) -> None:
        super().__init__()
        self.scheme = scheme or type(self).scheme
        self.scheme.check_dims(in_features, group_size)
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.register_buffer(
            "weight",
            torch.zeros(out_features, in_features // self.scheme.words_per_int, dtype=torch.int32),
        )
        self.register_buffer(
            "scales",
            torch.zeros(out_features, in_features // group_size, dtype=torch.float16),
        )
        self.register_buffer(
            "biases",
            torch.zeros(out_features, in_features // group_size, dtype=torch.float16),
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self._cached_weight: torch.Tensor | None = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 128,
        refine_iters: int = 0,
        thr_ratio: float = 0.5,
        scheme: QuantScheme | None = None,
    ) -> QuantizedLinear:
        scheme = scheme or cls.scheme
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
            scheme=scheme,
        )
        with torch.no_grad():
            packed, scales, biases = scheme.quantize(
                linear.weight.detach(),
                group_size,
                refine_iters=refine_iters,
                thr_ratio=thr_ratio,
            )
            module.weight = packed
            module.scales = scales
            module.biases = biases
            if linear.bias is not None:
                # Match the dequantized weight dtype so F.linear dtype-checks pass
                # even when the source linear was fp32 (weight becomes fp16).
                module.bias = nn.Parameter(
                    linear.bias.detach().clone().to(scales.dtype), requires_grad=False
                )
        return module

    def dequantize_weight(self) -> torch.Tensor:
        return self.scheme.dequantize(self.weight, self.scales, self.biases, self.group_size)

    def train(self, mode: bool = True) -> QuantizedLinear:
        if mode:
            self._cached_weight = None
        return super().train(mode)

    def _apply(self, fn, recurse=True):
        # Invalidate the cache when buffers move (`.to()`, `.cuda()`, ...).
        self._cached_weight = None
        return super()._apply(fn, recurse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.dequantize_weight()
        else:
            if self._cached_weight is None:
                self._cached_weight = self.dequantize_weight()
            weight = self._cached_weight
        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, group_size={self.group_size}, "
            f"scheme={self.scheme.label}"
        )


class QuantizedEmbedding(nn.Module):
    """Drop-in replacement for ``nn.Embedding`` storing a packed quantized weight."""

    scheme: QuantScheme

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        group_size: int = 128,
        *,
        scheme: QuantScheme | None = None,
    ) -> None:
        super().__init__()
        self.scheme = scheme or type(self).scheme
        self.scheme.check_dims(embedding_dim, group_size)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.group_size = group_size
        self.register_buffer(
            "weight",
            torch.zeros(num_embeddings, embedding_dim // self.scheme.words_per_int, dtype=torch.int32),
        )
        self.register_buffer(
            "scales",
            torch.zeros(num_embeddings, embedding_dim // group_size, dtype=torch.float16),
        )
        self.register_buffer(
            "biases",
            torch.zeros(num_embeddings, embedding_dim // group_size, dtype=torch.float16),
        )

    @classmethod
    def from_embedding(
        cls,
        embedding: nn.Embedding,
        group_size: int = 128,
        refine_iters: int = 0,
        thr_ratio: float = 0.5,
        scheme: QuantScheme | None = None,
    ) -> QuantizedEmbedding:
        scheme = scheme or cls.scheme
        module = cls(
            embedding.num_embeddings,
            embedding.embedding_dim,
            padding_idx=embedding.padding_idx,
            group_size=group_size,
            scheme=scheme,
        )
        with torch.no_grad():
            packed, scales, biases = scheme.quantize(
                embedding.weight.detach(),
                group_size,
                refine_iters=refine_iters,
                thr_ratio=thr_ratio,
            )
            module.weight = packed
            module.scales = scales
            module.biases = biases
        return module

    def dequantize_weight(self) -> torch.Tensor:
        return self.scheme.dequantize(self.weight, self.scales, self.biases, self.group_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Reference implementation: dequantize the whole table, then gather.
        # A gather-then-dequantize kernel would avoid materializing the table.
        weight = self.dequantize_weight()
        return F.embedding(input, weight, padding_idx=self.padding_idx)

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"padding_idx={self.padding_idx}, group_size={self.group_size}, "
            f"scheme={self.scheme.label}"
        )
