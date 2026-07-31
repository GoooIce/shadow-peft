"""Input gradients for frozen MLX quantized linear layers.

Prism's low-bit ``quantized_matmul`` is a custom kernel and intentionally has
no generic VJP. Shadow recovery keeps the packed base weights frozen but still
needs gradients with respect to layer inputs. For ``y = x @ W.T``, that VJP is
another quantized matmul: ``dx = dy @ W``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn


def _input_grad_quantized_linear(self: nn.QuantizedLinear, x: mx.array) -> mx.array:
    weight = self["weight"]
    scales = self["scales"]
    biases = self.get("biases")
    group_size = self.group_size
    bits = self.bits
    mode = self.mode

    @mx.custom_function
    def qmm(inputs: mx.array) -> mx.array:
        return mx.quantized_matmul(
            inputs,
            weight,
            scales=scales,
            biases=biases,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )

    @qmm.vjp
    def qmm_vjp(
        primals: tuple[mx.array],
        cotangent: mx.array,
        output: mx.array,
    ) -> mx.array:
        del primals, output
        return mx.quantized_matmul(
            cotangent,
            weight,
            scales=scales,
            biases=biases,
            transpose=False,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )

    result = qmm(x)
    if "bias" in self:
        result = result + self["bias"]
    return result


@contextmanager
def quantized_linear_input_grad() -> Iterator[None]:
    """Enable activation-only VJPs for frozen ``nn.QuantizedLinear`` modules.

    The packed weights, scales and biases are closure constants: this context
    does not make quantized parameters trainable. It is intended for adapter
    training through a frozen low-bit backbone.
    """
    original = nn.QuantizedLinear.__call__
    nn.QuantizedLinear.__call__ = _input_grad_quantized_linear
    try:
        yield
    finally:
        nn.QuantizedLinear.__call__ = original
