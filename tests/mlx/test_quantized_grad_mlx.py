from __future__ import annotations

import pytest

pytest.importorskip("mlx")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from shadow_peft_mlx.quantized_grad import quantized_linear_input_grad  # noqa: E402


def test_quantized_linear_input_vjp_matches_dequantized_reference():
    mx.random.seed(17)
    linear = nn.Linear(128, 32, bias=False)
    quantized = nn.QuantizedLinear.from_linear(linear, group_size=128, bits=2, mode="affine")
    x = mx.random.normal((2, 3, 128))

    dequantized = mx.dequantize(
        quantized.weight,
        quantized.scales,
        quantized.biases,
        group_size=quantized.group_size,
        bits=quantized.bits,
        mode=quantized.mode,
    )

    def reference(inputs):
        values = inputs @ dequantized.T
        return mx.sum(values * values)

    def candidate(inputs):
        values = quantized(inputs)
        return mx.sum(values * values)

    expected_loss, expected_grad = mx.value_and_grad(reference)(x)
    with quantized_linear_input_grad():
        actual_loss, actual_grad = mx.value_and_grad(candidate)(x)
    mx.eval(expected_loss, expected_grad, actual_loss, actual_grad)

    assert mx.allclose(actual_loss, expected_loss, rtol=2e-3, atol=2e-3)
    assert mx.allclose(actual_grad, expected_grad, rtol=3e-3, atol=3e-3)


def test_quantized_linear_input_grad_restores_original_call():
    original = nn.QuantizedLinear.__call__
    with quantized_linear_input_grad():
        assert nn.QuantizedLinear.__call__ is not original
    assert nn.QuantizedLinear.__call__ is original
