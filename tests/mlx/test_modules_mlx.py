"""Unit tests for the MLX Shadow injection/update adapter modules."""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from shadow_peft_mlx.modules import ShadowInjectionModel, ShadowUpdateModel  # noqa: E402


def _zero_module(module) -> None:
    module.update(
        tree_unflatten([(k, mx.zeros_like(v)) for k, v in tree_flatten(module.parameters())])
    )


def test_injection_zero_init_is_identity():
    m = ShadowInjectionModel(
        num_layers=2, hidden_size=8, injection_hidden_size=4, dropout=0.0, alpha=0.1
    )
    m.eval()
    h = mx.random.normal((2, 5, 8))
    s = mx.random.normal((2, 5, 8))
    for idx in (0, 1):
        out = m(h, s, idx)
        assert mx.abs(out - h).max().item() == 0.0


def test_injection_math_manual():
    # D=4, k=2: downs select the first two dims, ups write them back; alpha=2.
    m = ShadowInjectionModel(
        num_layers=1, hidden_size=4, injection_hidden_size=2, dropout=0.0, alpha=2.0
    )
    m.eval()
    m.injection_downs = mx.array([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]])
    m.injection_ups = mx.array([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])

    h = mx.array([[[1.0, 2.0, 3.0, 4.0]]])
    s = mx.array([[[0.5, 0.5, 0.5, 0.5]]])
    delta = h - s
    expected = h + 2.0 * mx.concatenate([delta[..., :2], mx.zeros((1, 1, 2))], axis=-1)
    out = m(h, s, 0)
    assert mx.allclose(out, expected, atol=1e-6)


def test_injection_alpha_scales_delta():
    base = {
        "num_layers": 1,
        "hidden_size": 4,
        "injection_hidden_size": 2,
        "dropout": 0.0,
    }
    m1 = ShadowInjectionModel(alpha=1.0, **base)
    m2 = ShadowInjectionModel(alpha=2.0, **base)
    for m in (m1, m2):
        m.eval()
        m.injection_downs = mx.ones((1, 4, 2)) * 0.5
        m.injection_ups = mx.ones((1, 2, 4)) * 0.5
    h = mx.random.normal((1, 3, 4))
    s = mx.random.normal((1, 3, 4))
    d1 = m1(h, s, 0) - h
    d2 = m2(h, s, 0) - h
    assert mx.allclose(d2, 2.0 * d1, atol=1e-6)


def test_injection_rejects_bad_args():
    with pytest.raises(ValueError):
        ShadowInjectionModel(
            num_layers=0, hidden_size=4, injection_hidden_size=2, dropout=0.0, alpha=0.1
        )
    with pytest.raises(ValueError):
        ShadowInjectionModel(
            num_layers=1, hidden_size=4, injection_hidden_size=0, dropout=0.0, alpha=0.1
        )


def test_update_shape_and_zero_gate_math():
    m = ShadowUpdateModel(num_layers=2, hidden_size=8, gate_hidden_size=4, dropout=0.0)
    m.eval()
    h = mx.random.normal((2, 5, 8))
    s = mx.random.normal((2, 5, 8))
    out = m(h, s, 1)
    assert out.shape == s.shape

    # Zero the gate MLP for layer 0 -> pre-sigmoid activations are 0 -> g = 0.5.
    _zero_module(m.update_gates[0])
    h_in = m.hidden_norm(h)
    ht = m.update_transforms[0](h_in)
    expected = s + 0.5 * (ht - s)
    assert mx.allclose(m(h, s, 0), expected, atol=1e-6)


def test_update_rejects_bad_args():
    with pytest.raises(ValueError):
        ShadowUpdateModel(num_layers=0, hidden_size=4, gate_hidden_size=4, dropout=0.0)
    with pytest.raises(ValueError):
        ShadowUpdateModel(num_layers=1, hidden_size=4, gate_hidden_size=1, dropout=0.0)
