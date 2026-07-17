"""Tests for the MLX Shadow task wrappers (causal LM + sequence classification)."""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from shadow_peft_mlx import (  # noqa: E402
    ShadowForCausalLM,
    ShadowForSequenceClassification,
    get_shadow_model,
    train,
)
from shadow_peft_mlx.task_models import _shifted_ce_loss  # noqa: E402


def test_causallm_outputs_and_loss(llama_factory, shadow_cfg):
    task = ShadowForCausalLM(get_shadow_model(llama_factory(seed=10), shadow_cfg))
    task.eval()
    ids = mx.array([[1, 5, 7, 9, 3]])

    out = task(ids, labels=ids)
    assert out.logits.shape == (1, 5, 64)
    assert out.shadow_logits.shape == out.logits.shape
    assert out.loss is not None and mx.isfinite(out.loss).item()

    # Combined loss = base CE + shadow_loss_weight * shadow CE.
    base_ce = _shifted_ce_loss(out.logits, ids)
    shadow_ce = _shifted_ce_loss(out.shadow_logits, ids)
    expected = base_ce + task.shadow_loss_weight * shadow_ce
    assert mx.allclose(out.loss, expected, atol=1e-6)

    out_nolabels = task(ids)
    assert out_nolabels.loss is None


def test_causallm_shadow_only_mode(llama_factory, shadow_cfg):
    task = ShadowForCausalLM(
        get_shadow_model(llama_factory(seed=11), shadow_cfg), inference_mode="shadow_only"
    )
    task.eval()
    ids = mx.array([[1, 5, 7, 9, 3]])

    out = task(ids, labels=ids)
    assert mx.abs(out.logits - out.shadow_logits).max().item() == 0.0
    # shadow_only loss is exactly the shadow CE.
    assert mx.allclose(out.loss, _shifted_ce_loss(out.shadow_logits, ids), atol=1e-6)

    task.set_inference_mode("base_shadow")
    out2 = task(ids)
    assert not mx.allclose(out2.logits, out2.shadow_logits, atol=1e-4)


def test_shifted_ce_ignores_minus_100():
    logits = mx.random.normal((1, 4, 16))
    labels_full = mx.array([[1, 2, 3, 4]])
    labels_masked = mx.array([[-100, -100, 3, 4]])

    def ce_on_tail(logits, labels):
        return _shifted_ce_loss(logits[:, -3:, :], labels[:, -3:])

    # Masking the first two positions == scoring only the tail.
    a = _shifted_ce_loss(logits, labels_masked)
    b = ce_on_tail(logits, labels_full)
    assert mx.allclose(a, b, atol=1e-6)


def test_gradients_only_on_shadow_params(llama_factory, shadow_cfg):
    task = ShadowForCausalLM(get_shadow_model(llama_factory(seed=12), shadow_cfg))
    ids = mx.array([[1, 5, 7, 9, 3]])

    def lf(m, a, b):
        return m(a, labels=b).loss

    _, grads = nn.value_and_grad(task, lf)(task, ids, ids)
    keys = [k for k, _ in tree_flatten(grads)]
    assert keys, "expected some trainable parameters"
    assert all(k.startswith("peft_model.shadow_") for k in keys), keys


def test_training_step_updates_only_adapters(llama_factory, shadow_cfg):
    base = llama_factory(seed=13)
    task = ShadowForCausalLM(get_shadow_model(base, shadow_cfg))
    ids = mx.array([[1, 5, 7, 9, 3]])

    embed_before = base.model.embed_tokens.weight * 1
    ups_before = task.peft_model.shadow_injection_model.injection_ups * 1
    assert mx.abs(ups_before).max().item() == 0.0  # zero-init

    history = train(task, [(ids, ids)], lr=1e-2, epochs=6, log_every=0)

    # Loss decreases on a fixed batch (overfit smoke test).
    assert history[-1][1] < history[0][1]
    # Injection ups moved off zero; frozen base embeddings unchanged.
    ups_after = task.peft_model.shadow_injection_model.injection_ups
    assert mx.abs(ups_after - ups_before).max().item() > 0.0
    assert mx.abs(base.model.embed_tokens.weight - embed_before).max().item() == 0.0


def test_seqcls_outputs_and_loss(llama_factory, shadow_cfg):
    cls = ShadowForSequenceClassification(
        get_shadow_model(llama_factory(seed=14), shadow_cfg), num_labels=4
    )
    cls.eval()
    ids = mx.array([[1, 5, 7, 9, 3]])
    labels = mx.array([2])

    out = cls(ids, labels=labels)
    assert out.logits.shape == (1, 4)
    assert out.shadow_logits.shape == (1, 4)
    assert out.loss is not None and mx.isfinite(out.loss).item()

    expected = nn.losses.cross_entropy(
        out.logits, labels, reduction="mean"
    ) + cls.shadow_loss_weight * nn.losses.cross_entropy(
        out.shadow_logits, labels, reduction="mean"
    )
    assert mx.allclose(out.loss, expected, atol=1e-6)

    # Both heads trainable by default; base stays frozen.
    trainable_keys = {k for k, _ in tree_flatten(cls.trainable_parameters())}
    assert any(k.startswith("classifier_head.") for k in trainable_keys)
    assert any(k.startswith("shadow_classifier_head.") for k in trainable_keys)
    assert not tree_flatten(cls.peft_model.base_model.trainable_parameters())


def test_seqcls_shadow_only_mode(llama_factory, shadow_cfg):
    cls = ShadowForSequenceClassification(
        get_shadow_model(llama_factory(seed=15), shadow_cfg),
        num_labels=3,
        inference_mode="shadow_only",
    )
    cls.eval()
    ids = mx.array([[1, 5, 7, 9, 3]])
    out = cls(ids, labels=mx.array([1]))
    assert mx.abs(out.logits - out.shadow_logits).max().item() == 0.0
    assert out.loss is not None
