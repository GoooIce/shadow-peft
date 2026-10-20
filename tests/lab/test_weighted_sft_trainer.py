"""Weighted SFT (issue #33) — external-behavior tests.

Spec: ``docs/specs/restrain-lite-structured-flywheel.md`` §Implementation
Decisions 4 and Testing Decisions「权重缝」:

- per-sample CE is the mean over non-``-100`` tokens; the batch loss is
  ``sum(w_i * CE_i) / sum(w_i)``;
- weight-0 samples do not contribute; all-zero weights return a differentiable
  zero instead of NaN;
- all-1 weights reproduce ``_shifted_ce_loss`` bit for bit (same batch, same
  seed), so weighted SFT degenerates to the existing SFT baseline;
- ``WeightedSFTTrainer`` only changes ``compute_loss`` when a batch carries
  ``sample_weights``; without them it is a plain ``ShadowSFTTrainer``.

Synthetic logits and a tiny random Llama, CPU, fixed seeds.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("trl")  # shadow_lab.trainers imports TRL at module load.

import torch.nn.functional as F  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM, Trainer  # noqa: E402

from shadow_lab.eval import _shifted_ce_loss  # noqa: E402
from shadow_lab.trainers import (  # noqa: E402
    ShadowSFTTrainer,
    WeightedSFTTrainer,
    weighted_shifted_ce_loss,
)


def _synthetic_batch(
    batch: int = 3, seq: int = 6, vocab: int = 32, seed: int = 7
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    logits = torch.randn(batch, seq, vocab)
    labels = torch.randint(0, vocab, (batch, seq))
    labels[:, 0] = -100  # prompt position; dropped by the shift anyway
    return logits, labels


def _tiny_llama(vocab_size: int = 32, seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
    )
    return LlamaForCausalLM(cfg)


def _trainer_batch(vocab_size: int = 32, batch: int = 4, seq: int = 8):
    torch.manual_seed(21)
    input_ids = torch.randint(2, vocab_size, (batch, seq))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :2] = -100
    return input_ids, attention_mask, labels


# ---------------------------------------------------------------------------
# weighted_shifted_ce_loss
# ---------------------------------------------------------------------------


def test_all_one_weights_match_shifted_ce_loss_bit_for_bit():
    logits, labels = _synthetic_batch()
    weights = torch.ones(logits.size(0))

    weighted = weighted_shifted_ce_loss(logits, labels, weights)
    expected = _shifted_ce_loss(logits, labels)

    assert torch.equal(weighted, expected)


def test_zero_weight_samples_do_not_contribute():
    logits, labels = _synthetic_batch(batch=3)

    loss = weighted_shifted_ce_loss(logits, labels, torch.tensor([1.0, 0.0, 0.0]))

    expected = F.cross_entropy(
        logits[0:1, :-1, :].reshape(-1, logits.size(-1)),
        labels[0:1, 1:].reshape(-1),
        ignore_index=-100,
    )
    assert torch.allclose(loss, expected, atol=1e-6, rtol=1e-6)
    # Dropping the filtered samples from the batch gives the same loss.
    solo = weighted_shifted_ce_loss(logits[0:1], labels[0:1], torch.ones(1))
    assert torch.allclose(loss, solo, atol=1e-6, rtol=1e-6)


def test_weights_average_per_sample_ce_not_per_token():
    # Sample 0 has one supervised target, sample 1 has four: a per-sample mean
    # must weight the two samples equally rather than by token count.
    torch.manual_seed(3)
    vocab, seq = 8, 6
    logits = torch.randn(2, seq, vocab)
    labels = torch.full((2, seq), -100, dtype=torch.long)
    labels[0, 1] = 3
    labels[1, 1:5] = torch.tensor([1, 2, 3, 4])

    def sample_ce(i: int) -> torch.Tensor:
        return F.cross_entropy(
            logits[i : i + 1, :-1, :].reshape(-1, vocab),
            labels[i : i + 1, 1:].reshape(-1),
            ignore_index=-100,
        )

    weights = torch.tensor([0.25, 0.75])
    loss = weighted_shifted_ce_loss(logits, labels, weights)
    expected = (weights[0] * sample_ce(0) + weights[1] * sample_ce(1)) / weights.sum()

    assert torch.allclose(loss, expected, atol=1e-6, rtol=1e-6)

    # Tilting the weights moves the loss toward the heavier sample.
    heavy_first = weighted_shifted_ce_loss(logits, labels, torch.tensor([0.9, 0.1]))
    heavy_second = weighted_shifted_ce_loss(logits, labels, torch.tensor([0.1, 0.9]))
    assert (heavy_first - sample_ce(0)).abs() < (heavy_second - sample_ce(0)).abs()
    assert (heavy_second - sample_ce(1)).abs() < (heavy_first - sample_ce(1)).abs()


def test_all_zero_weights_return_a_differentiable_zero():
    logits, labels = _synthetic_batch(batch=3)
    logits.requires_grad_(True)

    loss = weighted_shifted_ce_loss(logits, labels, torch.zeros(3))

    assert loss.item() == 0.0
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize("weights", [torch.ones(2), torch.ones(3, 1)])
def test_sample_weights_must_be_one_scalar_per_sample(weights):
    logits, labels = _synthetic_batch(batch=3)
    with pytest.raises(ValueError, match="sample_weights"):
        weighted_shifted_ce_loss(logits, labels, weights)


def test_negative_weights_are_rejected_with_index_and_value():
    logits, labels = _synthetic_batch(batch=3)

    with pytest.raises(ValueError) as excinfo:
        weighted_shifted_ce_loss(logits, labels, torch.tensor([1.0, -0.5, 2.0]))

    message = str(excinfo.value)
    assert "sample_weights" in message
    assert "non-negative" in message
    assert "index 1" in message
    assert "-0.5" in message


# ---------------------------------------------------------------------------
# WeightedSFTTrainer.compute_loss
# ---------------------------------------------------------------------------


def _bare_trainer() -> WeightedSFTTrainer:
    """``WeightedSFTTrainer`` without SFTTrainer.__init__ (model-agnostic seam)."""
    trainer = object.__new__(WeightedSFTTrainer)
    trainer.teacher_model = None
    return trainer


def test_weighted_trainer_is_a_shadow_sft_trainer():
    assert issubclass(WeightedSFTTrainer, ShadowSFTTrainer)


def test_compute_loss_without_weights_falls_back_to_the_parent(monkeypatch):
    calls = {}

    def record(self, model, inputs, return_outputs=False, **kwargs):
        calls.update(model=model, inputs=dict(inputs), return_outputs=return_outputs, kwargs=kwargs)
        return torch.tensor(1.0)

    monkeypatch.setattr(ShadowSFTTrainer, "compute_loss", record)

    trainer = _bare_trainer()
    model = object()
    batch = {
        "input_ids": torch.zeros(1, 2, dtype=torch.long),
        "labels": torch.zeros(1, 2, dtype=torch.long),
        "sample_weights": None,
    }
    loss = trainer.compute_loss(model, batch, return_outputs=False, num_items_in_batch=None)

    assert loss.item() == 1.0
    assert calls["model"] is model
    assert "sample_weights" not in calls["inputs"]
    assert calls["inputs"]["labels"] is batch["labels"]
    assert calls["return_outputs"] is False
    assert calls["kwargs"] == {"num_items_in_batch": None}


def test_compute_loss_with_weights_returns_weighted_ce_and_outputs():
    model = _tiny_llama()
    model.eval()
    input_ids, attention_mask, labels = _trainer_batch()
    weights = torch.tensor([1.0, 0.0, 0.0, 0.0])

    trainer = _bare_trainer()
    loss, outputs = trainer.compute_loss(
        model,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sample_weights": weights,
        },
        return_outputs=True,
    )

    expected = weighted_shifted_ce_loss(outputs.logits, labels, weights)
    assert torch.equal(loss, expected)
    # The weights steer the loss away from the unweighted token mean.
    assert not torch.allclose(loss, _shifted_ce_loss(outputs.logits, labels))
    # Labels were consumed by the trainer, not by the model.
    assert outputs.loss is None


def test_compute_loss_with_weights_without_labels_raises():
    trainer = _bare_trainer()
    batch = {"input_ids": torch.zeros(1, 2, dtype=torch.long), "sample_weights": torch.ones(1)}
    with pytest.raises(ValueError, match="labels"):
        trainer.compute_loss(object(), batch)


def test_evaluation_loop_strips_sample_weights(monkeypatch):
    seen: list[dict] = []
    sentinel = object()

    def record(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        for batch in dataloader:
            seen.append(dict(batch))
        return sentinel

    monkeypatch.setattr(ShadowSFTTrainer, "evaluation_loop", record)

    batches = [
        {
            "input_ids": torch.zeros(1, 2, dtype=torch.long),
            "labels": torch.zeros(1, 2, dtype=torch.long),
            "sample_weights": torch.ones(1),
        },
        {
            "input_ids": torch.ones(1, 2, dtype=torch.long),
            "labels": torch.ones(1, 2, dtype=torch.long),
        },
    ]

    result = _bare_trainer().evaluation_loop(batches, "eval")

    assert result is sentinel
    assert "sample_weights" not in seen[0]
    assert set(seen[0]) == {"input_ids", "labels"}
    assert set(seen[1]) == {"input_ids", "labels"}


def test_prediction_step_strips_sample_weights(monkeypatch):
    seen: dict = {}
    sentinel = object()

    def record(self, model, inputs, prediction_loss_only, ignore_keys=None, **kwargs):
        seen["inputs"] = dict(inputs)
        return sentinel

    monkeypatch.setattr(Trainer, "prediction_step", record)

    batch = {
        "input_ids": torch.zeros(1, 2, dtype=torch.long),
        "sample_weights": torch.ones(1),
    }

    result = _bare_trainer().prediction_step(object(), batch, True)

    assert result is sentinel
    assert "sample_weights" not in seen["inputs"]
    assert "sample_weights" in batch  # the caller's mapping is not mutated


def test_two_weighted_optimizer_steps_stay_finite():
    model = _tiny_llama(seed=5)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    input_ids, attention_mask, labels = _trainer_batch()
    trainer = _bare_trainer()

    weight_sets = (torch.tensor([1.0, 0.5, 0.0, 2.0]), torch.tensor([0.0, 1.0, 1.0, 0.0]))
    for weights in weight_sets:
        optimizer.zero_grad()
        loss = trainer.compute_loss(
            model,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "sample_weights": weights,
            },
        )
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads
        assert all(torch.isfinite(grad).all() for grad in grads)
        optimizer.step()

    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_all_zero_weights_leave_the_model_unchanged():
    model = _tiny_llama(seed=6)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    input_ids, attention_mask, labels = _trainer_batch()
    before = [p.detach().clone() for p in model.parameters()]

    loss = _bare_trainer().compute_loss(
        model,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sample_weights": torch.zeros(labels.size(0)),
        },
    )
    assert loss.item() == 0.0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.count_nonzero(grad) == 0 for grad in grads)
    optimizer.step()
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before, strict=True))
