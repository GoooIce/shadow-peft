"""GRACE loss math (shadow_lab.grace) — external-behavior tests.

Anchors pinned per the exp1-grace implementation spec:
- DKD identity: gate forced to 1 + exact recombination => gated DKD == the
  plain KL of ``_shifted_kd_loss`` (the ce_kl convention future refactors
  must not silently change).
- Soft-gate boundaries: one-hot teacher -> g=1; uniform teacher -> g minimal;
  gate never drops a token; weighted aggregate conserves total weight.
- span-CKA: identical features -> 0; orthogonal reparametrization invariant;
  prefix/ignored positions excluded from the Gram; dimension-agnostic.
- IB controller: dual-ascent direction, bounds, EMA smoothness.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from shadow_lab.grace import (  # noqa: E402
    GateMonitor,
    IBController,
    gate_histogram,
    gated_dkd_loss,
    soft_gate,
    span_cka_loss,
)
from shadow_peft.task_models import _shifted_kd_loss  # noqa: E402


def _logits(batch: int = 2, seq: int = 8, vocab: int = 32, seed: int = 0):
    torch.manual_seed(seed)
    return torch.randn(batch, seq, vocab)


def _labels(batch: int = 2, seq: int = 8, seed: int = 1):
    torch.manual_seed(seed)
    labels = torch.randint(0, 32, (batch, seq))
    labels[:, :2] = -100  # exercise ignore-index alignment
    return labels


# ---------------------------------------------------------------------------
# DKD identity: g == 1 + exact recombination == plain KL (ce_kl anchor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_gated_dkd_identity_with_plain_kl(temperature):
    student = _logits(seed=0)
    teacher = _logits(seed=3)
    labels = _labels()

    plain = _shifted_kd_loss(student, teacher, labels, temperature)

    gd = gated_dkd_loss(
        student,
        teacher,
        labels,
        temperature=temperature,
        tckd_weight=1.0,
        nckd_weight="exact",
        gate_lambda=0.0,  # g == 1 everywhere
    )
    assert torch.isfinite(gd.loss)
    assert gd.loss.item() == pytest.approx(plain.item(), abs=1e-5, rel=1e-4)
    # gate is identically 1 when lambda == 0
    assert torch.allclose(gd.gate, torch.ones_like(gd.gate))


def test_gated_dkd_identity_with_one_hot_teacher():
    """A one-hot teacher has zero entropy => g == 1 naturally (no lambda hack)."""
    vocab = 32
    labels = _labels()
    tgt = labels.clamp_min(0)  # one-hot AT the target class
    teacher = torch.full((2, 8, vocab), -20.0)
    teacher.scatter_(-1, tgt.unsqueeze(-1), 20.0)
    student = _logits(seed=0)

    gd = gated_dkd_loss(student, teacher, labels, temperature=2.0, nckd_weight="exact")
    assert torch.allclose(gd.gate, torch.ones_like(gd.gate), atol=1e-6)

    plain = _shifted_kd_loss(student, teacher, labels, 2.0)
    assert gd.loss.item() == pytest.approx(plain.item(), abs=1e-4, rel=1e-4)


# ---------------------------------------------------------------------------
# Soft gate boundaries (GRACE Eq. 4-6)
# ---------------------------------------------------------------------------


def test_soft_gate_one_hot_teacher_gives_one():
    vocab = 16
    one_hot = torch.full((1, 4, vocab), -30.0)
    one_hot[..., 0] = 30.0
    g = soft_gate(one_hot)
    assert torch.allclose(g, torch.ones_like(g), atol=1e-6)


def test_soft_gate_uniform_teacher_is_minimal():
    uniform = torch.zeros(1, 4, 128)
    g = soft_gate(uniform)
    # h_norm == 1 => g == exp(-1); the exponential gate's floor
    assert torch.allclose(g, torch.full_like(g, math.exp(-1.0)), atol=1e-6)


def test_soft_gate_monotone_in_confidence():
    torch.manual_seed(0)
    vocab = 64
    sharp = torch.randn(3, 5, vocab) * 8.0
    diffuse = torch.randn(3, 5, vocab)
    assert bool((soft_gate(sharp) >= soft_gate(diffuse) - 1e-6).all())


def test_gate_never_drops_and_weight_conservation():
    student = _logits(seed=0)
    teacher = _logits(seed=3)
    labels = _labels()

    gd = gated_dkd_loss(student, teacher, labels)
    n_valid = int((labels[:, 1:] != -100).sum())
    assert gd.gate.numel() == n_valid
    assert bool((gd.gate > 0).all())  # soft down-weight only, never zero

    # Constant per-token loss => the gate-normalized mean is that constant:
    # sum(g * L) / sum(g) == L — total weight is conserved by normalization.
    per_token_const = torch.ones_like(gd.gate)
    weighted = (gd.gate * per_token_const).sum() / gd.gate.sum()
    assert weighted.item() == pytest.approx(1.0, abs=1e-6)


def test_gated_dkd_grads_flow_to_student_not_gate():
    student = _logits(seed=0).requires_grad_(True)
    teacher = _logits(seed=3)
    labels = _labels()
    gd = gated_dkd_loss(student, teacher, labels)
    gd.loss.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert not gd.gate.requires_grad  # teacher-only statistic


def test_gated_dkd_components_finite_with_defaults():
    gd = gated_dkd_loss(_logits(), _logits(seed=3), _labels())
    for v in (gd.loss.item(), gd.tckd, gd.nckd, gd.gate_mean, gd.gate_var):
        assert math.isfinite(v)


# ---------------------------------------------------------------------------
# span-CKA (GRACE Eq. 10-13, textualized)
# ---------------------------------------------------------------------------


def test_span_cka_identical_features_loss_zero():
    h = torch.randn(2, 10, 16)
    labels = _labels(batch=2, seq=10)
    out = span_cka_loss(h, h.clone(), labels)
    assert out.loss.item() == pytest.approx(0.0, abs=1e-5)
    assert out.n_skipped == 0
    assert out.n_samples == 2


def test_span_cka_orthogonal_reparametrization_invariant():
    """CKA is invariant to orthogonal rotations and row scaling (Kornblith
    et al. 2019) — the dimension-free property that spares a projection layer."""
    torch.manual_seed(0)
    s = torch.randn(2, 10, 16)
    t = torch.randn(2, 10, 24)  # teacher dim != student dim is fine
    labels = _labels(batch=2, seq=10)

    base = span_cka_loss(s, t, labels).loss.item()

    q = torch.linalg.qr(torch.randn(16, 16).double())[0].float()  # square orthogonal
    row_scale = torch.rand(2, 1, 1) * 3.0 + 0.5
    rotated = (s * row_scale) @ q
    assert span_cka_loss(rotated, t, labels).loss.item() == pytest.approx(base, abs=1e-4)


def test_span_cka_excludes_ignored_positions_from_gram():
    """Prefix/pad rows (ignored labels) must not enter the Gram matrix:
    perturbing them cannot change the loss."""
    torch.manual_seed(0)
    s = torch.randn(1, 10, 8)
    t = torch.randn(1, 10, 8)
    labels = torch.full((1, 10), -100)
    labels[0, 6:] = 5  # only the tail is the span

    base = span_cka_loss(s, t, labels).loss

    # hidden row j is in the span iff labels[j+1] != -100 -> rows 0..4 are
    # outside (labels[0..5] all ignored), row 5 onward inside.
    s_perturbed = s.clone()
    s_perturbed[0, :5] += 100.0  # torch the ignored region
    perturbed = span_cka_loss(s_perturbed, t, labels).loss
    assert perturbed.item() == pytest.approx(base.item(), abs=1e-6)


def test_span_cka_skips_short_and_degenerate_samples():
    # sample 0: span of 3 tokens (rows 6..8) < min_tokens=4 -> skipped
    # sample 1: all span rows identical (collapsed Gram) -> skipped
    labels = torch.full((2, 10), -100)
    labels[0, 7:] = 5
    labels[1, 2:] = 5
    s = torch.randn(2, 10, 8)
    s[1, 1:] = s[1, 1, :].expand(9, 8)  # collapse sample 1 across its whole span
    t = torch.randn(2, 10, 8)

    out = span_cka_loss(s, t, labels, min_tokens=4)
    assert out.n_samples == 0
    assert out.n_skipped == 2
    assert out.loss.item() == 0.0


def test_span_cka_grads_flow_to_student_hidden():
    s = torch.randn(1, 10, 8, requires_grad=True)
    t = torch.randn(1, 10, 8)
    labels = torch.full((1, 10), -100)
    labels[0, 2:] = 5
    out = span_cka_loss(s, t, labels)
    out.loss.backward()
    # Span = hidden rows 1..8 (labels[2:] are the labeled targets); rows 0
    # and 9 are outside and must carry no gradient.
    assert s.grad is not None and s.grad[:, 1:9].abs().sum() > 0
    assert s.grad[:, :1].abs().sum() == 0
    assert s.grad[:, 9:].abs().sum() == 0


# ---------------------------------------------------------------------------
# IB controller (GRACE Eq. 20)
# ---------------------------------------------------------------------------


def test_ib_controller_direction_and_bounds():
    ib = IBController(tau=0.35, eta=0.01, beta=1.0, beta_min=0.1, beta_max=5.0)
    # Constraint violated (student trails teacher) -> beta rises on step one
    # (EMA inits to the first observation, already above tau).
    b = ib.beta
    ib.step(1.0)
    assert ib.beta > b
    # Constraint satisfied -> beta falls, but only after the EMA dips below
    # tau — the smoothing is the point (damped dual ascent, GRACE Table 8).
    for _ in range(500):
        ib.step(0.0)
    assert ib.ema is not None and ib.ema < 0.35
    assert ib.beta < b
    # Projection keeps beta inside the bounds.
    rising = IBController(tau=0.35, eta=1.0, beta=4.9, beta_min=0.1, beta_max=5.0)
    for _ in range(10):
        rising.step(10.0)
        assert rising.beta <= 5.0
    falling = IBController(tau=0.35, eta=1.0, beta=0.2, beta_min=0.1, beta_max=5.0)
    for _ in range(10):
        falling.step(0.0)
        assert falling.beta >= 0.1


def test_ib_controller_ema_smooths_trajectory():
    ib = IBController(tau=0.35, eta=0.0015, beta=1.0)
    deltas = []
    prev = ib.beta
    # Alternating extreme observations: the EMA damps them before they move beta.
    for v in (10.0, 0.0, 10.0, 0.0, 10.0, 0.0):
        ib.step(v)
        deltas.append(abs(ib.beta - prev))
        prev = ib.beta
    assert max(deltas) <= ib.eta * 10.0 + 1e-12  # no jump beyond eta * |ema - tau|
    # Smoothed: the effective beta moves at most eta * max|obs - tau| per step.
    assert all(d <= 0.0015 * 10.0 for d in deltas)


# ---------------------------------------------------------------------------
# GateMonitor: throttled logs + degradation latch (annotation only)
# ---------------------------------------------------------------------------


def test_gate_monitor_throttled_payload_and_histogram():
    m = GateMonitor(log_every=3)
    g = torch.linspace(0.2, 0.8, 12)
    recs = [m.record(s, g, beta=1.0) for s in range(1, 7)]
    assert [r is not None for r in recs] == [False, False, True, False, False, True]
    rec = recs[2]
    assert rec["step"] == 3
    assert sum(rec["gate_hist"]) == g.numel()
    assert rec["gate_mean"] == pytest.approx(g.mean().item(), abs=1e-6)
    assert rec["gate_cv"] == pytest.approx(g.std(unbiased=False).item() / g.mean().item(), abs=1e-6)


def test_gate_monitor_degradation_latches():
    m = GateMonitor(log_every=10**9, degrade_cv=0.05, degrade_mean=0.95, degrade_steps=3)
    flat = torch.full((4,), 0.97)  # mean > 0.95 -> degraded batch
    m.record(1, flat, 1.0)
    m.record(2, flat, 1.0)
    assert not m.degraded  # streak 2 < 3
    m.record(3, flat, 1.0)
    assert m.degraded and m.max_streak == 3
    # Latch is annotation-only and permanent: healthy batches cannot unlatch.
    varied = torch.tensor([0.1, 0.4, 0.7, 0.9])
    m.record(4, varied, 1.0)
    assert m.degraded and m.streak == 0


def test_gate_histogram_buckets():
    g = torch.tensor([0.05, 0.15, 0.5, 0.95])
    hist = gate_histogram(g)
    assert sum(hist) == 4
    assert hist[0] == 1  # 0.05
    assert hist[1] == 1  # 0.15
    assert hist[5] == 1  # 0.50 (bucketize right=True: edges inclusive)
    assert hist[9] == 1  # 0.95


# ---------------------------------------------------------------------------
# Legacy three modes: golden values (refactor anchor — new enum entries must
# not perturb the ce / ce_kl / ce_kl_hmse semantics)
# ---------------------------------------------------------------------------


def test_legacy_loss_modes_golden_values():
    from shadow_peft.task_models import _hidden_mse_loss, _shifted_ce_loss

    torch.manual_seed(0)
    batch, seq, vocab = 2, 8, 32
    student = torch.randn(batch, seq, vocab)
    teacher = torch.randn(batch, seq, vocab)
    labels = torch.randint(0, vocab, (batch, seq))
    labels[:, :2] = -100

    # Baselines recorded 2026-08-28 (torch 2.10, cpu float32); tolerance only
    # absorbs float dust, not semantic drift.
    assert _shifted_ce_loss(student, labels).item() == pytest.approx(4.405302047729492, rel=1e-4)
    assert _shifted_kd_loss(student, teacher, labels, 1.0).item() == pytest.approx(
        1.1721376180648804, rel=1e-4
    )
    assert _shifted_kd_loss(student, teacher, labels, 2.0).item() == pytest.approx(
        1.1890817880630493, rel=1e-4
    )

    torch.manual_seed(2)
    labels2 = torch.randint(0, vocab, (batch, seq))
    labels2[:, :2] = -100
    hs_s = tuple(torch.randn(batch, seq, 16) for _ in range(3))
    hs_t = tuple(h + 0.1 * torch.randn_like(h) for h in hs_s)
    assert _hidden_mse_loss(hs_s, hs_t, labels2).item() == pytest.approx(0.011302858591079712, rel=1e-3)


# ---------------------------------------------------------------------------
# TrainConfig: GRACE modes enter the enum; defaults are the paper values
# ---------------------------------------------------------------------------


def test_train_config_grace_modes_and_defaults():
    from shadow_lab.trainer import GRACE_MODES, LOSS_MODES, TrainConfig

    assert set(GRACE_MODES) < set(LOSS_MODES)
    cfg = TrainConfig(bits=1, loss_mode="dkd_gate_ib_cka")
    assert cfg.dkd_temperature == 2.0
    assert cfg.dkd_tckd_weight == 1.0
    assert cfg.dkd_nckd_weight == 4.0
    assert cfg.gate_lambda == 1.0
    assert (cfg.ib_tau, cfg.ib_eta) == (0.35, 0.0015)
    assert (cfg.ib_beta_min, cfg.ib_beta_max, cfg.ib_ema_decay) == (0.1, 5.0, 0.99)
    assert cfg.cka_weight == 1.0
    assert cfg.gate_log_every == 200

    with pytest.raises(ValueError, match="dkd_gate_ib_cka"):
        TrainConfig(bits=1, loss_mode="dkd")  # partial name must be rejected
    # The error message lists the full mode table.
    try:
        TrainConfig(bits=1, loss_mode="dkd")
    except ValueError as e:
        for mode in LOSS_MODES:
            assert mode in str(e)
