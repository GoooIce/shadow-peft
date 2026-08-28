"""GRACE distillation losses: confidence-gated DKD, adaptive IB controller, span-CKA.

Formula-level port from the GRACE paper ("Gated Relational Alignment via
Confidence-based Distillation for Efficient VLMs", arXiv:2601.22709, ICML 2026)
and its Apache-2.0 reference implementation
(github.com/ForeverBlue816/GRACE, ``qwenvl/train/grace_modules.py``), checked
against the formula-level record in
``.scratch/grace-fusion/research/01-grace-paper.md``:

- **DKD** (Eq. 1-3): KL split into target-class (TCKD) and non-target (NCKD)
  terms, per token, scaled by ``T^2``. With ``nckd_weight="exact"`` the
  natural ``(1 - p_t)`` factor is used and the decomposition recombines to
  the plain KL exactly (the regression anchor against ``_shifted_kd_loss``).
- **Soft gate** (Eq. 4-6): ``g = exp(-lam * h_norm)`` where ``h_norm`` is the
  teacher softmax entropy at T=1 normalized by ``log|V|``. Soft down-weight
  only — no thresholding, no dropping (hard filtering hurt in GRACE Fig. 6).
  Aggregation is the gate-normalized mean ``sum(g_i * L_i) / sum(g_i)``.
- **IB controller** (Eq. 20): projected dual ascent on the distill weight
  beta with EMA-smoothed GDKD observations, updated once per optimizer step.
- **span-CKA** (Eq. 10-13): per-sample linear-kernel centered CKA over the
  continuation span (label-mask non-ignored positions; prefixes/pads never
  enter the Gram matrix), dimension-agnostic, no projection layer.

Stability details carried over from the reference implementation: non-finite
loss terms are replaced by 0, and per-token softmax computations are chunked
over the valid-token axis to bound peak memory at large vocabularies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

# GRACE paper Table 8 defaults (research/01 §5.3).
DKD_TEMPERATURE = 2.0
DKD_TCKD_WEIGHT = 1.0
DKD_NCKD_WEIGHT = 4.0
GATE_LAMBDA = 1.0
IB_TAU = 0.35
IB_ETA = 0.0015
IB_BETA_INIT = 1.0
IB_BETA_MIN = 0.1
IB_BETA_MAX = 5.0
IB_EMA_DECAY = 0.99
CKA_MIN_TOKENS = 4


@torch.no_grad()
def soft_gate(teacher_logits_t1: torch.Tensor, lam: float = GATE_LAMBDA) -> torch.Tensor:
    """g = exp(-lam * h_norm) per position (GRACE Eq. 4-5).

    The entropy uses the teacher softmax at T=1 — NOT the DKD temperature —
    (the reference implementation notes this explicitly). Returned detached:
    the gate depends only on the frozen teacher, so it carries no student
    gradient either way.
    """
    probs = torch.softmax(teacher_logits_t1.float(), dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    h_norm = entropy / math.log(teacher_logits_t1.size(-1))
    return torch.exp(-lam * h_norm)


def _dkd_terms(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token (TCKD, NCKD) over already-flattened [N, V] valid rows.

    Both terms are scaled by ``T^2`` (standard KD scaling). ``xlogy`` and the
    clamps keep the one-hot teacher (``p_t == 1``, empty non-target support)
    finite.
    """
    n = student_logits.size(0)
    device = student_logits.device
    tckd_out = torch.zeros(n, device=device, dtype=torch.float32)
    nckd_out = torch.zeros(n, device=device, dtype=torch.float32)
    for s in range(0, n, chunk):
        sl = student_logits[s : s + chunk].float()
        tl = teacher_logits[s : s + chunk].float()
        tg = target[s : s + chunk]
        sel = tg.unsqueeze(-1)

        t_logp = torch.log_softmax(tl / temperature, dim=-1)
        s_logp = torch.log_softmax(sl / temperature, dim=-1)
        p_t = t_logp.gather(-1, sel).squeeze(-1).exp()
        q_t = s_logp.gather(-1, sel).squeeze(-1).exp()

        # TCKD: KL on the binary [target, non-target] distributions (Eq. 1).
        tckd = torch.xlogy(p_t, p_t / q_t.clamp_min(1e-12)) + torch.xlogy(
            1 - p_t, (1 - p_t) / (1 - q_t).clamp_min(1e-12)
        )

        # NCKD: KL over the non-target classes, renormalized by masking the
        # target logit to -inf (Eq. 2; reference-implementation idiom). The
        # student log-prob is masked where the teacher's non-target mass is
        # zero — otherwise 0 * (log 0 - (-inf)) is NaN (one-hot teachers).
        t_masked = (tl / temperature).scatter(-1, sel, float("-inf"))
        s_masked = (sl / temperature).scatter(-1, sel, float("-inf"))
        t_nt = torch.softmax(t_masked, dim=-1)
        s_nt_logp = torch.log_softmax(s_masked, dim=-1).masked_fill(t_nt == 0, 0.0)
        nckd = (t_nt * (torch.log(t_nt.clamp_min(1e-12)) - s_nt_logp)).sum(dim=-1)

        t2 = temperature * temperature
        tckd_out[s : s + chunk] = torch.nan_to_num(tckd) * t2
        nckd_out[s : s + chunk] = nckd * t2
    return tckd_out, nckd_out


@dataclass
class GatedDKD:
    """Result of one gated-DKD evaluation (loss tensor + monitoring floats)."""

    loss: torch.Tensor
    gate_mean: float
    gate_var: float
    tckd: float
    nckd: float
    gate: torch.Tensor  # detached per-valid-token weights, for monitoring


def gated_dkd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = DKD_TEMPERATURE,
    tckd_weight: float = DKD_TCKD_WEIGHT,
    nckd_weight: float | str = DKD_NCKD_WEIGHT,
    gate_lambda: float = GATE_LAMBDA,
    chunk: int = 2048,
) -> GatedDKD:
    """Gate-normalized DKD over valid tokens (GRACE Eq. 3 + Eq. 6).

    Shares the shift/ignore-index alignment with ``_shifted_ce_loss`` /
    ``_shifted_kd_loss``: position ``j`` is valid iff ``labels[j+1] != -100``.
    ``gate_lambda=0`` forces ``g == 1`` (used by the identity regression).
    """
    shift_student = student_logits[..., :-1, :].contiguous()
    shift_teacher = teacher_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid = shift_labels.view(-1) != -100
    if not bool(valid.any()):
        zero = student_logits.sum() * 0.0  # graph-connected zero (empty valid set)
        return GatedDKD(zero, float("nan"), float("nan"), 0.0, 0.0, torch.empty(0))

    s_flat = shift_student.view(-1, shift_student.size(-1))[valid]
    t_flat = shift_teacher.view(-1, shift_teacher.size(-1))[valid].to(s_flat.dtype)
    tgt = shift_labels.view(-1)[valid]
    # Gate entropy uses the T=1 teacher distribution.
    g = soft_gate(t_flat, lam=gate_lambda)

    tckd, nckd = _dkd_terms(s_flat, t_flat, tgt, temperature=temperature, chunk=chunk)
    if nckd_weight == "exact":
        p_t = torch.softmax(t_flat.float() / temperature, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        w = 1 - p_t
    else:
        w = float(nckd_weight)
    per_token = tckd_weight * tckd + w * nckd
    loss = (g * per_token).sum() / g.sum().clamp_min(1e-12)
    with torch.no_grad():
        gate_mean = g.mean().item()
        gate_var = g.var(unbiased=False).item()
    return GatedDKD(
        loss=loss,
        gate_mean=gate_mean,
        gate_var=gate_var,
        tckd=tckd.mean().item(),
        nckd=nckd.mean().item(),
        gate=g,
    )


@dataclass
class IBController:
    """Projected dual ascent on the distill weight beta (GRACE Eq. 20, Table 8).

    Update rule, once per optimizer step:
        ema  <- decay * ema + (1 - decay) * L_GDKD   (init: first observation)
        beta <- clip(beta + eta * (ema - tau), beta_min, beta_max)
    Constraint violated (student trails the teacher, EMA > tau) -> beta rises.
    """

    tau: float = IB_TAU
    eta: float = IB_ETA
    beta: float = IB_BETA_INIT
    beta_min: float = IB_BETA_MIN
    beta_max: float = IB_BETA_MAX
    ema_decay: float = IB_EMA_DECAY
    ema: float | None = None
    seen_min: float = field(init=False)  # trajectory bounds for the verdict
    seen_max: float = field(init=False)

    def __post_init__(self) -> None:
        self.seen_min = self.beta
        self.seen_max = self.beta

    def step(self, gdkd_value: float) -> float:
        self.ema = (
            gdkd_value if self.ema is None else self.ema_decay * self.ema + (1 - self.ema_decay) * gdkd_value
        )
        self.beta = min(self.beta_max, max(self.beta_min, self.beta + self.eta * (self.ema - self.tau)))
        self.seen_min = min(self.seen_min, self.beta)
        self.seen_max = max(self.seen_max, self.beta)
        return self.beta


def _cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor | None:
    """Linear-kernel centered CKA between two [n, d] feature blocks (Eq. 10-12).

    Rows are L2-normalized first (cosine Gram), then double-centered. The
    1/(n-1)^2 HSIC factor cancels in the normalization and is omitted, as in
    the reference implementation. Returns a differentiable 0-dim tensor, or
    None when either side's centered Gram is degenerate (collapsed rows) —
    the caller skips that sample.
    """
    xn = torch.nn.functional.normalize(x.float(), dim=-1)
    yn = torch.nn.functional.normalize(y.float(), dim=-1)
    k = xn @ xn.T
    ly = yn @ yn.T
    n = x.size(0)
    h = torch.eye(n, device=x.device, dtype=torch.float32) - 1.0 / n
    k = h @ k @ h
    ly = h @ ly @ h
    hsic_kl = (k * ly).sum()
    hsic_kk = (k * k).sum()
    hsic_ll = (ly * ly).sum()
    with torch.no_grad():
        if hsic_kk.item() < eps or hsic_ll.item() < eps:
            return None
    denom = (hsic_kk * hsic_ll).clamp_min(0.0).sqrt()
    return (hsic_kl / denom).clamp(0.0, 1.0)


@dataclass
class SpanCKA:
    """Result of one span-CKA evaluation."""

    loss: torch.Tensor
    n_samples: int  # samples that entered the mean
    n_skipped: int  # samples skipped (span too short or degenerate Gram)


def span_cka_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    min_tokens: int = CKA_MIN_TOKENS,
    eps: float = 1e-8,
) -> SpanCKA:
    """L = 1 - CKA over per-sample continuation spans (GRACE Eq. 13, textualized).

    Hidden states are [B, L, d] position-aligned with the input; position
    ``j`` supervises token ``j+1``, so the span is exactly the rows whose
    next-token label is not ignored — prefixes and pads never enter the Gram
    matrix. Teacher/student dimensions may differ (CKA acts on n x n Grams).
    Per-sample, averaged over the batch; degenerate samples are skipped and
    counted (all-degenerate batches contribute 0).
    """
    span = labels[:, 1:] != -100  # [B, L-1] — hidden[:, :-1] rows to keep
    total = None
    n_used = 0
    n_skipped = 0
    for b in range(student_hidden.size(0)):
        idx = span[b]
        if int(idx.sum()) < min_tokens:
            n_skipped += 1
            continue
        x = student_hidden[b, :-1][idx]
        y = teacher_hidden[b, :-1][idx]
        cka = _cka(x, y, eps=eps)
        if cka is None:
            n_skipped += 1
            continue
        term = 1.0 - cka
        total = term if total is None else total + term
        n_used += 1
    if total is None:
        return SpanCKA(student_hidden.sum() * 0.0, 0, n_skipped)
    return SpanCKA(total / n_used, n_used, n_skipped)


GATE_HIST_EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def gate_histogram(gate: torch.Tensor, edges=GATE_HIST_EDGES) -> list[int]:
    """Fixed-bucket counts of the gate weights (compact log summary)."""
    if gate.numel() == 0:
        return [0] * (len(edges) - 1)
    bounds = torch.tensor(list(edges[1:-1]), device=gate.device)
    counts = torch.bucketize(gate.detach().float(), bounds, right=True)
    return [int((counts == i).sum()) for i in range(len(edges) - 1)]


@dataclass
class GateMonitor:
    """Soft-gate statistics + self-distillation degradation tracking.

    Degradation criterion (experiment spec): batch CV (std/mean) < 0.05 or
    mean > 0.95 sustained for ``degrade_steps`` consecutive steps -> latch
    ``degraded``. Annotation only — never rolls back or alters training.
    """

    log_every: int = 200
    degrade_cv: float = 0.05
    degrade_mean: float = 0.95
    degrade_steps: int = 500
    degraded: bool = False
    streak: int = 0
    max_streak: int = 0
    last_mean: float = float("nan")
    last_var: float = float("nan")
    last_cv: float = float("nan")

    def record(self, step: int, gate: torch.Tensor, beta: float) -> dict | None:
        """Fold one batch in; every ``log_every`` steps return the log record."""
        if gate.numel() == 0:
            return None
        mean = gate.mean().item()
        var = gate.var(unbiased=False).item()
        cv = (math.sqrt(var) / mean) if mean > 0 else float("inf")
        self.last_mean, self.last_var, self.last_cv = mean, var, cv
        if cv < self.degrade_cv or mean > self.degrade_mean:
            self.streak += 1
            self.max_streak = max(self.max_streak, self.streak)
            if self.streak >= self.degrade_steps:
                self.degraded = True  # latch: annotation only, no rollback
        else:
            self.streak = 0
        if step % self.log_every != 0:
            return None
        return {
            "step": step,
            "gate_mean": mean,
            "gate_var": var,
            "gate_cv": cv,
            "gate_hist": gate_histogram(gate),
            "gate_degrade_streak": self.streak,
            "gate_degraded": self.degraded,
            "beta": beta,
        }
