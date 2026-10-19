"""Base/shadow logit features for handoff policies (双路 logits 特征).

ShadowPEFT can expose the base and shadow distributions for the same forward
pass. Their disagreement is the strongest signal of "this prefix is drifting",
but it must be available only when both paths exist: the extractor returns
``None`` for every shadow-side feature when no shadow logits are supplied, so a
policy cannot silently treat "no shadow" as "perfect agreement".

The module is pure Python (no numpy/torch) and deterministic: ties in the
top-k ranking resolve to the lower vocabulary index and identical logits give
exactly KL = 0 and agreement = 1.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .handoff import HandoffFeatures

# Same cutoff as ConfidenceEstimator.low_logprob, kept as a module constant so
# features and the legacy confidence path rank tokens consistently.
LOW_LOGPROB_THRESHOLD = -2.0

# Floor used only when a shadow probability underflows to exactly 0 for a token
# the base path assigns mass to, keeping KL finite. Probabilities that are
# actually equal are never floored, so identical logits still yield KL == 0.
_PROB_FLOOR = 1e-12


def _softmax(logits: Sequence[float]) -> list[float]:
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def _entropy(probs: Sequence[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def _margin(probs: Sequence[float]) -> float:
    ranked = sorted(probs, reverse=True)
    second = ranked[1] if len(ranked) > 1 else 0.0
    return ranked[0] - second


def _check_logits(logits: Sequence[float], name: str) -> list[float]:
    values = [float(value) for value in logits]
    if not values:
        raise ValueError(f"{name} 不能为空")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} 必须全部是有限数值")
    return values


def compute_handoff_features(
    base_logits: Sequence[float],
    shadow_logits: Sequence[float] | None = None,
    *,
    topk: int = 5,
    token_logprobs: Sequence[float] = (),
) -> HandoffFeatures:
    """Summarize one step's base/shadow distributions and prefix logprobs.

    Args:
        base_logits: Base-model logits for the next token (one value per vocab).
        shadow_logits: Shadow-model logits for the same position, or ``None``
            when the request runs without a shadow path.
        topk: Size of the shadow top-K set used for membership; clamped to the
            vocabulary size. Must be at least 1.
        token_logprobs: Log probabilities of the tokens generated so far; used
            for ``low_prob_frac`` and ``prefix_tokens``.

    Returns:
        ``HandoffFeatures`` with shadow-side fields set to ``None`` when
        ``shadow_logits`` is ``None`` (never 0). ``kl_base_shadow`` is measured
        in nats. ``boundary_kind``/``task``/``format_ok`` carry neutral
        defaults; the streaming backend fills the real context.

    Raises:
        ValueError: empty or non-finite logits, mismatched lengths, topk < 1.
    """
    base = _check_logits(base_logits, "base_logits")
    shadow: list[float] | None = None
    if shadow_logits is not None:
        shadow = _check_logits(shadow_logits, "shadow_logits")
        if len(shadow) != len(base):
            raise ValueError("base_logits 与 shadow_logits 长度必须一致")
    if topk < 1:
        raise ValueError("topk 必须至少为 1")

    base_probs = _softmax(base)
    base_top1 = max(range(len(base)), key=base.__getitem__)

    kl: float | None = None
    agreement: float | None = None
    shadow_entropy: float | None = None
    shadow_margin: float | None = None
    in_shadow_topk: bool | None = None

    if shadow is not None:
        shadow_probs = _softmax(shadow)
        kl = sum(
            p * math.log(p / (q if q > 0.0 else _PROB_FLOOR))
            for p, q in zip(base_probs, shadow_probs, strict=True)
            if p > 0.0
        )
        shadow_top1 = max(range(len(shadow)), key=shadow.__getitem__)
        agreement = 1.0 if base_top1 == shadow_top1 else 0.0
        k = min(topk, len(shadow))
        shadow_topk = set(sorted(range(len(shadow)), key=shadow.__getitem__, reverse=True)[:k])
        in_shadow_topk = base_top1 in shadow_topk
        shadow_entropy = _entropy(shadow_probs)
        shadow_margin = _margin(shadow_probs)

    finite_logprobs = [float(lp) for lp in token_logprobs if math.isfinite(float(lp))]
    if finite_logprobs:
        low = sum(1 for lp in finite_logprobs if lp < LOW_LOGPROB_THRESHOLD)
        low_prob_frac = low / len(finite_logprobs)
    else:
        low_prob_frac = 0.0

    return HandoffFeatures(
        kl_base_shadow=None if kl is None else round(kl, 6),
        top1_agreement=agreement,
        base_entropy=round(_entropy(base_probs), 6),
        shadow_entropy=None if shadow_entropy is None else round(shadow_entropy, 6),
        base_margin=round(_margin(base_probs), 6),
        shadow_margin=None if shadow_margin is None else round(shadow_margin, 6),
        base_top1_in_shadow_topk=in_shadow_topk,
        low_prob_frac=round(low_prob_frac, 6),
        prefix_tokens=len(token_logprobs),
        boundary_kind="token",
        task="general",
        format_ok=1.0,
    )
