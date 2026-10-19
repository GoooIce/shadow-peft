"""Public types for token-level handoff (token 级交接原语).

These dataclasses are the contract between the streaming backend, the handoff
policy and the router. They contain no behaviour beyond serialization; the
decision logic lives in replaceable ``HandoffPolicy`` implementations so the
router's control flow does not change when a new policy is tried.

Missing-value semantics: a ``None`` field means "this signal was not available"
and must never be read as zero. In particular, when the backend runs without a
shadow checkpoint the shadow-side features (``kl_base_shadow``,
``top1_agreement``, ``shadow_entropy``, ``shadow_margin``,
``base_top1_in_shadow_topk``) are ``None``; a policy must take its explicit
degraded branch instead of treating the absence as agreement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from .cost import CostEstimate

HandoffAction = Literal["continue", "handoff"]

HandoffReason = Literal[
    "legacy_threshold",
    "low_prefix_trust",
    "format_invalid",
    "task_requires_cloud",
    "deadline",
    "budget",
]

# "token" is only used when the boundary wait budget or the deadline forces a
# cut; ``boundaries.detect_boundary`` returns the other four kinds.
BoundaryKind = Literal["token", "sentence", "newline", "step", "json_field"]


@dataclass(slots=True)
class HandoffFeatures:
    """Per-step features visible to a ``HandoffPolicy``.

    Logit-derived fields are floats; a shadow-side field is ``None`` when the
    request has no shadow logits (no shadow checkpoint, or a backend that only
    exposes the base path), never 0.0. ``low_prob_frac`` is the fraction of
    finite token logprobs below the low-probability threshold; with no scored
    tokens it is 0.0 because no token was observed, not because all tokens were
    confident. ``prefix_tokens`` counts generated tokens observed so far.

    ``boundary_kind``, ``task`` and ``format_ok`` are context fields: the
    feature extractor fills neutral defaults (``"token"``, ``"general"``, 1.0)
    and the streaming backend replaces them with the real step context.
    """

    kl_base_shadow: float | None = None
    top1_agreement: float | None = None
    base_entropy: float = 0.0
    shadow_entropy: float | None = None
    base_margin: float = 0.0
    shadow_margin: float | None = None
    base_top1_in_shadow_topk: bool | None = None
    low_prob_frac: float = 0.0
    prefix_tokens: int = 0
    boundary_kind: BoundaryKind = "token"
    task: str = "general"
    format_ok: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class HandoffStep:
    """One observation passed to ``HandoffPolicy.observe``.

    ``token_logprob`` is ``None`` when the backend did not report a logprob for
    the observed token; that is missing information, not a probability of 0.
    The policy sees the step *before* the risky token is committed downstream,
    so it can decide on "current prefix + next distribution".
    """

    features: HandoffFeatures
    prefix_text: str = ""
    token: str = ""
    token_logprob: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class HandoffDecision:
    """A policy's verdict for the current step.

    ``action="handoff"`` means "hand off at the next qualifying boundary", not
    "truncate now"; the router resolves the boundary and enforces the
    single-handoff rule. ``reason`` is ``None`` for ``action="continue"`` and
    one of ``HandoffReason`` otherwise. ``prefix_trust`` is ``None`` when the
    policy has no calibrated trust estimate; it is never defaulted to 0.
    ``boundary`` is the requested or last observed boundary kind, and
    ``handoff_index`` is the token index at which the cut is requested (0
    before any token has been emitted).
    """

    action: HandoffAction
    reason: HandoffReason | None = None
    boundary: BoundaryKind = "token"
    handoff_index: int = 0
    prefix_trust: float | None = None
    estimated_cost: CostEstimate | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class HandoffOutcome:
    """Result of resolving a request's handoff state (交接结果).

    ``handoff_index`` and ``prefix_trust`` are ``None`` when no handoff took
    place. ``partial_trace`` is the text the cloud receives when
    ``handed_off`` is true; it is empty for edge-only or full-cloud paths.
    """

    handed_off: bool = False
    reason: HandoffReason | None = None
    boundary: BoundaryKind | None = None
    handoff_index: int | None = None
    partial_trace: str = ""
    prefix_trust: float | None = None
    estimated_cost: CostEstimate | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class HandoffPolicy(Protocol):
    """Replaceable upgrade policy; the router owns cloud calls and budgets.

    ``reset`` is called once per request, optionally with request context such
    as task, deadline or budget. ``observe`` is called once per generated
    token and returns a decision; a policy never calls the cloud itself.
    """

    def reset(self, context: dict[str, Any] | None = None) -> None: ...

    def observe(self, step: HandoffStep) -> HandoffDecision: ...
