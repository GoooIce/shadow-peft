"""Small public data interface for the annotation module."""

from __future__ import annotations

from dataclasses import dataclass

DECISIONS = frozenset({"left", "tie", "right", "skip"})
REASON_CODES = frozenset(
    {
        "prompt_following",
        "visual_quality",
        "artifact_or_structure",
        "text_accuracy",
        "other",
    }
)
REASON_TARGETS = frozenset({"left", "right", "both", "none"})
ROLES = frozenset({"annotator", "reviewer", "experiment_owner"})


@dataclass(frozen=True)
class Principal:
    user_id: str
    display_name: str
    role: str


@dataclass(frozen=True)
class BlindAssignment:
    assignment_id: str
    experiment_id: str
    comparison_id: str
    prompt_text: str
    seed: int
    width: int
    height: int
    left_media_url: str
    right_media_url: str
    lease_token: str
    lease_expires_at: str
    guideline_version: str
    has_text_target: bool


@dataclass(frozen=True)
class AnswerCommand:
    assignment_id: str
    lease_token: str
    idempotency_key: str
    decision: str
    reason_codes: tuple[str, ...] = ()
    reason_target: str = "none"
    comment: str | None = None
    flagged: bool = False
    client_version: str = "unknown"


@dataclass(frozen=True)
class AnswerReceipt:
    event_id: str
    assignment_id: str
    decision: str | None
    event_type: str
    saved_at: str
    idempotent_replay: bool


@dataclass(frozen=True)
class Progress:
    experiment_id: str
    assigned: int
    answered: int
    remaining: int
    target_votes: int
    saved_events: int


@dataclass(frozen=True)
class BlindReview:
    review_assignment_id: str
    experiment_id: str
    comparison_id: str
    prompt_text: str
    seed: int
    width: int
    height: int
    left_media_url: str
    right_media_url: str
    lease_token: str
    lease_expires_at: str
    guideline_version: str
    queue_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ReviewCommand:
    review_assignment_id: str
    lease_token: str
    idempotency_key: str
    decision: str
    comment: str | None = None
    client_version: str = "unknown"


@dataclass(frozen=True)
class ReviewReceipt:
    event_id: str
    review_assignment_id: str
    decision: str | None
    event_type: str
    saved_at: str
    idempotent_replay: bool


@dataclass(frozen=True)
class ReviewProgress:
    experiment_id: str
    queued: int
    adjudicated: int
    remaining: int
    saved_events: int
