"""Declarative configuration for the Mixture-of-Shadow harness (backend-free)."""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

InferenceMode = Literal["base_shadow", "shadow_only"]


@dataclass(frozen=True)
class SpecialistConfig:
    """A single shadow adapter that the harness can route requests to.

    Parameters
    ----------
    name:
        Unique identifier for this specialist (e.g. ``"medical"``, ``"code"``).
    adapter_path:
        Directory containing ``shadow_adapter.safetensors`` (and optional
        ``shadow_modules.safetensors`` for trained task heads). When ``None``
        the harness uses the live adapter weights already on the model — useful
        for tests or for a single randomly-initialized specialist.
    description:
        Human-readable description; used for debugging / logging.
    keywords:
        Substrings that trigger this specialist via keyword matching. Matching
        is case-insensitive. Empty list means "keyword routing never selects
        this specialist".
    metadata:
        Metadata field expectations for guardrail routing, e.g.
        ``{"domain": "medical"}``.
    priority:
        Tie-breaker when multiple specialists match by keyword. Higher wins.
    inference_mode:
        Which inference mode to use when serving this specialist.
    """

    name: str
    adapter_path: str | None = None
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    priority: int = 0
    inference_mode: InferenceMode = "base_shadow"


@dataclass(frozen=True)
class RoutingRule:
    """A metadata guardrail that forces routing to a specific specialist.

    When the request metadata contains ``field == value`` the router returns
    ``specialist`` regardless of keyword or session-sticky results.
    """

    field: str
    value: str
    specialist: str


@dataclass(frozen=True)
class ShadowHarnessConfig:
    """Top-level harness configuration.

    Parameters
    ----------
    specialists:
        All specialists available to the harness. Order is not significant.
    default_specialist:
        Name of the L0 fallback specialist. Must be present in ``specialists``.
        Used when no rule, keyword, or sticky binding applies.
    rules:
        Metadata guardrail rules, evaluated in order. The first match wins and
        overrides all other routing signals.
    sticky_sessions:
        When True (default), once a session is bound to a specialist all
        subsequent requests in that session stay with it unless a metadata rule
        forces otherwise.
    """

    specialists: list[SpecialistConfig]
    default_specialist: str
    rules: list[RoutingRule] = field(default_factory=list)
    sticky_sessions: bool = True

    def __post_init__(self) -> None:
        names = [s.name for s in self.specialists]
        if len(names) != len(set(names)):
            dupes = {n for n in names if names.count(n) > 1}
            raise ValueError(f"Duplicate specialist names: {sorted(dupes)}")
        if self.default_specialist not in names:
            raise ValueError(
                f"default_specialist {self.default_specialist!r} not in specialists {names!r}"
            )

    def specialist(self, name: str) -> SpecialistConfig:
        """Look up a specialist by name, raising ``KeyError`` if absent."""
        for s in self.specialists:
            if s.name == name:
                return s
        raise KeyError(f"Unknown specialist: {name!r}")

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)
