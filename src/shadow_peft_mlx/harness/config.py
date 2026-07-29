"""Declarative configuration for the Mixture-of-Shadow harness (MLX backend).

Mirror of ``shadow_peft/harness/config.py`` -- pure-Python, backend-agnostic.
Keep both files in sync; they intentionally define identical schemas so that
``ShadowHarnessConfig`` is portable across the torch and MLX harnesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

InferenceMode = Literal["base_shadow", "shadow_only"]


@dataclass(frozen=True)
class SpecialistConfig:
    """A single shadow adapter that the harness can route requests to."""

    name: str
    adapter_path: str | None = None
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    priority: int = 0
    inference_mode: InferenceMode = "base_shadow"


@dataclass(frozen=True)
class RoutingRule:
    """A metadata guardrail that forces routing to a specific specialist."""

    field: str
    value: str
    specialist: str


@dataclass(frozen=True)
class ShadowHarnessConfig:
    """Top-level harness configuration."""

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
