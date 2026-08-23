"""Mixture-of-Shadow harness: request-level routing across multiple shadow adapters.

One base model serves many specialists. Adapters are swapped in place via
``ShadowPeftModel.swap_adapter`` (weights + shadow-state reset, atomically). See :class:`~shadow_peft.harness.ShadowEngine`.

Public API
----------
- :class:`ShadowEngine` -- route a request to a specialist and generate.
- :class:`Router` / :class:`RouteDecision` -- routing logic (pure Python).
- :class:`Session` -- conversation state for sticky routing + history isolation.
- :class:`ShadowHarnessConfig` / :class:`SpecialistConfig` / :class:`RoutingRule`
  -- declarative configuration.
- :class:`AdapterRegistry` -- adapter weight pool + in-place swap (torch).
"""

from __future__ import annotations

from .config import RoutingRule, ShadowHarnessConfig, SpecialistConfig
from .engine import ShadowEngine
from .registry import AdapterRegistry
from .router import RouteDecision, Router
from .session import Session

__all__ = [
    "AdapterRegistry",
    "RouteDecision",
    "Router",
    "RoutingRule",
    "Session",
    "ShadowEngine",
    "ShadowHarnessConfig",
    "SpecialistConfig",
]
