"""Mixture-of-Shadow harness (MLX backend): request-level routing across adapters.

MLX counterpart of ``shadow_peft.harness``. One base model serves many
specialists; adapters are swapped in place via
``ShadowPeftModel.swap_adapter`` (weights + shadow-state reset, atomically). See
:class:`~shadow_peft_mlx.harness.ShadowEngineMlx`.

Public API
----------
- :class:`ShadowEngineMlx` -- route a request to a specialist and generate.
- :class:`Router` / :class:`RouteDecision` -- routing logic (pure Python).
- :class:`Session` -- conversation state for sticky routing + history isolation.
- :class:`ShadowHarnessConfig` / :class:`SpecialistConfig` / :class:`RoutingRule`
  -- declarative configuration (mirrors the torch harness).
- :class:`AdapterRegistryMlx` -- adapter weight pool + in-place swap (MLX).
"""

from __future__ import annotations

from .config import RoutingRule, ShadowHarnessConfig, SpecialistConfig
from .engine_mlx import ShadowEngineMlx
from .registry_mlx import AdapterRegistryMlx
from .router import RouteDecision, Router
from .session import Session

__all__ = [
    "AdapterRegistryMlx",
    "RouteDecision",
    "Router",
    "RoutingRule",
    "Session",
    "ShadowEngineMlx",
    "ShadowHarnessConfig",
    "SpecialistConfig",
]
