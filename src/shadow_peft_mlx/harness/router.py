"""Router lives in shadow_core; re-exported for import-path compatibility."""

from __future__ import annotations

from shadow_core.harness.router import RouteDecision, Router  # noqa: F401

__all__ = ['RouteDecision', 'Router']
