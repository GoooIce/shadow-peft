"""Harness config lives in shadow_core; re-exported for import-path compatibility."""

from __future__ import annotations

from shadow_core.harness.config import (  # noqa: F401
    InferenceMode,
    RoutingRule,
    ShadowHarnessConfig,
    SpecialistConfig,
)

__all__ = ['InferenceMode', 'RoutingRule', 'ShadowHarnessConfig', 'SpecialistConfig']
