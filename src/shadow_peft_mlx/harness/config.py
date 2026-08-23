"""Harness config lives in shadow_core; re-exported for import-path compatibility."""

from __future__ import annotations

from shadow_core.harness.config import RoutingRule, ShadowHarnessConfig, SpecialistConfig  # noqa: F401

__all__ = ['RoutingRule', 'ShadowHarnessConfig', 'SpecialistConfig']
