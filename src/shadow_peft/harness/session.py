"""Session lives in shadow_core; re-exported for import-path compatibility."""

from __future__ import annotations

from shadow_core.harness.session import Session  # noqa: F401

__all__ = ['Session']
