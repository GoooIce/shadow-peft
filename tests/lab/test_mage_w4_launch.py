"""Changing the evidence output must never create another pilot allowance."""

import importlib.util
from pathlib import Path

import pytest

from shadow_lab.mage_w4_search.budget import BudgetLedger


def launcher():
    path = Path(__file__).parents[2] / "scripts/mage_w4_launch.py"
    spec = importlib.util.spec_from_file_location("mage_w4_launch_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_canonical_budget_is_not_silently_initialized(tmp_path, monkeypatch):
    module = launcher()
    path = tmp_path / "pilot-budget.sqlite3"
    monkeypatch.setattr(module, "PILOT_LEDGER", path)
    with pytest.raises(ValueError, match="existing ticket 01"):
        module.open_pilot_ledger()
    assert not path.exists()


def test_reopened_canonical_budget_retains_previous_device_occupancy(tmp_path, monkeypatch):
    module = launcher()
    path = tmp_path / "pilot-budget.sqlite3"
    original = BudgetLedger(path, clock=lambda: 1000.0)
    lease = original.reserve("old-run", "02", "gooo", 10)
    monkeypatch.setattr(module, "PILOT_LEDGER", path)
    recovered = module.open_pilot_ledger().snapshot()
    assert recovered["devices"]["gooo"]["active_lease_id"] == lease.lease_id
    assert recovered["devices"]["gooo"]["used_seconds"] > 0
