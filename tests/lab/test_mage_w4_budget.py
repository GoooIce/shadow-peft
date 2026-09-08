"""Device occupancy is recovered at the public budget boundary."""

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from shadow_lab.mage_w4_search.budget import (
    BudgetError,
    BudgetLedger,
    DeadlineWatchdog,
    DeviceLease,
    run_guarded,
)


class Clock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now


def test_interrupted_request_recovers_same_occupancy_and_cannot_reclaim_budget(tmp_path):
    clock = Clock()
    path = tmp_path / "budget.sqlite"
    ledger = BudgetLedger(path, clock=clock)
    lease = ledger.reserve("request-1", "01", "gooo", 28_800)
    clock.now += 3_600

    recovered = BudgetLedger(path, clock=clock)
    assert recovered.reserve("request-1", "01", "gooo", 28_800) == lease
    assert recovered.snapshot()["devices"]["gooo"]["used_seconds"] == 3_600
    assert recovered.snapshot()["devices"]["gooo"]["committed_seconds"] == 28_800
    with pytest.raises(BudgetError, match="device-busy"):
        recovered.reserve("request-2", "02", "gooo", 1)

    recovered.release(lease.lease_id, reason="interrupted-and-stopped")
    assert recovered.snapshot()["devices"]["gooo"]["used_seconds"] == 3_600
    with pytest.raises(BudgetError, match="device-budget-exhausted"):
        recovered.reserve("request-3", "02", "gooo", 25_201)


def test_expired_lease_remains_busy_and_late_stop_is_fully_charged(tmp_path):
    clock = Clock()
    ledger = BudgetLedger(tmp_path / "budget.sqlite", clock=clock)
    lease = ledger.reserve("first", "02", "gooo", 10)
    clock.now += 12
    with pytest.raises(BudgetError, match="lease-deadline-exceeded"):
        ledger.assert_active(lease.lease_id)
    with pytest.raises(BudgetError, match="device-busy"):
        ledger.reserve("second", "02", "gooo", 1)
    ledger.release(lease.lease_id, reason="watchdog-stop-acknowledged")
    snap = ledger.snapshot()
    assert snap["used_seconds"] == 12
    assert snap["overrun_seconds"] == 2


def test_one_occupancy_can_bind_many_candidates_without_charging_them_twice(tmp_path):
    clock = Clock()
    ledger = BudgetLedger(tmp_path / "budget.sqlite", clock=clock)
    lease = ledger.reserve("occupancy-1", "07", "gooo", 100)
    ledger.bind_request(lease.lease_id, "candidate-request-1", "a" * 64)
    ledger.bind_request(lease.lease_id, "candidate-request-2", "b" * 64)
    ledger.bind_request(lease.lease_id, "candidate-request-1", "a" * 64)
    with pytest.raises(BudgetError, match="request-identity-conflict"):
        ledger.bind_request(lease.lease_id, "candidate-request-1", "c" * 64)
    assert ledger.snapshot()["committed_seconds"] == 100


def test_watchdog_reserves_cleanup_time_and_does_not_wait_for_candidate_completion(tmp_path):
    clock = Clock()
    ledger = BudgetLedger(tmp_path / "budget.sqlite", clock=clock)
    lease = ledger.reserve("occupancy-1", "02", "gooo", 10)
    signals = []
    watchdog = DeadlineWatchdog(
        lease,
        terminate=lambda: signals.append("terminate"),
        kill=lambda: signals.append("kill"),
        clock=clock,
        cleanup_seconds=2,
    )
    clock.now = 1_007
    assert watchdog.check() == "running"
    clock.now = 1_008
    assert watchdog.check() == "terminating"
    clock.now = 1_009
    assert watchdog.check() == "killed"
    assert watchdog.check() == "killed"
    assert signals == ["terminate", "kill"]


def test_rejected_operation_does_not_erase_observed_time_or_allow_clock_refund(tmp_path):
    clock = Clock()
    ledger = BudgetLedger(tmp_path / "budget.sqlite", clock=clock)
    lease = ledger.reserve("occupancy", "01", "gooo", 10)
    clock.now += 12
    with pytest.raises(BudgetError, match="lease-deadline-exceeded"):
        ledger.assert_active(lease.lease_id)
    clock.now -= 5
    with pytest.raises(BudgetError, match="clock-regressed"):
        ledger.snapshot()


def test_nested_component_and_cpu_time_do_not_inflate_device_occupancy(tmp_path):
    clock = Clock()
    ledger = BudgetLedger(tmp_path / "budget.sqlite", clock=clock)
    lease = ledger.reserve("occupancy", "02", "gooo", 30)
    clock.now += 20
    ledger.record_timing(
        "install", "device_component", "install", "02", 1000, 1015, lease_id=lease.lease_id
    )
    ledger.record_timing(
        "kernel",
        "device_component",
        "kernel-load",
        "02",
        1002,
        1007,
        lease_id=lease.lease_id,
        parent_event_id="install",
    )
    ledger.record_timing("manual", "manual", "transcription", "04", 990, 1000)
    ledger.release(lease.lease_id, reason="completed")
    snap = ledger.snapshot()
    assert snap["used_seconds"] == 20
    assert len(snap["timing_events"]) == 3
    with pytest.raises(BudgetError, match="timing-outside-occupancy"):
        ledger.record_timing(
            "bad", "device_component", "missed-start", "02", 999, 1005, lease_id=lease.lease_id
        )


def test_worker_supervisor_kills_its_stuck_process_group_and_returns_stop_evidence():
    now = time.time()
    lease = DeviceLease("lease-fixture", "occupancy", "01", "gooo", now, now + 2, 2, "run-1")
    result = run_guarded(
        lease,
        [
            sys.executable,
            "-c",
            "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)",
        ],
        cleanup_seconds=0.8,
        poll_seconds=0.02,
    )
    assert result["stop_confirmed"] is True
    assert result["exit_code"] == -9
    assert result["termination"] == "killed"
    assert result["stopped_at"] <= lease.hard_deadline


def test_concurrent_coordinators_cannot_allocate_the_same_device_twice(tmp_path):
    path = tmp_path / "budget.sqlite"
    BudgetLedger(path, clock=lambda: 1000)

    def attempt(number):
        ledger = BudgetLedger(path, clock=lambda: 1000)
        try:
            return ledger.reserve(f"request-{number}", "01", "gooo", 100).lease_id
        except BudgetError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt, range(8)))
    assert results.count("device-busy") == 7
    assert BudgetLedger(path, clock=lambda: 1000).snapshot()["committed_seconds"] == 100


def test_total_limit_applies_across_devices_and_settings_cannot_reset_on_reopen(tmp_path):
    path = tmp_path / "budget.sqlite"
    ledger = BudgetLedger(path, total_seconds=10, per_device_seconds=8, clock=lambda: 1000)
    ledger.reserve("first", "01", "gooo", 8)
    with pytest.raises(BudgetError, match="total-budget-exhausted"):
        ledger.reserve("second", "02", "goooinext", 3)
    with pytest.raises(BudgetError, match="budget-settings-mismatch"):
        BudgetLedger(path, total_seconds=20, per_device_seconds=8, clock=lambda: 1000)
