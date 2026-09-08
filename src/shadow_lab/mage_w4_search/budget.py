"""Durable GB10 occupancy: one coordinator, one ledger, no implicit refunds."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import sqlite3
import subprocess
import time
import uuid
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path


class BudgetError(ValueError):
    """A request cannot safely consume this campaign's device budget."""


@dataclass(frozen=True)
class DeviceLease:
    lease_id: str
    request_id: str
    ticket_id: str
    device_id: str
    started_at: float
    hard_deadline: float
    reserved_seconds: float
    run_id: str

    def __post_init__(self):
        if not all((self.lease_id, self.request_id, self.run_id, self.ticket_id, self.device_id)):
            raise BudgetError("invalid-lease-identity")
        if not all(
            math.isfinite(value)
            for value in (self.started_at, self.hard_deadline, self.reserved_seconds)
        ):
            raise BudgetError("invalid-lease-time")
        if self.reserved_seconds <= 0 or not math.isclose(
            self.started_at + self.reserved_seconds, self.hard_deadline, rel_tol=0, abs_tol=1e-6
        ):
            raise BudgetError("invalid-lease-duration")

    def to_dict(self) -> dict:
        return asdict(self)


class DeadlineWatchdog:
    """Worker-local deadline policy, independent of candidate progress callbacks.

    The caller must poll this from an independently surviving supervisor. Start
    cleanup early; force termination halfway through the cleanup reserve so
    acknowledgement has time before the hard deadline even if work is stuck.
    These callbacks must only affect the supervisor's own process group.
    """

    def __init__(self, lease, *, terminate, kill, cleanup_seconds=2.0, clock=time.time):
        if not math.isfinite(cleanup_seconds) or not 0 < cleanup_seconds < lease.reserved_seconds:
            raise BudgetError("invalid-cleanup-reserve")
        self.lease = lease
        self.terminate = terminate
        self.kill = kill
        self.cleanup_seconds = cleanup_seconds
        self.clock = clock
        self.state = "running"
        self.last_time = lease.started_at

    def check(self) -> str:
        now = float(self.clock())
        invalid_clock = not math.isfinite(now) or now < self.last_time
        self.last_time = max(self.last_time, now)
        if invalid_clock or now >= self.lease.hard_deadline - self.cleanup_seconds / 2:
            if self.state != "killed":
                self.kill()
                self.state = "killed"
        elif now >= self.lease.hard_deadline - self.cleanup_seconds and self.state == "running":
            self.terminate()
            self.state = "terminating"
        return self.state


def run_guarded(
    lease: DeviceLease,
    command: list[str],
    *,
    cleanup_seconds: float = 2,
    poll_seconds: float = 0.1,
    cwd: str | None = None,
    clock=time.time,
    stdout=None,
    stderr=None,
) -> dict:
    """Run one owned POSIX process group under a worker-local frozen lease.

    Launch this supervisor independently of SSH lifetime (for example via
    nohup/systemd). It deliberately does not open a second budget database.
    The coordinator releases its lease only after receiving stop_confirmed.
    Failure to prove the whole owned group stopped keeps the device occupied.
    """
    if os.name != "posix" or not command or not all(isinstance(part, str) for part in command):
        raise BudgetError("invalid-guarded-command")
    if not math.isfinite(poll_seconds) or not 0 < poll_seconds <= cleanup_seconds / 4:
        raise BudgetError("invalid-watchdog-poll-interval")
    started = float(clock())
    if started < lease.started_at or started >= lease.hard_deadline - cleanup_seconds:
        raise BudgetError("lease-not-startable")
    process = None

    def send(sig):
        if process is not None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, sig)

    def group_exists():
        try:
            os.killpg(process.pid, 0)
            return True
        except ProcessLookupError:
            return False

    watchdog = DeadlineWatchdog(
        lease,
        terminate=lambda: send(signal.SIGTERM),
        kill=lambda: send(signal.SIGKILL),
        cleanup_seconds=cleanup_seconds,
        clock=clock,
    )
    process = subprocess.Popen(
        command, cwd=cwd, stdout=stdout, stderr=stderr, start_new_session=True
    )
    stop_confirmed = False
    try:
        while True:
            watchdog.check()
            code = process.poll()
            if code is not None:
                if not group_exists():
                    stop_confirmed = True
                    break
                # A normally exiting parent must not leave a GPU child behind.
                send(signal.SIGKILL)
            now = float(clock())
            if now >= lease.hard_deadline or not math.isfinite(now):
                send(signal.SIGKILL)
                break
            time.sleep(min(poll_seconds, max(0, lease.hard_deadline - now)))
    except BaseException:
        send(signal.SIGKILL)
        # No receipt or release on supervisor failure: recover the occupied lease.
        raise
    observed_at = float(clock())
    return {
        "lease_id": lease.lease_id,
        "request_id": lease.request_id,
        "run_id": lease.run_id,
        "ticket_id": lease.ticket_id,
        "device_id": lease.device_id,
        "process_group": process.pid,
        "started_at": started,
        "stopped_at": observed_at if stop_confirmed else None,
        "observed_at": observed_at,
        "hard_deadline": lease.hard_deadline,
        "exit_code": process.poll(),
        "termination": watchdog.state,
        "stop_confirmed": stop_confirmed,
    }


def _main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Worker-local GB10 lease watchdog; no independent budget."
    )
    parser.add_argument("--lease", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--cleanup-seconds", type=float, default=2.0)
    parser.add_argument("--poll-seconds", type=float, default=0.1)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    lease = DeviceLease(**json.loads(args.lease.read_text()))
    result = run_guarded(
        lease, command, cleanup_seconds=args.cleanup_seconds, poll_seconds=args.poll_seconds
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.result.with_name(args.result.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, args.result)
    return 0 if result["stop_confirmed"] and result["exit_code"] == 0 else 1


class BudgetLedger:
    """A coordinator-local SQLite file must be shared by every campaign ticket.

    Reservation starts occupancy immediately. An interrupted caller leaves its
    lease occupied; only explicit stop acknowledgement releases the device.
    Expired reservations still occupy the device, and late release records the
    actual overrun instead of hiding it by clamping charges to the deadline.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        campaign_id: str = "mage-w4a4-gb10-pilot",
        devices: tuple[str, ...] = ("gooo", "goooinext"),
        total_seconds: float = 57_600,
        per_device_seconds: float = 28_800,
        clock=time.time,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.settings = {
            "campaign_id": campaign_id,
            "devices": list(devices),
            "total_seconds": float(total_seconds),
            "per_device_seconds": float(per_device_seconds),
        }
        if not campaign_id or not devices or len(set(devices)) != len(devices):
            raise BudgetError("invalid-budget-identity")
        for amount in (total_seconds, per_device_seconds):
            if not math.isfinite(amount) or amount <= 0:
                raise BudgetError("invalid-budget-limit")
        with self._transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS leases (lease_id TEXT PRIMARY KEY, "
                "request_id TEXT UNIQUE NOT NULL, ticket_id TEXT NOT NULL, "
                "device_id TEXT NOT NULL, started_at REAL NOT NULL, hard_deadline REAL NOT NULL, "
                "reserved_seconds REAL NOT NULL, stopped_at REAL, stop_reason TEXT)"
            )
            if "run_id" not in {row[1] for row in db.execute("PRAGMA table_info(leases)")}:
                db.execute("ALTER TABLE leases ADD COLUMN run_id TEXT NOT NULL DEFAULT ''")
                db.execute("UPDATE leases SET run_id=request_id WHERE run_id=''")
            db.execute(
                "CREATE TABLE IF NOT EXISTS requests (request_id TEXT PRIMARY KEY, "
                "request_digest TEXT NOT NULL, lease_id TEXT NOT NULL, registered_at REAL NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS timings (event_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            existing = db.execute("SELECT value FROM metadata WHERE key='settings'").fetchone()
            if existing and json.loads(existing[0]) != self.settings:
                raise BudgetError("budget-settings-mismatch")
            db.execute(
                "INSERT OR IGNORE INTO metadata VALUES ('settings', ?)",
                (json.dumps(self.settings, sort_keys=True),),
            )
            self._now(db)

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BudgetError:
            # Rejections must preserve the latest observed clock. Mutating
            # methods validate all conditions before writing their state.
            db.commit()
            raise
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _now(self, db):
        now = float(self.clock())
        previous = db.execute("SELECT value FROM metadata WHERE key='last_time'").fetchone()
        if not math.isfinite(now) or (previous and now < float(previous[0])):
            raise BudgetError("clock-regressed-or-invalid")
        db.execute("INSERT OR REPLACE INTO metadata VALUES ('last_time', ?)", (str(now),))
        return now

    @staticmethod
    def _lease(row):
        return DeviceLease(**{key: row[key] for key in DeviceLease.__dataclass_fields__})

    def _snapshot(self, db, now):
        devices = {
            name: {
                "used_seconds": 0.0,
                "committed_seconds": 0.0,
                "overrun_seconds": 0.0,
                "active_lease_id": None,
            }
            for name in self.settings["devices"]
        }
        leases = [
            dict(row) for row in db.execute("SELECT * FROM leases ORDER BY started_at, lease_id")
        ]
        for row in leases:
            used = (row["stopped_at"] if row["stopped_at"] is not None else now) - row["started_at"]
            item = devices[row["device_id"]]
            item["used_seconds"] += used
            item["overrun_seconds"] += max(0, used - row["reserved_seconds"])
            item["committed_seconds"] += (
                used if row["stopped_at"] is not None else max(used, row["reserved_seconds"])
            )
            if row["stopped_at"] is None:
                item["active_lease_id"] = row["lease_id"]
        return {
            "campaign_id": self.settings["campaign_id"],
            "at": now,
            "leases": leases,
            "devices": devices,
            "used_seconds": sum(item["used_seconds"] for item in devices.values()),
            "committed_seconds": sum(item["committed_seconds"] for item in devices.values()),
            "overrun_seconds": sum(item["overrun_seconds"] for item in devices.values()),
            "timing_events": [
                json.loads(row[0])
                for row in db.execute("SELECT payload FROM timings ORDER BY event_id")
            ],
        }

    def snapshot(self) -> dict:
        with self._transaction() as db:
            return self._snapshot(db, self._now(db))

    def reserve(
        self,
        request_id: str,
        ticket_id: str,
        device_id: str,
        seconds: float,
        *,
        run_id: str | None = None,
    ) -> DeviceLease:
        run_id = request_id if run_id is None else run_id
        if (
            not request_id
            or not ticket_id
            or not run_id
            or device_id not in self.settings["devices"]
        ):
            raise BudgetError("invalid-reservation-identity")
        if not math.isfinite(seconds) or seconds <= 0:
            raise BudgetError("invalid-reservation-duration")
        with self._transaction() as db:
            now = self._now(db)
            previous = db.execute(
                "SELECT * FROM leases WHERE request_id=?", (request_id,)
            ).fetchone()
            if previous:
                lease = self._lease(previous)
                if (lease.ticket_id, lease.device_id, lease.reserved_seconds, lease.run_id) != (
                    ticket_id,
                    device_id,
                    seconds,
                    run_id,
                ):
                    raise BudgetError("reservation-replay-mismatch")
                return lease
            snap = self._snapshot(db, now)
            device = snap["devices"][device_id]
            if device["active_lease_id"]:
                raise BudgetError("device-busy")
            if device["committed_seconds"] + seconds > self.settings["per_device_seconds"]:
                raise BudgetError("device-budget-exhausted")
            if snap["committed_seconds"] + seconds > self.settings["total_seconds"]:
                raise BudgetError("total-budget-exhausted")
            lease = DeviceLease(
                str(uuid.uuid4()),
                request_id,
                ticket_id,
                device_id,
                now,
                now + seconds,
                seconds,
                run_id,
            )
            db.execute(
                "INSERT INTO leases (lease_id, request_id, ticket_id, device_id, started_at, "
                "hard_deadline, reserved_seconds, run_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                tuple(asdict(lease).values()),
            )
            return lease

    def assert_active(self, lease_id: str) -> DeviceLease:
        with self._transaction() as db:
            now = self._now(db)
            row = db.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
            if row is None:
                raise BudgetError("unknown-lease")
            if row["stopped_at"] is not None:
                raise BudgetError("lease-already-stopped")
            if now >= row["hard_deadline"]:
                raise BudgetError("lease-deadline-exceeded")
            return self._lease(row)

    def bind_request(self, lease_id: str, request_id: str, request_digest: str) -> None:
        """Bind many candidate requests to one occupancy without double charging."""
        if not request_id or re.fullmatch(r"[0-9a-f]{64}", request_digest) is None:
            raise BudgetError("invalid-request-identity")
        with self._transaction() as db:
            now = self._now(db)
            lease = db.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
            if lease is None or lease["stopped_at"] is not None or now >= lease["hard_deadline"]:
                raise BudgetError("lease-not-active")
            existing = db.execute(
                "SELECT * FROM requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if existing and (existing["request_digest"], existing["lease_id"]) != (
                request_digest,
                lease_id,
            ):
                raise BudgetError("request-identity-conflict")
            db.execute(
                "INSERT OR IGNORE INTO requests VALUES (?, ?, ?, ?)",
                (request_id, request_digest, lease_id, now),
            )

    def release(self, lease_id: str, *, reason: str) -> dict:
        """Acknowledge the owned process group has stopped; never call on disconnect."""
        if not reason:
            raise BudgetError("missing-stop-reason")
        with self._transaction() as db:
            now = self._now(db)
            row = db.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
            if row is None:
                raise BudgetError("unknown-lease")
            if row["stopped_at"] is None:
                db.execute(
                    "UPDATE leases SET stopped_at=?, stop_reason=? WHERE lease_id=?",
                    (now, reason, lease_id),
                )
            return self._snapshot(db, now)

    def record_timing(
        self,
        event_id: str,
        kind: str,
        name: str,
        ticket_id: str,
        started_at: float,
        ended_at: float,
        *,
        lease_id: str | None = None,
        parent_event_id: str | None = None,
    ) -> None:
        """Keep component/CPU/manual intervals as attribution, never extra device charges."""
        if not all((event_id, name, ticket_id)) or kind not in {
            "device_component",
            "host_cpu",
            "manual",
        }:
            raise BudgetError("invalid-timing-identity")
        if not all(math.isfinite(t) for t in (started_at, ended_at)) or ended_at < started_at:
            raise BudgetError("invalid-timing-interval")
        event = {
            "event_id": event_id,
            "kind": kind,
            "name": name,
            "ticket_id": ticket_id,
            "started_at": started_at,
            "ended_at": ended_at,
            "lease_id": lease_id,
            "parent_event_id": parent_event_id,
        }
        with self._transaction() as db:
            now = self._now(db)
            if ended_at > now:
                raise BudgetError("timing-in-future")
            if kind == "device_component":
                row = db.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
                if row is None or row["ticket_id"] != ticket_id:
                    raise BudgetError("timing-lease-mismatch")
                stopped = row["stopped_at"] if row["stopped_at"] is not None else now
                if started_at < row["started_at"] or ended_at > stopped:
                    raise BudgetError("timing-outside-occupancy")
            elif lease_id is not None:
                raise BudgetError("nondevice-time-has-lease")
            if parent_event_id is not None:
                parent_row = db.execute(
                    "SELECT payload FROM timings WHERE event_id=?", (parent_event_id,)
                ).fetchone()
                if parent_row is None:
                    raise BudgetError("unknown-parent-timing")
                parent = json.loads(parent_row[0])
                if (
                    parent["lease_id"] != lease_id
                    or parent["kind"] != kind
                    or not (parent["started_at"] <= started_at <= ended_at <= parent["ended_at"])
                ):
                    raise BudgetError("timing-outside-parent")
            old = db.execute("SELECT payload FROM timings WHERE event_id=?", (event_id,)).fetchone()
            if old and json.loads(old[0]) != event:
                raise BudgetError("timing-identity-conflict")
            db.execute(
                "INSERT OR IGNORE INTO timings VALUES (?, ?)",
                (event_id, json.dumps(event, sort_keys=True)),
            )


if __name__ == "__main__":
    raise SystemExit(_main())
