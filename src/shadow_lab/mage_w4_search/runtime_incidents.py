"""Verify one narrow late-settlement disposition without altering budget history."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .budget import DeviceLease

SOURCE_NAMES = frozenset(
    {
        "shadow_lab/__init__.py",
        "shadow_lab/mage_w4_search/__init__.py",
        "shadow_lab/mage_w4_search/budget.py",
        "worker.py",
    }
)
OWNER_FIELDS = ("lease_id", "request_id", "run_id", "ticket_id", "device_id")


def _read(reference, *, json_object=True):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256", "size_bytes"}:
        raise ValueError("explicit artifact path, SHA and byte size required")
    size = reference["size_bytes"]
    if type(size) is not int or not 0 < size <= 1024 * 1024:
        raise ValueError("incident artifacts must be nonempty small files")
    path = Path(reference["path"])
    before = path.stat()
    if before.st_size != size:
        raise ValueError("incident artifact size mismatch")
    raw = path.read_bytes()
    after = path.stat()
    if (
        any(
            getattr(before, k) != getattr(after, k)
            for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        )
        or hashlib.sha256(raw).hexdigest() != reference["sha256"]
    ):
        raise ValueError("incident artifact SHA mismatch or concurrent modification")
    if not json_object:
        return raw
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("incident artifact must contain a JSON object")
    return value


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _run(references, leases, source_hashes, *, harness_sha256=None):
    if not isinstance(references, dict) or set(references) != {
        "lease",
        "stop",
        "clock",
        "audit",
        "worker_log",
        "prepared",
    }:
        raise ValueError("complete raw worker/lease/clock/stop/audit references required")
    values = {key: _read(reference) for key, reference in references.items()}
    lease, stop, clock, audit, worker, prepared = (
        values[key] for key in ("lease", "stop", "clock", "audit", "worker_log", "prepared")
    )
    DeviceLease(**lease)
    saved = leases[lease["lease_id"]]
    if (
        any(saved.get(key) != value for key, value in lease.items())
        or lease["ticket_id"] != "06"
        or saved["stopped_at"] is None
    ):
        raise ValueError("worker lease differs from closed ticket 06 ledger record")
    if (
        any(stop.get(key) != lease[key] for key in (*OWNER_FIELDS, "hard_deadline"))
        or audit["lease"] != lease
        or audit["stop"] != stop
        or audit["clock"] != clock
    ):
        raise ValueError("raw stop/clock/lease identity does not match its audit")
    if audit["source_sha256"] != source_hashes or prepared["source_sha256"] != source_hashes:
        raise ValueError("worker source snapshot differs from verified source files")
    if harness_sha256 is not None and (
        audit.get("harness_sha256") != harness_sha256
        or prepared.get("harness_sha256") != harness_sha256
        or audit.get("transport_events") != []
    ):
        raise ValueError("remediation source or transport evidence does not qualify")
    offset, uncertainty = clock["offset_seconds"], clock["uncertainty_seconds"]
    if (
        not all(_finite(v) for v in (offset, uncertainty, clock["at"]))
        or uncertainty < 0
        or abs(offset) + uncertainty > 2
        or clock["at"] > lease["started_at"]
    ):
        raise ValueError("clock evidence cannot bound the worker stop")
    times = [stop.get(key) for key in ("started_at", "stopped_at", "observed_at")]
    if (
        not all(_finite(value) for value in times)
        or not times[0] <= times[1] <= times[2]
        or times[0] - offset + uncertainty < lease["started_at"]
        or stop["stop_confirmed"] is not True
        or stop["exit_code"] != -9
        or stop["termination"] != "killed"
    ):
        raise ValueError("missing owned-group supervised SIGKILL stop evidence")
    stop_upper = stop["stopped_at"] - offset + uncertainty
    if stop_upper > lease["hard_deadline"]:
        raise ValueError("actual worker stop was late or timing is inconclusive")
    group = audit["independent_group_check"]
    if "receipt" in group:
        if group["receipt"] != stop or group.get("process_group_absent") is not True:
            raise ValueError("independent group check differs from raw stop")
        checked_at = group["checked_at"]
    else:
        if group.get("absent") is not True or group["process_group"] != stop["process_group"]:
            raise ValueError("independent group disappearance is unproven")
        checked_at = group["at"]
    if (
        not _finite(checked_at)
        or checked_at < stop["observed_at"]
        or checked_at - offset - uncertainty > saved["stopped_at"]
    ):
        raise ValueError("group disappearance check occurred outside the recorded recovery")
    if (
        type(stop["process_group"]) is not int
        or stop["process_group"] <= 1
        or worker.get("event") != "worker_started"
        or worker.get("sigterm") != "ignored"
        or worker.get("cuda_visible_devices") != ""
        or worker.get("pid") != stop["process_group"]
        or worker.get("process_group") != stop["process_group"]
        or not _finite(worker.get("at"))
        or not stop["started_at"] <= worker["at"] <= stop["stopped_at"]
    ):
        raise ValueError("raw sleep-worker identity or ignored SIGTERM is unproven")
    return {
        "lease_id": lease["lease_id"],
        "device_id": lease["device_id"],
        "started_at": lease["started_at"],
        "settled_at": saved["stopped_at"],
        "stop_upper_coordinator_time": stop_upper,
        "charge_overrun_seconds": max(0, saved["stopped_at"] - lease["hard_deadline"]),
    }


def _verify(reference, accounting):
    value = _read(reference)
    if (
        set(value)
        != {"schema", "kind", "campaign_id", "lease_id", "original", "remediation", "source_files"}
        or value["schema"] != "mage-late-settlement-disposition/v1"
        or value["kind"] != "late_settlement_after_verified_stop"
        or value["campaign_id"] != accounting["campaign_id"]
    ):
        raise ValueError("unsupported or unbound late-settlement disposition")
    sources = value["source_files"]
    if set(sources) != SOURCE_NAMES:
        raise ValueError("complete supervisor and sleep-worker source snapshot required")
    hashes = {
        name: hashlib.sha256(_read(ref, json_object=False)).hexdigest()
        for name, ref in sources.items()
    }
    leases = {row["lease_id"]: row for row in accounting["leases"]}
    old = _run(value["original"], leases, hashes)
    if old["lease_id"] != value["lease_id"] or old["charge_overrun_seconds"] <= 0:
        raise ValueError("disposition must identify an existing late-settlement charge")
    remedy = value["remediation"]
    if set(remedy) != {"runs", "harness", "original_harness"} or set(remedy["runs"]) != {
        "gooo",
        "goooinext",
    }:
        raise ValueError("both GB10 remediation runs and old/new harness sources required")
    harness = hashlib.sha256(_read(remedy["harness"], json_object=False)).hexdigest()
    if harness == hashlib.sha256(_read(remedy["original_harness"], json_object=False)).hexdigest():
        raise ValueError("remediation must identify changed harness source")
    repaired = {}
    for host, refs in remedy["runs"].items():
        run = _run(refs, leases, hashes, harness_sha256=harness)
        if (
            run["device_id"] != host
            or run["started_at"] <= old["settled_at"]
            or run["charge_overrun_seconds"] != 0
        ):
            raise ValueError("remediation requires later closed, timely runs on both hosts")
        repaired[host] = run
    if len({old["lease_id"], *(run["lease_id"] for run in repaired.values())}) != 3:
        raise ValueError("remediation leases must be distinct from the original incident")
    return {
        "lease_id": old["lease_id"],
        "kind": value["kind"],
        "artifact": reference,
        "original": old,
        "remediation": repaired,
        "harness_sha256": harness,
        "scope": "worker-hard-deadline-engineering-only",
        "charges_preserved": True,
    }


def assess_runtime_incidents(accounting, *, dispositions=()):
    """Classify verified historical settlement delays; never grant execution authority."""
    resolved, errors = {}, []
    seen = set()
    for reference in dispositions:
        try:
            item = _verify(reference, accounting)
            identity = item["lease_id"]
            if identity in seen:
                resolved.pop(identity, None)
                raise ValueError("duplicate incident disposition")
            seen.add(identity)
            resolved[identity] = item
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            errors.append({"artifact": reference, "reason": str(error)})
    unresolved = [
        {**row, "reason": "late settlement lacks verified on-time stop and remediation"}
        for row in accounting["deadline_overruns"]
        if row["lease_id"] not in resolved
    ]
    unresolved.extend(errors)
    return {
        "resolved": list(resolved.values()),
        "unresolved": unresolved,
        "active_leases": list(accounting["active_leases"]),
        "budget_exhausted": accounting["budget_exhausted"],
        "runtime_blocked": bool(unresolved or accounting["active_leases"]),
        "charge_overruns": list(accounting["deadline_overruns"]),
    }
