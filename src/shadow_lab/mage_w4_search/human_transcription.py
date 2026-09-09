"""Offline human transcription intake and evidence-bound OCR comparison.

These checks preserve supplied human text and verify artifact relationships; they
cannot authenticate a human author or turn malformed glyphs into objective truth.
Device evidence must come from the coordinator's authenticated artifact intake.
This module never starts an OCR process or a remote job.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

from .budget import DeviceLease
from .contracts import CandidateRequest, DataManifest, digest_json, preflight
from .ocr import (
    CalibrationPlan,
    CalibrationReport,
    HumanTranscription,
    RawSegment,
    RecognitionReceipt,
    TextSegment,
    compare_ocr,
    freeze_scorer,
)
from .scorers import REQUEST_SCHEMA, RESPONSE_SCHEMA, decode_ocr_config


def _unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_artifact(path):
    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    data = Path(path).read_bytes()
    value = json.loads(data, object_pairs_hook=_unique_pairs, parse_constant=invalid_constant)
    return value, hashlib.sha256(data).hexdigest(), len(data)


def read_json(path):
    """Read exact JSON, rejecting duplicate keys and nonfinite numeric constants."""
    return _read_artifact(path)[0]


def _sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _utc_timestamp(value):
    if not isinstance(value, str):
        raise ValueError("UTC timestamp is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("UTC timestamp is required") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("UTC timestamp is required")
    return parsed.timestamp()


def decode_plan(value):
    """Use the public OCR typed identity, distinct from a JSON/file digest."""
    value = dict(value)
    value["configs"] = tuple(decode_ocr_config(row) for row in value["configs"])
    value["image_sha256"] = tuple(value["image_sha256"])
    value["generation_receipt_sha256"] = tuple(value["generation_receipt_sha256"])
    return CalibrationPlan(**value)


@dataclass(frozen=True)
class HumanRow:
    index: int
    image_sha256: str
    complete: bool
    actual_segments: tuple[str, ...]


@dataclass(frozen=True)
class ImportedHumanWork:
    artifact_sha256: str
    artifact_size_bytes: int
    inventory_file_sha256: str
    transcribed_by: str
    transcribed_at_utc: str
    rows: tuple[HumanRow, ...]
    status: str = "structurally_valid_author_declarations_only"

    def for_plan(self, plan: CalibrationPlan) -> tuple[HumanTranscription, ...]:
        """An explicit pre-observation subset is allowed; incomplete truth is not."""
        if _utc_timestamp(self.transcribed_at_utc) > _utc_timestamp(plan.registered_at_utc):
            raise ValueError("human transcription must precede plan registration")
        by_sha = {row.image_sha256: row for row in self.rows}
        result = []
        for identity in plan.image_sha256:
            if identity not in by_sha:
                raise ValueError("planned image is absent from human inventory")
            row = by_sha[identity]
            if not row.complete:
                raise ValueError("incomplete human image cannot enter the registered comparison")
            result.append(
                HumanTranscription(
                    identity,
                    row.actual_segments,
                    "human_image_transcription",
                    self.transcribed_by,
                    self.artifact_sha256,
                    True,
                )
            )
        return tuple(result)

    def exclusions_for(self, plan: CalibrationPlan) -> tuple[dict, ...]:
        self.for_plan(plan)
        selected = set(plan.image_sha256)
        return tuple(
            {
                "index": row.index,
                "image_sha256": row.image_sha256,
                "reason": "not_preregistered" if row.complete else "author_declared_incomplete",
            }
            for row in self.rows
            if row.image_sha256 not in selected
        )


def import_human_work(source_path, inventory_path) -> ImportedHumanWork:
    """Validate all inventory rows without correcting or inferring supplied text."""
    source, source_sha, source_size = _read_artifact(source_path)
    required = {"schema", "purpose", "transcribed_by", "transcribed_at_utc", "rows"}
    if not isinstance(source, dict) or set(source) != required:
        raise ValueError("human transcription schema fields mismatch")
    if (
        source["schema"] != "mage-ocr-human-transcription-entry/v1"
        or source["purpose"] != "ocr_selection"
    ):
        raise ValueError("human transcription schema/purpose mismatch")
    if not isinstance(source["transcribed_by"], str) or not source["transcribed_by"].strip():
        raise ValueError("human annotator is required")
    _utc_timestamp(source["transcribed_at_utc"])
    inventory, inventory_sha, _ = _read_artifact(inventory_path)
    if inventory.get("purpose") != "ocr_selection":
        raise ValueError("only the complete OCR selection inventory is allowed")
    identities = [row["image_sha256"] for row in inventory["rows"]]
    if not identities or len(set(identities)) != len(identities) or not all(map(_sha, identities)):
        raise ValueError("inventory image identities must be unique SHA-256 values")
    if not isinstance(source["rows"], list) or len(source["rows"]) != len(identities):
        raise ValueError("human rows must cover the complete inventory exactly")
    rows = []
    for value in source["rows"]:
        if not isinstance(value, dict) or set(value) != {
            "index",
            "image_sha256",
            "complete",
            "actual_segments",
        }:
            raise ValueError("human row schema fields mismatch")
        index = value["index"]
        if (
            type(index) is not int
            or not 0 <= index < len(identities)
            or value["image_sha256"] != identities[index]
        ):
            raise ValueError("human index/image SHA mismatch")
        if type(value["complete"]) is not bool or not isinstance(value["actual_segments"], list):
            raise ValueError("human completeness/segments types mismatch")
        if any(not isinstance(text, str) or not text.strip() for text in value["actual_segments"]):
            raise ValueError("human segments must contain nonempty strings")
        rows.append(
            HumanRow(index, identities[index], value["complete"], tuple(value["actual_segments"]))
        )
    if {row.index for row in rows} != set(range(len(identities))):
        raise ValueError("duplicate or missing human inventory index")
    return ImportedHumanWork(
        source_sha,
        source_size,
        inventory_sha,
        source["transcribed_by"],
        source["transcribed_at_utc"],
        tuple(sorted(rows, key=lambda row: row.index)),
    )


class _Evidence:
    def __init__(self):
        self.files = {}

    def read(self, path):
        path = Path(path).resolve()
        value, sha, size = _read_artifact(path)
        identity = {"sha256": sha, "size_bytes": size}
        if str(path) in self.files and self.files[str(path)] != identity:
            raise ValueError("evidence artifact changed during closeout")
        self.files[str(path)] = identity
        return value

    def sha(self, path):
        return self.files[str(Path(path).resolve())]["sha256"]


def _finite(value, *, nonnegative=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError("finite numeric evidence is required")
    return value


def _closed_job(directory, snapshot, manifest, evidence, *, measured):
    directory = Path(directory)
    lease = DeviceLease(**evidence.read(directory / "lease.json"))
    matches = [row for row in snapshot["leases"] if row["lease_id"] == lease.lease_id]
    if len(matches) != 1 or any(
        matches[0].get(key) != value for key, value in asdict(lease).items()
    ):
        raise ValueError("lease does not match the unique coordinator ledger")
    ledger_row = matches[0]
    closed_at = _finite(ledger_row.get("stopped_at"))
    if closed_at < lease.started_at or closed_at > _finite(snapshot["at"]):
        raise ValueError("ledger lease must be closed before the snapshot")
    stop = evidence.read(directory / "stopped.json")
    if stop.get("stop_confirmed") is not True or stop.get("exit_code") != 0:
        raise ValueError("comparison requires a confirmed successful worker stop")
    for key in ("lease_id", "request_id", "run_id", "ticket_id", "device_id", "hard_deadline"):
        if stop.get(key) != getattr(lease, key):
            raise ValueError("worker stop receipt/lease identity mismatch")
    if type(stop.get("process_group")) is not int or stop["process_group"] <= 0:
        raise ValueError("worker stop needs an owned process group")
    clock = evidence.read(directory / "clock.json")
    offset = _finite(clock["offset_seconds"])
    uncertainty = _finite(clock["uncertainty_seconds"], nonnegative=True)
    if abs(offset) + uncertainty > 2 or _finite(clock["at"]) > lease.started_at:
        raise ValueError(
            "clock evidence exceeds the preregistered two-second bound or postdates lease"
        )
    started = _finite(stop["started_at"]) - offset
    stopped = _finite(stop["stopped_at"]) - offset
    if (
        started < lease.started_at - uncertainty
        or stopped < started
        or stopped > closed_at + uncertainty
    ):
        raise ValueError("worker stop times are outside the accounted interval")
    if stopped + uncertainty > lease.hard_deadline:
        raise ValueError("worker was not conservatively confirmed stopped by its hard deadline")
    check = evidence.read(directory / "independent-group-check.json")
    if (
        check.get("process_group") != stop["process_group"]
        or check.get("absent") is not True
        or check.get("device_id") != lease.device_id
        or check.get("lease_id") != lease.lease_id
        or _finite(check["checked_at"]) - offset < stopped - uncertainty
    ):
        raise ValueError("independent process-group absence check mismatch")
    grant = evidence.read(directory / "grant.json")
    request = CandidateRequest.from_dict(grant["request"])
    if (
        grant.get("status") != "authorized"
        or grant.get("request_digest") != request.request_digest
        or request.lease_id != lease.lease_id
        or request.device_id != lease.device_id
        or request.run_id != lease.run_id
        or request.ticket_id != lease.ticket_id
        or grant.get("lease") != asdict(lease)
    ):
        raise ValueError("coordinator grant identity mismatch")
    # Authenticated mirrors can retain remote asset paths. Metadata preflight
    # checks the original declared identities without pretending local files are
    # those remote model assets; model verification is in the child receipt.
    metadata_request = replace(
        request,
        lease_id=None,
        dependencies=tuple(replace(row, path=None) for row in request.dependencies),
    )
    precheck = preflight(metadata_request, manifest, expected_center_version=request.center_version)
    if (
        precheck.status != "ready_for_reservation"
        or request.ticket_id != "04"
        or request.purpose != "ocr_selection"
    ):
        raise ValueError("OCR development request metadata rejected")
    if measured:
        device = request.runtime_identity.get("device", {})
        if (
            not isinstance(device, dict)
            or device.get("id") != lease.device_id
            or "NVIDIA GB10" not in device.get("nvidia_identity", "")
        ):
            raise ValueError("measured comparison requires the registered GB10 device identity")
    return lease, request, stop, offset, uncertainty, closed_at - lease.started_at


def _decode_observation(value):
    value = dict(value)
    value["config"] = decode_ocr_config(value["config"])
    value["raw_segments"] = tuple(
        RawSegment(**{**row, "box": tuple(row["box"])}) for row in value["raw_segments"]
    )
    value["segments"] = tuple(
        TextSegment(**{**row, "box": tuple(row["box"]), "raw_indices": tuple(row["raw_indices"])})
        for row in value["segments"]
    )
    return RecognitionReceipt(**value)


def _validate_child(child, observation, image_bytes, stop):
    request = {
        "schema": REQUEST_SCHEMA,
        "action": "recognize",
        "config": asdict(observation.config),
        "config_sha256": observation.config.sha256,
        "image_sha256": observation.image_sha256,
        "image_base64": base64.b64encode(image_bytes).decode("ascii"),
    }
    if (
        not isinstance(child, dict)
        or child.get("action") != "recognize"
        or child.get("request_sha256") != digest_json(request)
        or child.get("config_sha256") != observation.config.sha256
        or child.get("image_sha256") != observation.image_sha256
    ):
        raise ValueError("target-blind child request identity mismatch")
    _finite(child["parent_wall_seconds"], nonnegative=True)
    if observation.status == "engine_failure":
        if child.get("status") != "failed" or not child.get("error"):
            raise ValueError("failed recognition must retain its child failure evidence")
        return
    worker = child.get("worker", {})
    settings = json.loads(observation.config.settings_json)
    if (
        child.get("schema") != RESPONSE_SCHEMA
        or child.get("status") != "ok"
        or child.get("returncode") != 0
        or worker.get("process_group") != stop["process_group"]
        or type(worker.get("pid")) is not int
        or worker["pid"] <= 0
        or not _sha(settings.get("adapter_source_sha256"))
        or worker.get("adapter_source_sha256") != settings["adapter_source_sha256"]
        or digest_json(child.get("segments"))
        != digest_json([asdict(row) for row in observation.raw_segments])
        or not _sha(child.get("stdout_sha256"))
        or not _sha(child.get("stderr_sha256"))
    ):
        raise ValueError("child output/source/process evidence mismatch")
    for key in ("wall_seconds", "host_cpu_seconds"):
        _finite(child["timing"][key], nonnegative=True)


@dataclass(frozen=True)
class ComparisonCloseout:
    report: CalibrationReport
    scorer_lock: dict | None
    accounting: dict
    evidence: dict


def close_comparison(
    *,
    source_path,
    inventory_path,
    plan_path,
    registration_path,
    manifest_path,
    budget_snapshot_path,
    generation_directory,
    generation_report_path,
    worker_directories,
    execution_kind,
    image_directory=None,
) -> ComparisonCloseout:
    """Close authenticated observations; never perform OCR or reserve hardware.

    ``gb10_measured`` needs the complete artifact chain, not just a caller label.
    The caller remains responsible for authenticated intake; unsigned JSON cannot
    prove remote execution by itself. Fixture evidence never mints a scorer lock.
    Costs include all reserved-device occupancy, even CPU OCR and late settlement.
    """
    if execution_kind not in ("fixture", "gb10_measured") or len(worker_directories) != 2:
        raise ValueError("two workers and an explicit execution kind are required")
    if execution_kind == "gb10_measured" and (
        platform.system() != "Linux" or platform.machine().lower() not in {"aarch64", "arm64"}
    ):
        raise ValueError("real OCR comparison/freeze must execute on the GB10 Linux ARM host")
    evidence = _Evidence()
    work = import_human_work(source_path, inventory_path)
    evidence.read(source_path)
    inventory = evidence.read(inventory_path)
    if (
        evidence.sha(source_path) != work.artifact_sha256
        or evidence.sha(inventory_path) != work.inventory_file_sha256
    ):
        raise ValueError("human source/inventory changed during closeout")
    plan = decode_plan(evidence.read(plan_path))
    truth = work.for_plan(plan)
    exclusions = list(work.exclusions_for(plan))
    registration = evidence.read(registration_path)
    accepted = [row.index for row in work.rows if row.image_sha256 in plan.image_sha256]
    if (
        registration.get("schema") != "mage-w4-ocr-comparison-registration/v1"
        or registration.get("plan_sha256") != plan.sha256
        or registration.get("plan_file_sha256") != evidence.sha(plan_path)
        or registration.get("registered_at_utc") != plan.registered_at_utc
        or registration.get("maximum_mean_error") != plan.maximum_mean_error
        or registration.get("human_transcription_artifact_sha256") != work.artifact_sha256
        or registration.get("accepted_indices") != accepted
        or registration.get("exclusions") != exclusions
    ):
        raise ValueError("preregistered plan/human selection identity mismatch")
    manifest = DataManifest.from_dict(evidence.read(manifest_path))
    snapshot = evidence.read(budget_snapshot_path)
    if len({row["lease_id"] for row in snapshot["leases"]}) != len(snapshot["leases"]):
        raise ValueError("budget snapshot contains duplicate leases")
    measured = execution_kind == "gb10_measured"
    generation_lease, generation_request, _, _, _, generation_seconds = _closed_job(
        generation_directory, snapshot, manifest, evidence, measured=measured
    )
    generation = evidence.read(generation_report_path)
    generation_sha = evidence.sha(generation_report_path)
    if (
        generation.get("qualified") is not True
        or generation.get("center_sha256") != generation_request.q0_digest
        or inventory.get("q0_sha256") != generation_request.q0_digest
        or inventory.get("source_job") != generation_lease.run_id
        or set(plan.generation_receipt_sha256) != {generation_sha}
    ):
        raise ValueError("shared native q0 generation identity mismatch")
    generation_rows = {row["sample_id"]: row for row in generation["rows"]}
    if len(generation_rows) != len(generation["rows"]) or set(generation_rows) != {
        row.sample_id for row in generation_request.samples
    }:
        raise ValueError("generation samples do not match their authorized request")
    local_images = {}
    for row in inventory["rows"]:
        path = Path(row["image_path"])
        if image_directory is not None:
            path = Path(image_directory) / path.name
        data = path.read_bytes()
        if (
            hashlib.sha256(data).hexdigest() != row["image_sha256"]
            or generation_rows.get(row["sample_id"], {}).get("image_file_sha256")
            != row["image_sha256"]
        ):
            raise ValueError("local image does not match the shared native generation receipt")
        local_images[row["image_sha256"]] = (row, data)
    cost_allocation = {
        "schema": "mage-w4-ocr-cost-allocation-v1",
        "method": "own_ocr_lease_plus_shared_generation",
        "shared_generation_lease_id": generation_lease.lease_id,
        "shared_generation_report_sha256": generation_sha,
        "comparison_total_method": "sum_unique_leases_once",
    }
    if registration.get("cost_allocation") != cost_allocation:
        raise ValueError("shared generation cost allocation was not preregistered")
    observations, costs, worker_costs = [], [], []
    seen_configs, seen_leases = set(), {generation_lease.lease_id}
    registered_jobs = registration.get("jobs", [])
    if len(registered_jobs) != 2 or len({row["job_id"] for row in registered_jobs}) != 2:
        raise ValueError("exactly two distinct worker jobs must be preregistered")
    for directory in worker_directories:
        directory = Path(directory)
        lease, request, stop, offset, uncertainty, seconds = _closed_job(
            directory, snapshot, manifest, evidence, measured=measured
        )
        if lease.lease_id in seen_leases:
            raise ValueError("comparison leases must be distinct; no double counting")
        seen_leases.add(lease.lease_id)
        job = evidence.read(directory / "ocr-job.json")
        config_index = job.get("config_index")
        if (
            type(config_index) is not int
            or config_index not in (0, 1)
            or config_index in seen_configs
        ):
            raise ValueError("each registered OCR configuration must appear exactly once")
        seen_configs.add(config_index)
        config = plan.configs[config_index]
        if (
            job.get("schema") != "mage-w4-ocr-worker-v1"
            or decode_plan(job["plan"]).sha256 != plan.sha256
            or job.get("plan_sha256") != plan.sha256
            or request.runtime_identity.get("ocr_comparison_job_sha256") != digest_json(job)
            or job.get("human_transcription_artifact_sha256") != work.artifact_sha256
            or job.get("exclusions") != exclusions
            or job.get("cost_allocation") != cost_allocation
            or request.q0_digest != generation_request.q0_digest
        ):
            raise ValueError("worker job differs from frozen plan, human artifact or allocation")
        registered_job = [row for row in registered_jobs if row["job_id"] == lease.run_id]
        if (
            len(registered_job) != 1
            or registered_job[0].get("job_sha256") != digest_json(job)
            or registered_job[0].get("config_sha256") != config.sha256
            or registered_job[0].get("request_digest")
            != replace(request, lease_id=None).request_digest
        ):
            raise ValueError("worker authorization differs from comparison preregistration")
        if _utc_timestamp(plan.registered_at_utc) > lease.started_at:
            raise ValueError("worker lease precedes comparison preregistration")
        if (
            tuple(row["sha256"] for row in job["images"]) != plan.image_sha256
            or tuple(row["sample_id"] for row in job["images"])
            != tuple(sample.sample_id for sample in request.samples)
            or any(row["generation_report_sha256"] != generation_sha for row in job["images"])
            or any(
                local_images[row["sha256"]][0]["sample_id"] != row["sample_id"]
                for row in job["images"]
            )
        ):
            raise ValueError("worker images differ from registered ordered samples")
        package = evidence.read(directory / "ocr-observations.json")
        required_package = {
            "schema": "mage-w4-ocr-observations-v1",
            "request_digest": request.request_digest,
            "lease_id": lease.lease_id,
            "device_id": lease.device_id,
            "plan_sha256": plan.sha256,
            "config_sha256": config.sha256,
            "job_sha256": digest_json(job),
            "cost_basis": "pending_coordinator_verified_lease_stop",
            "scorer_frozen": False,
        }
        if any(package.get(key) != value for key, value in required_package.items()) or len(
            package["rows"]
        ) != len(plan.image_sha256):
            raise ValueError("worker observation package identity or row count mismatch")
        partials = set(directory.glob("ocr-observation-*.json"))
        expected_partials = {
            directory / f"ocr-observation-{index:04d}.json"
            for index in range(len(plan.image_sha256))
        }
        if partials != expected_partials:
            raise ValueError("missing or extra per-image raw observation artifacts")
        for index, row in enumerate(package["rows"]):
            if evidence.read(directory / f"ocr-observation-{index:04d}.json") != row:
                raise ValueError("per-image raw artifact differs from combined observation")
            observation = _decode_observation(row["observation"])
            if (
                observation.config.sha256 != config.sha256
                or observation.image_sha256 != plan.image_sha256[index]
            ):
                raise ValueError("observation image/config/order identity mismatch")
            started = _utc_timestamp(observation.started_at_utc)
            if (
                started - offset < lease.started_at - uncertainty
                or started + observation.elapsed_seconds > stop["stopped_at"] + uncertainty
            ):
                raise ValueError("recognition does not fall inside the worker lifetime")
            _validate_child(
                row["child"], observation, local_images[observation.image_sha256][1], stop
            )
            observations.append(observation)
        costs.append((config.sha256, seconds + generation_seconds))
        worker_costs.append(
            {
                "config_sha256": config.sha256,
                "lease_id": lease.lease_id,
                "ocr_occupied_seconds": seconds,
                "selection_cost_seconds": seconds + generation_seconds,
            }
        )
    accounting = {
        "method": cost_allocation,
        "shared_generation_seconds": generation_seconds,
        "workers": worker_costs,
        "comparison_total_seconds": generation_seconds
        + sum(row["ocr_occupied_seconds"] for row in worker_costs),
        "unique_lease_ids": sorted(seen_leases),
        "campaign_used_seconds": snapshot["used_seconds"],
        "note": "Selection costs each include the same shared generation cost; comparison total counts each lease once.",
    }
    proof = {
        "schema": "mage-w4-ocr-comparison-evidence-v1",
        "execution_kind": execution_kind,
        "artifacts": evidence.files,
        "accounting": accounting,
        "exclusions": exclusions,
        "plan_typed_sha256": plan.sha256,
        "human_artifact_file_sha256": work.artifact_sha256,
        "scope": "Agreement with the supplied human-readable transcription, not general OCR accuracy; malformed glyphs and small samples limit interpretation.",
        "trust_boundary": "Authenticated coordinator intake is required. JSON linkage checks do not authenticate people or machines.",
        "group_check_timing": "Independent absence checks are later observations, not retroactive stop times.",
    }
    report = compare_ocr(
        plan,
        truth,
        observations,
        execution_kind=execution_kind,
        occupied_seconds_by_config=tuple(costs),
        device_evidence_sha256=digest_json(proof),
    )
    lock = freeze_scorer(report) if measured and report.selected_engine is not None else None
    return ComparisonCloseout(report, lock, accounting, proof)
