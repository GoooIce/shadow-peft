"""Frozen two-device engineering replay; never an optimizer assignment.

GB10 verifies real batches and differences. The budget control boundary only
checks metadata, file identities and the original authority's closed lease.
"""

from __future__ import annotations

import copy
import math
import platform
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from .contracts import CandidateRequest, digest_json, preflight
from .human_transcription import read_json_artifact

REPLAY_SCHEMA = "mage-w4-development-replay-job/v1"
REPLAY_PREFIX = "evaluation-replay-"


def _epoch(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("replay registration requires an explicit timezone")
    return parsed.timestamp()


def _host(execution_kind):
    if execution_kind not in {"fixture", "gb10_measured"}:
        raise ValueError("explicit replay execution kind required")
    if execution_kind == "gb10_measured" and (
        platform.system() != "Linux" or platform.machine() != "aarch64"
    ):
        raise ValueError("real batch validation and replay comparison require GB10")


def prepare_development_replay(
    source, manifest, *, source_artifact, run_id, device, registered_at_utc, rule
):
    """Freeze both requests before source execution; copy metadata, never model values."""
    from .evaluation_worker import reconstruct_evaluation_job

    source_job = source["evaluation_job"]
    if source_job["schema"] != "mage-w4-center-job/v1":
        raise ValueError("replay source requires an ordinary complete center job")
    if digest_json(reconstruct_evaluation_job(source_job, manifest)) != digest_json(source):
        raise ValueError("replay source preparation identity mismatch")
    original = CandidateRequest.from_dict(source["request"])
    if (
        original.ticket_id != "06"
        or original.effective_edits <= 0
        or original.lease_id is not None
        or original.request_id.startswith(REPLAY_PREFIX)
        or device.get("id") not in {"gooo", "goooinext"}
        or device["id"] == original.device_id
        or run_id == original.run_id
    ):
        raise ValueError("replay needs an unleased ticket06 candidate and a distinct destination")
    _epoch(registered_at_utc)
    if (
        set(rule) != {"max_abs_pixel_error", "require_identical_png", "max_abs_score_error"}
        or rule["require_identical_png"] is not True
        or any(
            type(rule[key]) not in (int, float) or not math.isfinite(rule[key]) or rule[key] != 0
            for key in ("max_abs_pixel_error", "max_abs_score_error")
        )
    ):
        raise ValueError("this finite replay requires the frozen zero-error/identical-PNG rule")
    if (
        set(source_artifact) != {"path", "sha256", "size_bytes"}
        or not Path(source_artifact["path"]).is_absolute()
        or len(source_artifact["sha256"]) != 64
        or type(source_artifact["size_bytes"]) is not int
        or source_artifact["size_bytes"] <= 0
    ):
        raise ValueError("source preparation requires a frozen file identity")
    center_job = copy.deepcopy(source_job)
    center_job["run_id"] = run_id
    center_job["shared"]["runtime_identity"]["device"] = device
    destination = reconstruct_evaluation_job(center_job, manifest)
    job = {
        **center_job,
        "schema": REPLAY_SCHEMA,
        "source_prepared": source,
        "source_artifact": source_artifact,
        "registered_at_utc": registered_at_utc,
        "rule": rule,
        "eligible_for_center_update": False,
    }
    request = replace(
        CandidateRequest.from_dict(destination["request"]),
        request_id=REPLAY_PREFIX + digest_json(job),
    )
    comparable = replace(
        request,
        request_id=original.request_id,
        run_id=original.run_id,
        device_id=original.device_id,
        runtime_identity=original.runtime_identity,
    )
    if comparable != original or {
        k: v for k, v in request.runtime_identity.items() if k != "device"
    } != {k: v for k, v in original.runtime_identity.items() if k != "device"}:
        raise ValueError("replay may change only device and execution identities")
    checked = preflight(request, manifest, expected_center_version=request.center_version)
    if checked.status != "ready_for_reservation":
        raise ValueError(checked.to_dict())
    return {
        "request": asdict(request),
        "preflight": checked.to_dict(),
        "evaluation_job": job,
        "evaluation_job_sha256": digest_json(job),
    }


def reconstruct_replay_job(job, manifest):
    return prepare_development_replay(
        job["source_prepared"],
        manifest,
        source_artifact=job["source_artifact"],
        run_id=job["run_id"],
        device=job["shared"]["runtime_identity"]["device"],
        registered_at_utc=job["registered_at_utc"],
        rule=job["rule"],
    )


def verify_replay_prepared(prepared, manifest):
    job = prepared["evaluation_job"]
    if job["schema"] != REPLAY_SCHEMA or digest_json(
        reconstruct_replay_job(job, manifest)
    ) != digest_json(prepared):
        raise ValueError("frozen development replay identity mismatch")
    return job


def verify_source_artifact(job):
    value, sha, size = read_json_artifact(job["source_artifact"]["path"])
    if (sha, size) != (
        job["source_artifact"]["sha256"],
        job["source_artifact"]["size_bytes"],
    ) or digest_json(value) != digest_json(job["source_prepared"]):
        raise ValueError("frozen source preparation file changed")


def replay_registration_binding(prepared, manifest, source_request):
    """Pure control metadata bound into the source grant before its first generation."""
    job = verify_replay_prepared(prepared, manifest)
    source = CandidateRequest.from_dict(job["source_prepared"]["request"])
    if source_request != source:
        raise ValueError("replay registration does not bind this source request")
    return {
        "schema": "mage-w4-replay-registration-binding/v1",
        "source_job_sha256": digest_json(job["source_prepared"]["evaluation_job"]),
        "source_request_digest": source.request_digest,
        "destination_job_sha256": digest_json(job),
        "destination_request_digest": CandidateRequest.from_dict(
            prepared["request"]
        ).request_digest,
        "registered_at_utc": job["registered_at_utc"],
    }


def _closed_source(snapshot, source, lease_id):
    rows = [row for row in snapshot["leases"] if row["lease_id"] == lease_id]
    if len(rows) != 1 or rows[0].get("stopped_at") is None:
        raise ValueError("replay requires the original source lease closed in its authority")
    row = rows[0]
    if (row["ticket_id"], row["run_id"], row["device_id"]) != (
        source.ticket_id,
        source.run_id,
        source.device_id,
    ) or row["request_id"] not in {source.request_id, source.run_id + "-occupancy"}:
        raise ValueError("replay source lease identity mismatch")
    return row


def verify_replay_source(prepared, manifest, snapshot, *, execution_kind="gb10_measured"):
    """Compute-host-only full source validation including OCR, stop and pair preregistration."""
    from .development import verify_evaluation_job

    _host(execution_kind)
    job = verify_replay_prepared(prepared, manifest)
    verify_source_artifact(job)
    source = CandidateRequest.from_dict(job["source_prepared"]["request"])
    directory = Path(job["source_artifact"]["path"]).parent
    batch, proof = verify_evaluation_job(
        directory, source, manifest, snapshot, execution_kind=execution_kind
    )
    row = _closed_source(snapshot, source, proof["lease_id"])
    binding = replay_registration_binding(prepared, manifest, source)
    grant, _, _ = read_json_artifact(directory / "grant.json")
    if (
        grant.get("development_replay_registration") != binding
        or _epoch(job["registered_at_utc"]) >= row["started_at"]
    ):
        raise ValueError("replay pair must be frozen in the grant before source execution")
    return {
        "schema": "mage-w4-replay-source-verification/v1",
        "status": "verified",
        "execution_kind": execution_kind,
        "registration": binding,
        "source_lease": row,
        "source_evidence": proof,
        "source_batch_sha256": batch.sha256,
        "eligible_for_center_update": False,
    }


def authorize_replay_control(
    prepared, manifest, receipt, snapshot, *, execution_kind="gb10_measured"
):
    """No scoring or artifact interpretation: compare authenticated GB10 receipt with authority."""
    job = verify_replay_prepared(prepared, manifest)
    source = CandidateRequest.from_dict(job["source_prepared"]["request"])
    binding = replay_registration_binding(prepared, manifest, source)
    if (
        execution_kind not in {"fixture", "gb10_measured"}
        or receipt.get("schema") != "mage-w4-replay-source-verification/v1"
        or receipt.get("status") != "verified"
        or receipt.get("execution_kind") != execution_kind
        or receipt.get("registration") != binding
        or receipt.get("eligible_for_center_update") is not False
    ):
        raise ValueError("replay source verification identity/status mismatch")
    row = _closed_source(snapshot, source, receipt["source_lease"]["lease_id"])
    if (
        digest_json(row) != digest_json(receipt["source_lease"])
        or _epoch(job["registered_at_utc"]) >= row["started_at"]
        or receipt["source_evidence"]["lease_id"] != row["lease_id"]
        or receipt["source_evidence"]["batch_sha256"] != receipt["source_batch_sha256"]
    ):
        raise ValueError("replay source verification differs from original closed occupancy")
    return {
        **binding,
        "source_lease_id": row["lease_id"],
        "source_verification_sha256": digest_json(receipt),
    }


def verify_replay_worker(prepared, manifest, grant, receipt_path, *, execution_kind):
    """A replay grant authorizes only its previously verified source receipt bytes."""
    binding = grant.get("development_replay")
    if not isinstance(binding, dict):
        raise ValueError("development replay requires its source verification grant")
    receipt, sha, _ = read_json_artifact(receipt_path)
    expected = authorize_replay_control(
        prepared,
        manifest,
        receipt,
        {"leases": [receipt["source_lease"]]},
        execution_kind=execution_kind,
    )
    expected["source_verification_file_sha256"] = sha
    if binding != expected:
        raise ValueError("development replay source verification changed after authorization")
    verify_source_artifact(prepared["evaluation_job"])
    return expected


def close_development_replay(
    prepared, manifest, snapshot, directory, *, execution_kind="gb10_measured"
):
    """Compare two fully verified batches on GB10; failures remain explicit engineering results."""
    import hashlib

    from PIL import Image

    from .development import verify_evaluation_job

    _host(execution_kind)
    job = verify_replay_prepared(prepared, manifest)
    directory = Path(directory).resolve()
    source_directory = Path(job["source_artifact"]["path"]).parent
    recomputed = verify_replay_source(prepared, manifest, snapshot, execution_kind=execution_kind)
    receipt, _, _ = read_json_artifact(directory / "replay-source-verification.json")
    if digest_json(receipt) != digest_json(recomputed):
        raise ValueError("source verification does not replay from original closed evidence")
    grant, _, _ = read_json_artifact(directory / "grant.json")
    verify_replay_worker(
        prepared,
        manifest,
        grant,
        directory / "replay-source-verification.json",
        execution_kind=execution_kind,
    )
    source_request = CandidateRequest.from_dict(job["source_prepared"]["request"])
    destination_request = CandidateRequest.from_dict(prepared["request"])
    source, source_proof = verify_evaluation_job(
        source_directory, source_request, manifest, snapshot, execution_kind=execution_kind
    )
    target, target_proof = verify_evaluation_job(
        directory, destination_request, manifest, snapshot, execution_kind=execution_kind
    )
    target_lease = next(
        row for row in snapshot["leases"] if row["lease_id"] == target_proof["lease_id"]
    )
    if target_lease["started_at"] < recomputed["source_lease"]["stopped_at"]:
        raise ValueError("replay occupancy precedes source closure")
    reports = [
        read_json_artifact(root / "generation/report.json")[0]
        for root in (source_directory, directory)
    ]
    native = [{row["sample_id"]: row for row in report["rows"]} for report in reports]
    scored = [{row.sample_id: row for row in batch.samples} for batch in (source, target)]
    rows = []
    for sample in source_request.samples:
        native_pair = [side[sample.sample_id] for side in native]
        images = [Path(row["image_path"]) for row in native_pair]
        pixels, sizes = [], []
        for image, row in zip(images, native_pair, strict=True):
            with Image.open(image) as decoded:
                rgb = decoded.convert("RGB")
                pixels.append(rgb.tobytes())
                sizes.append(rgb.size)
            if hashlib.sha256(pixels[-1]).hexdigest() != row["pixel_sha256"]:
                raise ValueError("replay pixel bytes differ from their native report")
        same_size = sizes[0] == sizes[1]
        maximum = (
            max((abs(a - b) for a, b in zip(*pixels, strict=True)), default=0)
            if same_size
            else None
        )
        left, right = (side[sample.sample_id] for side in scored)
        applicable = left.ocr_status != "not_applicable"
        same_applicability = applicable == (right.ocr_status != "not_applicable")
        differences = {
            axis: abs(getattr(left, axis) - getattr(right, axis)) for axis in ("clip", "hps")
        }
        differences["ocr"] = (
            abs(left.ocr - right.ocr) if applicable and same_applicability else None
        )
        identical = images[0].read_bytes() == images[1].read_bytes()
        matched = (
            same_size
            and identical
            and maximum == 0
            and same_applicability
            and all(value == 0 for value in differences.values() if value is not None)
        )
        rows.append(
            {
                "sample_id": sample.sample_id,
                "identical_png": identical,
                "same_dimensions": same_size,
                "max_abs_pixel_error": maximum,
                "ocr_applicability_equal": same_applicability,
                "absolute_score_differences": differences,
                "matched": matched,
            }
        )
    return {
        "schema": "mage-w4-development-replay-closeout/v1",
        "status": "matched" if all(row["matched"] for row in rows) else "failed",
        "execution_kind": execution_kind,
        "rule": job["rule"],
        "rows": rows,
        "source_evidence": source_proof,
        "destination_evidence": target_proof,
        "registration": recomputed["registration"],
        "accounted_lease_ids": [source_proof["lease_id"], target_proof["lease_id"]],
        "accounting": "existing source and replay leases; no additional occupancy or refund",
        "eligible_for_center_update": False,
        "optimizer_candidate_count": 0,
        "quality_improvement_claimed": False,
        "formal_execution_authorized": False,
    }
