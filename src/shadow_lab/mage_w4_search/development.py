"""Explicit development sample plans, fixed scales and paired screening closeout.

Pure public functions accept fixtures. CLI execution of actual selection,
registration and numerical closeout belongs on the GB10 CPU host. No function
reserves a device, changes a center, or opens confirmation/final observations.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from .contracts import CandidateRequest, DataManifest, RequestedSample, digest_json, preflight
from .human_transcription import (
    decode_plan,
    decode_recognition_receipt,
    read_json_artifact,
    validate_recognizer_process,
)
from .ocr import (
    CalibrationImageScore,
    CalibrationReport,
    EngineCalibration,
    HumanTranscription,
    SegmentMatch,
)
from .rewards import (
    ScalePlan,
    ScaleReceipt,
    ScoredBatch,
    calibrate_scales,
    fixed_generation_identity,
    screen_candidates,
)
from .scorers import decode_ocr_config


def decode_calibration_report(value) -> CalibrationReport:
    """Decode the public OCR report without replacing its typed SHA algorithm."""
    engines = []
    for engine in value["engines"]:
        images = []
        for image in engine["images"]:
            matches = tuple(
                SegmentMatch(**{**row, "edits": tuple(tuple(edit) for edit in row["edits"])})
                for row in image["matches"]
            )
            images.append(CalibrationImageScore(**{**image, "matches": matches}))
        engines.append(
            EngineCalibration(
                **{**engine, "config": decode_ocr_config(engine["config"]), "images": tuple(images)}
            )
        )
    return CalibrationReport(
        **{
            **value,
            "plan": decode_plan(value["plan"]),
            "transcriptions": tuple(
                HumanTranscription(**{**row, "actual_segments": tuple(row["actual_segments"])})
                for row in value["transcriptions"]
            ),
            "engines": tuple(engines),
        }
    )


def select_development_samples(
    manifest: DataManifest, purpose: str, *, text_prompts: int, nontext_prompts: int, seeds
) -> tuple[RequestedSample, ...]:
    """Select all applicable text plus fixed sorted nontext, with distinct sources."""
    if purpose not in {"scale_calibration", "candidate_development"}:
        raise ValueError("only development scale/candidate sample selection is allowed")
    if (
        type(text_prompts) is not int
        or text_prompts < 1
        or type(nontext_prompts) is not int
        or nontext_prompts < 0
        or not isinstance(seeds, (tuple, list))
        or not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
    ):
        raise ValueError("explicit valid prompt counts and unique generation seeds required")
    eligible = sorted(
        (
            row
            for row in manifest.records
            if row.pool == "development" and purpose in row.development_uses and not row.retired
        ),
        key=lambda row: row.prompt_id,
    )
    text = [row for row in eligible if row.complete_target_inventory]
    if len(text) != text_prompts:
        raise ValueError(
            "registered text count must cover all complete text prompts in this purpose"
        )
    selected = list(text)
    sources = {row.source_group for row in selected}
    if len(sources) != len(selected):
        raise ValueError("text prompts must have independent source groups")
    for row in eligible:
        if len(selected) == text_prompts + nontext_prompts:
            break
        if (
            not row.target_text
            and not row.complete_target_inventory
            and row.source_group not in sources
        ):
            selected.append(row)
            sources.add(row.source_group)
    if len(selected) != text_prompts + nontext_prompts:
        raise ValueError("insufficient independent nontext source groups")
    return tuple(
        RequestedSample(row.prompt_id, digest_json(row.prompt), row.source_group, seed)
        for row in sorted(selected, key=lambda row: row.prompt_id)
        for seed in seeds
    )


def _registered_time(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError("registration must use an explicit UTC timestamp")
    return stamp


def _complete_development_request(request, manifest):
    without_paths = replace(
        request,
        lease_id=None,
        dependencies=tuple(replace(row, path=None) for row in request.dependencies),
    )
    checked = preflight(without_paths, manifest, expected_center_version=request.center_version)
    if (
        checked.status != "ready_for_reservation"
        or request.ticket_id not in {"05", "06", "08"}
        or (request.ticket_id != "05" and request.purpose != "candidate_development")
        or request.evaluation_mode != "complete"
        or request.lease_id is not None
    ):
        raise ValueError("an unleased complete development request is required")


def _complete_request(request, manifest):
    if request.ticket_id != "05":
        raise ValueError("register an unleased complete ticket 05 development request")
    _complete_development_request(request, manifest)


def _seal(value):
    return {**value, "sha256": digest_json(value)}


def _unseal(value, schema):
    contents = {key: item for key, item in value.items() if key != "sha256"}
    if contents.get("schema") != schema or value.get("sha256") != digest_json(contents):
        raise ValueError("development registration/receipt identity mismatch")
    return contents


def register_scale_plan(
    request: CandidateRequest, manifest: DataManifest, *, selection, reliability, registered_at_utc
):
    """Bind deterministic independent sources and explicit reliability before scores."""
    _registered_time(registered_at_utc)
    _complete_request(request, manifest)
    selected = select_development_samples(manifest, "scale_calibration", **selection)
    if (
        request.purpose != "scale_calibration"
        or request.samples != selected
        or request.center_version != 0
        or request.effective_edits != 0
        or request.candidate_id != "q0"
        or not request.q0_digest == request.center_digest == request.candidate_digest
    ):
        raise ValueError("scale registration must bind the exact selected fixed q0 request")
    records = {row.prompt_id: row for row in manifest.records}
    prompts = sorted({row.prompt_id for row in selected})
    text = tuple(prompt for prompt in prompts if records[prompt].complete_target_inventory)
    counts = {"clip": len(prompts), "hps": len(prompts), "ocr": len(text)}
    if reliability["min_source_groups"] > counts["ocr"]:
        raise ValueError("minimum source count exceeds the available independent OCR source groups")
    if (
        reliability["min_prompts"] > counts["clip"]
        or reliability["min_ocr_prompts"] > counts["ocr"]
    ):
        raise ValueError("minimum prompt count exceeds the explicitly selected sample design")
    scorer_ids = tuple(
        sorted(
            (row.role.removesuffix("_scorer"), row.sha256)
            for row in request.dependencies
            if row.role.endswith("_scorer")
        )
    )
    plan = ScalePlan(
        request.q0_digest,
        selected,
        text,
        scorer_ids,
        manifest.digest,
        registered_at_utc,
        **reliability,
    )
    return _seal(
        {
            "schema": "mage-w4-development-scale-registration/v1",
            "plan": asdict(plan),
            "plan_sha256": plan.sha256,
            "request": asdict(request),
            "request_digest": request.request_digest,
            "selection": selection,
            "source_group_counts": counts,
            "one_source_group_per_prompt": True,
            "selection_rule": "all-text-then-nontext-prompt-id-order-distinct-sources-v1",
            "stability_interpretation": "engineering reliability gate, not a coverage or power guarantee",
            "stage": "development_scale_calibration",
            "formal_execution_authorized": False,
        }
    )


def decode_scale_plan(value):
    return ScalePlan(
        **{
            **value,
            "samples": tuple(RequestedSample(**row) for row in value["samples"]),
            "ocr_prompt_ids": tuple(value["ocr_prompt_ids"]),
            "scorer_sha256": tuple(tuple(row) for row in value["scorer_sha256"]),
        }
    )


def decode_scale_receipt(value):
    from .rewards import AxisValues

    return ScaleReceipt(
        **{
            **value,
            "plan": decode_scale_plan(value["plan"]),
            "scales": AxisValues(**value["scales"]) if value["scales"] else None,
            "margins": AxisValues(**value["margins"]) if value["margins"] else None,
            "reasons": tuple(value["reasons"]),
        }
    )


def close_scale_registration(registration, baseline: ScoredBatch) -> ScaleReceipt:
    """Replay the preregistered design; caller verifies the underlying job artifacts."""
    value = _unseal(registration, "mage-w4-development-scale-registration/v1")
    request = CandidateRequest.from_dict(value["request"])
    if replace(baseline.request, lease_id=None).request_digest != value["request_digest"]:
        raise ValueError("scale batch differs from the registered request")
    plan = decode_scale_plan(value["plan"])
    replay = register_scale_plan(
        request,
        baseline.manifest,
        selection=value["selection"],
        reliability={
            key: getattr(plan, key)
            for key in (
                "min_prompts",
                "min_ocr_prompts",
                "min_source_groups",
                "maximum_leave_group_out_relative_change",
            )
        },
        registered_at_utc=plan.registered_at_utc,
    )
    if digest_json(replay) != digest_json(registration):
        raise ValueError("scale registration does not replay from its request and manifest")
    return calibrate_scales(plan, baseline)


def register_screen_plan(
    scales: ScaleReceipt,
    candidates,
    references_by_device,
    manifest,
    *,
    selection,
    candidate_order,
    registered_at_utc,
):
    """Freeze the initial q0 development screen; it cannot install a new center."""
    registered = _registered_time(registered_at_utc)
    if scales.status != "ready" or scales.scales is None or scales.reasons:
        raise ValueError("reliable fixed scales are required before candidate registration")
    if registered < _registered_time(scales.plan.registered_at_utc):
        raise ValueError("screen registration precedes the scale protocol")
    selected = select_development_samples(manifest, "candidate_development", **selection)
    by_id = {row.candidate_id: row for row in candidates}
    if (
        not by_id
        or len(by_id) != len(candidates)
        or len(set(candidate_order)) != len(candidate_order)
        or set(by_id) != set(candidate_order)
        or set(references_by_device) != {row.device_id for row in candidates}
    ):
        raise ValueError("candidate order and q0 devices must exactly cover the frozen screen")
    calibrated_sources = {row.source_group for row in scales.plan.samples}
    calibrated_prompts = {row.prompt_digest for row in scales.plan.samples}
    if any(
        row.source_group in calibrated_sources or row.prompt_digest in calibrated_prompts
        for row in selected
    ):
        raise ValueError("candidate development must be independent of scale calibration sources")
    for row in (*candidates, *references_by_device.values()):
        _complete_request(row, manifest)
        scorer_ids = dict(
            (artifact.role.removesuffix("_scorer"), artifact.sha256)
            for artifact in row.dependencies
            if artifact.role.endswith("_scorer")
        )
        if (
            row.purpose != "candidate_development"
            or row.samples != selected
            or row.center_version != 0
            or row.center_digest != scales.plan.q0_digest
            or row.q0_digest != scales.plan.q0_digest
            or row.manifest_digest != scales.plan.manifest_digest
            or scorer_ids != dict(scales.plan.scorer_sha256)
        ):
            raise ValueError("screen request differs from fixed q0/sample/scorer identities")
        if fixed_generation_identity(row) != scales.fixed_generation_sha256:
            raise ValueError("screen runtime or fixed components differ from calibrated generation")
    for device, reference in references_by_device.items():
        if (
            reference.device_id != device
            or reference.candidate_id != "q0"
            or reference.candidate_digest != scales.plan.q0_digest
            or reference.effective_edits != 0
        ):
            raise ValueError(
                "the first development screen requires one unmodified q0 reference per device"
            )
    for candidate in candidates:
        reference = references_by_device[candidate.device_id]
        if (
            candidate.effective_edits <= 0
            or candidate.candidate_digest == scales.plan.q0_digest
            or candidate.runtime_identity != reference.runtime_identity
        ):
            raise ValueError("candidate/reference fixed runtime or components differ")
    return _seal(
        {
            "schema": "mage-w4-development-screen-registration/v1",
            "registered_at_utc": registered_at_utc,
            "scales_sha256": scales.sha256,
            "selection": selection,
            "candidate_order": list(candidate_order),
            "candidates": [asdict(by_id[key]) for key in candidate_order],
            "references_by_device": {key: asdict(row) for key, row in references_by_device.items()},
            "manifest_digest": manifest.digest,
            "initial_center": "q0",
            "double_reference_reuses_q0": True,
            "stage": "development_screening",
            "formal_execution_authorized": False,
            "formal_density_frozen": False,
        }
    )


def close_screen_registration(registration, scales, candidates, references_by_device):
    """Retain every planned candidate; noninferior winners remain unconfirmed."""
    value = _unseal(registration, "mage-w4-development-screen-registration/v1")
    by_id = {row.request.candidate_id: row for row in candidates}
    if (
        len(by_id) != len(candidates)
        or set(by_id) != set(value["candidate_order"])
        or set(references_by_device) != set(value["references_by_device"])
    ):
        raise ValueError(
            "observed candidates and references must exactly cover the registered screen"
        )
    if scales.sha256 != value["scales_sha256"]:
        raise ValueError("screening scale receipt identity mismatch")
    expected = {row["candidate_id"]: CandidateRequest.from_dict(row) for row in value["candidates"]}
    refs = {
        key: CandidateRequest.from_dict(row) for key, row in value["references_by_device"].items()
    }
    for batch, request in [(row, expected[key]) for key, row in by_id.items()] + [
        (row, refs[key]) for key, row in references_by_device.items()
    ]:
        if replace(batch.request, lease_id=None).request_digest != request.request_digest:
            raise ValueError("screened batch differs from registered request")
        if _registered_time(batch.started_at_utc) < _registered_time(value["registered_at_utc"]):
            raise ValueError("candidate/reference scores precede screen preregistration")
    manifest = candidates[0].manifest
    replay = register_screen_plan(
        scales,
        list(expected.values()),
        refs,
        manifest,
        selection=value["selection"],
        candidate_order=tuple(value["candidate_order"]),
        registered_at_utc=value["registered_at_utc"],
    )
    if digest_json(replay) != digest_json(registration):
        raise ValueError("screen registration does not replay from the frozen inputs")
    references = {key: references_by_device[row.request.device_id] for key, row in by_id.items()}
    return screen_candidates(
        scales, candidates, references, references, candidate_order=tuple(value["candidate_order"])
    )


def _source(path, evidence):
    value, sha, size = read_json_artifact(path)
    reference = {"path": str(Path(path).resolve()), "sha256": sha, "size_bytes": size}
    if reference["path"] in evidence and evidence[reference["path"]] != reference:
        raise ValueError("development evidence changed during closeout")
    evidence[reference["path"]] = reference
    return value


def _finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("finite numeric execution evidence is required")
    return value


def verify_evaluation_job(job_directory, expected_request, manifest, snapshot, *, execution_kind):
    """Verify complete worker output and a closed lease; no models or trace replay.

    The complete worker already verifies raw native traces while scoring. This
    closeout retains their native report identity and the actual worker source
    identity; it does not parse multi-GB traces a second time. Artifact intake
    must be authenticated by the coordinator, not an arbitrary claimed label.
    """
    from .budget import DeviceLease
    from .evaluation import GenerationBinding, load_scored_batch
    from .evaluation_worker import reconstruct_evaluation_job
    from .ocr import TargetInventory, match_text
    from .pilot import audit_occupancy, validate_native_sample_binding

    if execution_kind not in {"fixture", "gb10_measured"}:
        raise ValueError("explicit execution kind required")
    _complete_development_request(expected_request, manifest)
    directory, evidence = Path(job_directory).resolve(), {}
    accounting = audit_occupancy(snapshot)
    lease = DeviceLease(**_source(directory / "lease.json", evidence))
    saved = [row for row in accounting["leases"] if row["lease_id"] == lease.lease_id]
    if (
        len(saved) != 1
        or saved[0]["stopped_at"] is None
        or any(saved[0].get(key) != value for key, value in asdict(lease).items())
    ):
        raise ValueError("complete evaluation requires its exact closed coordinator lease")
    saved = saved[0]
    grant = _source(directory / "grant.json", evidence)
    request = CandidateRequest.from_dict(grant["request"])
    if (
        grant.get("status") != "authorized"
        or grant.get("request_digest") != request.request_digest
        or grant.get("lease") != asdict(lease)
        or request.lease_id != lease.lease_id
        or request.run_id != lease.run_id
        or request.ticket_id != lease.ticket_id
        or request.device_id != lease.device_id
        or replace(request, lease_id=None).request_digest != expected_request.request_digest
    ):
        raise ValueError("complete evaluation grant differs from registered request or lease")
    prepared = _source(directory / "prepared.json", evidence)
    job = prepared["evaluation_job"]
    replay = reconstruct_evaluation_job(job, manifest)
    if (
        digest_json(replay) != digest_json(prepared)
        or request.request_id != replay["request"]["request_id"]
        or CandidateRequest.from_dict(prepared["request"]).request_digest
        != expected_request.request_digest
    ):
        raise ValueError("prepared complete job differs from its authorized source plan")
    clock = _source(directory / "clock.json", evidence)
    offset, uncertainty = _finite(clock["offset_seconds"]), _finite(clock["uncertainty_seconds"])
    if uncertainty < 0 or abs(offset) + uncertainty > 2 or _finite(clock["at"]) > lease.started_at:
        raise ValueError("clock evidence cannot conservatively bound this worker")
    stop = _source(directory / "stopped.json", evidence)
    if (
        stop.get("stop_confirmed") is not True
        or stop.get("exit_code") != 0
        or any(
            stop.get(key) != getattr(lease, key)
            for key in (
                "lease_id",
                "request_id",
                "run_id",
                "ticket_id",
                "device_id",
                "hard_deadline",
            )
        )
        or type(stop.get("process_group")) is not int
        or stop["process_group"] <= 0
    ):
        raise ValueError("complete evaluation lacks a successful owned-group stop")
    start, end, observed = (
        _finite(stop[key]) for key in ("started_at", "stopped_at", "observed_at")
    )
    if (
        not start <= end <= observed
        or start - offset < lease.started_at - uncertainty
        or end - offset + uncertainty > lease.hard_deadline
        or end - offset > saved["stopped_at"] + uncertainty
    ):
        raise ValueError("worker stop is late or outside the accounted interval")
    group = _source(directory / "independent-group-check.json", evidence)
    if (
        group.get("absent") is not True
        or group.get("process_group") != stop["process_group"]
        or group.get("lease_id") != lease.lease_id
        or group.get("device_id") != lease.device_id
        or _finite(group["checked_at"]) < observed
    ):
        raise ValueError("independent process-group stop check mismatch")
    result = _source(directory / "result.json", evidence)
    started_receipt = _source(directory / "started.json", evidence)
    if (
        result.get("schema") != "mage-w4-evaluation-result/v1"
        or result.get("status") != "ok"
        or result.get("execution_kind") != execution_kind
        or result.get("request_digest") != request.request_digest
        or result.get("lease_id") != lease.lease_id
        or result.get("device_id") != lease.device_id
        or result.get("evaluation_job_sha256") != digest_json(job)
        or result.get("quality_improvement_claimed") is not False
    ):
        raise ValueError("complete result identity/status mismatch")
    if (
        not start <= _finite(result["started_at"]) <= _finite(result["finished_at"]) <= end
        or not math.isclose(
            _finite(result["wall_seconds"]),
            result["finished_at"] - result["started_at"],
            abs_tol=1e-6,
        )
        or _finite(result["host_cpu_seconds"]) < 0
    ):
        raise ValueError("result time does not fit its supervised execution")
    report_path, binding_path, batch_path = (
        directory / "generation/report.json",
        directory / "generation-binding.json",
        directory / "scores/batch.json",
    )
    report = _source(report_path, evidence)
    binding = GenerationBinding(**_source(binding_path, evidence))
    batch_payload = _source(batch_path, evidence)
    batch = load_scored_batch(batch_path)
    for path, field in (
        (report_path, "native_report_file_sha256"),
        (binding_path, "generation_binding_file_sha256"),
        (batch_path, "scored_batch_file_sha256"),
    ):
        if result.get(field) != evidence[str(path)]["sha256"]:
            raise ValueError("complete result artifact file SHA mismatch")
    if (
        batch.sha256 != digest_json(batch_payload)
        or batch.request != request
        or batch.manifest != manifest
        or batch.execution_kind != execution_kind
        or binding.execution_kind != execution_kind
        or binding.request_digest != request.request_digest
        or result.get("generation_binding") != asdict(binding)
        or binding.native_report_sha256 != digest_json(report)
        or result.get("native_report_sha256") != digest_json(report)
        or batch.native_receipt_sha256 != digest_json(report)
        or result.get("scored_batch_sha256") != batch.sha256
        or result.get("image_count") != len(batch.samples)
        or report.get("center_sha256") != request.candidate_digest
        or report.get("fixed_sha256") != binding.fixed_state_sha256
        or binding.fixed_state_sha256 != job["shared"]["fixed_state_sha256"]
        or binding.runtime_sha256 != digest_json(report.get("runtime"))
        or binding.generation_sha256 != digest_json(report.get("generation"))
        or report.get("generation") != request.runtime_identity["generation"]
        or report.get("schema_version") != 1
        or report.get("status") != "ok"
    ):
        raise ValueError("native/binding/scored batch identities disagree")
    validate_native_sample_binding(request, manifest, report)
    if execution_kind == "gb10_measured":
        layers = report.get("installation", {}).get("installed_layers", [])
        if (
            result.get("process_group") != stop["process_group"]
            or type(result.get("pid")) is not int
            or result["pid"] <= 0
            or started_receipt.get("pid") != result["pid"]
            or started_receipt.get("process_group") != result["process_group"]
            or started_receipt.get("at") != result["started_at"]
            or report.get("native_compute") is not True
            or report.get("qualified") is not True
            or report.get("runtime", {}).get("compute_capability") != [12, 1]
            or report.get("installation", {}).get("backend") != "torchao_nvfp4"
            or len(layers) != 174
            or len(set(layers)) != 174
        ):
            raise ValueError("claimed measured batch lacks GB10 native MMDiT execution evidence")
    native_rows = {row["sample_id"]: row for row in report["rows"]}
    if len(native_rows) != len(report["rows"]) or set(native_rows) != {
        row.sample_id for row in batch.samples
    }:
        raise ValueError("native generation must cover each scored sample exactly once")
    requested = {row.sample_id: row for row in request.samples}
    prompts = {row.prompt_id: row for row in manifest.records}
    expected_files = {directory / "scores" / (row.sample_id + ".json") for row in batch.samples} | {
        batch_path
    }
    if set((directory / "scores").glob("*.json")) != expected_files:
        raise ValueError("missing or extra per-image score receipt files")
    for row in batch.samples:
        native, sample = native_rows[row.sample_id], requested[row.sample_id]
        prompt = prompts[sample.prompt_id]
        image = Path(native["image_path"]).resolve()
        if not image.is_relative_to(directory):
            raise ValueError("generated image is outside its complete worker directory")
        data = image.read_bytes()
        if (
            hashlib.sha256(data).hexdigest() != row.image_sha256
            or native.get("image_file_sha256") != row.image_sha256
            or native.get("seed") != sample.seed
            or native.get("status") != "ok"
            or native.get("prompt_sha256") != hashlib.sha256(prompt.prompt.encode()).hexdigest()
        ):
            raise ValueError("scored image/prompt/seed differs from native generation")
        score = _source(directory / "scores" / (row.sample_id + ".json"), evidence)
        if (
            digest_json(score) != row.score_receipt_sha256
            or score.get("request_digest") != request.request_digest
            or score.get("native_report_sha256") != batch.native_receipt_sha256
            or score.get("sample_id") != row.sample_id
            or score.get("image_sha256") != row.image_sha256
            or score.get("execution_kind") != execution_kind
            or score.get("scorer_sha256") != dict(batch.scorer_sha256)
            or score.get("clip") != row.clip
            or score.get("hps") != row.hps
        ):
            raise ValueError("per-image score receipt identity/value mismatch")
        if row.ocr_status == "not_applicable":
            if (
                score.get("recognition") is not None
                or score.get("text_score") is not None
                or score.get("recognizer_process") is not None
            ):
                raise ValueError("nontext sample must keep OCR not applicable")
            continue
        recognition = decode_recognition_receipt(score["recognition"])
        text_score = match_text(
            recognition, TargetInventory(prompt.target_text, prompt.complete_target_inventory)
        )
        if (
            recognition.image_sha256 != row.image_sha256
            or recognition.config.sha256 != dict(batch.scorer_sha256)["ocr"]
            or digest_json(asdict(text_score)) != digest_json(score["text_score"])
            or text_score.status != row.ocr_status
            or text_score.character_correctness != row.ocr
        ):
            raise ValueError("OCR correctness does not replay from target-blind recognition")
        if execution_kind == "gb10_measured":
            validate_recognizer_process(score["recognizer_process"], recognition, data, stop)
        elif score.get("recognizer_process") is not None:
            raise ValueError("fixture OCR cannot claim an actual child process")
    return batch, {
        "schema": "mage-w4-development-job-evidence/v1",
        "batch_sha256": batch.sha256,
        "request_digest": request.request_digest,
        "execution_kind": execution_kind,
        "lease_id": lease.lease_id,
        "occupied_seconds": saved["stopped_at"] - lease.started_at,
        "device_id": lease.device_id,
        "artifacts": list(evidence.values()),
        "formal_execution_authorized": False,
        "trust_boundary": "coordinator-authenticated artifact intake; hashes alone do not authenticate a remote host",
        "raw_native_trace_policy": "verified by the complete worker; retained on GB10, not reparsed during closeout",
    }
