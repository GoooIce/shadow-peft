"""Register a finite grid and close complete GB10 groups without starting workers.

This adapter owns file/provenance wiring. The frozen pilot and reward modules
remain the only numerical implementation; incomplete groups are never filled
with synthetic failed outcomes or zero rewards.
"""

import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .contracts import CandidateRequest, digest_json
from .development import (
    close_scale_registration,
    decode_scale_receipt,
    select_development_samples,
    verify_evaluation_job,
)
from .human_transcription import read_json_artifact
from .pilot import ProposalOutcome, calibrate_group, register_group_pilot


def read(path):
    return read_json_artifact(path)[0]


def load_fixed_scale(inputs, manifest, snapshot):
    registration = read(inputs["scale_registration_path"])
    request = CandidateRequest.from_dict(registration["request"])
    directory = Path(inputs["scale_job_directory"])
    verify_registration_time(directory, registration["plan"]["registered_at_utc"])
    batch, evidence = verify_evaluation_job(
        directory, request, manifest, snapshot, execution_kind=inputs["execution_kind"]
    )
    scales = close_scale_registration(registration, batch)
    saved = decode_scale_receipt(read(inputs["scale_receipt_path"]))
    if saved.sha256 != scales.sha256 or scales.status != "ready":
        raise ValueError("complete ready scales must replay from the original registered batch")
    return scales, evidence


def verify_registration_time(directory, registered_at_utc):
    registered = datetime.fromisoformat(registered_at_utc.replace("Z", "+00:00"))
    if registered.utcoffset() is None or registered.utcoffset().total_seconds() != 0:
        raise ValueError("explicit UTC registration is required")
    if read(Path(directory) / "lease.json")["started_at"] < registered.timestamp():
        raise ValueError("job lease predates its finite preregistration")


def verify_candidate_preparation(directory, request, registered_at_utc):
    """Freeze a direction before CPU proposal generation, not merely before GPU scoring."""
    path = Path(directory) / "preparation-timing.json"
    timing, sha, size = read_json_artifact(path)
    expected_schema = (
        "mage-w4-evaluation-preparation-timing/v1"
        if request.ticket_id == "05"
        else "mage-w4-center-preparation-timing/v1"
    )
    start, end = timing.get("started_at"), timing.get("finished_at")
    registered = datetime.fromisoformat(registered_at_utc.replace("Z", "+00:00")).timestamp()
    lease = read(Path(directory) / "lease.json")
    clock = read(Path(directory) / "clock.json")
    offset, uncertainty = clock["offset_seconds"], clock["uncertainty_seconds"]
    if (
        timing.get("schema") != expected_schema
        or timing.get("status") != "prepared"
        or timing.get("job_id") != request.run_id
        or timing.get("device_id") != request.device_id
        or request.ticket_id == "08"
        and timing.get("command") != "prepare"
        or any(type(t) not in (float, int) or not math.isfinite(t) for t in (start, end))
        or not registered <= start <= end
        or end - offset > lease["started_at"] + uncertainty
    ):
        raise ValueError(
            "candidate preparation must follow grid registration and precede its lease"
        )
    return {"path": str(path.resolve()), "sha256": sha, "size_bytes": size}


def register_execution(manifest, scales, *, config, registered_at_utc):
    """Freeze samples, grid, routing and ticket ownership before any candidate work."""
    if set(config) != {
        "selection",
        "edit_counts",
        "proposals_per_count",
        "max_attempts",
        "random_seed",
        "group_devices",
        "execution_kind",
    }:
        raise ValueError(
            "explicit finite grid, selection and all thirteen device assignments required"
        )
    assignments = config["group_devices"]
    if set(assignments) != {str(i) for i in range(13)} or any(
        device not in {"gooo", "goooinext"} for device in assignments.values()
    ):
        raise ValueError("each group needs one of the two authorized GB10 devices")
    samples = select_development_samples(manifest, "candidate_development", **config["selection"])
    plan = register_group_pilot(
        manifest,
        q0_digest=scales.plan.q0_digest,
        prompt_ids=tuple(dict.fromkeys(row.prompt_id for row in samples)),
        seeds=config["selection"]["seeds"],
        edit_counts=config["edit_counts"],
        proposals_per_count=config["proposals_per_count"],
        max_attempts=config["max_attempts"],
        random_seed=config["random_seed"],
        registered_at_utc=registered_at_utc,
        scales=scales,
        execution_kind=config["execution_kind"],
    )
    if plan.samples != samples:
        raise ValueError("grid and deterministic development selector disagree")
    jobs = []
    for index, row in enumerate(plan.candidates()):
        group = row["group_id"]
        ticket = "05" if group == 0 else "08"
        worker_plan = {
            "schema": "mage-w4-evaluation-plan/v1" if ticket == "05" else "mage-w4-center-plan/v1",
            "purpose": "candidate_development",
            "samples": [asdict(sample) for sample in samples],
            "candidate": {
                "kind": "adjacent",
                "candidate_id": row["candidate_id"],
                "group_id": group,
                "mutation_count": row["mutation_count"],
                "seed": row["mutation_seed"],
                "candidate_index": index,
                "max_attempts": row["max_attempts"],
            },
        }
        if ticket == "08":
            worker_plan.update(ticket_id="08", center_version=0)
        jobs.append(
            {
                "candidate_id": row["candidate_id"],
                "group_id": group,
                "device_id": assignments[str(group)],
                "ticket_id": ticket,
                "plan": worker_plan,
            }
        )
    value = {
        "schema": "mage-w4-group-execution-registration/v1",
        "config": config,
        "plan": asdict(plan),
        "jobs": jobs,
        "formal_execution_authorized": False,
        "source_independence_proven": False,
        "formal_density_frozen": False,
    }
    return {**value, "sha256": digest_json(value)}


def replay_registration(registration, manifest, scales):
    value = dict(registration)
    sha = value.pop("sha256")
    if digest_json(value) != sha:
        raise ValueError("finite grid registration content was changed")
    expected = register_execution(
        manifest,
        scales,
        config=value["config"],
        registered_at_utc=value["plan"]["registered_at_utc"],
    )
    if digest_json(expected) != digest_json(registration):
        raise ValueError("finite grid does not replay from its original settings and scales")
    return expected


def close_group(
    registration, manifest, scales, snapshot, *, group_id, candidate_jobs, reference_jobs
):
    """Accept only complete original jobs; a missing or failed job leaves a visible gap."""
    registration = replay_registration(registration, manifest, scales)
    if type(group_id) is not int or not 0 <= group_id < 13:
        raise ValueError("a registered group from zero through twelve is required")
    jobs = {row["candidate_id"]: row for row in registration["jobs"] if row["group_id"] == group_id}
    if set(candidate_jobs) != set(jobs):
        raise ValueError("complete group requires every registered candidate directory")
    devices = {row["device_id"] for row in jobs.values()}
    if set(reference_jobs) != devices:
        raise ValueError("exact same-device q0 references are required")
    config = registration["config"]
    plan = register_group_pilot(
        manifest,
        q0_digest=scales.plan.q0_digest,
        prompt_ids=tuple(
            dict.fromkeys(row["prompt_id"] for row in registration["plan"]["samples"])
        ),
        seeds=config["selection"]["seeds"],
        edit_counts=config["edit_counts"],
        proposals_per_count=config["proposals_per_count"],
        max_attempts=config["max_attempts"],
        random_seed=config["random_seed"],
        registered_at_utc=registration["plan"]["registered_at_utc"],
        scales=scales,
        execution_kind=config["execution_kind"],
    )
    batches, references, proofs, outcomes, sources = [], {}, [], [], {}

    def verified(directory):
        prepared = read(Path(directory) / "prepared.json")
        request = CandidateRequest.from_dict(prepared["request"])
        batch, proof = verify_evaluation_job(
            directory, request, manifest, snapshot, execution_kind=config["execution_kind"]
        )
        proofs.append(proof)
        for ref in proof["artifacts"]:
            path = Path(ref["path"])
            if (
                path.name == "report.json"
                and path.parent.name == "generation"
                or path.parent.name == "scores"
                and path.suffix == ".json"
            ):
                sources[ref["path"]] = ref
        return prepared, batch, proof

    for device, directory in reference_jobs.items():
        _, batch, _ = verified(directory)
        if (
            batch.request.device_id != device
            or batch.request.effective_edits != 0
            or batch.request.candidate_digest != plan.q0_digest
            or batch.request.center_digest != plan.q0_digest
            or batch.request.center_version != 0
        ):
            raise ValueError("reference must be the original same-device q0/version-zero batch")
        references[device] = batch
    for candidate_id, expected in jobs.items():
        directory = candidate_jobs[candidate_id]
        verify_registration_time(directory, plan.registered_at_utc)
        prepared, batch, proof = verified(directory)
        request = batch.request
        if (
            digest_json(prepared["evaluation_job"]["plan"]) != digest_json(expected["plan"])
            or request.candidate_id != candidate_id
            or request.device_id != expected["device_id"]
            or request.ticket_id != expected["ticket_id"]
            or request.center_version != 0
            or request.request_id.startswith("evaluation-replay-")
        ):
            raise ValueError(
                "candidate differs from its preregistered routing, ticket or worker plan"
            )
        proof["artifacts"].append(
            verify_candidate_preparation(directory, request, plan.registered_at_utc)
        )
        receipt = prepared["evaluation_job"]["candidate_receipt"]
        outcomes.append(
            ProposalOutcome(
                candidate_id,
                "scored",
                len(receipt["attempts"]),
                request.effective_edits,
                request.candidate_digest,
                proof["occupied_seconds"],
            )
        )
        batches.append(batch)
    if len({proof["lease_id"] for proof in proofs}) != len(proofs):
        raise ValueError("one group cannot count the same worker lease as multiple jobs")
    refs = {row.request.candidate_id: references[row.request.device_id] for row in batches}
    receipt = calibrate_group(
        plan,
        scales,
        group_id=group_id,
        outcomes=outcomes,
        candidates=batches,
        current_references=refs,
        q0_references=refs,
        source_artifacts=tuple(sources.values()),
    )
    return receipt, {
        "schema": "mage-w4-group-execution-evidence/v1",
        "registration_sha256": registration["sha256"],
        "jobs": proofs,
        "unique_lease_ids": [proof["lease_id"] for proof in proofs],
        "sum_of_referenced_occupancies_seconds": sum(proof["occupied_seconds"] for proof in proofs),
        "reference_occupancy_is_shared_across_groups": True,
        "these_are_not_new_charges": True,
        "formal_execution_authorized": False,
    }
