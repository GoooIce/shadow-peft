"""Finite development calibration and an explicit, evidence-linked freeze table."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime

from .contracts import DataManifest, RequestedSample, digest_json
from .rewards import ScaleReceipt

FORMAL_FIELDS = (
    "native_two_host",
    "ocr_lock",
    "scale_receipt",
    "measured_group_calibrations",
    "candidate_batch_size",
    "search_samples",
    "confirmation_slices",
    "confirmation_attempts",
    "confirmation_alphas",
    "joint_interval_method",
    "method_applicability",
    "score_support_audit",
    "independence_audit",
    "final_samples",
    "detectable_improvement",
    "formal_total_device_seconds",
    "formal_per_device_seconds",
    "final_reserved_device_seconds",
)


@dataclass(frozen=True)
class GroupPilotPlan:
    q0_digest: str
    manifest_digest: str
    samples: tuple[RequestedSample, ...]
    edit_counts: tuple[int, ...]
    proposals_per_count: int
    max_attempts: int
    random_seed: int
    registered_at_utc: str
    scales_sha256: str
    fixed_generation_sha256: str
    scorer_sha256: tuple[tuple[str, str], ...]
    execution_kind: str
    purpose: str = "candidate_development"
    selection_rule: str = "best-positive-feasible-reward-then-fewer-edits-then-candidate-order"

    def __post_init__(self):
        hashes = (
            self.q0_digest,
            self.manifest_digest,
            self.scales_sha256,
            self.fixed_generation_sha256,
        )
        if any(
            not isinstance(item, str)
            or len(item) != 64
            or any(c not in "0123456789abcdef" for c in item)
            for item in hashes
        ):
            raise ValueError("pilot needs complete frozen content identities")
        if (
            self.purpose != "candidate_development"
            or self.execution_kind not in {"fixture", "gb10_measured"}
            or self.selection_rule
            != "best-positive-feasible-reward-then-fewer-edits-then-candidate-order"
            or not self.samples
            or len({row.sample_id for row in self.samples}) != len(self.samples)
            or set(dict(self.scorer_sha256)) != {"clip", "hps", "ocr"}
            or len(self.scorer_sha256) != 3
            or not self.edit_counts
            or tuple(sorted(set(self.edit_counts))) != self.edit_counts
            or any(
                type(n) is not int or n <= 0
                for n in (*self.edit_counts, self.proposals_per_count, self.max_attempts)
            )
            or type(self.random_seed) is not int
            or self.random_seed < 0
            or datetime.fromisoformat(self.registered_at_utc).tzinfo is None
        ):
            raise ValueError("invalid frozen finite pilot plan")

    @property
    def sha256(self):
        return digest_json(asdict(self))

    def candidates(self):
        rng = random.Random(self.random_seed)
        return tuple(
            {
                "candidate_id": f"g{group:02d}-m{count}-r{replicate}",
                "group_id": group,
                "mutation_count": count,
                "replicate": replicate,
                "mutation_seed": rng.randrange(2**32),
                "max_attempts": self.max_attempts,
                "plan_sha256": self.sha256,
            }
            for group in range(13)
            for count in self.edit_counts
            for replicate in range(self.proposals_per_count)
        )


def register_group_pilot(
    manifest: DataManifest,
    *,
    q0_digest,
    prompt_ids,
    seeds,
    edit_counts,
    proposals_per_count,
    max_attempts,
    random_seed,
    registered_at_utc,
    scales: ScaleReceipt,
    execution_kind: str,
) -> GroupPilotPlan:
    if manifest.validate():
        raise ValueError(manifest.validate())
    records = {row.prompt_id: row for row in manifest.records}
    if (
        not prompt_ids
        or len(set(prompt_ids)) != len(prompt_ids)
        or any(
            i not in records or records[i].development_uses != ("candidate_development",)
            for i in prompt_ids
        )
    ):
        raise ValueError("pilot samples must be frozen candidate development prompts")
    if not any(records[i].complete_target_inventory for i in prompt_ids):
        raise ValueError("pilot requires a nonempty complete OCR subset")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
    ):
        raise ValueError("pilot seeds must be unique uint32 values")
    if (
        not edit_counts
        or tuple(sorted(set(edit_counts))) != tuple(edit_counts)
        or any(type(n) is not int or n <= 0 for n in edit_counts)
        or any(type(n) is not int or n <= 0 for n in (proposals_per_count, max_attempts))
        or type(random_seed) is not int
        or random_seed < 0
    ):
        raise ValueError("register a finite ordered mutation grid and finite retry limit")
    if len(q0_digest) != 64 or any(c not in "0123456789abcdef" for c in q0_digest):
        raise ValueError("q0 content identity is required")
    if datetime.fromisoformat(registered_at_utc).tzinfo is None:
        raise ValueError("registration timestamp requires an explicit timezone")
    if (
        execution_kind not in {"fixture", "gb10_measured"}
        or scales.execution_kind != execution_kind
    ):
        raise ValueError(
            "real calibration requires measured scales; fixture identity must stay explicit"
        )
    if (
        scales.status != "ready"
        or scales.scales is None
        or scales.reasons
        or scales.plan.q0_digest != q0_digest
    ):
        raise ValueError("ready fixed three-axis q0 scales are required before calibration")
    samples = tuple(
        RequestedSample(i, digest_json(records[i].prompt), records[i].source_group, seed)
        for i in prompt_ids
        for seed in seeds
    )
    return GroupPilotPlan(
        q0_digest,
        manifest.digest,
        samples,
        tuple(edit_counts),
        proposals_per_count,
        max_attempts,
        random_seed,
        registered_at_utc,
        scales.sha256,
        scales.fixed_generation_sha256,
        scales.plan.scorer_sha256,
        execution_kind,
    )


def formal_readiness(evidence, *, budget_used_seconds, per_device_seconds):
    """Report missing measured facts. This report never grants formal authority."""
    if (
        not math.isfinite(budget_used_seconds)
        or budget_used_seconds < 0
        or set(per_device_seconds) != {"gooo", "goooinext"}
        or any(not math.isfinite(v) or v < 0 for v in per_device_seconds.values())
        or not math.isclose(sum(per_device_seconds.values()), budget_used_seconds, abs_tol=1e-6)
    ):
        raise ValueError("inconsistent authoritative occupancy totals")
    missing, problems, fields = [], {}, {}
    for key in FORMAL_FIELDS:
        try:
            fields[key] = _resolve_formal_field(key, evidence.get(key))
        except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError) as error:
            missing.append(key)
            problems[key] = str(error)
            fields[key] = None
    if not missing:
        try:
            _crosscheck_formal_fields(fields)
        except (ValueError, TypeError, KeyError) as error:
            missing.append("formal-field-consistency")
            problems["formal-field-consistency"] = str(error)
    exceeded = budget_used_seconds > 57600 or any(v > 28800 for v in per_device_seconds.values())
    if exceeded:
        missing.append("pilot-budget-compliance")
    return {
        "schema": "mage-formal-readiness/v2",
        "status": "runtime-blocked"
        if evidence.get("runtime_blocker")
        else "insufficient-evidence"
        if missing
        else "ready-to-freeze",
        "missing": missing,
        "problems": problems,
        "pilot_used_seconds": budget_used_seconds,
        "pilot_per_device_seconds": per_device_seconds,
        "formal_execution_authorized": False,
        "formal_budget_includes_final": not missing,
        "fields": fields,
        "evidence": {key: evidence.get(key) for key in FORMAL_FIELDS},
    }


def _number(value):
    return type(value) in {int, float} and math.isfinite(value)


def _artifact(reference):
    """Read an explicitly supplied small JSON evidence file, never discover data."""
    import hashlib
    import json
    from pathlib import Path

    if not isinstance(reference, dict) or set(reference) != {"path", "sha256", "size_bytes"}:
        raise ValueError("artifact requires a path, raw file SHA-256 and byte size")
    if (
        type(reference["size_bytes"]) is not int
        or not 0 < reference["size_bytes"] <= 64 * 1024 * 1024
        or not isinstance(reference["path"], str)
    ):
        raise ValueError("invalid artifact size/path")
    path = Path(reference["path"])
    before = path.stat()
    if before.st_size != reference["size_bytes"]:
        raise ValueError("artifact size mismatch")
    raw = path.read_bytes()
    after = path.stat()
    if (
        any(
            getattr(before, key) != getattr(after, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        )
        or hashlib.sha256(raw).hexdigest() != reference["sha256"]
    ):
        raise ValueError("artifact SHA-256 mismatch or concurrent modification")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("artifact must contain a JSON object")
    return value


def audit_occupancy(snapshot):
    """Recompute unique lease intervals. Nested timing is descriptive, never billed twice."""
    if not isinstance(snapshot.get("campaign_id"), str) or not _number(snapshot.get("at")):
        raise ValueError("authoritative campaign snapshot identity/time required")
    leases, duplicates = {}, []
    for row in snapshot["leases"]:
        identity = row["lease_id"]
        if identity in leases:
            if row != leases[identity]:
                raise ValueError("conflicting duplicate occupancy lease")
            duplicates.append(identity)
            continue
        leases[identity] = row
    devices = {name: [] for name in ("gooo", "goooinext")}
    active, overruns = [], []
    for row in leases.values():
        start = row["started_at"]
        end = snapshot["at"] if row["stopped_at"] is None else row["stopped_at"]
        deadline = row["hard_deadline"]
        if (
            row["device_id"] not in devices
            or not all(_number(v) for v in (start, end, deadline))
            or end < start
            or end > snapshot["at"]
            or deadline <= start
        ):
            raise ValueError("invalid occupancy interval")
        devices[row["device_id"]].append((start, end, row["lease_id"]))
        if row["stopped_at"] is None:
            active.append(row["lease_id"])
        if end > deadline:
            overruns.append({"lease_id": row["lease_id"], "seconds": end - deadline})
    per_device = {}
    for device, intervals in devices.items():
        intervals.sort()
        if any(
            current[0] < previous[1]
            for previous, current in zip(intervals, intervals[1:], strict=False)
        ):
            raise ValueError("overlapping occupancy intervals on the same device")
        per_device[device] = math.fsum(end - start for start, end, _ in intervals)
        declared = snapshot["devices"][device]["used_seconds"]
        if not _number(declared) or not math.isclose(
            declared, per_device[device], abs_tol=1e-6, rel_tol=0
        ):
            raise ValueError("device total does not match unique occupancy intervals")
    total = math.fsum(per_device.values())
    if not _number(snapshot.get("used_seconds")) or not math.isclose(
        total, snapshot["used_seconds"], abs_tol=1e-6, rel_tol=0
    ):
        raise ValueError("campaign total does not match unique occupancy intervals")
    events = {}
    for event in snapshot.get("timing_events", ()):
        if event["event_id"] in events and event != events[event["event_id"]]:
            raise ValueError("conflicting duplicate timing event")
        if (
            not all(_number(event.get(key)) for key in ("started_at", "ended_at"))
            or event["ended_at"] < event["started_at"]
        ):
            raise ValueError("invalid component timing interval")
        if event["kind"] == "device_component":
            if event["lease_id"] not in leases:
                raise ValueError("component references an unknown occupancy lease")
            lease = leases[event["lease_id"]]
            end = snapshot["at"] if lease["stopped_at"] is None else lease["stopped_at"]
            if event["started_at"] < lease["started_at"] or event["ended_at"] > end:
                raise ValueError("component interval outside its occupancy lease")
        events[event["event_id"]] = event
    intervals = [interval for rows in devices.values() for interval in rows]
    return {
        "campaign_id": snapshot["campaign_id"],
        "at": snapshot["at"],
        "used_seconds": total,
        "per_device_seconds": per_device,
        "remaining_seconds": max(0, 57600 - total),
        "remaining_per_device_seconds": {
            key: max(0, 28800 - value) for key, value in per_device.items()
        },
        "wall_seconds": max(row[1] for row in intervals) - min(row[0] for row in intervals)
        if intervals
        else 0,
        "leases": list(leases.values()),
        "duplicate_lease_ids": sorted(set(duplicates)),
        "active_leases": active,
        "deadline_overruns": overruns,
        "budget_exhausted": total >= 57600 or any(value >= 28800 for value in per_device.values()),
        "component_intervals": list(events.values()),
        "component_intervals_are_additional_charges": False,
    }


def closeout_report(snapshot, *, components, evidence, runtime_dispositions=()):
    """Merge explicit component receipts; a component claim is not scientific qualification."""
    accounting = audit_occupancy(snapshot)
    leases = {row["lease_id"]: row for row in accounting["leases"]}
    receipts, images, measured_groups = [], {}, {}
    for component in components:
        if component["ticket_id"] not in {f"{i:02d}" for i in range(1, 9)}:
            raise ValueError("only pilot tickets may enter this closeout")
        value = _artifact(component["artifact"])
        if value.get("purpose") in {"confirmation", "final"} or value.get("scope") in {
            "confirmation",
            "final",
        }:
            raise ValueError("sealed evaluation content cannot enter pilot reports")
        lease_id = value.get("lease_id", component.get("lease_id"))
        if lease_id is not None:
            lease = leases.get(lease_id)
            if (
                lease is None
                or lease["ticket_id"] != component["ticket_id"]
                or lease["device_id"] != component["device_id"]
            ):
                raise ValueError("component identity does not match authoritative occupancy")
        if value.get("schema") == "mage-group-calibration/v1":
            _replay_group_receipt(value)
            group = str(value["group_id"])
            if (
                group in measured_groups
                and measured_groups[group]["artifact"] != component["artifact"]
            ):
                raise ValueError("duplicate conflicting group calibration evidence")
            measured_groups[group] = {
                "status": value["status"],
                "selected_edit_count": value["selected_edit_count"],
                "selected_candidate_id": value["selected_candidate_id"],
                "counts": value["counts"],
                "outcomes": value["outcomes"],
                "component_seconds": value["component_seconds"],
                "artifact": component["artifact"],
            }
        for row in value.get("rows", ()):
            image_hash = row.get(
                "image_file_sha256", row.get("image_sha256", row.get("png_sha256"))
            )
            if image_hash:
                images.setdefault(image_hash, []).append(
                    {"artifact": component["artifact"], "sample_id": row.get("sample_id")}
                )
        receipts.append(
            {
                **component,
                "file_verified": True,
                "schema": value.get("schema", value.get("schema_version")),
                "scope": value.get("scope", value.get("purpose")),
                "claimed_qualified": value.get("qualified"),
                "status": value.get("status"),
                "lease_id": lease_id,
                "execution_kind": value.get("execution_kind"),
                "cpu_seconds": value.get("cpu_seconds"),
                "wall_seconds": value.get("wall_seconds"),
                "limitation": value.get("limitation"),
                "formal_applicability": value.get("formal_applicability"),
                "initialization_summary": {
                    key: value.get(key)
                    for key in ("rows", "images_recognized", "human_calibration_completed")
                }
                if value.get("schema") == "mage-w4-ocr-initialization-summary-v1"
                else None,
            }
        )
    readiness = formal_readiness(
        evidence,
        budget_used_seconds=accounting["used_seconds"],
        per_device_seconds=accounting["per_device_seconds"],
    )
    from .runtime_incidents import assess_runtime_incidents

    incidents = assess_runtime_incidents(accounting, dispositions=runtime_dispositions)
    if incidents["runtime_blocked"]:
        readiness["status"] = "runtime-blocked"
        readiness["missing"].append("closed-compliant-occupancy")
    return {
        "schema": "mage-pilot-closeout/v1",
        "accounting": accounting,
        "runtime_incidents": incidents,
        "components": receipts,
        "duplicate_image_evidence": {key: value for key, value in images.items() if len(value) > 1},
        "duplicate_images_are_additional_generation_charges": False,
        "groups": {
            str(i): measured_groups.get(
                str(i), {"status": "unmeasured", "selected_edit_count": None}
            )
            for i in range(13)
        },
        "readiness": readiness,
        "formal_execution_authorized": False,
        "stop_reason": "budget-exhausted"
        if accounting["budget_exhausted"]
        else ("calibration-grid-complete" if len(measured_groups) == 13 else "incomplete-evidence"),
        "formal_budget_seconds": math.fsum(
            readiness["fields"]["formal_total_device_seconds"].values()
        )
        if readiness["status"] == "ready-to-freeze"
        else None,
        "confirmation_and_final_results_opened": False,
    }


@dataclass(frozen=True)
class ProposalOutcome:
    candidate_id: str
    status: str  # scored / failed / duplicate; failure is not a reward of zero
    attempts: int
    actual_changes: int
    candidate_digest: str | None
    component_seconds: float  # descriptive; occupancy remains in the shared ledger
    reason: str | None = None


def calibrate_group(
    plan: GroupPilotPlan,
    scales: ScaleReceipt,
    *,
    group_id,
    outcomes,
    candidates,
    current_references,
    q0_references,
    source_artifacts=(),
):
    """Choose a density from one complete preregistered grid against fixed q0.

    Uses ticket 05's actual paired three-axis screen. It does not accept a new
    center: ticket 06 owns that transaction, with development mock confirmation.
    The fixed q0 center across the grid avoids changing the density comparison's
    baseline. Missing/failed proposals stay visible; no runner-up trial is added.
    """
    from .rewards import screen_candidates

    if (
        plan.scales_sha256 != scales.sha256
        or plan.fixed_generation_sha256 != scales.fixed_generation_sha256
        or scales.status != "ready"
        or scales.execution_kind != plan.execution_kind
        or plan.scorer_sha256 != scales.plan.scorer_sha256
    ):
        raise ValueError("pilot fixed scales/runtime/scorers changed")
    if type(group_id) is not int or not 0 <= group_id < 13:
        raise ValueError("invalid registered layer group")
    proposals = {
        row["candidate_id"]: row for row in plan.candidates() if row["group_id"] == group_id
    }
    received = {row.candidate_id: row for row in outcomes}
    if len(received) != len(outcomes) or set(received) != set(proposals):
        raise ValueError("every proposal in the complete finite grid needs one terminal outcome")
    valid_ids, seen_payloads = [], set()
    for candidate_id in proposals:
        row = received[candidate_id]
        proposal = proposals[candidate_id]
        if (
            row.status not in {"scored", "failed", "duplicate"}
            or type(row.attempts) is not int
            or not 1 <= row.attempts <= plan.max_attempts
            or type(row.actual_changes) is not int
            or row.actual_changes < 0
            or not _number(row.component_seconds)
            or row.component_seconds < 0
        ):
            raise ValueError("invalid finite proposal outcome")
        if row.status == "scored":
            if (
                row.actual_changes != proposal["mutation_count"]
                or row.candidate_digest in seen_payloads
            ):
                raise ValueError(
                    "actual changes differ from the registered proposal or duplicate payload"
                )
            valid_ids.append(candidate_id)
            seen_payloads.add(row.candidate_digest)
        elif not row.reason:
            raise ValueError("failed/duplicate proposals need an explicit reason")
        if row.status == "duplicate" and row.candidate_digest not in seen_payloads:
            raise ValueError("duplicate proposal must reference an earlier scored payload")
    batches = {row.request.candidate_id: row for row in candidates}
    if len(batches) != len(candidates) or set(batches) != set(valid_ids):
        raise ValueError("only scored proposals may provide candidate batches")
    registered = datetime.fromisoformat(plan.registered_at_utc)
    for candidate_id, batch in batches.items():
        request = batch.request
        proposal = proposals[candidate_id]
        if (
            request.group_id != group_id
            or request.mutation_seed != proposal["mutation_seed"]
            or request.effective_edits != proposal["mutation_count"]
            or request.candidate_digest != received[candidate_id].candidate_digest
            or request.samples != plan.samples
            or request.manifest_digest != plan.manifest_digest
            or request.q0_digest != plan.q0_digest
            or request.center_digest != plan.q0_digest
            or batch.execution_kind != plan.execution_kind
            or datetime.fromisoformat(batch.started_at_utc) < registered
        ):
            raise ValueError(
                "batch differs from registered proposal/data/center or predates registration"
            )
    if set(current_references) != set(valid_ids) or set(q0_references) != set(valid_ids):
        raise ValueError("every scored proposal requires both paired references")
    if plan.execution_kind == "gb10_measured":
        _verify_batch_sources(
            tuple(batches.values())
            + tuple(current_references.values())
            + tuple(q0_references.values()),
            source_artifacts,
        )
    screened = (
        screen_candidates(
            scales, candidates, current_references, q0_references, candidate_order=tuple(valid_ids)
        )
        if valid_ids
        else None
    )
    positives = (
        [
            row
            for row in screened.candidates
            if row.feasible and row.current.composite_improvement > 0
        ]
        if screened
        else []
    )
    winner = (
        min(
            positives,
            key=lambda row: (
                -row.current.composite_improvement,
                received[row.candidate_id].actual_changes,
                tuple(proposals).index(row.candidate_id),
            ),
        )
        if positives
        else None
    )
    return {
        "schema": "mage-group-calibration/v1",
        "plan_sha256": plan.sha256,
        "group_id": group_id,
        "plan": asdict(plan),
        "scales": asdict(scales),
        "execution_kind": plan.execution_kind,
        "q0_digest": plan.q0_digest,
        "scales_sha256": plan.scales_sha256,
        "fixed_generation_sha256": plan.fixed_generation_sha256,
        "status": "no-positive-feasible-candidate"
        if winner is None
        else (
            "measured-selection"
            if plan.execution_kind == "gb10_measured"
            else "fixture-selection-only"
        ),
        "selected_candidate_id": winner.candidate_id if winner else None,
        "selected_edit_count": received[winner.candidate_id].actual_changes if winner else None,
        "formal_density_frozen": False,
        "selection_rule": plan.selection_rule,
        "counts": {
            **{
                status: sum(row.status == status for row in outcomes)
                for status in ("scored", "failed", "duplicate")
            },
            "positive_feasible": len(positives),
        },
        "component_seconds": math.fsum(row.component_seconds for row in outcomes),
        "component_seconds_are_additional_charges": False,
        "outcomes": [asdict(received[key]) for key in proposals],
        "screening": asdict(screened) if screened else None,
        "source_artifacts": list(source_artifacts),
        "inputs": {
            "candidates": [asdict(batches[key]) for key in valid_ids],
            "current_references": {key: asdict(current_references[key]) for key in valid_ids},
            "q0_references": {key: asdict(q0_references[key]) for key in valid_ids},
        },
    }


def validate_native_sample_binding(request, manifest, report):
    """Check native.generate's sample-content hash, not the full request hash.

    This is an identity check only: CPU fixture reports retain native_compute
    false. Full request-to-score binding is checked by the scored receipts.
    """
    from ..mage_qad_checkpoint import content_hash

    prompts = {row.prompt_id: row.prompt for row in manifest.records}
    try:
        samples = [
            {"sample_id": row.sample_id, "prompt": prompts[row.prompt_id], "seed": row.seed}
            for row in request.samples
        ]
    except KeyError as error:
        raise ValueError("native sample references an unknown prompt") from error
    if report.get("request_sha256") != content_hash(samples):
        raise ValueError(
            "native sample-content identity differs from the requested prompt/seed sequence"
        )


def _verify_batch_sources(batches, references):
    """Measured labels need content-checkable native and per-image score receipts."""
    docs = {}
    for reference in references:
        value = _artifact(reference)
        # Artifact reference is the raw file hash; receipts use canonical JSON.
        docs[digest_json(value)] = value
    for batch in batches:
        native = docs.get(batch.native_receipt_sha256)
        if (
            not native
            or native.get("native_compute") is not True
            or native.get("qualified") is not True
            or native.get("center_sha256") != batch.request.candidate_digest
        ):
            raise ValueError("measured batch requires its verified native generation artifact")
        validate_native_sample_binding(batch.request, batch.manifest, native)
        native_rows = {row["sample_id"]: row for row in native.get("rows", ())}
        if (
            set(native_rows) != {row.sample_id for row in batch.samples}
            or len(native_rows) != len(native.get("rows", ()))
            or native.get("generation") != batch.request.runtime_identity["generation"]
            or native.get("runtime", {}).get("compute_capability") != [12, 1]
            or native.get("installation", {}).get("backend") != "torchao_nvfp4"
        ):
            raise ValueError("native receipt lacks exact sample/config/GB10 compute binding")
        for row in batch.samples:
            score = docs.get(row.score_receipt_sha256)
            if not score:
                raise ValueError("measured batch requires every actual three-axis score artifact")
            if (
                score.get("image_sha256") != row.image_sha256
                or score.get("sample_id") != row.sample_id
            ):
                raise ValueError("score artifact sample/image identity mismatch")
            if native_rows[row.sample_id].get("image_file_sha256") != row.image_sha256:
                raise ValueError("scored image differs from the native generation receipt")
            text_score = score.get("text_score")
            if (
                score.get("request_digest") != batch.request.request_digest
                or score.get("native_report_sha256") != batch.native_receipt_sha256
                or score.get("scorer_sha256") != dict(batch.scorer_sha256)
                or score.get("execution_kind") != "gb10_measured"
                or score.get("clip") != row.clip
                or score.get("hps") != row.hps
                or (text_score.get("character_correctness") if text_score else None) != row.ocr
                or (text_score.get("status") if text_score else "not_applicable") != row.ocr_status
            ):
                raise ValueError("score artifact differs from the actual three-axis batch")
            if row.ocr_status == "ok":
                _verify_scored_ocr(batch, row, score)


def _resolve_formal_field(key, reference):
    """Every field has an inspectable value and independently hashed basis files.

    The field file is a proposal, never an authority. Type-specific checks below
    replay the measurement where possible. Numerical design choices additionally
    require an explicit rationale and development/method evidence; hashes alone
    cannot establish scientific premises or grant formal execution permission.
    """
    doc = _artifact(reference)
    if (
        doc.get("schema") != "mage-formal-field/v1"
        or doc.get("field") != key
        or not isinstance(doc.get("rationale"), str)
        or not doc["rationale"].strip()
        or not isinstance(doc.get("basis"), list)
        or not doc["basis"]
    ):
        raise ValueError("missing typed field, explicit rationale or verifiable basis")
    basis = [_artifact(item) for item in doc["basis"]]
    value = doc["value"]
    if key == "native_two_host":
        if set(value["hosts"]) != {"gooo", "goooinext"}:
            raise ValueError("both registered GB10 native receipts are required")
        hosts = [_artifact(ref) for ref in value["hosts"].values()]
        for host in hosts:
            replay = host.get("replay", {})
            if (
                host.get("schema") != "mage-native-probe/v1"
                or host.get("qualified") is not True
                or host.get("zero_update_images_equal") is not True
                or replay.get("native_compute") is not True
                or replay.get("packed_reload_equal") is not True
                or replay.get("images_match") is not True
                or not replay.get("comparisons")
                or any(row.get("passed") is not True for row in replay["comparisons"])
            ):
                raise ValueError("native replay evidence is incomplete")
        comparison = _artifact(value["comparison"])
        if (
            hosts[0]["q0_sha256"] != hosts[1]["q0_sha256"]
            or hosts[0]["fixed_state_sha256"] != hosts[1]["fixed_state_sha256"]
            or set(comparison.get("identity_checks", {}))
            != {"center", "fixed_state", "generation", "request", "native"}
            or not all(item is True for item in comparison["identity_checks"].values())
            or not comparison.get("rows")
            or any(row.get("passed") is not True for row in comparison["rows"])
        ):
            raise ValueError("two-host native identity/comparison evidence failed")
    elif key == "ocr_lock":
        _validate_ocr_lock(value)
    elif key == "scale_receipt":
        _validate_scales(value)
    elif key == "measured_group_calibrations":
        _validate_groups(value)
    else:
        supported = False
        for item in basis:
            if item.get("schema") == "mage-group-calibration/v1":
                _replay_group_receipt(item)
                supported = supported or item["status"] == "measured-selection"
            elif item.get("schema") == "mage-method-audit/v1":
                from .statistical import METHOD

                if (
                    item.get("method") != METHOD
                    or not item.get("analysis")
                    or not item.get("premises")
                    or not item.get("source_artifacts")
                ):
                    raise ValueError(
                        "method audit needs explicit analysis, premises and source artifacts"
                    )
                for source in item["source_artifacts"]:
                    _artifact(source)
                supported = True
        if not supported:
            raise ValueError(
                "formal choices require actual group measurement or explicit method audit evidence"
            )
        if key in {"candidate_batch_size", "confirmation_attempts"}:
            if type(value) is not int or value <= 0:
                raise ValueError("a positive measured/planned integer is required")
        elif key in {"search_samples", "final_samples"}:
            _sample_design(value)
        elif key == "confirmation_slices":
            if not isinstance(value, list) or not value:
                raise ValueError("explicit confirmation slices are required")
            for item in value:
                _sample_design(item)
        elif key == "confirmation_alphas":
            if (
                not isinstance(value, list)
                or not value
                or any(not _number(a) or not 0 < a <= 0.05 for a in value)
                or math.fsum(value) > 0.05
            ):
                raise ValueError("confirmation alpha allocation must total at most 0.05")
        elif key == "joint_interval_method":
            from .statistical import METHOD

            if value != METHOD:
                raise ValueError("unsupported joint interval method")
        elif key == "method_applicability":
            if (
                value.get("applicable") is not True
                or not value.get("premises")
                or not all(item is True for item in value["premises"].values())
                or not any(item.get("status") == "synthetic-coverage-supported" for item in basis)
            ):
                raise ValueError(
                    "method needs supported coverage and explicitly audited real premises"
                )
        elif key == "score_support_audit":
            if set(value) != {"clip", "hps", "ocr"} or list(value["ocr"]) != [0, 1]:
                raise ValueError("three frozen score supports, including OCR [0,1], are required")
            if any(
                len(bound) != 2 or not all(_number(n) for n in bound) or bound[0] >= bound[1]
                for bound in value.values()
            ):
                raise ValueError("invalid analytical score support")
        elif key == "independence_audit":
            if (
                value.get("independent_source_clusters") is not True
                or value.get("sampling_protocol") is None
                or value.get("source_group_overlap") != []
                or value.get("prompt_overlap") != []
            ):
                raise ValueError(
                    "independent sampling and absence of cross-pool overlap need explicit audit"
                )
        elif key == "detectable_improvement":
            for stage in ("confirmation", "final"):
                design = value[stage]
                if (
                    set(design["joint_lower_bound_widths"]) != {"clip", "hps", "ocr", "composite"}
                    or any(
                        not _number(n) or n <= 0
                        for n in design["joint_lower_bound_widths"].values()
                    )
                    or not _number(design["detectable_positive_composite"])
                    or design["detectable_positive_composite"]
                    < design["joint_lower_bound_widths"]["composite"]
                    or not _number(design["power"])
                    or not 0 < design["power"] <= 1
                ):
                    raise ValueError(
                        "each stage needs finite joint widths and an explicit effect/power assessment"
                    )
        elif key == "formal_total_device_seconds":
            _stage_budget(value)
        elif key == "formal_per_device_seconds":
            if set(value) != {"gooo", "goooinext"}:
                raise ValueError("formal budget needs both registered devices")
            for row in value.values():
                _stage_budget(row)
        elif key == "final_reserved_device_seconds" and (not _number(value) or value <= 0):
            raise ValueError("final execution needs a positive separate reservation")
    return value


def _sample_design(value):
    if (
        not isinstance(value.get("samples"), list)
        or not value["samples"]
        or not value.get("ocr_prompt_ids")
    ):
        raise ValueError("frozen prompts/seeds and complete OCR subset required")
    samples = [RequestedSample(**row) for row in value["samples"]]
    if (
        len({row.sample_id for row in samples}) != len(samples)
        or len(set(value["ocr_prompt_ids"])) != len(value["ocr_prompt_ids"])
        or not set(value["ocr_prompt_ids"]) <= {row.prompt_id for row in samples}
        or any(type(row.seed) is not int or not 0 <= row.seed < 2**32 for row in samples)
    ):
        raise ValueError("duplicate or invalid sample design")


def _stage_budget(value):
    if set(value) != {"search", "confirmation", "final"} or any(
        not _number(seconds) or seconds <= 0 for seconds in value.values()
    ):
        raise ValueError("explicit positive search, confirmation and final time estimates required")


def _batch_from_dict(value):
    from .contracts import CandidateRequest
    from .rewards import SampleScores, ScoredBatch

    return ScoredBatch(
        CandidateRequest.from_dict(value["request"]),
        DataManifest.from_dict(value["manifest"]),
        tuple(SampleScores(**row) for row in value["samples"]),
        tuple(value["ocr_prompt_ids"]),
        tuple(tuple(row) for row in value["scorer_sha256"]),
        value["native_receipt_sha256"],
        value["started_at_utc"],
        value["execution_kind"],
    )


def _scales_from_dict(value):
    from .rewards import AxisValues, ScalePlan

    plan = value["plan"]
    plan = ScalePlan(
        **{
            **plan,
            "samples": tuple(RequestedSample(**row) for row in plan["samples"]),
            "ocr_prompt_ids": tuple(plan["ocr_prompt_ids"]),
            "scorer_sha256": tuple(tuple(row) for row in plan["scorer_sha256"]),
        }
    )
    return ScaleReceipt(
        **{
            **value,
            "plan": plan,
            "scales": AxisValues(**value["scales"]) if value["scales"] else None,
            "margins": AxisValues(**value["margins"]) if value["margins"] else None,
            "reasons": tuple(value["reasons"]),
        }
    )


def _validate_scales(value):
    from .rewards import calibrate_scales

    receipt = _scales_from_dict(_artifact(value["receipt"]))
    baseline = _batch_from_dict(_artifact(value["baseline"]))
    if receipt.execution_kind != "gb10_measured" or receipt.status != "ready":
        raise ValueError("real reliable three-axis calibration is required")
    _verify_batch_sources((baseline,), value["source_artifacts"])
    replay = calibrate_scales(receipt.plan, baseline)
    if replay.sha256 != receipt.sha256:
        raise ValueError("scale receipt does not replay from its actual baseline")


def _validate_ocr_lock(value):
    from .ocr import freeze_scorer

    report = validate_ocr_archive(value)
    if digest_json(freeze_scorer(report)) != digest_json(_artifact(value["lock"])):
        raise ValueError("OCR lock does not replay from actual transcriptions and observations")


def validate_ocr_archive(value):
    """Replay OCR archive identities without claiming measured qualification.

    OCR objects own their SHA algorithm. Raw artifact hashes and canonical JSON
    comparisons verify storage/content only, never replace object identities.
    The formal wrapper separately invokes freeze_scorer's measured-data gate.
    """
    from .ocr import CalibrationPlan, HumanTranscription, compare_ocr, scorer_lock_identity
    from .scorers import decode_ocr_config

    lock = _artifact(value["lock"])
    report = _artifact(value["calibration"])
    truths = [_artifact(ref) for ref in value["human_transcriptions"]]
    raw_plan = report["plan"]
    plan = CalibrationPlan(
        **{
            **raw_plan,
            "image_sha256": tuple(raw_plan["image_sha256"]),
            "configs": tuple(decode_ocr_config(row) for row in raw_plan["configs"]),
            "generation_receipt_sha256": tuple(raw_plan["generation_receipt_sha256"]),
        }
    )
    config = decode_ocr_config(lock["config"])
    if (
        lock.get("schema") != "mage-w4-ocr-scorer-lock-v1"
        or plan.sha256 != lock["plan_sha256"]
        or config.sha256 != lock["config_sha256"]
    ):
        raise ValueError("OCR archive requires exact typed object identities")
    device_digest = report.get("device_evidence_sha256")
    if value.get("device_evidence") is None:
        if device_digest is not None:
            raise ValueError("missing OCR device evidence artifact")
    else:
        device = _artifact(value["device_evidence"])
        # The public OCR contract accepts an opaque external receipt identity.
        # Verify it against its supplied raw-file or canonical-JSON artifact;
        # feed the unchanged identity back into the typed replay.
        if device_digest not in {value["device_evidence"]["sha256"], digest_json(device)}:
            raise ValueError("OCR device evidence identity mismatch")
    expected = {row["image_sha256"]: row for row in report["transcriptions"]}
    if (
        len(expected) != len(plan.image_sha256)
        or len(truths) != len(expected)
        or {row["image_sha256"] for row in truths} != set(expected)
    ):
        raise ValueError("complete actual human transcription artifacts are required")
    for truth, reference in zip(truths, value["human_transcriptions"], strict=True):
        recorded = expected[truth["image_sha256"]]
        if (
            truth.get("source_kind") != "human_image_transcription"
            or not truth.get("annotator_id")
            or truth.get("complete") is not True
            or truth.get("actual_segments") != recorded["actual_segments"]
            or truth["annotator_id"] != recorded["annotator_id"]
            or reference["sha256"] != recorded["artifact_sha256"]
        ):
            raise ValueError("OCR truth must be complete actual human image transcription")
    observed = tuple(_recognition_from_dict(_artifact(ref)) for ref in value["observations"])
    costs = (
        ()
        if report["cost_basis"] == "fixture_recognition_seconds"
        else tuple(
            (decode_ocr_config(row["config"]).sha256, row["occupied_seconds"])
            for row in report["engines"]
        )
    )
    replay = compare_ocr(
        plan,
        tuple(
            HumanTranscription(**{**row, "actual_segments": tuple(row["actual_segments"])})
            for row in report["transcriptions"]
        ),
        observed,
        execution_kind=report["execution_kind"],
        device_evidence_sha256=device_digest,
        occupied_seconds_by_config=costs,
    )
    if (
        digest_json(asdict(replay)) != digest_json(report)
        or replay.sha256 != lock["calibration_report_sha256"]
        or digest_json(scorer_lock_identity(replay)) != digest_json(lock)
    ):
        raise ValueError("OCR archive does not replay from its typed identities and observations")
    return replay


def _validate_groups(value):
    if set(value["groups"]) != {str(i) for i in range(13)}:
        raise ValueError("all 13 actual group calibrations are required")
    plan_data = _artifact(value["plan"])
    plan = GroupPilotPlan(
        **{
            **plan_data,
            "samples": tuple(RequestedSample(**row) for row in plan_data["samples"]),
            "edit_counts": tuple(plan_data["edit_counts"]),
            "scorer_sha256": tuple(tuple(row) for row in plan_data["scorer_sha256"]),
        }
    )
    scales = _scales_from_dict(_artifact(value["scales"]))
    if plan.execution_kind != "gb10_measured":
        raise ValueError("fixture plan cannot support formal densities")
    for group_id, reference in value["groups"].items():
        receipt = _artifact(reference)
        if (
            receipt["group_id"] != int(group_id)
            or receipt["plan_sha256"] != plan.sha256
            or receipt["scales_sha256"] != scales.sha256
        ):
            raise ValueError("group measurement belongs to a different plan/scales/group")
        _replay_group_receipt(receipt)
        if receipt["status"] != "measured-selection":
            raise ValueError(
                "group density is not reproducible from positive feasible actual three-axis evidence"
            )


def _replay_group_receipt(receipt):
    raw = receipt["plan"]
    plan = GroupPilotPlan(
        **{
            **raw,
            "samples": tuple(RequestedSample(**row) for row in raw["samples"]),
            "edit_counts": tuple(raw["edit_counts"]),
            "scorer_sha256": tuple(tuple(row) for row in raw["scorer_sha256"]),
        }
    )
    if plan.execution_kind != "gb10_measured":
        raise ValueError("fixture group receipt cannot support formal fields")
    inputs = receipt["inputs"]
    replay = calibrate_group(
        plan,
        _scales_from_dict(receipt["scales"]),
        group_id=receipt["group_id"],
        outcomes=tuple(ProposalOutcome(**row) for row in receipt["outcomes"]),
        candidates=tuple(_batch_from_dict(row) for row in inputs["candidates"]),
        current_references={
            key: _batch_from_dict(row) for key, row in inputs["current_references"].items()
        },
        q0_references={key: _batch_from_dict(row) for key, row in inputs["q0_references"].items()},
        source_artifacts=receipt["source_artifacts"],
    )
    if digest_json(replay) != digest_json(receipt):
        raise ValueError("group measurement cannot be reproduced from its inputs")


def _crosscheck_formal_fields(fields):
    scales = _artifact(fields["scale_receipt"]["receipt"])
    group_plan = _artifact(fields["measured_group_calibrations"]["plan"])
    ocr_lock = _artifact(fields["ocr_lock"]["lock"])
    hosts = [_artifact(ref) for ref in fields["native_two_host"]["hosts"].values()]
    if (
        any(row["q0_sha256"] != scales["plan"]["q0_digest"] for row in hosts)
        or group_plan["q0_digest"] != scales["plan"]["q0_digest"]
        or group_plan["scales_sha256"] != digest_json(scales)
        or dict(scales["plan"]["scorer_sha256"])["ocr"] != ocr_lock["config_sha256"]
        or group_plan["fixed_generation_sha256"] != scales["fixed_generation_sha256"]
    ):
        raise ValueError(
            "native/OCR/scales/group evidence chains describe different frozen identities"
        )
    count = fields["confirmation_attempts"]
    slices, alphas = fields["confirmation_slices"], fields["confirmation_alphas"]
    if len(slices) != count or len(alphas) != count:
        raise ValueError("confirmation attempts/slices/alphas disagree")
    all_groups, all_prompts, all_ids = set(), set(), set()
    for design in (fields["search_samples"], *slices, fields["final_samples"]):
        groups = {row["source_group"] for row in design["samples"]}
        prompts = {row["prompt_digest"] for row in design["samples"]}
        ids = {row["prompt_id"] for row in design["samples"]}
        if all_groups & groups or all_prompts & prompts or all_ids & ids:
            raise ValueError("search, confirmation slices and final samples must be disjoint")
        all_groups.update(groups)
        all_prompts.update(prompts)
        all_ids.update(ids)
    total = fields["formal_total_device_seconds"]
    per_device = fields["formal_per_device_seconds"]
    if (
        any(
            not math.isclose(
                total[stage],
                math.fsum(row[stage] for row in per_device.values()),
                rel_tol=0,
                abs_tol=1e-6,
            )
            for stage in total
        )
        or total["final"] != fields["final_reserved_device_seconds"]
    ):
        raise ValueError("formal total/per-device/final reservation disagree")


def artifact_reference(path):
    """Describe one explicitly selected JSON file; does not discover files."""
    import hashlib
    from pathlib import Path

    path = Path(path).resolve()
    raw = path.read_bytes()
    reference = {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }
    _artifact(reference)
    return reference


def render_closeout_html(report):
    """Standalone factual closeout. Missing values stay visibly missing."""
    import html
    import json

    accounting = report["accounting"]
    labels = {
        "native_two_host": "双 GB10 原生执行",
        "ocr_lock": "OCR 评分器冻结",
        "scale_receipt": "三轴固定尺度",
        "measured_group_calibrations": "13 个层组实测",
        "candidate_batch_size": "每批候选数量",
        "search_samples": "搜索提示词、种子与文字子集",
        "confirmation_slices": "独立确认分片",
        "confirmation_attempts": "确认次数",
        "confirmation_alphas": "确认风险额度",
        "joint_interval_method": "联合区间方法",
        "method_applicability": "统计方法适用性",
        "score_support_audit": "评分取值范围依据",
        "independence_audit": "来源独立性",
        "final_samples": "一次终检样本",
        "detectable_improvement": "区间宽度与可检测改善",
        "formal_total_device_seconds": "正式阶段总机时",
        "formal_per_device_seconds": "正式阶段每机预算",
        "final_reserved_device_seconds": "一次终检预留机时",
        "closed-compliant-occupancy": "设备占用闭合及截止合规",
        "pilot-budget-compliance": "先导预算合规",
        "formal-field-consistency": "冻结字段一致性",
    }

    def shown(value):
        if value is None:
            return "未声明/不适用"
        return html.escape(str(value))

    def component_status(item):
        initialization = item.get("initialization_summary")
        if initialization is not None:
            engines = " / ".join(
                f"{row['engine']}：{row['status']}" for row in initialization["rows"]
            )
            return shown(
                f"{engines}；已识图 {initialization['images_recognized']} 张；人工校准未完成"
            )
        return shown(
            item.get("status") if item.get("status") is not None else item.get("claimed_qualified")
        )

    missing = "".join(
        f"<li><b>{html.escape(labels.get(key, key))}</b>："
        f"{'尚无已验证的依据文件' if report['readiness'].get('evidence', {}).get(key) is None else '依据尚未通过核验，详情见完整 JSON'}。</li>"
        for key in report["readiness"]["missing"]
    )
    rows = "".join(
        f"<tr><td>{html.escape(item['ticket_id'])}</td><td>{shown(item.get('device_id'))}</td>"
        f"<td>{shown(item.get('scope') or item.get('execution_kind') or item.get('schema'))}</td>"
        f"<td>{component_status(item)}</td>"
        f"<td><code>{html.escape(item['artifact']['sha256'][:16])}</code></td></tr>"
        for item in report["components"]
    )
    incidents = report.get("runtime_incidents", {})
    incident_summary = (
        f"已核验的迟到结算：{len(incidents.get('resolved', []))} 项；"
        f"未解决的运行异常：{len(incidents.get('unresolved', []))} 项。"
        f"历史预留超时仍计费：{sum(row['seconds'] for row in accounting['deadline_overruns']):.6f} 秒。"
        "运行异常处置不减少占用、不新增预算，也不补齐三轴或正式协议证据。"
    )
    payload = html.escape(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mage W4A4 · 组件阶段证据快照</title><style>
:root{{color-scheme:light dark;font-family:system-ui,sans-serif}}body{{max-width:1080px;margin:40px auto;padding:0 24px;line-height:1.65}}
h1{{font-size:2rem}}code,pre{{font-size:.85rem}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #8885}}
.kpis{{display:flex;gap:24px;flex-wrap:wrap}}.card{{padding:16px;border:1px solid #8885;border-radius:12px}}pre{{overflow:auto;max-height:650px}}.state{{color:#be7700;font-weight:700}}
</style><h1>Mage 直接 W4A4 · 组件阶段证据快照</h1>
<p class="state">{html.escape(report["readiness"]["status"])}</p>
<p>本页按已提供的开发证据与唯一占用台账生成。05–08 真实闭环尚未验收，等待实际图中文字的人工转录，再完成 OCR 比较与固定尺度标定。组件回执中的通过声明保留其原始范围；正式参数未核验时保持缺失。此快照不表示整个先导已完成或 campaign 已停止，也不授予正式执行权限。</p>
<div class="kpis"><div class="card">总设备时间<br><b>{accounting["used_seconds"] / 3600:.4f} / 16 小时</b></div>
<div class="card">gooo<br><b>{accounting["per_device_seconds"]["gooo"] / 3600:.4f} / 8 小时</b></div>
<div class="card">goooinext<br><b>{accounting["per_device_seconds"]["goooinext"] / 3600:.4f} / 8 小时</b></div>
<div class="card">首末占用墙钟跨度<br><b>{accounting["wall_seconds"] / 3600:.4f} 小时</b></div></div>
<h2>已有组件证据</h2><table><thead><tr><th>票</th><th>设备</th><th>回执范围</th><th>原始状态</th><th>文件 SHA</th></tr></thead><tbody>{rows}</tbody></table>
<p>组件计时为占用内的描述区间，未重复加到机时。相同图像仅标识复用/重放，不再额外计费。</p>
<p>{html.escape(incident_summary)}</p>
<h2>正式冻结缺口</h2><ul>{missing}</ul>
<p>13 组各自的未完成状态、实际改码、失败与重复记录见完整 JSON。正式搜索、独立确认与一次终检的预算均须另行由证据冻结，不填默认数。</p>
<details><summary>展开可重算的完整证据报告</summary><pre>{payload}</pre></details></html>"""


def _main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Read explicit pilot metadata and emit a factual closeout; never run GPU work."
    )
    parser.add_argument("--budget-snapshot", type=Path, required=True)
    parser.add_argument(
        "--evidence-manifest",
        type=Path,
        required=True,
        help="JSON with components[] and evidence{}; paths are explicit, no recursive discovery",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path)
    args = parser.parse_args()
    budget_ref, evidence_ref = (
        artifact_reference(args.budget_snapshot),
        artifact_reference(args.evidence_manifest),
    )
    manifest = _artifact(evidence_ref)
    report = closeout_report(
        _artifact(budget_ref),
        components=manifest.get("components", []),
        evidence=manifest.get("evidence", {}),
        runtime_dispositions=manifest.get("runtime_dispositions", []),
    )
    report["inputs"] = {"budget_snapshot": budget_ref, "evidence_manifest": evidence_ref}
    outputs = [args.output_json] + ([args.output_html] if args.output_html else [])
    if any(path.exists() for path in outputs):
        parser.error("output already exists; choose new report paths")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    if args.output_html:
        args.output_html.write_text(render_closeout_html(report))
    print(
        json.dumps(
            {
                "status": report["readiness"]["status"],
                "output": str(args.output_json),
                "used_seconds": report["accounting"]["used_seconds"],
                "formal_execution_authorized": False,
            }
        )
    )


def _recognition_from_dict(raw):
    from .ocr import RawSegment, RecognitionReceipt, TextSegment
    from .scorers import decode_ocr_config

    return RecognitionReceipt(
        **{
            **raw,
            "config": decode_ocr_config(raw["config"]),
            "raw_segments": tuple(
                RawSegment(**{**row, "box": tuple(row["box"])}) for row in raw["raw_segments"]
            ),
            "segments": tuple(
                TextSegment(
                    **{**row, "box": tuple(row["box"]), "raw_indices": tuple(row["raw_indices"])}
                )
                for row in raw["segments"]
            ),
        }
    )


def _verify_scored_ocr(batch, row, score):
    import json

    from .ocr import TargetInventory, match_text

    recognition = _recognition_from_dict(score["recognition"])
    sample = next(item for item in batch.request.samples if item.sample_id == row.sample_id)
    target = next(item for item in batch.manifest.records if item.prompt_id == sample.prompt_id)
    replay = match_text(
        recognition, TargetInventory(target.target_text, target.complete_target_inventory)
    )
    process = score.get("recognizer_process", {})
    settings = json.loads(recognition.config.settings_json)
    if (
        digest_json(asdict(replay)) != digest_json(score["text_score"])
        or recognition.config.sha256 != dict(batch.scorer_sha256)["ocr"]
        or recognition.image_sha256 != row.image_sha256
        or process.get("status") != "ok"
        or process.get("returncode") != 0
        or process.get("config_sha256") != recognition.config.sha256
        or process.get("image_sha256") != row.image_sha256
        or process.get("worker", {}).get("adapter_source_sha256")
        != settings.get("adapter_source_sha256")
        or digest_json(process.get("segments"))
        != digest_json([asdict(item) for item in recognition.raw_segments])
    ):
        raise ValueError("OCR score does not replay from its isolated target-blind recognition")


if __name__ == "__main__":
    _main()
