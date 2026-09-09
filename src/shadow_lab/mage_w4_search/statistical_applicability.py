"""Development diagnostics for the existing joint method, never formal authorization.

Distinct source labels and finite observed ranges do not establish its assumptions.
The CLI verifies the raw complete jobs before passing their costs to this module.
"""

from __future__ import annotations

import hashlib
import math
import platform
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .contracts import CandidateRequest, DataManifest, digest_json
from .development import (
    close_scale_registration,
    close_screen_registration,
    decode_scale_receipt,
    verify_evaluation_job,
)
from .human_transcription import read_json_artifact
from .statistical import METHOD

OCR_REVIEWED_SHA256 = "39dee97d2dc5ee8c9307e0765482c8ad403f326b709117ba3cd4c93b69571fb9"
ADAPTER_REVIEWED_SHA256 = "ee327f6ad6233ac59dee8b12c6da5232745ec8658e63b3861cd25fa2b2eddaad"


def support_audit(request, *, external_model_paths=None):
    """Bind reviewed local code and external model files to the frozen runtime.

    A file hash establishes identity, not a floating-point error bound. Even when
    all source identities match, no CLIP/HPS support proof is claimed here.
    """
    root = Path(__file__).resolve().parents[3]
    numerical = request.runtime_identity.get("numerical_implementation_sha256", {})
    paths = {
        "ocr": ("src/shadow_lab/mage_w4_search/ocr.py", OCR_REVIEWED_SHA256),
        "adapter": ("src/shadow_lab/image_reward_models.py", ADAPTER_REVIEWED_SHA256),
    }
    sources = {}
    for role, (relative, reviewed) in paths.items():
        path = root / relative
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        sources[role] = {
            "path": str(path),
            "file_sha256": actual,
            "size_bytes": len(data),
            "reviewed_sha256": reviewed,
            "reviewed_code_matches": actual == reviewed,
            "frozen_request_matches": numerical.get(relative) == actual,
        }
    external = request.runtime_identity.get("external_source_sha256", {})
    provided = external_model_paths or {}
    if set(provided) - {"clip", "hps"}:
        raise ValueError("only explicit CLIP/HPS model source paths are accepted")
    for axis, package, relative in (
        ("clip", "open_clip", "model.py"),
        ("hps", "hpsv2", "src/open_clip/model.py"),
    ):
        expected = external.get(package, {}).get(relative)
        if axis not in provided:
            sources[axis] = {"frozen_file_sha256": expected, "file_verified": False}
            continue
        path = Path(provided[axis]).resolve()
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError("external model source differs from the frozen scoring package")
        sources[axis] = {
            "path": str(path),
            "file_sha256": actual,
            "size_bytes": len(data),
            "frozen_file_sha256": expected,
            "file_verified": True,
        }
    ocr_proven = sources["ocr"]["reviewed_code_matches"]
    return {
        "axes": {
            "ocr": {
                "bound": [0, 1] if ocr_proven else None,
                "basis": "nonnegative integer edits / positive target length; max(0,1-CER)",
                "scope": "successful applicable scores; engine failure is missing, never zero",
                "request_bound": ocr_proven and sources["ocr"]["frozen_request_matches"],
            },
            **{
                axis: {
                    "bound": None,
                    "basis": "floating normalization, division, multiplication and reduction error not bounded",
                    "ideal_cosine_is_not_a_numerical_bound": True,
                }
                for axis in ("clip", "hps")
            },
        },
        "sources": sources,
        "observed_extrema_used_as_support": False,
        "finite_output_rejection_proves_no_narrow_support": True,
    }


def _cost_audit(batches, proofs):
    if not proofs:
        return None
    known = {batch.sha256: batch for batch in batches}
    unique = {}
    for proof in proofs:
        batch = known.get(proof.get("batch_sha256"))
        seconds = proof.get("occupied_seconds")
        if (
            batch is None
            or proof.get("schema") != "mage-w4-development-job-evidence/v1"
            or proof.get("request_digest") != batch.request.request_digest
            or proof.get("execution_kind") != batch.execution_kind
            or not isinstance(proof.get("lease_id"), str)
            or not proof["lease_id"]
            or proof.get("lease_id") != batch.request.lease_id
            or proof.get("device_id") != batch.request.device_id
            or type(seconds) not in {int, float}
            or not math.isfinite(seconds)
            or seconds < 0
            or not proof.get("artifacts")
            or proof.get("formal_execution_authorized") is not False
        ):
            raise ValueError("job evidence differs from the complete development batch")
        key = proof["lease_id"]
        if key in unique and digest_json(unique[key]) != digest_json(proof):
            raise ValueError("job evidence reuses one lease for different outputs or costs")
        unique[key] = proof
    rows = [
        {
            **proof,
            "image_count": len(known[proof["batch_sha256"]].samples),
            "occupied_seconds_per_image": proof["occupied_seconds"]
            / len(known[proof["batch_sha256"]].samples),
        }
        for _, proof in sorted(unique.items())
    ]
    return {
        "jobs": rows,
        "unique_occupied_seconds": math.fsum(row["occupied_seconds"] for row in rows),
        "missing_batch_costs": sorted(set(known) - {row["batch_sha256"] for row in rows}),
        "interpretation": "closed lease gross cost; nested generation/scoring timers are not added",
        "formal_budget": None,
    }


def conditional_ocr_gates(samples, ocr_prompt_ids, scenarios, *, cost=None):
    """Compute algebraic OCR gate thresholds, conditional on independent sources.

    These are required observed effects, not a power calculation or valid intervals
    for the selected development pool. No CLIP/HPS support is supplied by analogy.
    """
    prompts = {
        row.prompt_id: row.source_group for row in samples if row.prompt_id in ocr_prompt_ids
    }
    counts = Counter(prompts.values())
    if not counts:
        raise ValueError("OCR gate diagnostic requires a frozen nonempty applicable subset")
    seeds = {row.seed for row in samples}
    result = []
    for scenario in scenarios:
        if set(scenario) != {"stage", "attempt_alpha"}:
            raise ValueError("risk scenario must explicitly specify stage and attempt_alpha")
        stage, alpha = scenario["stage"], scenario["attempt_alpha"]
        if (
            stage not in {"confirmation", "final"}
            or type(alpha) not in {int, float}
            or not math.isfinite(alpha)
            or not 0 < alpha <= 0.05
            or (stage == "final" and alpha != 0.05)
        ):
            raise ValueError("invalid conditional risk scenario")
        per_check = alpha / (7 if stage == "confirmation" else 4)
        mass_squared = math.fsum((count / len(prompts)) ** 2 for count in counts.values())
        radius = math.sqrt(-2 * math.log(per_check) * mass_squared)
        required = math.ceil(-2 * math.log(per_check) / 0.01**2)
        projection = None
        if cost:
            rates = [row["occupied_seconds_per_image"] for row in cost["jobs"]]
            high = max(rates)
            projection = {
                "two_models_seconds": required * len(seeds) * 2 * high,
                "three_models_seconds": required * len(seeds) * 3 * high,
                "observed_job_rate_range": [min(rates), high],
                "interpretation": "linear illustration at the slowest measured gross batch rate, not an upper cost bound",
                "assumptions": "one OCR prompt per independent source, same seed count, repeated q0/current/candidate scoring; no cache savings",
            }
        result.append(
            {
                **scenario,
                "per_check_alpha": per_check,
                "ocr_prompts": len(prompts),
                "ocr_source_clusters": len(counts),
                "source_prompt_counts": dict(sorted(counts.items())),
                "weight_concentration": mass_squared,
                "ocr_radius": radius,
                "required_empirical_ocr_improvement": radius - 0.01,
                "can_pass_with_any_supported_ocr_difference": radius - 0.01 <= 1,
                "zero_effect_single_prompt_source_requirement": required,
                "projected_zero_effect_ocr_cost": projection,
                "joint_composite_radius": None,
                "power": None,
                "formal_applicability": False,
                "interpretation": "hypothetical independent-source OCR gate only; absent CLIP/HPS bounds and sampling proof block joint inference",
            }
        )
    return result


def applicability_report(
    registration,
    baseline,
    *,
    external_model_paths=None,
    job_evidence=(),
    risk_scenarios=(),
    screen_inputs=None,
):
    """Retain unreliable scales and explicit unknowns without creating a protocol."""
    scales = close_scale_registration(registration, baseline)
    samples = baseline.request.samples
    prompts = {row.prompt_id for row in samples}
    audit = support_audit(baseline.request, external_model_paths=external_model_paths)
    reasons = [
        "clip-numerical-support-unproved",
        "hps-numerical-support-unproved",
        "source-sampling-independence-unproved",
        "no-paired-candidate-screen",
        "no-verified-complete-job-cost",
    ]
    if scales.status != "ready":
        reasons.append("fixed-scales-unreliable")
    if not audit["axes"]["ocr"]["request_bound"]:
        reasons.append("ocr-code-support-not-bound-to-request")
    if not audit["sources"]["adapter"]["reviewed_code_matches"]:
        reasons.append("scoring-adapter-reviewed-source-mismatch")
    batches = [baseline]
    screened = None
    gate_batch = baseline
    if screen_inputs is not None:
        if set(screen_inputs) != {"registration", "candidates", "references"}:
            raise ValueError(
                "complete screen inputs require registration, candidates and references"
            )
        screened = close_screen_registration(
            screen_inputs["registration"],
            scales,
            screen_inputs["candidates"],
            screen_inputs["references"],
        )
        batches.extend(screen_inputs["candidates"])
        batches.extend(screen_inputs["references"].values())
        gate_batch = screen_inputs["candidates"][0]
        reasons.remove("no-paired-candidate-screen")
    cost = _cost_audit(batches, job_evidence)
    if cost is not None:
        reasons.remove("no-verified-complete-job-cost")
    value = {
        "schema": "mage-w4-statistical-applicability/v1",
        "method": METHOD,
        "status": "insufficient-evidence",
        "execution_kind": baseline.execution_kind,
        "baseline_batch_sha256": baseline.sha256,
        "registration_sha256": registration["sha256"],
        "scale": {**asdict(scales), "sha256": scales.sha256},
        "support": audit,
        "independence": {
            "selection_rule": registration["selection_rule"],
            "prompts": len(prompts),
            "source_clusters": len({row.source_group for row in samples}),
            "ocr_source_clusters": len(
                {row.source_group for row in samples if row.prompt_id in baseline.ocr_prompt_ids}
            ),
            "generation_seeds": sorted({row.seed for row in samples}),
            "independence_established": False,
            "basis": "deterministic development selection and distinct source labels establish structure only",
            "missing": [
                "target population and independently sampled source mechanism",
                "source provenance/near-duplicate audit, beyond distinct identifiers",
                "conditional independence under fixed model and shared generation seed rules",
            ],
            "seeds_count_as_independent_prompts": False,
        },
        "screen": None if screened is None else {**asdict(screened), "sha256": screened.sha256},
        "paired_effects": None
        if screened is None
        else [asdict(row) for row in screened.candidates],
        "paired_variance_interpretation": "descriptive sample variance of per-prompt paired seed means; not an independence certificate or standard error",
        "cost": cost,
        "conditional_gate_diagnostics": conditional_ocr_gates(
            gate_batch.request.samples, gate_batch.ocr_prompt_ids, risk_scenarios, cost=cost
        )
        if audit["axes"]["ocr"]["bound"] == [0, 1]
        else None,
        "reasons": reasons,
        "formal_execution_authorized": False,
        "independently_confirmed": False,
        "formal_protocol_created": False,
    }
    return {**value, "sha256": digest_json(value)}


def load_applicability_report(inputs):
    """Validate raw complete job/lease evidence, then compute on the GB10 CPU.

    No budget DB, models, confirmation images or final-test scores are opened.
    The authenticated coordinator supplies the input paths and closed snapshot.
    """
    if inputs.get("schema") != "mage-w4-statistical-applicability-inputs/v1":
        raise ValueError("explicit applicability input schema required")
    kind = inputs["execution_kind"]
    if kind not in {"fixture", "gb10_measured"}:
        raise ValueError("explicit fixture or gb10_measured execution kind required")
    if kind == "gb10_measured" and (
        platform.system() != "Linux" or platform.machine() not in {"aarch64", "arm64"}
    ):
        raise ValueError("actual applicability calculations require the GB10 Linux ARM host")
    artifacts = []

    def read(path):
        value, sha, size = read_json_artifact(path)
        artifacts.append(
            {"path": str(Path(path).resolve()), "file_sha256": sha, "size_bytes": size}
        )
        return value

    manifest = DataManifest.from_dict(read(inputs["manifest_path"]))
    snapshot = read(inputs["budget_snapshot_path"])
    registration = read(inputs["scale_registration_path"])
    proofs = []

    def verified(directory, request, registered_at):
        at = datetime.fromisoformat(registered_at.replace("Z", "+00:00"))
        if (
            at.utcoffset() is None
            or read(Path(directory) / "lease.json")["started_at"] < at.timestamp()
        ):
            raise ValueError("complete job predates its development preregistration")
        batch, proof = verify_evaluation_job(
            directory, request, manifest, snapshot, execution_kind=kind
        )
        proofs.append(proof)
        return batch

    baseline = verified(
        inputs["scale_job_directory"],
        CandidateRequest.from_dict(registration["request"]),
        registration["plan"]["registered_at_utc"],
    )
    screen_inputs = None
    if "screen" in inputs:
        screen = inputs["screen"]
        frozen = read(screen["registration_path"])
        candidates = {
            row["candidate_id"]: CandidateRequest.from_dict(row) for row in frozen["candidates"]
        }
        references = {
            device: CandidateRequest.from_dict(row)
            for device, row in frozen["references_by_device"].items()
        }
        if set(screen["candidate_job_directories"]) != set(candidates) or set(
            screen["reference_job_directories"]
        ) != set(references):
            raise ValueError("screen job directories must exactly cover the frozen plan")
        screen_inputs = {
            "registration": frozen,
            "candidates": [
                verified(
                    screen["candidate_job_directories"][key], request, frozen["registered_at_utc"]
                )
                for key, request in candidates.items()
            ],
            "references": {
                device: verified(
                    screen["reference_job_directories"][device],
                    request,
                    frozen["registered_at_utc"],
                )
                for device, request in references.items()
            },
        }
    report = applicability_report(
        registration,
        baseline,
        external_model_paths=inputs.get("external_model_paths"),
        job_evidence=proofs,
        risk_scenarios=inputs["risk_scenarios"],
        screen_inputs=screen_inputs,
    )
    if "scale_receipt_path" in inputs:
        saved = decode_scale_receipt(read(inputs["scale_receipt_path"]))
        if saved.sha256 != report["scale"]["sha256"]:
            raise ValueError(
                "saved scale receipt does not replay from registered complete q0 evidence"
            )
    value = {key: value for key, value in report.items() if key != "sha256"}
    value["input_artifacts"] = artifacts
    value["prior_synthetic_coverage"] = (
        read_coverage_evidence(inputs["coverage_paths"]) if "coverage_paths" in inputs else None
    )
    return {**value, "sha256": digest_json(value)}


def read_coverage_evidence(paths):
    """Link the existing CPU simulation and source snapshot without rerunning it."""
    if set(paths) != {"coverage", "resources", "source_manifest"}:
        raise ValueError("explicit coverage, resources and source_manifest paths required")
    loaded = {key: read_json_artifact(path) for key, path in paths.items()}
    coverage, resources, manifest = (
        loaded[key][0] for key in ("coverage", "resources", "source_manifest")
    )
    source = Path(__file__).with_name("statistical.py")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if (
        resources.get("output_sha256") != loaded["coverage"][1]
        or resources.get("source_manifest_sha256") != loaded["source_manifest"][1]
        or resources.get("source_sha256") != manifest.get("source_sha256")
        or manifest.get("source_sha256", {}).get("shadow_lab/mage_w4_search/statistical.py")
        != source_sha
        or resources.get("exit_status") != 0
        or resources.get("timed_out") is not False
        or resources.get("execution_kind") != "synthetic-cpu-validation"
        or coverage.get("execution_kind") != "synthetic-cpu-validation"
        or coverage.get("method") != METHOD
    ):
        raise ValueError("prior coverage/source/resource identity mismatch")
    return {
        "recorded_diagnostic": coverage,
        "recorded_resources": resources,
        "source_sha256": source_sha,
        "artifacts": {
            key: {"path": str(Path(paths[key]).resolve()), "file_sha256": sha, "size_bytes": size}
            for key, (_, sha, size) in loaded.items()
        },
        "establishes_real_applicability": False,
        "interpretation": "identity-linked prior simulation under its declared synthetic model; not rerun or generalized to real distributions",
    }
