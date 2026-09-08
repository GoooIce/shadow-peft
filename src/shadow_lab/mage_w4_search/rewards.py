"""Frozen q0 reward scales and development-only paired candidate screening."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from decimal import Decimal, localcontext
from statistics import mean, stdev, variance

from .contracts import CandidateRequest, DataManifest, RequestedSample, digest_json, preflight

AXES = ("clip", "hps", "ocr")


def _hash(value: str) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _time(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("explicit time offset required")
    return result


@dataclass(frozen=True)
class SampleScores:
    sample_id: str
    image_sha256: str
    clip: float
    hps: float
    ocr: float | None
    ocr_status: str
    score_receipt_sha256: str


@dataclass(frozen=True)
class ScoredBatch:
    request: CandidateRequest
    manifest: DataManifest
    samples: tuple[SampleScores, ...]
    ocr_prompt_ids: tuple[str, ...]
    scorer_sha256: tuple[tuple[str, str], ...]
    native_receipt_sha256: str
    started_at_utc: str
    execution_kind: str

    @property
    def sha256(self) -> str:
        return digest_json(asdict(self))


@dataclass(frozen=True)
class AxisValues:
    clip: float
    hps: float
    ocr: float


@dataclass(frozen=True)
class ScalePlan:
    q0_digest: str
    samples: tuple[RequestedSample, ...]
    ocr_prompt_ids: tuple[str, ...]
    scorer_sha256: tuple[tuple[str, str], ...]
    manifest_digest: str
    registered_at_utc: str
    min_prompts: int
    min_ocr_prompts: int
    min_source_groups: int
    maximum_leave_group_out_relative_change: float

    def __post_init__(self) -> None:
        if not _hash(self.q0_digest) or not _hash(self.manifest_digest):
            raise ValueError("scale plan needs q0 and manifest content identities")
        if not self.samples or len({row.sample_id for row in self.samples}) != len(self.samples):
            raise ValueError("scale plan needs unique frozen samples")
        if any(
            type(value) is not int or value < 3
            for value in (
                self.min_prompts,
                self.min_ocr_prompts,
                self.min_source_groups,
            )
        ):
            raise ValueError("predeclare minimum counts of at least three independent units")
        if (
            not math.isfinite(self.maximum_leave_group_out_relative_change)
            or self.maximum_leave_group_out_relative_change < 0
        ):
            raise ValueError("predeclare a finite nonnegative stability bound")
        _time(self.registered_at_utc)

    @property
    def sha256(self) -> str:
        return digest_json(asdict(self))


@dataclass(frozen=True)
class ScaleReceipt:
    status: str
    plan: ScalePlan
    q0_batch_sha256: str
    scales: AxisValues | None
    margins: AxisValues | None
    prompt_counts: dict[str, int]
    diagnostics: dict
    reasons: tuple[str, ...]
    execution_kind: str
    fixed_generation_sha256: str
    method: str = "sample-standard-deviation-ddof-1"

    @property
    def sha256(self) -> str:
        return digest_json(asdict(self))


def validate_scored_batch(batch: ScoredBatch) -> None:
    """Reject an incomplete or identity-mismatched complete development result."""
    # Revalidate declared identities without rereading multi-GB artifact paths;
    # actual byte verification is the native runner's cited receipt.
    request = replace(
        batch.request,
        lease_id=None,
        dependencies=tuple(replace(artifact, path=None) for artifact in batch.request.dependencies),
    )
    checked = preflight(request, batch.manifest, expected_center_version=request.center_version)
    if checked.status != "ready_for_reservation":
        raise ValueError(f"invalid scored request: {checked.reasons}")
    if request.evaluation_mode != "complete":
        raise ValueError("component scoring cannot form a complete reward batch")
    if request.device_id not in {"gooo", "goooinext"}:
        raise ValueError("only the two registered GB10 devices may provide search scores")
    if batch.execution_kind not in {"fixture", "gb10_measured"} or not _hash(
        batch.native_receipt_sha256
    ):
        raise ValueError("explicit execution kind and native receipt identity are required")
    _time(batch.started_at_utc)
    scorer_ids = dict(batch.scorer_sha256)
    if len(scorer_ids) != len(batch.scorer_sha256) or set(scorer_ids) != set(AXES):
        raise ValueError("exactly three frozen scorer identities are required")
    declared_scorers = {
        artifact.role.removesuffix("_scorer"): artifact.sha256
        for artifact in request.dependencies
        if artifact.role.endswith("_scorer")
    }
    if scorer_ids != declared_scorers:
        raise ValueError("scorer identity differs from frozen request")
    requested = {sample.sample_id: sample for sample in request.samples}
    scores = {sample.sample_id: sample for sample in batch.samples}
    if len(scores) != len(batch.samples) or scores.keys() != requested.keys():
        raise ValueError("missing, duplicate or mismatched sample scores")
    prompt_ids = {sample.prompt_id for sample in request.samples}
    expected_ocr = {
        row.prompt_id
        for row in batch.manifest.records
        if row.prompt_id in prompt_ids and row.complete_target_inventory
    }
    if (
        not expected_ocr
        or set(batch.ocr_prompt_ids) != expected_ocr
        or len(set(batch.ocr_prompt_ids)) != len(batch.ocr_prompt_ids)
    ):
        raise ValueError("OCR subset differs from the frozen complete target inventories")
    for row in batch.samples:
        if not _hash(row.image_sha256) or not _hash(row.score_receipt_sha256):
            raise ValueError("image and score receipt identities are required")
        if not all(
            type(value) in {int, float} and math.isfinite(value) for value in (row.clip, row.hps)
        ):
            raise ValueError("failed/nonfinite CLIP or HPS cannot form complete scores")
        if requested[row.sample_id].prompt_id in expected_ocr:
            if (
                row.ocr_status != "ok"
                or type(row.ocr) not in {int, float}
                or not math.isfinite(row.ocr)
                or not 0 <= row.ocr <= 1
            ):
                raise ValueError("applicable OCR must have a successful finite correctness score")
        elif row.ocr_status != "not_applicable" or row.ocr is not None:
            raise ValueError("nonapplicable OCR must remain missing, not a numeric zero")


def _prompt_scores(batch: ScoredBatch) -> dict[str, dict[str, float]]:
    requested = {sample.sample_id: sample for sample in batch.request.samples}
    buckets: dict[str, dict[str, list[float]]] = {axis: {} for axis in AXES}
    for row in batch.samples:
        prompt = requested[row.sample_id].prompt_id
        for axis in AXES:
            value = getattr(row, axis)
            if value is not None:
                buckets[axis].setdefault(prompt, []).append(value)
    return {
        axis: {prompt: mean(values) for prompt, values in prompts.items()}
        for axis, prompts in buckets.items()
    }


def _fixed_generation_identity(request: CandidateRequest) -> str:
    return digest_json(
        {
            "runtime": {
                key: value for key, value in request.runtime_identity.items() if key != "device"
            },
            "components": sorted(
                (item.role, item.sha256, item.size_bytes)
                for item in request.dependencies
                if item.role != "quantized_packed"
            ),
        }
    )


def calibrate_scales(plan: ScalePlan, baseline: ScoredBatch) -> ScaleReceipt:
    """Fix ddof=1 cross-prompt q0 scales after averaging seeds within prompt."""
    validate_scored_batch(baseline)
    request = baseline.request
    if request.purpose != "scale_calibration" or not (
        request.candidate_digest == request.center_digest == request.q0_digest == plan.q0_digest
    ):
        raise ValueError("scales can only come from fixed q0 in scale_calibration")
    if request.samples != plan.samples or request.manifest_digest != plan.manifest_digest:
        raise ValueError("calibration samples or manifest differ from the registered plan")
    if dict(baseline.scorer_sha256) != dict(plan.scorer_sha256) or set(
        baseline.ocr_prompt_ids
    ) != set(plan.ocr_prompt_ids):
        raise ValueError("calibration scorers or OCR subset differ from the registered plan")
    if _time(baseline.started_at_utc) < _time(plan.registered_at_utc):
        raise ValueError("scale observations precede preregistration")
    values = _prompt_scores(baseline)
    groups = {sample.prompt_id: sample.source_group for sample in request.samples}
    reasons = []
    measured = {}
    diagnostics = {}
    for axis in AXES:
        points = values[axis]
        minimum = plan.min_ocr_prompts if axis == "ocr" else plan.min_prompts
        source_groups = sorted({groups[prompt] for prompt in points})
        if len(points) < minimum:
            reasons.append(f"{axis}:insufficient-prompts")
        if len(source_groups) < plan.min_source_groups:
            reasons.append(f"{axis}:insufficient-source-groups")
        scale = stdev(tuple(points.values())) if len(points) >= 2 else None
        removals = []
        if scale is None or not math.isfinite(scale) or scale <= 0:
            reasons.append(f"{axis}:nonpositive-scale")
        else:
            measured[axis] = scale
            for group in source_groups:
                remaining = [value for prompt, value in points.items() if groups[prompt] != group]
                reduced = stdev(remaining) if len(remaining) >= 2 else None
                relative = abs(reduced / scale - 1) if reduced is not None else None
                stable = (
                    reduced is not None
                    and reduced > 0
                    and relative <= plan.maximum_leave_group_out_relative_change
                )
                removals.append(
                    {
                        "removed_source_group": group,
                        "remaining_prompts": len(remaining),
                        "scale": reduced,
                        "relative_change": relative,
                        "stable": stable,
                    }
                )
                if not stable:
                    reasons.append(f"{axis}:unstable-under-source-group-removal")
        diagnostics[axis] = {
            "scale": scale,
            "minimum_prompts": minimum,
            "source_group_count": len(source_groups),
            "leave_group_out": removals,
        }
    scales = AxisValues(**measured) if not reasons else None
    return ScaleReceipt(
        "ready" if scales is not None else "unreliable",
        plan,
        baseline.sha256,
        scales,
        AxisValues(0.05 * scales.clip, 0.05 * scales.hps, 0.01) if scales else None,
        {axis: len(values[axis]) for axis in AXES},
        diagnostics,
        tuple(dict.fromkeys(reasons)),
        baseline.execution_kind,
        _fixed_generation_identity(request),
    )


@dataclass(frozen=True)
class PromptDifference:
    prompt_id: str
    source_group: str
    clip: float
    hps: float
    ocr: float | None


@dataclass(frozen=True)
class PairedComparison:
    reference_batch_sha256: str
    differences: AxisValues
    composite_improvement: float
    passed: bool
    failed_axes: tuple[str, ...]
    paired_variances: dict[str, float | None]
    prompt_counts: dict[str, int]
    prompt_differences: tuple[PromptDifference, ...]


@dataclass(frozen=True)
class CandidateScreen:
    candidate_id: str
    candidate_digest: str
    candidate_batch_sha256: str
    device_id: str
    feasible: bool
    current: PairedComparison
    q0: PairedComparison


@dataclass(frozen=True)
class ScreeningReceipt:
    status: str
    selected_candidate_id: str | None
    candidates: tuple[CandidateScreen, ...]
    candidate_order: tuple[str, ...]
    scales_sha256: str
    center_version: int
    center_digest: str
    q0_digest: str
    execution_kind: str
    stage: str = "development_screening"
    independently_confirmed: bool = False

    @property
    def sha256(self) -> str:
        return digest_json(asdict(self))


def _same_pair(candidate: ScoredBatch, reference: ScoredBatch, model_digest: str) -> None:
    left, right = candidate.request, reference.request
    if right.candidate_digest != model_digest:
        raise ValueError("reference model does not match the frozen center/q0")
    for attribute in (
        "center_version",
        "center_digest",
        "q0_digest",
        "device_id",
        "protocol",
        "manifest_digest",
    ):
        if getattr(left, attribute) != getattr(right, attribute):
            raise ValueError(f"paired {attribute} mismatch")
    if left.runtime_identity != right.runtime_identity:
        raise ValueError("paired runtime/sampler identity mismatch")
    if {sample.sample_id for sample in left.samples} != {
        sample.sample_id for sample in right.samples
    }:
        raise ValueError("paired prompt/seed/sample identity mismatch")

    def fixed_dependencies(request: CandidateRequest) -> set[tuple[str, str, int]]:
        return {
            (item.role, item.sha256, item.size_bytes)
            for item in request.dependencies
            if item.role != "quantized_packed"
        }

    if fixed_dependencies(left) != fixed_dependencies(right):
        raise ValueError("paired fixed component/scorer identity mismatch")
    if dict(candidate.scorer_sha256) != dict(reference.scorer_sha256) or set(
        candidate.ocr_prompt_ids
    ) != set(reference.ocr_prompt_ids):
        raise ValueError("paired scoring identity mismatch")
    if candidate.execution_kind != reference.execution_kind:
        raise ValueError("fixture and measured paired results cannot mix")


def _compare(
    candidate: ScoredBatch, reference: ScoredBatch, scales: ScaleReceipt
) -> PairedComparison:
    request = {sample.sample_id: sample for sample in candidate.request.samples}
    reference_scores = {sample.sample_id: sample for sample in reference.samples}
    buckets: dict[str, dict[str, list[Decimal]]] = {axis: {} for axis in AXES}
    # Decimal differences use the declared decimal score values. This avoids
    # e.g. 0.95-1.0 falling just below the exact -0.05 boundary in binary float,
    # without enlarging the scientific non-inferiority margin.
    with localcontext() as context:
        context.prec = 34
        for sample in candidate.samples:
            prompt = request[sample.sample_id].prompt_id
            other = reference_scores[sample.sample_id]
            for axis in AXES:
                value = getattr(sample, axis)
                if value is not None:
                    difference = Decimal(str(value)) - Decimal(str(getattr(other, axis)))
                    buckets[axis].setdefault(prompt, []).append(difference)
        prompt_values = {
            axis: {prompt: sum(points) / len(points) for prompt, points in rows.items()}
            for axis, rows in buckets.items()
        }
        averages = {axis: sum(rows.values()) / len(rows) for axis, rows in prompt_values.items()}
        failed = tuple(
            axis for axis in AXES if averages[axis] < -Decimal(str(getattr(scales.margins, axis)))
        )
        composite = sum(
            weight * averages[axis] / Decimal(str(getattr(scales.scales, axis)))
            for axis, weight in (
                ("clip", Decimal("0.5")),
                ("hps", Decimal("0.25")),
                ("ocr", Decimal("0.25")),
            )
        )
    groups = {sample.prompt_id: sample.source_group for sample in candidate.request.samples}
    prompt_differences = tuple(
        PromptDifference(
            prompt,
            groups[prompt],
            float(prompt_values["clip"][prompt]),
            float(prompt_values["hps"][prompt]),
            float(prompt_values["ocr"][prompt]) if prompt in prompt_values["ocr"] else None,
        )
        for prompt in sorted(groups)
    )
    paired_variances = {
        axis: variance([float(value) for value in rows.values()]) if len(rows) >= 2 else None
        for axis, rows in prompt_values.items()
    }
    return PairedComparison(
        reference.sha256,
        AxisValues(**{axis: float(value) for axis, value in averages.items()}),
        float(composite),
        not failed,
        failed,
        paired_variances,
        {axis: len(rows) for axis, rows in prompt_values.items()},
        prompt_differences,
    )


def screen_candidates(
    scales: ScaleReceipt,
    candidates: Sequence[ScoredBatch],
    current_references: Mapping[str, ScoredBatch],
    q0_references: Mapping[str, ScoredBatch],
    *,
    candidate_order: tuple[str, ...],
) -> ScreeningReceipt:
    """Select only the best feasible candidate; this is never confirmation."""
    if (
        scales.status != "ready"
        or scales.scales is None
        or scales.margins is None
        or scales.reasons
    ):
        raise ValueError("unreliable scales cannot screen a complete three-axis batch")
    if any(
        not math.isfinite(getattr(scales.scales, axis)) or getattr(scales.scales, axis) <= 0
        for axis in AXES
    ):
        raise ValueError("nonpositive/nonfinite scales cannot screen candidates")
    if scales.margins != AxisValues(0.05 * scales.scales.clip, 0.05 * scales.scales.hps, 0.01):
        raise ValueError("fixed non-inferiority margins were changed")
    by_id = {batch.request.candidate_id: batch for batch in candidates}
    if (
        not by_id
        or len(by_id) != len(candidates)
        or len(set(candidate_order)) != len(candidate_order)
        or set(candidate_order) != set(by_id)
        or set(current_references) != set(by_id)
        or set(q0_references) != set(by_id)
    ):
        raise ValueError("candidate order and both reference maps must exactly cover the batch")
    anchor = by_id[candidate_order[0]].request
    calibrated_groups = {sample.source_group for sample in scales.plan.samples}
    calibrated_prompts = {sample.prompt_digest for sample in scales.plan.samples}
    checked = set()
    results = []
    for candidate_id in candidate_order:
        candidate = by_id[candidate_id]
        current, q0 = current_references[candidate_id], q0_references[candidate_id]
        for item in (candidate, current, q0):
            identity = item.sha256
            if identity not in checked:
                validate_scored_batch(item)
                checked.add(identity)
            if item.request.purpose != "candidate_development":
                raise ValueError("screening requires candidate_development data")
            if item.execution_kind != scales.execution_kind or dict(item.scorer_sha256) != dict(
                scales.plan.scorer_sha256
            ):
                raise ValueError(
                    "scale calibration and screening scorer/execution identities differ"
                )
            if _fixed_generation_identity(item.request) != scales.fixed_generation_sha256:
                raise ValueError(
                    "screening differs from scale calibration fixed generation identity"
                )
        request = candidate.request
        if request.q0_digest != scales.plan.q0_digest:
            raise ValueError("screening q0 differs from fixed scale origin")
        if any(
            getattr(request, key) != getattr(anchor, key)
            for key in ("center_version", "center_digest", "q0_digest", "group_id")
        ):
            raise ValueError("mixed center versions or layer groups in candidate batch")
        if request.manifest_digest != anchor.manifest_digest or {
            sample.sample_id for sample in request.samples
        } != {sample.sample_id for sample in anchor.samples}:
            raise ValueError("candidate batch sample identities must be identical")
        if {key: value for key, value in request.runtime_identity.items() if key != "device"} != {
            key: value for key, value in anchor.runtime_identity.items() if key != "device"
        }:
            raise ValueError("candidate batch runtime/sampler identities differ")
        if calibrated_groups & {
            sample.source_group for sample in request.samples
        } or calibrated_prompts & {sample.prompt_digest for sample in request.samples}:
            raise ValueError("candidate development consumed independent scale calibration data")
        _same_pair(candidate, current, request.center_digest)
        _same_pair(candidate, q0, request.q0_digest)
        relative_current = _compare(candidate, current, scales)
        relative_q0 = _compare(candidate, q0, scales)
        results.append(
            CandidateScreen(
                candidate_id,
                request.candidate_digest,
                candidate.sha256,
                request.device_id,
                relative_current.passed and relative_q0.passed,
                relative_current,
                relative_q0,
            )
        )
    feasible = [row for row in results if row.feasible]
    # max returns the first equal item; results already follow frozen order.
    winner = max(feasible, key=lambda row: row.current.composite_improvement) if feasible else None
    return ScreeningReceipt(
        "best_feasible_candidate" if winner else "no_feasible_candidate",
        winner.candidate_id if winner else None,
        tuple(results),
        candidate_order,
        scales.sha256,
        anchor.center_version,
        anchor.center_digest,
        anchor.q0_digest,
        scales.execution_kind,
    )
