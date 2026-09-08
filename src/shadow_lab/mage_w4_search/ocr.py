"""Target-blind OCR recognition and auditable, strict text-content matching.

This module imports no OCR/model runtime. An external recognizer receives only
image bytes and a frozen generic configuration; expected text enters afterwards.
The software contract is not a claim of measured OCR accuracy on GB10.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from time import perf_counter


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _sha256(value: str) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _no_answers(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {
                "target",
                "target_text",
                "target_segments",
                "expected_text",
                "answer",
                "prompt",
            }:
                raise ValueError(f"target/answer-specific OCR setting is forbidden: {key}")
            _no_answers(item)
    elif isinstance(value, list):
        for item in value:
            _no_answers(item)


@dataclass(frozen=True)
class EngineIdentity:
    engine: str
    package_version: str
    model_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if (
            not self.engine
            or not self.package_version
            or self.package_version.lower() in {"latest", "unknown", "master"}
        ):
            raise ValueError("OCR engine and exact package version are required")
        if (
            not isinstance(self.model_sha256, tuple)
            or not self.model_sha256
            or any(
                not isinstance(row, tuple) or len(row) != 2 or not row[0] or not _sha256(row[1])
                for row in self.model_sha256
            )
            or len({row[0] for row in self.model_sha256}) != len(self.model_sha256)
        ):
            raise ValueError("unique model names and actual SHA-256 identities are required")


@dataclass(frozen=True)
class GeometryPolicy:
    """Frozen conservative horizontal-fragment policy in original image pixels.

    Mutual nearest right/left neighbors merge only when vertical overlap covers
    at least 80% of the shorter box, heights differ by at most 2×, and the gap
    is at most half the smaller height. Gaps over 0.15 height insert one space.
    Rotated/vertical and overlapping boxes stay separate; no paragraph merging,
    text-based deduplication or answer-based correction is performed.
    """

    version: str = "horizontal-mutual-neighbor-v1"
    minimum_vertical_overlap: float = 0.8
    minimum_height_ratio: float = 0.5
    maximum_gap_height: float = 0.5
    space_gap_height: float = 0.15
    horizontal_angle_tolerance: float = 5.0

    def __post_init__(self) -> None:
        values = (
            self.minimum_vertical_overlap,
            self.minimum_height_ratio,
            self.maximum_gap_height,
            self.space_gap_height,
            self.horizontal_angle_tolerance,
        )
        if (
            self.version != "horizontal-mutual-neighbor-v1"
            or not all(math.isfinite(value) for value in values)
            or not 0 < self.minimum_vertical_overlap <= 1
            or not 0 < self.minimum_height_ratio <= 1
            or not 0 <= self.space_gap_height <= self.maximum_gap_height
            or not 0 <= self.horizontal_angle_tolerance <= 45
        ):
            raise ValueError("invalid frozen geometry policy")


@dataclass(frozen=True)
class OCRConfig:
    identity: EngineIdentity
    geometry: GeometryPolicy = GeometryPolicy()
    # JSON is immutable and part of the identity. It contains generic engine
    # options frozen before scoring, never a per-image answer/prompt.
    settings_json: str = "{}"
    matching_version: str = "nfc-codepoint-whitespace-hungarian-v1"
    unicode_version: str = unicodedata.unidata_version

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EngineIdentity) or not isinstance(
            self.geometry, GeometryPolicy
        ):
            raise ValueError("OCR identity and geometry policy are required")
        if self.matching_version != "nfc-codepoint-whitespace-hungarian-v1":
            raise ValueError("unsupported OCR matching version")
        if self.unicode_version != unicodedata.unidata_version:
            raise ValueError("Unicode version differs from the frozen OCR policy")
        settings = json.loads(self.settings_json)
        if not isinstance(settings, dict):
            raise ValueError("generic OCR settings must be an object")
        _no_answers(settings)
        _digest(settings)

    @property
    def sha256(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class RawSegment:
    text: str
    box: tuple[float, float, float, float]
    confidence: float | None = None
    orientation_degrees: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("recognized text must be a string")
        if (
            not isinstance(self.box, tuple)
            or len(self.box) != 4
            or not all(math.isfinite(value) and value >= 0 for value in self.box)
            or self.box[2] <= self.box[0]
            or self.box[3] <= self.box[1]
        ):
            raise ValueError("box must contain finite positive-area image coordinates")
        if self.confidence is not None and not (
            math.isfinite(self.confidence) and 0 <= self.confidence <= 1
        ):
            raise ValueError("invalid recognition confidence")
        if not math.isfinite(self.orientation_degrees):
            raise ValueError("invalid text orientation")


@dataclass(frozen=True)
class TextSegment:
    text: str
    box: tuple[float, float, float, float]
    raw_indices: tuple[int, ...]


@dataclass(frozen=True)
class RecognitionReceipt:
    image_sha256: str
    config: OCRConfig
    status: str
    raw_segments: tuple[RawSegment, ...]
    segments: tuple[TextSegment, ...]
    elapsed_seconds: float
    error: str | None = None
    started_at_utc: str = ""

    def __post_init__(self) -> None:
        if not _sha256(self.image_sha256) or not isinstance(self.config, OCRConfig):
            raise ValueError("recognition image and config identity are required")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("recognition elapsed seconds must be finite and nonnegative")
        if not isinstance(self.raw_segments, tuple) or not isinstance(self.segments, tuple):
            raise ValueError("recognition segments must be immutable")
        if self.status == "ok":
            if self.error is not None or self.segments != merge_geometry(
                self.raw_segments, self.config.geometry
            ):
                raise ValueError(
                    "recognition segments must equal frozen target-blind geometry output"
                )
        elif self.status == "engine_failure":
            if not self.error or self.segments:
                raise ValueError("failed OCR needs an error and no valid merged output")
        else:
            raise ValueError("unknown recognition status")
        _timestamp(self.started_at_utc)

    @property
    def sha256(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class TargetInventory:
    segments: tuple[str, ...]
    complete: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.complete, bool) or not isinstance(self.segments, tuple):
            raise ValueError("target completeness and immutable segments are required")
        if (self.complete and not self.segments) or any(
            not isinstance(text, str) or not _normalize(text) for text in self.segments
        ):
            raise ValueError("a complete target needs nonempty text segments")


@dataclass(frozen=True)
class SegmentMatch:
    target_index: int | None
    recognized_index: int | None
    target: str
    recognized: str
    substitutions: int
    deletions: int
    insertions: int
    # The complete alignment makes the edit counts independently reviewable.
    edits: tuple[tuple[str, str, str], ...]

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


@dataclass(frozen=True)
class TextScore:
    status: str
    recognition_sha256: str
    target_sha256: str
    character_correctness: float | None
    cer: float | None
    full_inventory_correct: bool | None
    target_characters: int
    substitutions: int
    deletions: int
    insertions: int
    matches: tuple[SegmentMatch, ...]


def _normalize(text: str) -> str:
    # NFC is canonical equivalence, not compatibility folding or correction.
    # Code points are the counting unit; Unicode whitespace runs collapse to
    # one ASCII space, and segment-edge layout whitespace is stripped.
    return " ".join(unicodedata.normalize("NFC", text).split())


def merge_geometry(raw: Sequence[RawSegment], policy: GeometryPolicy) -> tuple[TextSegment, ...]:
    """Merge solely by frozen geometry; this interface has no target argument."""
    if any(not isinstance(segment, RawSegment) for segment in raw):
        raise ValueError("recognizer must return RawSegment records")
    next_choices: dict[int, tuple[float, int]] = {}
    previous_choices: dict[int, tuple[float, int]] = {}
    indices = [index for index, segment in enumerate(raw) if _normalize(segment.text)]
    for i in indices:
        left = raw[i]
        if abs(left.orientation_degrees) > policy.horizontal_angle_tolerance:
            continue
        for j in indices:
            if i == j:
                continue
            right = raw[j]
            if abs(right.orientation_degrees) > policy.horizontal_angle_tolerance:
                continue
            left_height = left.box[3] - left.box[1]
            right_height = right.box[3] - right.box[1]
            height = min(left_height, right_height)
            overlap = min(left.box[3], right.box[3]) - max(left.box[1], right.box[1])
            gap = right.box[0] - left.box[2]
            if (
                height / max(left_height, right_height) < policy.minimum_height_ratio
                or overlap / height < policy.minimum_vertical_overlap
                or not 0 <= gap <= height * policy.maximum_gap_height
            ):
                continue
            choice = (gap / height, j)
            if i not in next_choices or choice < next_choices[i]:
                next_choices[i] = choice
            choice = (gap / height, i)
            if j not in previous_choices or choice < previous_choices[j]:
                previous_choices[j] = choice
    following = {
        i: choice[1] for i, choice in next_choices.items() if previous_choices[choice[1]][1] == i
    }
    starts = set(indices) - set(following.values())
    segments = []
    for start in sorted(starts, key=lambda index: (raw[index].box[1], raw[index].box[0], index)):
        chain = [start]
        text = _normalize(raw[start].text)
        while chain[-1] in following:
            i, j = chain[-1], following[chain[-1]]
            height = min(raw[i].box[3] - raw[i].box[1], raw[j].box[3] - raw[j].box[1])
            separator = (
                " " if raw[j].box[0] - raw[i].box[2] > height * policy.space_gap_height else ""
            )
            text += separator + _normalize(raw[j].text)
            chain.append(j)
        box = (
            min(raw[i].box[0] for i in chain),
            min(raw[i].box[1] for i in chain),
            max(raw[i].box[2] for i in chain),
            max(raw[i].box[3] for i in chain),
        )
        segments.append(TextSegment(text, box, tuple(chain)))
    return tuple(segments)


def recognize(
    image_bytes: bytes,
    config: OCRConfig,
    recognizer: Callable[[bytes, OCRConfig], Sequence[RawSegment]],
) -> RecognitionReceipt:
    """Recognize without a prompt, target inventory or target-dependent routing."""
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("nonempty image bytes are required")
    started = perf_counter()
    started_at_utc = datetime.now(timezone.utc).isoformat()
    raw: tuple[RawSegment, ...] = ()
    try:
        output = tuple(recognizer(image_bytes, config))
        if any(not isinstance(segment, RawSegment) for segment in output):
            raise ValueError("recognizer must return RawSegment records")
        raw = output
        segments = merge_geometry(raw, config.geometry)
    except Exception as error:
        return RecognitionReceipt(
            hashlib.sha256(image_bytes).hexdigest(),
            config,
            "engine_failure",
            raw,
            (),
            perf_counter() - started,
            f"{type(error).__name__}: {error}",
            started_at_utc,
        )
    return RecognitionReceipt(
        hashlib.sha256(image_bytes).hexdigest(),
        config,
        "ok",
        raw,
        segments,
        perf_counter() - started,
        started_at_utc=started_at_utc,
    )


def _edit(target: str, recognized: str, ti: int | None, ri: int | None) -> SegmentMatch:
    table = [[0] * (len(recognized) + 1) for _ in range(len(target) + 1)]
    for i in range(len(target) + 1):
        table[i][0] = i
    for j in range(len(recognized) + 1):
        table[0][j] = j
    for i, expected in enumerate(target, 1):
        for j, observed in enumerate(recognized, 1):
            table[i][j] = min(
                table[i - 1][j - 1] + (expected != observed),
                table[i - 1][j] + 1,
                table[i][j - 1] + 1,
            )
    i, j = len(target), len(recognized)
    edits: list[tuple[str, str, str]] = []
    # Fixed trace tie order: equal/substitute, delete, insert.
    while i or j:
        if i and j and table[i][j] == table[i - 1][j - 1] + (target[i - 1] != recognized[j - 1]):
            operation = "equal" if target[i - 1] == recognized[j - 1] else "substitute"
            edits.append((operation, target[i - 1], recognized[j - 1]))
            i, j = i - 1, j - 1
        elif i and table[i][j] == table[i - 1][j] + 1:
            edits.append(("delete", target[i - 1], ""))
            i -= 1
        else:
            edits.append(("insert", "", recognized[j - 1]))
            j -= 1
    edits.reverse()
    return SegmentMatch(
        ti,
        ri,
        target,
        recognized,
        sum(edit[0] == "substitute" for edit in edits),
        sum(edit[0] == "delete" for edit in edits),
        sum(edit[0] == "insert" for edit in edits),
        tuple(edits),
    )


def _assignment(costs: list[list[int]]) -> list[int]:
    """Hungarian shortest augmenting paths, O(n³), deterministic index ties.

    Rows are visited in ascending order; equal reduced-cost choices retain the
    first (lowest) column. This is a fixed tie rule, not a claimed lexicographic
    minimum over all optimal assignments.
    """
    size = len(costs)
    u, v, matched, previous = ([0] * (size + 1) for _ in range(4))
    for row in range(1, size + 1):
        matched[0] = row
        column = 0
        minimum = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            current_row = matched[column]
            delta, next_column = math.inf, 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = costs[current_row - 1][candidate - 1] - u[current_row] - v[candidate]
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    previous[candidate] = column
                if minimum[candidate] < delta:
                    delta, next_column = minimum[candidate], candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    u[matched[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched[column] == 0:
                break
        while column:
            prior = previous[column]
            matched[column] = matched[prior]
            column = prior
    assigned = [0] * size
    for column in range(1, size + 1):
        assigned[matched[column] - 1] = column - 1
    return assigned


def _match_segments(
    expected: tuple[str, ...], observed: tuple[str, ...]
) -> tuple[SegmentMatch, ...]:
    size = len(expected) + len(observed)
    costs = []
    for i in range(size):
        row = []
        for j in range(size):
            target = expected[i] if i < len(expected) else ""
            recognized = observed[j] if j < len(observed) else ""
            row.append(_edit(target, recognized, None, None).errors)
        costs.append(row)
    assigned = _assignment(costs)
    matches = []
    for i, j in enumerate(assigned):
        if i < len(expected):
            ri = j if j < len(observed) else None
            matches.append(_edit(expected[i], observed[j] if ri is not None else "", i, ri))
        elif j < len(observed):
            matches.append(_edit("", observed[j], None, j))
    return tuple(
        sorted(
            matches,
            key=lambda row: (
                row.target_index is None,
                row.target_index if row.target_index is not None else row.recognized_index,
            ),
        )
    )


def match_text(recognition: RecognitionReceipt, target: TargetInventory) -> TextScore:
    """Compare frozen OCR output with the complete expected text inventory."""
    if not target.complete or recognition.status == "engine_failure":
        return TextScore(
            "not_applicable" if not target.complete else "engine_failure",
            recognition.sha256,
            _digest(asdict(target)),
            None,
            None,
            None,
            0,
            0,
            0,
            0,
            (),
        )
    if recognition.status != "ok":
        raise ValueError("unknown recognition status")
    expected = tuple(_normalize(text) for text in target.segments)
    observed = tuple(segment.text for segment in recognition.segments)
    matches = _match_segments(expected, observed)
    substitutions = sum(row.substitutions for row in matches)
    deletions = sum(row.deletions for row in matches)
    insertions = sum(row.insertions for row in matches)
    denominator = sum(map(len, expected))
    cer = (substitutions + deletions + insertions) / denominator
    return TextScore(
        "ok",
        recognition.sha256,
        _digest(asdict(target)),
        max(0.0, 1.0 - cer),
        cer,
        cer == 0.0,
        denominator,
        substitutions,
        deletions,
        insertions,
        matches,
    )


def _timestamp(value: str) -> datetime:
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("an explicit UTC offset is required")
    return stamp


@dataclass(frozen=True)
class HumanTranscription:
    """An external human artifact, never synthesized by this module.

    Provenance fields are mandatory assertions for audit, not an automated
    proof that a person produced the artifact. Actual empty text is valid;
    uncertain/incomplete transcription cannot enter scorer selection.
    """

    image_sha256: str
    actual_segments: tuple[str, ...]
    source_kind: str
    annotator_id: str
    artifact_sha256: str
    complete: bool = True

    def __post_init__(self) -> None:
        if self.source_kind != "human_image_transcription" or not self.annotator_id:
            raise ValueError("actual human image transcription provenance is required")
        if not _sha256(self.image_sha256) or not _sha256(self.artifact_sha256):
            raise ValueError("human transcription needs image and artifact SHA-256 identities")
        if (
            self.complete is not True
            or not isinstance(self.actual_segments, tuple)
            or any(
                not isinstance(text, str) or not _normalize(text) for text in self.actual_segments
            )
        ):
            raise ValueError("complete human transcription with nonempty segments is required")


@dataclass(frozen=True)
class CalibrationPlan:
    """Register before comparison; no prompt targets occur in this manifest.

    Selection order is fixed: only failure-free engines under the explicitly
    registered mean-error limit qualify; lower per-image normalized edit error
    wins, then lower occupied device seconds, then the registered engine order.
    Empty actual transcription uses errors / 1 (reported as insertion count,
    not CER); nonempty images use errors / actual character count. Images have
    equal weight. This is OCR-tool accuracy, not generation target correctness.
    """

    image_sha256: tuple[str, ...]
    configs: tuple[OCRConfig, OCRConfig]
    generation_receipt_sha256: tuple[str, ...]
    registered_at_utc: str
    maximum_mean_error: float
    purpose: str = "development_ocr_selection"
    version: str = "ocr-human-calibration-v1"

    def __post_init__(self) -> None:
        if (
            self.purpose != "development_ocr_selection"
            or self.version != "ocr-human-calibration-v1"
        ):
            raise ValueError("OCR comparison only accepts the frozen development purpose")
        if (
            not isinstance(self.image_sha256, tuple)
            or not self.image_sha256
            or len(set(self.image_sha256)) != len(self.image_sha256)
            or not all(_sha256(value) for value in self.image_sha256)
        ):
            raise ValueError("unique development image SHA-256 identities are required")
        if (
            not isinstance(self.generation_receipt_sha256, tuple)
            or len(self.generation_receipt_sha256) != len(self.image_sha256)
            or not all(_sha256(value) for value in self.generation_receipt_sha256)
        ):
            raise ValueError("every image needs its quantized generation receipt SHA-256")
        if (
            not isinstance(self.configs, tuple)
            or len(self.configs) != 2
            or {item.identity.engine for item in self.configs} != {"PaddleOCR", "EasyOCR"}
        ):
            raise ValueError("register one frozen PaddleOCR and one EasyOCR configuration")
        if not math.isfinite(self.maximum_mean_error) or self.maximum_mean_error < 0:
            raise ValueError("register a finite nonnegative accuracy threshold")
        _timestamp(self.registered_at_utc)

    @property
    def sha256(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class CalibrationImageScore:
    image_sha256: str
    recognition_sha256: str
    transcription_sha256: str
    status: str
    actual_characters: int
    normalized_edit_error: float | None
    full_inventory_correct: bool | None
    matches: tuple[SegmentMatch, ...]


@dataclass(frozen=True)
class EngineCalibration:
    config: OCRConfig
    failure_count: int
    mean_normalized_edit_error: float | None
    occupied_seconds: float
    eligible: bool
    images: tuple[CalibrationImageScore, ...]


@dataclass(frozen=True)
class CalibrationReport:
    plan: CalibrationPlan
    transcriptions: tuple[HumanTranscription, ...]
    engines: tuple[EngineCalibration, ...]
    selected_engine: str | None
    execution_kind: str
    device_evidence_sha256: str | None
    # Recognition-only fixture timing excludes model initialization; measured
    # comparisons require externally metered full occupied-device costs.
    cost_basis: str

    @property
    def sha256(self) -> str:
        return _digest(asdict(self))


def compare_ocr(
    plan: CalibrationPlan,
    transcriptions: Sequence[HumanTranscription],
    observations: Sequence[RecognitionReceipt],
    *,
    execution_kind: str,
    occupied_seconds_by_config: tuple[tuple[str, float], ...] = (),
    device_evidence_sha256: str | None = None,
) -> CalibrationReport:
    """Compare both tools on registered images and independent actual text.

    Real model adapters must verify packages/weight bytes against OCRConfig
    before inference. This pure receipt interface does not load either engine.
    """
    if execution_kind not in {"fixture", "gb10_measured"}:
        raise ValueError("explicit fixture or gb10_measured execution kind is required")
    truth = {row.image_sha256: row for row in transcriptions}
    if len(truth) != len(transcriptions) or set(truth) != set(plan.image_sha256):
        raise ValueError("every registered image needs exactly one actual human transcription")
    costs = dict(occupied_seconds_by_config)
    if len(costs) != len(occupied_seconds_by_config) or any(
        not math.isfinite(value) or value < 0 for value in costs.values()
    ):
        raise ValueError("occupied device cost must be unique, finite and nonnegative")
    if execution_kind == "gb10_measured" and (
        not _sha256(device_evidence_sha256)
        or set(costs) != {config.sha256 for config in plan.configs}
    ):
        raise ValueError("GB10 comparison requires device evidence and both full occupied costs")
    expected_keys = {
        (config.sha256, image) for config in plan.configs for image in plan.image_sha256
    }
    received = {(row.config.sha256, row.image_sha256): row for row in observations}
    if len(received) != len(observations) or set(received) != expected_keys:
        raise ValueError("missing, duplicate or unexpected OCR observations")
    registered = _timestamp(plan.registered_at_utc)
    if any(_timestamp(row.started_at_utc) < registered for row in observations):
        raise ValueError("OCR observations precede comparison preregistration")
    engines = []
    for config in plan.configs:
        images = []
        for image in plan.image_sha256:
            row = received[config.sha256, image]
            actual = truth[image]
            normalized_actual = tuple(_normalize(text) for text in actual.actual_segments)
            actual_characters = sum(map(len, normalized_actual))
            if row.status == "engine_failure":
                images.append(
                    CalibrationImageScore(
                        image,
                        row.sha256,
                        actual.artifact_sha256,
                        row.status,
                        actual_characters,
                        None,
                        None,
                        (),
                    )
                )
                continue
            if row.status != "ok":
                raise ValueError("unknown OCR observation status")
            matches = _match_segments(
                normalized_actual, tuple(segment.text for segment in row.segments)
            )
            error_count = sum(item.errors for item in matches)
            images.append(
                CalibrationImageScore(
                    image,
                    row.sha256,
                    actual.artifact_sha256,
                    "ok",
                    actual_characters,
                    error_count / max(1, actual_characters),
                    error_count == 0,
                    matches,
                )
            )
        failures = sum(row.status != "ok" for row in images)
        # Never silently average a smaller set after engine failure.
        mean_error = (
            None if failures else sum(row.normalized_edit_error for row in images) / len(images)
        )
        seconds = costs.get(
            config.sha256,
            sum(received[config.sha256, image].elapsed_seconds for image in plan.image_sha256),
        )
        engines.append(
            EngineCalibration(
                config,
                failures,
                mean_error,
                seconds,
                mean_error is not None and mean_error <= plan.maximum_mean_error,
                tuple(images),
            )
        )
    eligible = [(index, row) for index, row in enumerate(engines) if row.eligible]
    selected = (
        min(
            eligible,
            key=lambda item: (
                item[1].mean_normalized_edit_error,
                item[1].occupied_seconds,
                item[0],
            ),
        )[1].config.identity.engine
        if eligible
        else None
    )
    return CalibrationReport(
        plan,
        tuple(truth[image] for image in plan.image_sha256),
        tuple(engines),
        selected,
        execution_kind,
        device_evidence_sha256,
        "occupied_device_seconds"
        if execution_kind == "gb10_measured"
        else ("fixture_declared_cost_seconds" if costs else "fixture_recognition_seconds"),
    )


def freeze_scorer(report: CalibrationReport) -> dict:
    """Return a lock receipt only for an eligible, human-calibrated GB10 run."""
    if report.execution_kind != "gb10_measured" or not _sha256(report.device_evidence_sha256):
        raise ValueError("a real GB10 OCR comparison is required; fixtures cannot freeze a scorer")
    selected = next(
        (row for row in report.engines if row.config.identity.engine == report.selected_engine),
        None,
    )
    if selected is None or not selected.eligible or selected.failure_count:
        raise ValueError("no OCR engine satisfies the registered accuracy/availability rule")
    if not report.transcriptions or len(report.transcriptions) != len(report.plan.image_sha256):
        raise ValueError("complete actual human transcription evidence is required")
    return scorer_lock_identity(report)


def scorer_lock_identity(report: CalibrationReport) -> dict:
    """Describe selected-scorer identities only; this does not qualify or freeze it.

    Kept separate so archive readers can verify fixture identities without
    relabelling fixtures as GB10 measurements. Qualification stays exclusively
    in freeze_scorer; the report's execution kind remains part of its identity.
    """
    selected = next(
        (row for row in report.engines if row.config.identity.engine == report.selected_engine),
        None,
    )
    if selected is None:
        raise ValueError("scorer identity requires a selected engine")
    return {
        "schema": "mage-w4-ocr-scorer-lock-v1",
        "config": asdict(selected.config),
        "config_sha256": selected.config.sha256,
        "calibration_report_sha256": report.sha256,
        "plan_sha256": report.plan.sha256,
        "device_evidence_sha256": report.device_evidence_sha256,
    }
