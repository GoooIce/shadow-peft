"""Bind generated images to complete, frozen, teacher-free development scores."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path

from .contracts import CandidateRequest, DataManifest, digest_json, preflight
from .ocr import OCRConfig, TargetInventory, match_text, recognize
from .rewards import SampleScores, ScoredBatch, validate_scored_batch


class EvaluationError(ValueError):
    """Incomplete or identity-mismatched generation cannot become scored evidence."""


@dataclass(frozen=True)
class GenerationBinding:
    request_digest: str
    native_report_sha256: str
    fixed_state_sha256: str
    runtime_sha256: str
    generation_sha256: str
    execution_kind: str

    def __post_init__(self):
        if any(
            not _hash(value) for name, value in asdict(self).items() if name != "execution_kind"
        ):
            raise EvaluationError("missing-generation-binding-identity")
        if self.execution_kind not in {"fixture", "gb10_measured"}:
            raise EvaluationError("invalid-execution-kind")


def load_scored_batch(path) -> ScoredBatch:
    """Decode and validate a batch; source files/native evidence still need auditing.

    The batch and native receipt identities are canonical JSON digests. An archive
    file's byte digest remains a separate caller-verified identity.
    """
    payload = json.loads(Path(path).read_bytes())
    if not isinstance(payload, dict) or set(payload) != {
        field.name for field in fields(ScoredBatch)
    }:
        raise EvaluationError("invalid-scored-batch-schema")
    try:
        batch = ScoredBatch(
            request=CandidateRequest.from_dict(payload["request"]),
            manifest=DataManifest.from_dict(payload["manifest"]),
            samples=tuple(SampleScores(**row) for row in payload["samples"]),
            ocr_prompt_ids=tuple(payload["ocr_prompt_ids"]),
            scorer_sha256=tuple(tuple(pair) for pair in payload["scorer_sha256"]),
            native_receipt_sha256=payload["native_receipt_sha256"],
            started_at_utc=payload["started_at_utc"],
            execution_kind=payload["execution_kind"],
        )
    except (KeyError, TypeError) as error:
        raise EvaluationError("invalid-scored-batch-schema") from error
    validate_scored_batch(batch)
    return batch


def evaluate_generated_candidate(
    request: CandidateRequest,
    manifest: DataManifest,
    report: dict,
    *,
    binding: GenerationBinding,
    expected_center_version: int,
    image_root,
    receipt_directory,
    ocr_config: OCRConfig,
    ocr_lock=None,
    ocr_calibration=None,
    clip_checkpoint=None,
    hps_checkpoint=None,
    ocr_model_root=None,
    ocr_python_executable=None,
    fixture_scorers=None,
    before_stage=None,
) -> ScoredBatch:
    """Score only complete matching images, retaining image/recognition/edit receipts.

    Fixtures replace only the expensive external score/recognition boundary and
    always remain fixture evidence. Real mode loads the existing fixed adapters.
    """
    checked_request = replace(
        request,
        lease_id=None,
        dependencies=tuple(replace(artifact, path=None) for artifact in request.dependencies),
    )
    checked = preflight(checked_request, manifest, expected_center_version=expected_center_version)
    if checked.status != "ready_for_reservation" or request.evaluation_mode != "complete":
        raise EvaluationError(f"invalid-complete-request:{checked.reasons}")
    if binding.request_digest != request.request_digest:
        raise EvaluationError("generation-request-binding-mismatch")
    if digest_json(report) != binding.native_report_sha256:
        raise EvaluationError("native-report-identity-mismatch")
    if report.get("center_sha256") != request.candidate_digest:
        raise EvaluationError("native-candidate-identity-mismatch")
    if report.get("fixed_sha256") != binding.fixed_state_sha256:
        raise EvaluationError("native-fixed-state-mismatch")
    if (
        digest_json(report.get("runtime")) != binding.runtime_sha256
        or digest_json(report.get("generation")) != binding.generation_sha256
    ):
        raise EvaluationError("native-runtime-generation-binding-mismatch")
    if report.get("generation") != request.runtime_identity["generation"]:
        raise EvaluationError("generation-config-differs-from-request")
    if report.get("schema_version") != 1 or report.get("status") != "ok":
        raise EvaluationError("native-generation-not-complete")
    if binding.execution_kind == "gb10_measured":
        if fixture_scorers is not None:
            raise EvaluationError("fixture-scorers-cannot-create-measured-evidence")
        if (
            report.get("native_compute") is not True
            or report.get("qualified") is not True
            or report.get("installation", {}).get("backend") != "torchao_nvfp4"
        ):
            raise EvaluationError("native-compute-evidence-required")
        layers = report["installation"].get("installed_layers", [])
        if (
            len(layers) != 174
            or len(set(layers)) != 174
            or report["runtime"].get("compute_capability") != [12, 1]
        ):
            raise EvaluationError("incomplete-native-mmdit-or-non-gb10-runtime")
    elif fixture_scorers is None or len(fixture_scorers) != 3:
        raise EvaluationError("explicit-fixture-external-scorers-required")
    identities = {sample.sample_id: sample for sample in request.samples}
    rows = report.get("rows", [])
    by_id = {row.get("sample_id"): row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(identities):
        raise EvaluationError("missing-duplicate-or-mismatched-generation-samples")
    prompts = {record.prompt_id: record for record in manifest.records}
    root = Path(image_root).resolve(strict=True)
    images = {}
    for sample_id, sample in identities.items():
        row, prompt = by_id[sample_id], prompts[sample.prompt_id]
        if (
            row.get("status") != "ok"
            or row.get("seed") != sample.seed
            or row.get("prompt_sha256") != hashlib.sha256(prompt.prompt.encode()).hexdigest()
        ):
            raise EvaluationError("generation-prompt-seed-or-status-mismatch")
        image = _content_under_root(row.get("image_path"), row.get("image_file_sha256"), root)
        _check_image(image, report["generation"], row.get("pixel_sha256"))
        if binding.execution_kind == "gb10_measured":
            _check_native_trace(row, root)
        images[sample_id] = image
    scorer_ids = tuple(
        sorted(
            (artifact.role.removesuffix("_scorer"), artifact.sha256)
            for artifact in request.dependencies
            if artifact.role.endswith("_scorer")
        )
    )
    if dict(scorer_ids)["ocr"] != ocr_config.sha256:
        raise EvaluationError("ocr-config-differs-from-request")
    started = datetime.now(timezone.utc).isoformat()
    output = Path(receipt_directory)
    output.mkdir(parents=True, exist_ok=False)
    if before_stage:
        before_stage("scorer_load")
    if binding.execution_kind == "fixture":
        clip_scorer, hps_scorer, recognizer = fixture_scorers
    else:
        clip_scorer, hps_scorer, recognizer = _real_scorers(
            request,
            report,
            ocr_config,
            ocr_lock,
            ocr_calibration,
            clip_checkpoint,
            hps_checkpoint,
            ocr_model_root,
            ocr_python_executable,
        )
    from ..image_eval import FrozenEvalSample, RewardInput

    ordered = tuple(request.samples)
    inputs = [
        RewardInput(
            images[sample.sample_id],
            FrozenEvalSample(
                sample.sample_id,
                "mage-direct-w4a4",
                request.protocol,
                sample.prompt_id,
                prompts[sample.prompt_id].prompt,
                "development",
                "es_train",
                sample.seed,
                json.dumps({"purpose": request.purpose, "source_group": sample.source_group}),
            ),
        )
        for sample in ordered
    ]
    if before_stage:
        before_stage("score_clip")
    clip = clip_scorer.score_batch(inputs)
    if before_stage:
        before_stage("score_hps")
    hps = hps_scorer.score_batch(inputs)
    if (
        len(clip) != len(ordered)
        or len(hps) != len(ordered)
        or any(
            type(value) not in {float, int} or not math.isfinite(value) for value in (*clip, *hps)
        )
    ):
        raise EvaluationError("missing-or-invalid-visual-scores")
    scores = []
    for sample, clip_value, hps_value in zip(ordered, clip, hps, strict=True):
        record, recognition, text_score = prompts[sample.prompt_id], None, None
        if record.complete_target_inventory:
            if before_stage:
                before_stage("score_ocr")
            recognition = recognize(images[sample.sample_id], ocr_config, recognizer)
            text_score = match_text(recognition, TargetInventory(tuple(record.target_text)))
            if text_score.status != "ok":
                raise EvaluationError(
                    f"ocr-engine-failure-cannot-form-complete-batch:{recognition.error}"
                )
        payload = {
            "sample_id": sample.sample_id,
            "image_sha256": by_id[sample.sample_id]["image_file_sha256"],
            "request_digest": request.request_digest,
            "native_report_sha256": binding.native_report_sha256,
            "scorer_sha256": dict(scorer_ids),
            "clip": float(clip_value),
            "hps": float(hps_value),
            "recognition": asdict(recognition) if recognition is not None else None,
            "text_score": asdict(text_score) if text_score is not None else None,
            "recognizer_process": recognizer.last_receipt
            if recognition is not None and binding.execution_kind == "gb10_measured"
            else None,
            "execution_kind": binding.execution_kind,
        }
        score_hash = digest_json(payload)
        _write_canonical(output / f"{sample.sample_id}.json", payload)
        scores.append(
            SampleScores(
                sample.sample_id,
                payload["image_sha256"],
                float(clip_value),
                float(hps_value),
                text_score.character_correctness if text_score else None,
                text_score.status if text_score else "not_applicable",
                score_hash,
            )
        )
    batch = ScoredBatch(
        request,
        manifest,
        tuple(scores),
        tuple(
            sorted(
                {
                    sample.prompt_id
                    for sample in ordered
                    if prompts[sample.prompt_id].complete_target_inventory
                }
            )
        ),
        scorer_ids,
        binding.native_report_sha256,
        started,
        binding.execution_kind,
    )
    validate_scored_batch(batch)
    _write_canonical(output / "batch.json", asdict(batch))
    return batch


def _hash(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _content_under_root(path, expected_hash, root):
    if not isinstance(path, str) or not _hash(expected_hash):
        raise EvaluationError("missing-image-or-trace-identity")
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise EvaluationError("generation-artifact-escapes-root")
    data = resolved.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise EvaluationError("generation-artifact-content-mismatch")
    return data


def _check_image(data, generation, pixel_hash):
    import io

    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.format != "PNG" or image.size != (
            generation.get("width"),
            generation.get("height"),
        ):
            raise EvaluationError("invalid-image-format-or-resolution")
        if hashlib.sha256(image.convert("RGB").tobytes()).hexdigest() != pixel_hash:
            raise EvaluationError("image-pixel-identity-mismatch")


def _check_native_trace(row, root):
    # This uses the same strict kernel predicate as generation. Storage type or
    # a CPU profiler range with an impressive name is not CUDA compute evidence.
    trace_bytes = _content_under_root(row.get("trace_path"), row.get("trace_file_sha256"), root)
    trace = json.loads(trace_bytes)
    if not isinstance(trace, dict) or not isinstance(trace.get("traceEvents"), list):
        raise EvaluationError("invalid-native-profiler-trace")
    kernels = {
        event["name"]
        for event in trace["traceEvents"]
        if isinstance(event, dict)
        and event.get("cat") == "kernel"
        and event.get("ph") == "X"
        and isinstance(event.get("name"), str)
        and type(event.get("dur")) in {float, int}
        and math.isfinite(event["dur"])
        and event["dur"] > 0
    }
    from ..nvfp4_runtime import native_kernel_evidence

    observed = native_kernel_evidence(kernels)
    declared_names = row.get("cuda_kernel_names")
    if not isinstance(declared_names, list) or any(
        not isinstance(name, str) for name in declared_names
    ):
        raise EvaluationError("missing-native-kernel-names")
    declared = native_kernel_evidence(declared_names)
    if (
        observed["native_compute"] is not True
        or observed != declared
        or row.get("native_evidence") != declared
    ):
        raise EvaluationError("native-row-and-cuda-trace-evidence-mismatch")


def _write_canonical(path, payload):
    path.write_text(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    )


def _real_scorers(
    request, report, config, lock, calibration, clip_path, hps_path, model_root, ocr_python
):
    import importlib.metadata

    from .. import image_reward_models
    from .ocr import freeze_scorer
    from .scorers import SubprocessOCRAdapter

    if lock is None or calibration is None:
        raise EvaluationError("missing-human-calibrated-ocr-lock")
    try:
        reconstructed = freeze_scorer(calibration)
    except (ValueError, AttributeError) as error:
        raise EvaluationError("invalid-human-calibrated-ocr-lock") from error
    if digest_json(lock) != digest_json(reconstructed) or lock["config_sha256"] != config.sha256:
        raise EvaluationError("ocr-lock-and-calibration-mismatch")
    if ocr_python is None:
        raise EvaluationError("missing-isolated-ocr-python")
    settings = request.runtime_identity.get("visual_scoring")
    required = {"package_versions", "adapter_source_sha256", "batch_size", "numerical_mode"}
    if (
        not isinstance(settings, dict)
        or set(settings) != required
        or settings["numerical_mode"] != "existing-bfloat16-dot-v1"
    ):
        raise EvaluationError("missing-frozen-visual-scorer-runtime")
    if type(settings["batch_size"]) is not int or settings["batch_size"] <= 0:
        raise EvaluationError("invalid-frozen-scorer-batch-size")
    versions = settings["package_versions"]
    if not isinstance(versions, dict) or not {"torch", "open-clip-torch", "hpsv2"} <= {
        name.lower().replace("_", "-") for name in versions
    }:
        raise EvaluationError("missing-visual-scorer-package-identities")
    for name, version in versions.items():
        if importlib.metadata.version(name) != version:
            raise EvaluationError(f"visual-scorer-version-mismatch:{name}")
    source_hash = hashlib.sha256(Path(image_reward_models.__file__).read_bytes()).hexdigest()
    if source_hash != settings["adapter_source_sha256"]:
        raise EvaluationError("visual-scorer-implementation-mismatch")
    dependencies = {artifact.role: artifact for artifact in request.dependencies}
    for axis, path in (("clip", clip_path), ("hps", hps_path)):
        if path is None:
            raise EvaluationError(f"missing-local-{axis}-checkpoint")
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != dependencies[f"{axis}_scorer"].sha256:
            raise EvaluationError(f"visual-scorer-checkpoint-mismatch:{axis}")
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        raise EvaluationError("measured-scoring-requires-gb10")
    if report["runtime"].get("device_name") != torch.cuda.get_device_name():
        raise EvaluationError("scoring-device-differs-from-generation-device")
    # Preserve the current metric, including its numerical behavior. Do not
    # silently clamp, re-normalize, multiply by 100 or advertise support bounds.
    clip = image_reward_models.OpenCLIPH14Scorer(
        clip_path, device="cuda", batch_size=settings["batch_size"]
    )
    hps = image_reward_models.HPSv21Scorer(
        hps_path, device="cuda", batch_size=settings["batch_size"]
    )
    return clip, hps, SubprocessOCRAdapter(config, model_root, python_executable=ocr_python)
