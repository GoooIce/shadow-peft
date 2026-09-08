"""Generated images reach complete scoring only through matching frozen identities."""

import hashlib
import json
from dataclasses import replace

import pytest
from PIL import Image

from shadow_lab.mage_w4_search.contracts import (
    ArtifactIdentity,
    CandidateRequest,
    DataManifest,
    PromptRecord,
    RequestedSample,
    digest_json,
)
from shadow_lab.mage_w4_search.evaluation import (
    GenerationBinding,
    evaluate_generated_candidate,
    load_scored_batch,
)
from shadow_lab.mage_w4_search.ocr import EngineIdentity, OCRConfig, RawSegment


class ConstantScorer:
    def __init__(self, value):
        self.value = value

    def score_batch(self, inputs):
        return [self.value for _ in inputs]


def fixture(tmp_path):
    config = OCRConfig(EngineIdentity("fixture-ocr", "1.0", (("fixture", "e" * 64),)))
    record = PromptRecord(
        "p-1",
        'Only the text "OPEN"',
        "source-1",
        "development",
        ("candidate_development",),
        ("OPEN",),
        True,
    )
    manifest = DataManifest((record,))
    sample = RequestedSample(record.prompt_id, digest_json(record.prompt), record.source_group, 7)
    runtime = {
        "software": "fixture",
        "kernel": "fixture",
        "sampler": "fixture",
        "generation": {"width": 2, "height": 2},
        "device": "gooo",
    }
    roles = (
        "quantized_packed",
        "weight_scales",
        "quantized_remainder",
        "layer_manifest",
        "text_encoder",
        "vae",
        "clip_scorer",
        "hps_scorer",
        "ocr_scorer",
    )
    request = CandidateRequest(
        "request-1",
        "run-1",
        "05",
        0,
        "a" * 64,
        "a" * 64,
        "candidate-1",
        "b" * 64,
        0,
        7,
        1,
        (sample,),
        "candidate_development",
        "gooo",
        tuple(
            ArtifactIdentity(role, config.sha256 if role == "ocr_scorer" else "c" * 64, 1)
            for role in roles
        ),
        runtime,
        manifest.digest,
    )
    image = tmp_path / "native.png"
    Image.new("RGB", (2, 2), "white").save(image)
    report = {
        "schema_version": 1,
        "center_sha256": request.candidate_digest,
        "fixed_sha256": "d" * 64,
        "runtime": {"fixture": True},
        "generation": runtime["generation"],
        "status": "ok",
        "native_compute": False,
        "qualified": False,
        "installation": {"backend": "fixture"},
        "rows": [
            {
                "sample_id": sample.sample_id,
                "seed": 7,
                "prompt_sha256": hashlib.sha256(record.prompt.encode()).hexdigest(),
                "status": "ok",
                "image_path": str(image),
                "image_file_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "pixel_sha256": hashlib.sha256(
                    Image.open(image).convert("RGB").tobytes()
                ).hexdigest(),
                "native_evidence": {"native_compute": False},
            }
        ],
    }
    binding = GenerationBinding(
        request.request_digest,
        digest_json(report),
        "d" * 64,
        digest_json(report["runtime"]),
        digest_json(report["generation"]),
        "fixture",
    )
    return request, manifest, config, report, binding


def test_fixture_generation_is_scored_with_target_blind_ocr_and_stays_fixture(tmp_path):
    request, manifest, config, report, binding = fixture(tmp_path)
    seen = []

    def recognize(image_bytes, generic_config):
        seen.append((image_bytes, generic_config))
        return (RawSegment("OPFN", (0.0, 0.0, 2.0, 1.0)),)

    batch = evaluate_generated_candidate(
        request,
        manifest,
        report,
        binding=binding,
        expected_center_version=0,
        image_root=tmp_path,
        receipt_directory=tmp_path / "scores",
        ocr_config=config,
        fixture_scorers=(ConstantScorer(0.3), ConstantScorer(0.2), recognize),
    )
    assert batch.execution_kind == "fixture"
    assert batch.samples[0].ocr == 0.75
    assert batch.samples[0].image_sha256 == report["rows"][0]["image_file_sha256"]
    assert len(seen) == 1
    assert seen[0][1] == config
    assert load_scored_batch(tmp_path / "scores" / "batch.json") == batch
    payload = json.loads((tmp_path / "scores" / "batch.json").read_text())
    payload["samples"].append(payload["samples"][0])
    (tmp_path / "scores" / "invalid.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicate"):
        load_scored_batch(tmp_path / "scores" / "invalid.json")


@pytest.mark.parametrize(
    "corruption",
    ("missing", "duplicate", "seed", "bytes", "pixels", "stale", "scorer", "native-promotion"),
)
def test_corrupt_or_promoted_artifacts_fail_before_external_scoring(tmp_path, corruption):
    request, manifest, config, report, binding = fixture(tmp_path)
    expected_version = 0
    if corruption == "missing":
        report["rows"] = []
    elif corruption == "duplicate":
        report["rows"].append(dict(report["rows"][0]))
    elif corruption == "seed":
        report["rows"][0]["seed"] = 8
    elif corruption == "bytes":
        (tmp_path / "native.png").write_bytes(b"different content")
    elif corruption == "pixels":
        report["rows"][0]["pixel_sha256"] = "0" * 64
    elif corruption == "stale":
        expected_version = 1
    elif corruption == "scorer":
        config = OCRConfig(EngineIdentity("different", "1.0", (("fixture", "f" * 64),)))
    else:
        binding = replace(binding, execution_kind="gb10_measured")
    binding = replace(binding, native_report_sha256=digest_json(report))

    class MustNotScore:
        def score_batch(self, inputs):
            raise AssertionError("invalid request reached expensive scoring")

    with pytest.raises(ValueError):
        evaluate_generated_candidate(
            request,
            manifest,
            report,
            binding=binding,
            expected_center_version=expected_version,
            image_root=tmp_path,
            receipt_directory=tmp_path / "scores",
            ocr_config=config,
            fixture_scorers=(MustNotScore(), MustNotScore(), lambda *_: ()),
        )


@pytest.mark.parametrize(
    "trace_problem", ("missing", "cpu-only", "false-row", "missing-human-lock")
)
def test_claimed_native_report_requires_cuda_trace_and_human_ocr_lock(tmp_path, trace_problem):
    """Synthetic reports exercise rejection only; they never produce measured scores."""
    request, manifest, config, report, binding = fixture(tmp_path)
    kernel = "fixture_sm121_block_scaled_ue4m3xe2m1_ue4m3xe2m1"
    row = report["rows"][0]
    row["cuda_kernel_names"] = [kernel]
    row["native_evidence"] = {
        "quant_representation": "packed_e2m1_with_e4m3_block_scale",
        "compute_kernel": [kernel],
        "native_compute": True,
        "fallback_reason": None,
    }
    report.update(native_compute=True, qualified=True)
    report["installation"] = {
        "backend": "torchao_nvfp4",
        "installed_layers": [f"layer-{i}" for i in range(174)],
    }
    report["runtime"] = {
        "device_name": "fixture is not physical GB10 evidence",
        "compute_capability": [12, 1],
    }
    trace = tmp_path / "fixture.trace.json"
    trace.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "name": kernel,
                        "cat": "cpu_op" if trace_problem == "cpu-only" else "kernel",
                        "ph": "X",
                        "dur": 1.0,
                    }
                ]
            }
        )
    )
    if trace_problem != "missing":
        row.update(
            trace_path=str(trace), trace_file_sha256=hashlib.sha256(trace.read_bytes()).hexdigest()
        )
    if trace_problem == "false-row":
        row["native_evidence"]["native_compute"] = False
    binding = replace(
        binding,
        native_report_sha256=digest_json(report),
        runtime_sha256=digest_json(report["runtime"]),
        execution_kind="gb10_measured",
    )
    expected = (
        "missing-human-calibrated-ocr-lock"
        if trace_problem == "missing-human-lock"
        else "native|trace"
    )
    with pytest.raises(ValueError, match=expected):
        evaluate_generated_candidate(
            request,
            manifest,
            report,
            binding=binding,
            expected_center_version=0,
            image_root=tmp_path,
            receipt_directory=tmp_path / "scores",
            ocr_config=config,
        )
