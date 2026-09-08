"""The OCR worker must reject changed inputs before loading either engine."""

import hashlib
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from shadow_lab.mage_w4_search.contracts import digest_json
from shadow_lab.mage_w4_search.ocr import CalibrationPlan, EngineIdentity, OCRConfig
from shadow_lab.mage_w4_search.ocr_worker import validate_comparison_job


def inputs(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture-image-never-recognized")
    image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
    report = tmp_path / "native.json"
    report.write_text(
        json.dumps(
            {
                "qualified": True,
                "center_sha256": "a" * 64,
                "rows": [{"sample_id": "sample", "image_file_sha256": image_sha}],
            }
        )
    )
    report_sha = hashlib.sha256(report.read_bytes()).hexdigest()
    configs = tuple(
        OCRConfig(EngineIdentity(name, "fixture", (("weight", "b" * 64),)))
        for name in ("PaddleOCR", "EasyOCR")
    )
    plan = CalibrationPlan((image_sha,), configs, (report_sha,), "2026-09-09T00:00:00Z", 0.1)
    job = {
        "schema": "mage-w4-ocr-worker-v1",
        "plan": asdict(plan),
        "plan_sha256": plan.sha256,
        "config_index": 0,
        "images": [
            {
                "path": str(image),
                "sha256": image_sha,
                "sample_id": "sample",
                "generation_report_path": str(report),
                "generation_report_sha256": report_sha,
            }
        ],
        "model_root": str(tmp_path),
        "python_executable": "/not/executed",
    }
    request = SimpleNamespace(
        ticket_id="04",
        purpose="ocr_selection",
        q0_digest="a" * 64,
        samples=(SimpleNamespace(sample_id="sample"),),
        runtime_identity={"ocr_comparison_job_sha256": digest_json(job)},
    )
    return job, request


def test_changed_image_or_job_is_rejected_before_model_loading(tmp_path):
    job, request = inputs(tmp_path)
    validate_comparison_job(job, request)
    job["images"][0]["path"] = str(tmp_path / "changed.png")
    with pytest.raises(ValueError, match="job identity"):
        validate_comparison_job(job, request)
    job, request = inputs(tmp_path)
    (tmp_path / "image.png").write_bytes(b"different")
    with pytest.raises(ValueError, match="image SHA"):
        validate_comparison_job(job, request)


def test_wrong_pool_or_generation_identity_is_rejected(tmp_path):
    job, request = inputs(tmp_path)
    request.purpose = "final"
    with pytest.raises(ValueError, match="OCR development"):
        validate_comparison_job(job, request)
    job, request = inputs(tmp_path)
    request.q0_digest = "c" * 64
    with pytest.raises(ValueError, match="native q0"):
        validate_comparison_job(job, request)
