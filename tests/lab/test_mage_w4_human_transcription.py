"""Human artifact ingestion never invents completeness or drops planned images."""

import base64
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from shadow_lab.mage_w4_search.budget import DeviceLease
from shadow_lab.mage_w4_search.contracts import (
    ArtifactIdentity,
    CandidateRequest,
    DataManifest,
    PromptRecord,
    RequestedSample,
    digest_json,
)
from shadow_lab.mage_w4_search.human_transcription import close_comparison, import_human_work
from shadow_lab.mage_w4_search.ocr import (
    CalibrationPlan,
    EngineIdentity,
    OCRConfig,
    RawSegment,
    recognize,
)
from shadow_lab.mage_w4_search.scorers import REQUEST_SCHEMA, RESPONSE_SCHEMA


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return path


def human_inputs(tmp_path):
    inventory = {
        "purpose": "ocr_selection",
        "rows": [
            {"image_sha256": "a" * 64},
            {"image_sha256": "b" * 64},
        ],
    }
    source = {
        "schema": "mage-ocr-human-transcription-entry/v1",
        "purpose": "ocr_selection",
        "transcribed_by": "human-fixture",
        "transcribed_at_utc": "2026-09-09T00:00:00Z",
        "rows": [
            {
                "index": 0,
                "image_sha256": "a" * 64,
                "complete": True,
                "actual_segments": ["Save Water", "请  水水"],
            },
            {
                "index": 1,
                "image_sha256": "b" * 64,
                "complete": False,
                "actual_segments": ["Bamboo 竹出"],
            },
        ],
    }
    inventory_path = write_json(tmp_path / "images.json", inventory)
    source_path = write_json(tmp_path / "human.json", source)
    return source_path, inventory_path, source


def plan_for(images):
    configs = tuple(
        OCRConfig(EngineIdentity(name, "fixture", (("model", "c" * 64),)))
        for name in ("PaddleOCR", "EasyOCR")
    )
    return CalibrationPlan(
        tuple(images), configs, ("d" * 64,) * len(images), "2026-09-09T00:01:00Z", 0.25
    )


def test_import_preserves_author_text_and_raw_artifact_identity(tmp_path):
    path, inventory, _ = human_inputs(tmp_path)
    work = import_human_work(path, inventory)
    assert work.artifact_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert work.rows[0].actual_segments == ("Save Water", "请  水水")
    assert work.rows[1].complete is False
    assert work.status == "structurally_valid_author_declarations_only"
    selected = work.for_plan(plan_for(["a" * 64]))
    assert selected[0].artifact_sha256 == work.artifact_sha256
    assert selected[0].annotator_id == "human-fixture"
    assert work.exclusions_for(plan_for(["a" * 64])) == (
        {"index": 1, "image_sha256": "b" * 64, "reason": "author_declared_incomplete"},
    )


def test_incomplete_planned_image_cannot_be_calibration_truth(tmp_path):
    path, inventory, _ = human_inputs(tmp_path)
    with pytest.raises(ValueError, match="incomplete"):
        import_human_work(path, inventory).for_plan(plan_for(["a" * 64, "b" * 64]))


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v.update(schema="unknown"),
        lambda v: v.update(purpose="final"),
        lambda v: v.update(transcribed_by=" "),
        lambda v: v.update(transcribed_at_utc="2026-09-09T00:00:00"),
        lambda v: v["rows"].append(v["rows"][0]),
        lambda v: v["rows"].pop(),
        lambda v: v["rows"][0].update(index=True),
        lambda v: v["rows"][0].update(index=3),
        lambda v: v["rows"][0].update(image_sha256="e" * 64),
        lambda v: v["rows"][0].update(complete="true"),
        lambda v: v["rows"][0].update(actual_segments=[1]),
        lambda v: v.update(unexpected="value"),
    ],
)
def test_invalid_human_boundaries_are_rejected(tmp_path, change):
    path, inventory, source = human_inputs(tmp_path)
    change(source)
    write_json(path, source)
    with pytest.raises(ValueError):
        import_human_work(path, inventory)


def test_duplicate_json_keys_are_rejected(tmp_path):
    path, inventory, _ = human_inputs(tmp_path)
    path.write_text(
        path.read_text().replace(
            '"purpose": "ocr_selection"', '"purpose": "final", "purpose": "ocr_selection"'
        )
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        import_human_work(path, inventory)


def stamp(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def comparison_inputs(tmp_path):
    """Recorded fixture evidence, never real device/model execution."""
    images = []
    for index in range(2):
        path = tmp_path / f"{index}.png"
        path.write_bytes(f"fixture image {index}".encode())
        images.append(
            {"image_path": str(path), "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    records = tuple(
        PromptRecord(
            f"p{i}",
            f"fixture prompt {i}",
            f"source{i}",
            "development",
            ("ocr_selection",),
            (),
            False,
        )
        for i in range(2)
    )
    manifest = DataManifest(records)
    manifest_path = write_json(tmp_path / "manifest.json", asdict(manifest))
    samples = tuple(
        RequestedSample(row.prompt_id, digest_json(row.prompt), row.source_group, 42)
        for row in records
    )
    for image, sample in zip(images, samples, strict=True):
        image["sample_id"] = sample.sample_id
    inventory_path = write_json(
        tmp_path / "images.json",
        {
            "purpose": "ocr_selection",
            "q0_sha256": "a" * 64,
            "source_job": "generation",
            "rows": images,
        },
    )
    human_path = write_json(
        tmp_path / "human.json",
        {
            "schema": "mage-ocr-human-transcription-entry/v1",
            "purpose": "ocr_selection",
            "transcribed_by": "fixture-human",
            "transcribed_at_utc": stamp(900),
            "rows": [
                {
                    "index": i,
                    "image_sha256": row["image_sha256"],
                    "complete": i == 0,
                    "actual_segments": ["OPEN" if i == 0 else "uncertain"],
                }
                for i, row in enumerate(images)
            ],
        },
    )
    dependencies = tuple(
        ArtifactIdentity(role, "d" * 64, 1)
        for role in (
            "quantized_packed",
            "weight_scales",
            "quantized_remainder",
            "layer_manifest",
            "text_encoder",
            "vae",
        )
    )
    base = CandidateRequest(
        "request",
        "generation",
        "04",
        0,
        "a" * 64,
        "a" * 64,
        "q0",
        "a" * 64,
        0,
        0,
        0,
        samples,
        "ocr_selection",
        "gooo",
        dependencies,
        {
            "software": "fixture",
            "kernel": "fixture",
            "sampler": "fixture",
            "generation": {"steps": 4},
            "device": {"id": "gooo", "nvidia_identity": "fixture"},
        },
        manifest.digest,
        evaluation_mode="generation",
    )
    leases = []

    def job_directory(name, request, start, stop, group):
        directory = tmp_path / name
        directory.mkdir()
        lease = DeviceLease(name, name + "-occupancy", "04", "gooo", start, start + 300, 300, name)
        request = replace(request, run_id=name, request_id=name, lease_id=lease.lease_id)
        write_json(directory / "lease.json", asdict(lease))
        write_json(
            directory / "grant.json",
            {
                "status": "authorized",
                "request_digest": request.request_digest,
                "request": asdict(request),
                "lease": asdict(lease),
            },
        )
        write_json(
            directory / "clock.json",
            {"offset_seconds": 0, "uncertainty_seconds": 0, "at": start - 1},
        )
        write_json(
            directory / "stopped.json",
            {
                **asdict(lease),
                "stop_confirmed": True,
                "stopped_at": stop,
                "observed_at": stop,
                "exit_code": 0,
                "process_group": group,
            },
        )
        write_json(
            directory / "independent-group-check.json",
            {
                "process_group": group,
                "absent": True,
                "checked_at": 1401,
                "device_id": "gooo",
                "lease_id": name,
            },
        )
        leases.append(
            {**asdict(lease), "stopped_at": stop, "stop_reason": "fixture-confirmed-stop"}
        )
        return directory, request

    gen, _ = job_directory("generation", base, 500, 600, 100)
    report_path = write_json(
        tmp_path / "native.json",
        {
            "qualified": True,
            "center_sha256": "a" * 64,
            "rows": [
                {"sample_id": s.sample_id, "image_file_sha256": i["image_sha256"]}
                for s, i in zip(samples, images, strict=True)
            ],
        },
    )
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    configs = tuple(
        OCRConfig(
            EngineIdentity(name, "fixture", (("model", "b" * 64),)),
            settings_json=json.dumps({"adapter_source_sha256": "f" * 64}),
        )
        for name in ("PaddleOCR", "EasyOCR")
    )
    plan = CalibrationPlan((images[0]["image_sha256"],), configs, (report_sha,), stamp(1000), 0.25)
    plan_path = write_json(tmp_path / "plan.json", asdict(plan))
    work = import_human_work(human_path, inventory_path)
    cost = {
        "schema": "mage-w4-ocr-cost-allocation-v1",
        "method": "own_ocr_lease_plus_shared_generation",
        "shared_generation_lease_id": "generation",
        "shared_generation_report_sha256": report_sha,
        "comparison_total_method": "sum_unique_leases_once",
    }
    directories = []
    registered_jobs = []
    for index, config in enumerate(configs):
        job = {
            "schema": "mage-w4-ocr-worker-v1",
            "config_index": index,
            "plan": asdict(plan),
            "plan_sha256": plan.sha256,
            "human_transcription_artifact_sha256": work.artifact_sha256,
            "exclusions": list(work.exclusions_for(plan)),
            "cost_allocation": cost,
            "images": [
                {
                    "path": images[0]["image_path"],
                    "sha256": images[0]["image_sha256"],
                    "generation_report_path": str(report_path),
                    "generation_report_sha256": report_sha,
                    "sample_id": samples[0].sample_id,
                }
            ],
        }
        request = replace(
            base,
            samples=samples[:1],
            runtime_identity={
                **base.runtime_identity,
                "ocr_comparison_job_sha256": digest_json(job),
            },
        )
        start = 1100 + index * 100
        directory, request = job_directory(
            f"engine{index}", request, start, start + 20 + 10 * index, 101 + index
        )
        write_json(directory / "ocr-job.json", job)
        data = Path(images[0]["image_path"]).read_bytes()
        observation = replace(
            recognize(data, config, lambda *_: [RawSegment("OPEN", (0, 0, 20, 10))]),
            elapsed_seconds=0.1,
            started_at_utc=stamp(start + 1),
        )
        child_request = {
            "schema": REQUEST_SCHEMA,
            "action": "recognize",
            "config": asdict(config),
            "config_sha256": config.sha256,
            "image_sha256": observation.image_sha256,
            "image_base64": base64.b64encode(data).decode("ascii"),
        }
        row = {
            "observation": asdict(observation),
            "child": {
                "schema": RESPONSE_SCHEMA,
                "action": "recognize",
                "request_sha256": digest_json(child_request),
                "config_sha256": config.sha256,
                "image_sha256": observation.image_sha256,
                "status": "ok",
                "segments": [asdict(segment) for segment in observation.raw_segments],
                "returncode": 0,
                "worker": {
                    "pid": 999,
                    "process_group": 101 + index,
                    "adapter_source_sha256": "f" * 64,
                },
                "timing": {"wall_seconds": 0.08, "host_cpu_seconds": 0.04},
                "parent_wall_seconds": 0.09,
                "stdout_sha256": "e" * 64,
                "stderr_sha256": "e" * 64,
            },
        }
        write_json(directory / "ocr-observation-0000.json", row)
        write_json(
            directory / "ocr-observations.json",
            {
                "schema": "mage-w4-ocr-observations-v1",
                "request_digest": request.request_digest,
                "lease_id": directory.name,
                "device_id": "gooo",
                "plan_sha256": plan.sha256,
                "config_sha256": config.sha256,
                "job_sha256": digest_json(job),
                "rows": [row],
                "cost_basis": "pending_coordinator_verified_lease_stop",
                "scorer_frozen": False,
            },
        )
        registered_jobs.append(
            {
                "job_id": directory.name,
                "config_sha256": config.sha256,
                "job_sha256": digest_json(job),
                "request_digest": replace(request, lease_id=None).request_digest,
            }
        )
        directories.append(directory)
    registration_path = write_json(
        tmp_path / "registration.json",
        {
            "schema": "mage-w4-ocr-comparison-registration/v1",
            "plan_sha256": plan.sha256,
            "plan_file_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "registered_at_utc": plan.registered_at_utc,
            "human_transcription_artifact_sha256": work.artifact_sha256,
            "accepted_indices": [0],
            "exclusions": list(work.exclusions_for(plan)),
            "maximum_mean_error": 0.25,
            "cost_allocation": cost,
            "jobs": registered_jobs,
        },
    )
    snapshot = write_json(
        tmp_path / "budget.json",
        {"campaign_id": "fixture", "at": 1400, "leases": leases, "used_seconds": 150},
    )
    return {
        "source_path": human_path,
        "inventory_path": inventory_path,
        "plan_path": plan_path,
        "registration_path": registration_path,
        "manifest_path": manifest_path,
        "budget_snapshot_path": snapshot,
        "generation_directory": gen,
        "generation_report_path": report_path,
        "worker_directories": directories,
        "execution_kind": "fixture",
    }


def test_closeout_uses_typed_identity_and_deduplicates_shared_generation(tmp_path):
    result = close_comparison(**comparison_inputs(tmp_path))
    assert result.report.execution_kind == "fixture"
    assert result.report.selected_engine == "PaddleOCR"
    assert result.scorer_lock is None
    assert [engine.occupied_seconds for engine in result.report.engines] == [120, 130]
    assert result.accounting["comparison_total_seconds"] == 150
    assert result.accounting["shared_generation_seconds"] == 100


@pytest.mark.parametrize(
    "file,key,value",
    [
        ("engine0/stopped.json", "stop_confirmed", False),
        ("engine0/ocr-observations.json", "lease_id", "unrelated"),
        ("engine0/ocr-observations.json", "rows", []),
        ("engine0/grant.json", "request_digest", "0" * 64),
        ("registration.json", "plan_sha256", "0" * 64),
    ],
)
def test_closeout_rejects_changed_or_incomplete_evidence(tmp_path, file, key, value):
    kwargs = comparison_inputs(tmp_path)
    path = tmp_path / file
    payload = json.loads(path.read_text())
    payload[key] = value
    write_json(path, payload)
    with pytest.raises(ValueError):
        close_comparison(**kwargs)


def test_measured_label_does_not_qualify_fixture_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    kwargs = comparison_inputs(tmp_path)
    kwargs["execution_kind"] = "gb10_measured"
    with pytest.raises(ValueError, match="GB10"):
        close_comparison(**kwargs)


@pytest.mark.parametrize(
    "patch",
    [
        {"uncertainty_seconds": 2.1},
        {"offset_seconds": -1.9, "uncertainty_seconds": 0.2},
        {"at": 1100.01},
    ],
)
def test_clock_outside_frozen_bound_cannot_qualify_closeout(tmp_path, patch):
    kwargs = comparison_inputs(tmp_path)
    path = tmp_path / "engine0/clock.json"
    value = json.loads(path.read_text())
    value.update(patch)
    write_json(path, value)
    with pytest.raises(ValueError, match="clock evidence"):
        close_comparison(**kwargs)


def test_late_settlement_is_fully_billed_but_actual_late_stop_is_rejected(tmp_path):
    kwargs = comparison_inputs(tmp_path)
    path = tmp_path / "budget.json"
    snapshot = json.loads(path.read_text())
    snapshot["at"] = 1600
    snapshot["leases"][1]["stopped_at"] = 1450
    snapshot["used_seconds"] = 480
    write_json(path, snapshot)
    assert close_comparison(**kwargs).accounting["comparison_total_seconds"] == 480
    stop_path = tmp_path / "engine0/stopped.json"
    stop = json.loads(stop_path.read_text())
    stop["stopped_at"] = 1400.01
    write_json(stop_path, stop)
    with pytest.raises(ValueError, match="hard deadline"):
        close_comparison(**kwargs)


def test_failed_engine_row_is_preserved_instead_of_silently_dropped(tmp_path):
    kwargs = comparison_inputs(tmp_path)
    path = tmp_path / "engine0/ocr-observations.json"
    package = json.loads(path.read_text())
    row = package["rows"][0]
    row["observation"].update(
        status="engine_failure", raw_segments=[], segments=[], error="fixture process failure"
    )
    row["child"].update(status="failed", error="fixture process failure", returncode=1)
    write_json(path, package)
    write_json(tmp_path / "engine0/ocr-observation-0000.json", row)
    report = close_comparison(**kwargs).report
    assert report.selected_engine == "EasyOCR"
    assert report.engines[0].failure_count == 1
    assert report.engines[0].mean_normalized_edit_error is None
    assert len(report.engines[0].images) == 1


def test_missing_partial_artifact_rejects_an_otherwise_complete_package(tmp_path):
    kwargs = comparison_inputs(tmp_path)
    (tmp_path / "engine0/ocr-observation-0000.json").unlink()
    with pytest.raises(ValueError, match="raw observation"):
        close_comparison(**kwargs)


def test_raw_human_content_and_sha_come_from_one_read(tmp_path, monkeypatch):
    source, inventory, _ = human_inputs(tmp_path)
    original = source.read_bytes()
    real_read = Path.read_bytes
    count = 0

    def changing_read(path):
        nonlocal count
        value = real_read(path)
        if path == source:
            count += 1
            source.write_bytes(b'{"different":"artifact"}')
        return value

    monkeypatch.setattr(Path, "read_bytes", changing_read)
    work = import_human_work(source, inventory)
    assert count == 1
    assert work.artifact_sha256 == hashlib.sha256(original).hexdigest()
    assert work.rows[0].actual_segments[0] == "Save Water"


def test_cli_fixture_closeout_writes_report_identity_without_a_lock(tmp_path):
    kwargs = comparison_inputs(tmp_path)
    inputs = write_json(
        tmp_path / "inputs.json",
        {
            key: [str(path) for path in value] if key == "worker_directories" else str(value)
            for key, value in kwargs.items()
        },
    )
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "closeout"
    command = [
        sys.executable,
        str(root / "scripts/mage_w4_ocr_calibrate.py"),
        "closeout",
        "--inputs",
        str(inputs),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        text=True,
        capture_output=True,
        check=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["status"] == "fixture_only"
    assert summary["comparison_total_seconds"] == 150
    assert not (output / "scorer-lock.json").exists()
    assert (
        summary["artifacts"]["calibration-report.json"]["file_sha256"]
        == hashlib.sha256((output / "calibration-report.json").read_bytes()).hexdigest()
    )
    repeated = subprocess.run(
        command,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        text=True,
        capture_output=True,
    )
    assert repeated.returncode != 0
    assert "already exists" in repeated.stderr
