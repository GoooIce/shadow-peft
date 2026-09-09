"""Complete evaluation jobs preserve authorization and all three scoring axes."""

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from shadow_lab.mage_w4_search.contracts import (
    ArtifactIdentity,
    DataManifest,
    PromptRecord,
    RequestedSample,
    digest_json,
)
from shadow_lab.mage_w4_search.evaluation_worker import (
    numerical_implementation_hashes,
    package_runtime_sources,
    prepare_evaluation_job,
    run_evaluation_job,
    verify_scoring_inputs,
)


def request_inputs():
    record = PromptRecord(
        "p1",
        'A sign saying "OPEN"',
        "source1",
        "development",
        ("candidate_development",),
        ("OPEN",),
        True,
    )
    manifest = DataManifest((record,))
    sample = RequestedSample(record.prompt_id, digest_json(record.prompt), record.source_group, 7)
    plan = {
        "schema": "mage-w4-evaluation-plan/v1",
        "purpose": "candidate_development",
        "samples": [asdict(sample)],
        "candidate": {"kind": "q0", "candidate_id": "q0", "group_id": 0},
    }
    runtime = {
        "software": "fixture",
        "kernel": "fixture",
        "sampler": "fixture",
        "generation": {"width": 2, "height": 2},
        "device": {"id": "gooo"},
    }
    dependencies = tuple(
        ArtifactIdentity(role, "c" * 64, 1)
        for role in (
            "quantized_packed",
            "quantized_remainder",
            "weight_scales",
            "layer_manifest",
            "text_encoder",
            "vae",
            "clip_scorer",
            "hps_scorer",
            "ocr_scorer",
        )
    )
    shared = {
        "runtime_identity": runtime,
        "dependencies": [asdict(item) for item in dependencies],
        "q0_digest": "a" * 64,
        "fixed_state_sha256": "d" * 64,
        "inputs": {},
    }
    return plan, manifest, shared


def test_complete_q0_request_binds_its_job_without_polluting_fixed_runtime():
    plan, manifest, shared = request_inputs()
    prepared = prepare_evaluation_job("q0-fixture", plan, manifest, shared=shared)
    request = prepared["request"]
    assert request["evaluation_mode"] == "complete"
    assert request["ticket_id"] == "05"
    assert request["candidate_digest"] == request["q0_digest"]
    assert request["request_id"] == "evaluation-" + digest_json(prepared["evaluation_job"])
    assert request["runtime_identity"] == shared["runtime_identity"]
    assert prepared["preflight"]["status"] == "ready_for_reservation"


def test_candidate_receipt_is_bound_while_runtime_matches_q0_reference():
    plan, manifest, shared = request_inputs()
    q0 = prepare_evaluation_job("q0-ref", plan, manifest, shared=shared)
    plan["candidate"] = {
        "kind": "adjacent",
        "candidate_id": "candidate-1",
        "group_id": 0,
        "mutation_count": 1,
        "seed": 20260909,
        "candidate_index": 0,
        "max_attempts": 1,
    }
    receipt = {
        "status": "candidate",
        "base_sha256": "a" * 64,
        "fixed_sha256": "d" * 64,
        "candidate_sha256": "b" * 64,
        "group_id": "block_00",
        "actual_changes": 1,
        "requested_changes": 1,
        "seed": 20260909,
        "candidate_index": 0,
        "max_attempts": 1,
    }
    candidate = prepare_evaluation_job(
        "candidate-1",
        plan,
        manifest,
        shared=shared,
        candidate_receipt=receipt,
    )
    assert candidate["request"]["candidate_digest"] == "b" * 64
    assert candidate["request"]["effective_edits"] == 1
    assert candidate["request"]["runtime_identity"] == q0["request"]["runtime_identity"]
    assert candidate["request"]["request_id"] != q0["request"]["request_id"]
    plan["purpose"] = "scale_calibration"
    with pytest.raises(ValueError, match="scale calibration requires q0"):
        prepare_evaluation_job(
            "bad-scale", plan, manifest, shared=shared, candidate_receipt=receipt
        )


def fixture_job(tmp_path):
    from shadow_lab.mage_w4_search.budget import DeviceLease
    from shadow_lab.mage_w4_search.ocr import EngineIdentity, OCRConfig

    plan, manifest, shared = request_inputs()
    config = OCRConfig(EngineIdentity("fixture", "1", (("weights", "e" * 64),)))
    shared["inputs"] = {"ocr_config": asdict(config)}
    next(row for row in shared["dependencies"] if row["role"] == "ocr_scorer")["sha256"] = (
        config.sha256
    )
    prepared = prepare_evaluation_job("q0-fixture", plan, manifest, shared=shared)
    now = time.time()
    lease = DeviceLease(
        "lease-fixture", "occupancy", "05", "gooo", now - 1, now + 300, 301, "q0-fixture"
    )
    request = prepared["request"] | {"lease_id": lease.lease_id}
    grant = {
        "status": "authorized",
        "request": request,
        "request_digest": prepared["preflight"]["request_digest"],
        "lease": asdict(lease),
    }
    for name, value in {
        "prepared.json": prepared,
        "grant.json": grant,
        "lease.json": asdict(lease),
    }.items():
        (tmp_path / name).write_text(json.dumps(value))
    return manifest, shared


class FixtureScorer:
    def __init__(self, score, stages):
        self.score, self.stages = score, stages

    def score_batch(self, inputs):
        self.stages.append("score")
        return [self.score for _ in inputs]


def fixture_generator(shared, stages):
    def generate(request, manifest, directory):
        from PIL import Image

        directory.mkdir()
        stages.append("generation")
        record = manifest.records[0]
        image = directory / "0000.png"
        Image.new("RGB", (2, 2), "white").save(image)
        report = {
            "schema_version": 1,
            "status": "ok",
            "qualified": False,
            "native_compute": False,
            "center_sha256": request.candidate_digest,
            "fixed_sha256": shared["fixed_state_sha256"],
            "runtime": {"fixture": True},
            "generation": request.runtime_identity["generation"],
            "installation": {"backend": "fixture"},
            "rows": [
                {
                    "sample_id": request.samples[0].sample_id,
                    "seed": 7,
                    "status": "ok",
                    "prompt_sha256": hashlib.sha256(record.prompt.encode()).hexdigest(),
                    "image_path": str(image),
                    "image_file_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    "pixel_sha256": hashlib.sha256(
                        Image.open(image).convert("RGB").tobytes()
                    ).hexdigest(),
                }
            ],
        }
        (directory / "report.json").write_text(json.dumps(report))
        return report

    return generate


def test_authenticated_complete_fixture_writes_all_three_axes_and_bound_artifacts(tmp_path):
    from shadow_lab.mage_w4_search.evaluation import load_scored_batch
    from shadow_lab.mage_w4_search.ocr import RawSegment

    manifest, shared = fixture_job(tmp_path)
    stages = []
    result = run_evaluation_job(
        tmp_path,
        manifest,
        device_id="gooo",
        fixture_generation=fixture_generator(shared, stages),
        fixture_scorers=(
            FixtureScorer(0.3, stages),
            FixtureScorer(0.2, stages),
            lambda image, config: (RawSegment("OPEN", (0.0, 0.0, 2.0, 1.0)),),
        ),
    )
    batch = load_scored_batch(tmp_path / "scores/batch.json")
    assert stages == ["generation", "score", "score"]
    assert batch.execution_kind == "fixture"
    assert batch.samples[0].ocr == 1.0
    assert result["scored_batch_sha256"] == batch.sha256
    assert result["execution_kind"] == "fixture"
    assert result["status"] == "ok"
    assert result["generation_binding"]["request_digest"] == batch.request.request_digest
    assert (
        result["native_report_file_sha256"]
        == hashlib.sha256((tmp_path / "generation/report.json").read_bytes()).hexdigest()
    )
    assert (
        result["scored_batch_file_sha256"]
        == hashlib.sha256((tmp_path / "scores/batch.json").read_bytes()).hexdigest()
    )
    assert json.loads((tmp_path / "result.json").read_text()) == result


def test_worker_rejects_changed_job_before_generation_and_retains_failure(tmp_path):
    manifest, shared = fixture_job(tmp_path)
    prepared_path = tmp_path / "prepared.json"
    prepared = json.loads(prepared_path.read_text())
    prepared["evaluation_job"]["plan"]["samples"][0]["seed"] = 8
    prepared_path.write_text(json.dumps(prepared))
    seen = []
    with pytest.raises(ValueError, match="frozen evaluation"):
        run_evaluation_job(
            tmp_path,
            manifest,
            device_id="gooo",
            fixture_generation=fixture_generator(shared, seen),
            fixture_scorers=(None, None, None),
        )
    assert seen == []
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "failed"
    with pytest.raises(ValueError, match="already started"):
        run_evaluation_job(tmp_path, manifest, device_id="gooo")


def test_incomplete_scoring_cannot_emit_a_successful_batch(tmp_path):
    manifest, shared = fixture_job(tmp_path)
    stages = []

    class MissingScorer:
        def score_batch(self, inputs):
            return []

    with pytest.raises(ValueError, match="missing-or-invalid-visual-scores"):
        run_evaluation_job(
            tmp_path,
            manifest,
            device_id="gooo",
            fixture_generation=fixture_generator(shared, stages),
            fixture_scorers=(FixtureScorer(0.3, stages), MissingScorer(), lambda image, config: ()),
        )
    assert not (tmp_path / "scores/batch.json").exists()
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "failed"


def test_scoring_asset_change_is_rejected_before_loading_models_or_decoding_a_lock(tmp_path):
    path = tmp_path / "checkpoint"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="scoring artifact changed"):
        verify_scoring_inputs(
            {
                "schema": "mage-w4-evaluation-inputs/v1",
                "clip_checkpoint": {"path": str(path), "sha256": "a" * 64, "size_bytes": 7},
                "hps_checkpoint": {},
                "ocr_lock": {},
                "ocr_calibration": {},
                "ocr_model_root": str(tmp_path),
                "ocr_python_executable": "/fixture/python",
                "visual_batch_size": 1,
                "generation": {},
            }
        )


def test_launcher_dispatches_only_complete_ticket05_requests_to_evaluation_worker():
    import importlib.util
    from dataclasses import replace

    from shadow_lab.mage_w4_search.contracts import CandidateRequest

    path = Path(__file__).parents[2] / "scripts/mage_w4_launch.py"
    spec = importlib.util.spec_from_file_location("evaluation_launch_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan, manifest, shared = request_inputs()
    request = CandidateRequest.from_dict(
        prepare_evaluation_job("q0", plan, manifest, shared=shared)["request"]
    )
    assert module.worker_command("evaluation", request) == [
        module.PYTHON,
        f"{module.REPO}/scripts/mage_w4_evaluate.py",
    ]
    with pytest.raises(ValueError, match="complete ticket 05"):
        module.worker_command("evaluation", replace(request, evaluation_mode="replay"))


def test_valid_scoring_metadata_preserves_the_ocr_virtualenv_executable_path(tmp_path):
    from shadow_lab.mage_w4_search.native import GenerationConfig
    from shadow_lab.mage_w4_search.ocr import (
        CalibrationPlan,
        CalibrationReport,
        EngineCalibration,
        EngineIdentity,
        HumanTranscription,
        OCRConfig,
        freeze_scorer,
    )

    configs = tuple(
        OCRConfig(EngineIdentity(name, "fixture", (("model", "e" * 64),)))
        for name in ("PaddleOCR", "EasyOCR")
    )
    plan = CalibrationPlan(("a" * 64,), configs, ("b" * 64,), "2026-09-09T00:00:00Z", 0.25)
    # Synthetic typed metadata for this file-boundary unit test; no device evidence is emitted.
    report = CalibrationReport(
        plan,
        (
            HumanTranscription(
                "a" * 64, ("OPEN",), "human_image_transcription", "fixture", "c" * 64, True
            ),
        ),
        tuple(EngineCalibration(config, 0, 0.0, 1.0, True, ()) for config in configs),
        "PaddleOCR",
        "gb10_measured",
        "d" * 64,
        "occupied_device_seconds",
    )
    inputs = {"schema": "mage-w4-evaluation-inputs/v1"}
    for role, value in (
        ("clip_checkpoint", {"fixture": "clip"}),
        ("hps_checkpoint", {"fixture": "hps"}),
        ("ocr_lock", freeze_scorer(report)),
        ("ocr_calibration", asdict(report)),
    ):
        path = tmp_path / role
        path.write_text(json.dumps(value))
        inputs[role] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    target = tmp_path / "base-python"
    target.write_text("fixture interpreter")
    venv_python = tmp_path / "ocr-env-python"
    venv_python.symlink_to(target)
    inputs.update(
        ocr_model_root=str(tmp_path),
        ocr_python_executable=str(venv_python),
        visual_batch_size=1,
        generation=asdict(GenerationConfig(watermark_key=20260909)),
    )
    loaded = verify_scoring_inputs(inputs)
    assert loaded["ocr_python_executable"] == str(venv_python)
    assert loaded["ocr_config"].sha256 == configs[0].sha256


def test_orchestration_proof_changes_request_but_not_frozen_numerical_identity():
    from shadow_lab.mage_w4_search.contracts import CandidateRequest
    from shadow_lab.mage_w4_search.rewards import fixed_generation_identity

    plan, manifest, shared = request_inputs()
    shared["implementation_sha256"] = {"runner": "1" * 64}
    shared["runtime_identity"]["numerical_implementation_sha256"] = {"native": "2" * 64}
    before = CandidateRequest.from_dict(
        prepare_evaluation_job("same-batch", plan, manifest, shared=shared)["request"]
    )
    shared["implementation_sha256"] = {"runner": "3" * 64}
    changed_runner = CandidateRequest.from_dict(
        prepare_evaluation_job("same-batch", plan, manifest, shared=shared)["request"]
    )
    assert before.request_digest != changed_runner.request_digest
    assert fixed_generation_identity(before) == fixed_generation_identity(changed_runner)
    shared["runtime_identity"]["numerical_implementation_sha256"] = {"native": "4" * 64}
    changed_numerics = CandidateRequest.from_dict(
        prepare_evaluation_job("same-batch", plan, manifest, shared=shared)["request"]
    )
    assert fixed_generation_identity(before) != fixed_generation_identity(changed_numerics)


def test_expired_lease_cannot_start_fixture_generation(tmp_path):
    manifest, shared = fixture_job(tmp_path)
    lease = json.loads((tmp_path / "lease.json").read_text())
    lease["hard_deadline"] = lease["started_at"] + 0.5
    lease["reserved_seconds"] = 0.5
    (tmp_path / "lease.json").write_text(json.dumps(lease))
    grant = json.loads((tmp_path / "grant.json").read_text())
    grant["lease"] = lease
    (tmp_path / "grant.json").write_text(json.dumps(grant))
    seen = []
    with pytest.raises(ValueError, match="expired"):
        run_evaluation_job(
            tmp_path,
            manifest,
            device_id="gooo",
            fixture_generation=fixture_generator(shared, seen),
            fixture_scorers=(None, None, None),
        )
    assert seen == []
    assert not (tmp_path / "generation").exists()


def test_fixed_numerical_source_set_excludes_only_orchestration_sources():
    sources = numerical_implementation_hashes()
    for name in ("native.py", "mutation.py", "evaluation.py", "ocr.py", "scorers.py", "rewards.py"):
        assert f"src/shadow_lab/mage_w4_search/{name}" in sources
    assert "src/shadow_lab/nvfp4_runtime.py" in sources
    assert "src/shadow_lab/image_reward_models.py" in sources
    assert "src/shadow_lab/mage_w4_search/evaluation_worker.py" not in sources
    assert "src/shadow_lab/mage_w4_search/development.py" not in sources
    assert "scripts/mage_w4_evaluate.py" not in sources


def test_editable_scorer_sources_freeze_actual_code_without_hidden_environments(tmp_path):
    package = tmp_path / "hpsv2"
    package.mkdir()
    (package / "__init__.py").write_text("# fixture package\n")
    (package / "model.py").write_text("SCORE = 1\n")
    hidden = package / ".venv"
    hidden.mkdir()
    (hidden / "ignore.py").write_text("do not inventory another environment\n")
    before = package_runtime_sources({"hpsv2": package})
    assert set(before["hpsv2"]) == {"__init__.py", "model.py"}
    (package / "model.py").write_text("SCORE = 2\n")
    after = package_runtime_sources({"hpsv2": package})
    assert before != after
    assert before["hpsv2"]["__init__.py"] == after["hpsv2"]["__init__.py"]


def test_scorer_model_configuration_and_vocabulary_are_part_of_runtime_identity(tmp_path):
    package = tmp_path / "hpsv2"
    package.mkdir()
    (package / "__init__.py").write_text("# fixture package\n")
    config = package / "model_configs"
    config.mkdir()
    (config / "ViT-H-14.json").write_text('{"width":1024}')
    vocab = package / "bpe_simple_vocab_16e6.txt.gz"
    vocab.write_bytes(b"fixture bpe")
    (package / "train_data.txt").write_text("unrelated training inputs")
    files = package_runtime_sources({"hpsv2": package})["hpsv2"]
    assert set(files) == {
        "__init__.py",
        "model_configs/ViT-H-14.json",
        "bpe_simple_vocab_16e6.txt.gz",
    }
    vocab.write_bytes(b"changed bpe")
    assert package_runtime_sources({"hpsv2": package})["hpsv2"] != files


def test_real_subprocess_worker_reports_its_group_separately_from_its_pid(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    child = """
import json, sys
from pathlib import Path
from test_mage_w4_evaluation_worker import fixture_job, fixture_generator, FixtureScorer
from shadow_lab.mage_w4_search.evaluation_worker import run_evaluation_job
from shadow_lab.mage_w4_search.ocr import RawSegment
root = Path(sys.argv[1]); manifest, shared = fixture_job(root); stages = []
result = run_evaluation_job(root, manifest, device_id='gooo',
    fixture_generation=fixture_generator(shared, stages),
    fixture_scorers=(FixtureScorer(.3, stages), FixtureScorer(.2, stages),
        lambda image, config: (RawSegment('OPEN', (0., 0., 2., 1.)),)))
print(json.dumps(result))
"""
    wrapper = """
import json, os, subprocess, sys
completed = subprocess.run([sys.executable, '-c', sys.argv[1], sys.argv[2]],
                           text=True, capture_output=True, check=True)
print(json.dumps({'wrapper_pid':os.getpid(), 'result':json.loads(completed.stdout.splitlines()[-1])}))
"""
    repo = Path(__file__).parents[2]
    completed = subprocess.run(
        [sys.executable, "-c", wrapper, child, str(tmp_path)],
        start_new_session=True,
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(repo / "src"), str(repo / "tests/lab"))),
        },
    )
    output = json.loads(completed.stdout)
    result = output["result"]
    assert result["execution_kind"] == "fixture"
    assert result["process_group"] == output["wrapper_pid"]
    assert result["pid"] != result["process_group"]
    started = json.loads((tmp_path / "started.json").read_text())
    assert (started["pid"], started["process_group"]) == (result["pid"], result["process_group"])


@pytest.mark.parametrize("preparation_fails", [False, True])
def test_prepare_cli_preserves_host_timing_without_changing_identity_or_leasing(
    tmp_path, monkeypatch, preparation_fails
):
    import importlib.util
    import sys

    script = Path(__file__).parents[2] / "scripts/mage_w4_evaluate.py"
    spec = importlib.util.spec_from_file_location("mage_w4_evaluate_timing", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    plan, manifest, shared = request_inputs()
    expected = prepare_evaluation_job("prepare-fixture", plan, manifest, shared=shared)
    plan_path, inputs_path = tmp_path / "plan.json", tmp_path / "inputs.json"
    plan_path.write_text(json.dumps(plan))
    inputs_path.write_text("{}")

    def cpu_asset_boundary(run_root, run_id, device_id, selected_plan, inputs):
        assert (run_root, run_id, device_id, selected_plan, inputs) == (
            tmp_path,
            "prepare-fixture",
            "gooo",
            plan,
            {},
        )
        if preparation_fails:
            raise ValueError("fixture asset verification failed")
        return expected

    monkeypatch.setattr(cli, "prepare_on_gb10", cpu_asset_boundary)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--run-root",
            str(tmp_path),
            "--job-id",
            "prepare-fixture",
            "--device-id",
            "gooo",
            "--prepare",
            "--plan",
            str(plan_path),
            "--inputs",
            str(inputs_path),
        ],
    )
    before = time.time()
    if preparation_fails:
        with pytest.raises(ValueError, match="fixture asset verification failed"):
            cli.main()
    else:
        cli.main()
    after = time.time()
    job = tmp_path / "jobs/prepare-fixture"
    timing_path = job / "preparation-timing.json"
    original = timing_path.read_bytes()
    timing = json.loads(original)
    assert timing["schema"] == "mage-w4-evaluation-preparation-timing/v1"
    assert timing["status"] == ("failed" if preparation_fails else "prepared")
    assert before <= timing["started_at"] <= timing["finished_at"] <= after
    assert timing["wall_seconds"] >= 0
    assert timing["host_cpu_seconds"] >= 0
    assert timing["job_id"] == "prepare-fixture" and timing["device_id"] == "gooo"
    assert "no new lease" in timing["accounting"]
    if preparation_fails:
        assert "fixture asset verification failed" in timing["error"]
        assert not (job / "prepared.json").exists()
    else:
        assert json.loads((job / "prepared.json").read_text()) == json.loads(json.dumps(expected))
        assert json.loads((job / "evaluation-job.json").read_text()) == json.loads(
            json.dumps(expected["evaluation_job"])
        )
    with pytest.raises(SystemExit):
        cli.main()
    assert timing_path.read_bytes() == original
    assert not tuple(tmp_path.rglob("*.sqlite3"))
    assert not (job / "lease.json").exists()
    assert not (job / "grant.json").exists()


def test_new_run_root_reuses_symlinked_q0_without_changing_fixed_runtime(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from test_mage_w4_native import _write, torch

    import shadow_lab.mage_w4_search.evaluation_worker as worker
    from shadow_lab.mage_w4_search.native import GenerationConfig
    from shadow_lab.mage_w4_search.ocr import EngineIdentity, OCRConfig

    original, successor = tmp_path / "original", tmp_path / "successor"
    (original / "input").mkdir(parents=True)
    successor.mkdir()
    (successor / "input").symlink_to(original / "input", target_is_directory=True)
    layer = {
        "weight_packed": torch.full((16, 8), 0x21, dtype=torch.uint8),
        "weight_scale": torch.ones((16, 1), dtype=torch.float8_e4m3fn),
        "tensor_scale": torch.tensor(0.25, dtype=torch.float32),
    }
    packed = original / "input/qad-nvfp4-best-673.pt"
    remainder = original / "input/qad-transformer-remainder-best-673.pt"
    allowed = {
        str(packed): _write(
            packed,
            {"schema_version": 1, "packed": {f"linear_{i}": layer for i in range(174)}},
        ),
        str(remainder): _write(remainder, {"bias": torch.zeros(16)}),
    }
    inventory = {"allowed_assets": allowed, "runtime": {"fixture": "1"}}
    plan, manifest, _ = request_inputs()
    for root in (original, successor):
        (root / "inventory.json").write_text(json.dumps(inventory))
        (root / "data-v1").mkdir()
        selected_manifest = manifest
        if root == successor:
            selected_manifest = DataManifest(
                manifest.records
                + (
                    PromptRecord(
                        "new-scale",
                        "A mountain valley",
                        "new-source",
                        "development",
                        ("scale_calibration",),
                    ),
                )
            )
        (root / "data-v1/manifest.json").write_text(json.dumps(asdict(selected_manifest)))
    config = OCRConfig(EngineIdentity("fixture", "1", (("weights", "e" * 64),)))
    inputs = {
        "generation": asdict(GenerationConfig(watermark_key=20260909)),
        "visual_batch_size": 8,
        **{
            role: {"sha256": "e" * 64, "size_bytes": 1}
            for role in ("clip_checkpoint", "hps_checkpoint", "ocr_lock", "ocr_calibration")
        },
    }
    # Replace only installed-host/scorer discovery; actual packed fixture loading/hash stays real.
    monkeypatch.setattr(
        worker, "platform", SimpleNamespace(system=lambda: "Linux", machine=lambda: "aarch64")
    )
    monkeypatch.setattr(worker, "verify_scoring_inputs", lambda _: {"ocr_config": config})
    monkeypatch.setattr(worker, "_installed_external_sources", lambda: {})
    monkeypatch.setattr(worker, "_visual_versions", lambda: {"fixture": "1"})
    monkeypatch.setattr(
        worker.subprocess, "check_output", lambda *a, **kw: "fixture, NVIDIA GB10, fixture"
    )
    first = worker.prepare_on_gb10(original, "original-job", "gooo", plan, inputs)
    second = worker.prepare_on_gb10(successor, "successor-job", "gooo", plan, inputs)
    assert first["request"]["runtime_identity"] == second["request"]["runtime_identity"]
    assert first["request"]["dependencies"] == second["request"]["dependencies"]
    assert first["request"]["q0_digest"] == second["request"]["q0_digest"]
    assert first["request"]["manifest_digest"] != second["request"]["manifest_digest"]
    assert second["evaluation_job"]["shared"]["run_root"] == str(successor)
    assert first["evaluation_job_sha256"] != second["evaluation_job_sha256"]
    assert (original / "inventory.json").read_bytes() == (successor / "inventory.json").read_bytes()
    assert not tuple(tmp_path.rglob("*.sqlite3"))
