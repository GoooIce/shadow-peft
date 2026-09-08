import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from shadow_lab.mage_w4_search.ocr import EngineIdentity, OCRConfig
from shadow_lab.mage_w4_search.scorers import (
    FrozenOCRAdapter,
    OCRProcessError,
    SubprocessOCRAdapter,
    adapter_source_sha256,
    decode_ocr_config,
    ocr_segments,
    verify_ocr_assets,
)


def test_external_ocr_outputs_preserve_text_and_rotation():
    box = [[1, 2], [11, 2], [11, 8], [1, 8]]
    easy = ocr_segments("EasyOCR", [(box, "OPFN", 0.9)])
    paddle = ocr_segments(
        "PaddleOCR", {"rec_polys": [box], "rec_texts": ["OPFN"], "rec_scores": [0.9]}
    )
    assert easy == paddle
    assert easy[0].text == "OPFN"
    assert easy[0].box == (1, 2, 11, 8)
    vertical = [[10, 1], [10, 11], [4, 11], [4, 1]]
    assert ocr_segments("EasyOCR", [(vertical, "OPEN", 0.9)])[0].orientation_degrees == 90
    with pytest.raises(ValueError, match="incomplete"):
        ocr_segments("PaddleOCR", {"rec_polys": [box], "rec_texts": [], "rec_scores": [0.9]})


def test_same_named_ocr_weights_cannot_replace_frozen_asset(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"registered model")
    config = OCRConfig(
        EngineIdentity(
            "EasyOCR", "1.7.2", (("weights.bin", hashlib.sha256(path.read_bytes()).hexdigest()),)
        )
    )
    assert verify_ocr_assets(config, tmp_path) == {"weights.bin": path.resolve()}
    path.write_bytes(b"different model")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_ocr_assets(config, tmp_path)


def subprocess_config(timeout=2):
    return OCRConfig(
        EngineIdentity("EasyOCR", "fixture", (("weights", "a" * 64),)),
        settings_json=json.dumps(
            {
                "runtime_versions": {"easyocr": "fixture"},
                "device": "cpu",
                "model_dirs": {"reader": "easy"},
                "init_options": {"languages": ["en"]},
                "predict_options": {
                    "decoder": "greedy",
                    "batch_size": 1,
                    "workers": 0,
                    "detail": 1,
                    "paragraph": False,
                },
                "adapter_source_sha256": adapter_source_sha256(),
                "worker_timeout_seconds": timeout,
            }
        ),
    )


def test_ocr_configuration_decoder_is_strict_and_preserves_identity():
    config = subprocess_config()
    assert decode_ocr_config(json.loads(json.dumps(asdict(config)))) == config
    with pytest.raises(ValueError, match="schema"):
        decode_ocr_config({**asdict(config), "target_text": "OPEN"})
    raw = asdict(config)
    raw["settings_json"] = json.dumps({"prompt": "OPEN"})
    with pytest.raises(ValueError, match="forbidden"):
        decode_ocr_config(raw)


def fake_ocr_python(tmp_path, mode="ok"):
    """Tiny external process validates IPC only; it is not an OCR accuracy fixture."""
    path = tmp_path / "fixture-ocr-python"
    path.write_text(f"""#!{sys.executable}
import hashlib, json, os, signal, sys, time
request = json.loads(sys.stdin.read())
if {mode!r} == "signal":
    os.kill(os.getpid(), signal.SIGTERM)
if {mode!r} == "timeout":
    time.sleep(5)
assert set(request) == {{"schema", "action", "config", "config_sha256", "image_sha256", "image_base64"}}
assert "prompt" not in request and "target" not in request
fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",",":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
payload = {{"schema":"mage-w4-ocr-response-v1", "action":request["action"], "request_sha256":fingerprint,
"config_sha256":request["config_sha256"], "image_sha256":request["image_sha256"], "status":"ok",
"segments":[{{"text":"fixture recognized pixels", "box":[0.,0.,2.,1.], "confidence":0.9,"orientation_degrees":0.}}],
"worker":{{"pid":os.getpid(), "process_group":os.getpgrp(), "adapter_source_sha256":json.loads(request["config"]["settings_json"])["adapter_source_sha256"]}},
"timing":{{"wall_seconds":0.0,"host_cpu_seconds":0.0}}, "error":None}}
if {mode!r} == "identity": payload["image_sha256"] = "f"*64
print("fixture library initialization log")
line = "MAGE_W4_OCR_RESULT_V1 " + json.dumps(payload)
print(line)
if {mode!r} == "duplicate": print(line)
""")
    path.chmod(0o700)
    return path


def test_ocr_subprocess_is_target_blind_and_inherits_budget_process_group(tmp_path):
    config = subprocess_config()
    adapter = SubprocessOCRAdapter(config, tmp_path, python_executable=fake_ocr_python(tmp_path))
    result = adapter(b"fixed tiny IPC fixture bytes", config)
    assert result[0].text == "fixture recognized pixels"
    assert adapter.last_receipt["worker"]["process_group"] == os.getpgrp()
    assert adapter.last_receipt["worker"]["pid"] != os.getpid()
    assert adapter.last_receipt["parent_wall_seconds"] >= 0
    with pytest.raises(ValueError, match="changed"):
        adapter(
            b"bytes", replace(config, identity=replace(config.identity, package_version="other"))
        )


@pytest.mark.parametrize("mode", ("signal", "timeout", "identity", "duplicate"))
def test_ocr_child_failure_or_invalid_receipt_never_returns_recognition(tmp_path, mode):
    config = subprocess_config(timeout=0.1 if mode == "timeout" else 2)
    adapter = SubprocessOCRAdapter(
        config, tmp_path, python_executable=fake_ocr_python(tmp_path, mode)
    )
    with pytest.raises(OCRProcessError):
        adapter(b"fixed tiny IPC fixture bytes", config)
    assert adapter.last_receipt["status"] == "failed"


def test_config_preparation_freezes_both_engines_before_isolated_load_attempts(
    tmp_path, monkeypatch
):
    script = Path(__file__).parents[2] / "scripts" / "mage_w4_ocr_configs.py"
    spec = importlib.util.spec_from_file_location("mage_w4_ocr_configs_fixture", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_root = tmp_path / "models"
    model_root.mkdir()
    (model_root / "inventory.json").write_text(
        json.dumps(
            {
                "runtime": {"paddleocr": "fixture", "easyocr": "fixture"},
                "files": {"paddle/weights": "a" * 64, "easy/weights": "b" * 64},
            }
        )
    )
    output = tmp_path / "configs"
    calls = []

    class IsolatedBoundary:
        def __init__(self, config, model_root, *, python_executable):
            self.config, self.last_receipt = config, None
            assert all((output / f"{name}.json").exists() for name in ("PaddleOCR", "EasyOCR"))

        def initialize(self):
            calls.append(self.config.identity.engine)
            if self.config.identity.engine == "PaddleOCR":
                self.last_receipt = {"status": "failed", "returncode": -11}
                raise OCRProcessError("fixture C++ signal exit")
            self.last_receipt = {"status": "ok", "timing": {"host_cpu_seconds": 0.0}}
            return self.last_receipt

    monkeypatch.setattr(module, "SubprocessOCRAdapter", IsolatedBoundary)
    records = module.prepare_configs(
        model_root, output, python_executable=Path(sys.executable), timeout_seconds=2
    )
    assert calls == ["PaddleOCR", "EasyOCR"]
    assert [record["status"] for record in records] == ["load_failed", "loaded"]
    assert all(record["images_scored"] == 0 for record in records)
    assert json.loads((output / "configs.json").read_text()) == records


def test_paddle_runtime_is_imported_before_paddleocr_constructor(tmp_path, monkeypatch):
    from shadow_lab.mage_w4_search import scorers

    names = ("detector/weights", "recognizer/weights")
    for name in names:
        path = tmp_path / name
        path.parent.mkdir()
        path.write_bytes(b"fixture frozen weights")
    settings = json.loads(subprocess_config().settings_json)
    settings.update(
        model_dirs={"detector": "detector", "recognizer": "recognizer"},
        runtime_versions={"paddleocr": "fixture"},
        init_options={"cpu_threads": 1, "enable_mkldnn": False},
        predict_options={},
    )
    config = OCRConfig(
        EngineIdentity(
            "PaddleOCR",
            "fixture",
            tuple(
                (name, hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()) for name in names
            ),
        ),
        settings_json=json.dumps(settings),
    )
    imports = []
    original = scorers.importlib.import_module

    def module_boundary(name, *args, **kwargs):
        if name == "paddle":
            imports.append(name)
            return SimpleNamespace()
        return original(name, *args, **kwargs)

    class ConstructorBoundary:
        def __init__(self, **kwargs):
            assert imports == ["paddle"]
            assert kwargs["device"] == "cpu"

    monkeypatch.setattr(scorers.importlib, "import_module", module_boundary)
    monkeypatch.setattr(scorers.importlib.metadata, "version", lambda _: "fixture")
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=ConstructorBoundary))
    adapter = FrozenOCRAdapter(config, tmp_path)
    assert isinstance(adapter.model, ConstructorBoundary)


def test_actual_ocr_worker_cli_rejects_answer_payload_before_loading_models(tmp_path):
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "shadow_lab.mage_w4_search.scorers",
            "--worker",
            "--model-root",
            str(tmp_path),
        ],
        input=json.dumps({"target_text": "OPEN"}),
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert child.returncode == 1
    response = json.loads(
        next(
            line.removeprefix("MAGE_W4_OCR_RESULT_V1 ")
            for line in child.stdout.splitlines()
            if line.startswith("MAGE_W4_OCR_RESULT_V1 ")
        )
    )
    assert response["status"] == "failed"
    assert "invalid OCR request schema" in response["error"]
