import importlib.util
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    path = Path(__file__).resolve().parents[2] / "scripts/serve_mage_text_pilot.py"
    spec = importlib.util.spec_from_file_location("pilot_collector", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_root_binding_rejects_different_data_and_unbound_completed_ballot(tmp_path):
    m = module()
    identity = dict(data_sha256="first", image_directory="image", auxiliary_directory="aux")
    first = m.bind_experiment(tmp_path, identity)
    assert m.bind_experiment(tmp_path, identity) == first
    with pytest.raises(ValueError, match="different"):
        m.bind_experiment(tmp_path, dict(identity, data_sha256="other"))
    other = tmp_path / "old"
    other.mkdir()
    m.atomic_json(other / "verdict.json", dict(status="completed"))
    with pytest.raises(ValueError, match="unbound"):
        m.bind_experiment(other, identity)


@pytest.mark.parametrize("mismatch,phase", [(False, "failed"), (True, "invalid")])
def test_collector_stops_on_dead_training_or_wrong_display_data(
    tmp_path, monkeypatch, mismatch, phase
):
    m = module()
    image = tmp_path / "source"
    image.mkdir()
    report = dict(status="running", data_sha256="data", train=[], baseline=[], validation=[])
    m.atomic_json(image / "report.json", report)
    monkeypatch.setattr(
        m.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=json.dumps(dict(alive=False, report=report))
        ),
    )
    ready = threading.Event()
    m.collect(
        tmp_path,
        image,
        "unused",
        Path("unused"),
        dict(data_sha256="other" if mismatch else "data"),
        "identity",
        ready,
        os.getpid(),
        999999999,
    )
    assert json.loads((tmp_path / "progress.json").read_text())["phase"] == phase
    assert not ready.is_set()


def test_completed_ballot_is_not_reopened_after_artifact_mutation(tmp_path):
    m = module()
    (tmp_path / "image.png").write_bytes(b"changed")
    m.atomic_json(tmp_path / "artifact-sha256.json", {"image.png": "original"})
    m.atomic_json(tmp_path / "verdict.json", dict(status="completed", experiment_sha256="identity"))
    ready = threading.Event()
    m.collect(tmp_path, tmp_path, "unused", tmp_path, {}, "identity", ready, 0, 0)
    assert not ready.is_set()
    assert json.loads((tmp_path / "progress.json").read_text())["phase"] == "invalid"
