"""Real subprocess tests for experiment supervision (no model execution)."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_monitor():
    path = Path(__file__).resolve().parents[2] / "scripts/orchestrate_mage_qad_exp008.py"
    spec = importlib.util.spec_from_file_location("qad_monitor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_monitor_restart_adopts_worker_without_repeating_the_stage(tmp_path):
    monitor = load_monitor()
    counter = tmp_path / "counter.txt"
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(counter)!r}).open('a').write('executed\\n')",
    ]

    def interrupted(**updates):
        raise OSError("monitor transport interrupted")

    with pytest.raises(OSError):
        monitor.monitored_command(
            command,
            "smoke",
            tmp_path,
            str(tmp_path),
            sys.executable,
            interrupted,
            0.05,
            tmp_path / "campaign",
        )
    for _ in range(2):
        monitor.monitored_command(
            command,
            "smoke",
            tmp_path,
            str(tmp_path),
            sys.executable,
            lambda **updates: None,
            0.05,
            tmp_path / "campaign",
        )
    assert counter.read_text().splitlines() == ["executed"]
    with pytest.raises(monitor.ExperimentFailed, match="identity"):
        monitor.monitored_command(
            [sys.executable, "-c", "pass"],
            "smoke",
            tmp_path,
            str(tmp_path),
            sys.executable,
            lambda **updates: None,
            0.05,
            tmp_path / "campaign",
        )


def test_monitor_stops_on_failed_child_and_retains_failure(tmp_path):
    monitor = load_monitor()
    command = [sys.executable, "-c", "raise SystemExit(7)"]
    for _ in range(2):
        with pytest.raises(monitor.ExperimentFailed, match="exited 7"):
            monitor.monitored_command(
                command,
                "smoke",
                tmp_path,
                str(tmp_path),
                sys.executable,
                lambda **updates: None,
                0.05,
                tmp_path / "campaign",
            )


@pytest.mark.parametrize("returncode", [0, 7])
def test_worker_result_committed_during_liveness_check(tmp_path, monkeypatch, returncode):
    monitor = load_monitor()
    command = [sys.executable, "-c", "pass"]
    path = tmp_path / "smoke.worker.json"
    monitor.atomic_json(path, {"command": command, "repository": str(tmp_path)})
    monitor.atomic_json(path.with_suffix(".process.json"), {"pid": 123})

    def finished_during_check(pid):
        monitor.atomic_json(path.with_suffix(".result.json"), {"returncode": returncode})
        return False

    monkeypatch.setattr(monitor, "process_alive", finished_during_check)

    def run():
        monitor.monitored_command(
            command,
            "smoke",
            tmp_path,
            str(tmp_path),
            sys.executable,
            lambda **updates: None,
            0.05,
            tmp_path / "campaign",
        )

    if returncode:
        with pytest.raises(monitor.ExperimentFailed, match="exited 7"):
            run()
    else:
        run()


def test_prepared_campaign_preserves_frozen_data_and_excludes_protocol_prompts(tmp_path):
    from shadow_lab.mage_qad_data import build_qad_pool

    m = load_monitor()
    sources = [
        dict(
            prompt=f"formal {i}",
            source="test",
            source_revision="v1",
            source_id=str(i),
            stratum="test",
        )
        for i in range(1152)
    ]
    evaluation = [dict(prompt=f"eval {i}", sample_id=f"e{i}", seed=i) for i in range(500)]
    rows, manifest = build_qad_pool(sources, evaluation)
    (tmp_path / "qad.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    m.atomic_json(tmp_path / "manifest.json", manifest)
    pilot_rows = [dict(prompt=f"pilot {i}") for i in range(16)]
    m.atomic_json(tmp_path / "pilot.json", dict(train=pilot_rows[:8], validation=pilot_rows[8:]))
    proof = dict(
        status="completed",
        arm="auxiliary",
        trainable_layers=174,
        train=[{}] * 8,
        validation=[{}] * 8,
        data_sha256=hashlib.sha256((tmp_path / "pilot.json").read_bytes()).hexdigest(),
    )
    m.atomic_json(tmp_path / "proof.json", proof)
    m.atomic_json(tmp_path / "config.json", dict(text_initial_fraction=0.1, adamw_foreach=False))
    m.atomic_json(tmp_path / "probe.json", dict(prompt="independent probe"))
    exclusion = tmp_path / "exclude.jsonl"
    exclusion.write_text("".join(json.dumps(r) + "\n" for r in pilot_rows))
    plan = dict(
        qad_data=str(tmp_path),
        config=str(tmp_path / "config.json"),
        probe_sample=str(tmp_path / "probe.json"),
        protocol_development_exclusions=[str(exclusion)],
        prepared=dict(
            approved_protocol="image_and_text_174",
            user_instruction="go",
            data_sha256=manifest["data_sha256"],
            pilot_data=str(tmp_path / "pilot.json"),
            auxiliary_pilot_report=str(tmp_path / "proof.json"),
            auxiliary_pilot_report_sha256=hashlib.sha256(
                (tmp_path / "proof.json").read_bytes()
            ).hexdigest(),
        ),
    )
    assert m.validate_prepared_campaign(plan)["excluded_prompts"] == 17
    with exclusion.open("a") as stream:
        stream.write(json.dumps(dict(prompt=rows[0]["prompt"])) + "\n")
    with pytest.raises(m.ExperimentFailed, match="leaked"):
        m.validate_prepared_campaign(plan)
    exclusion.write_text("")
    with pytest.raises(m.ExperimentFailed, match="missing"):
        m.validate_prepared_campaign(plan)
