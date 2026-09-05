"""Real subprocess tests for experiment supervision (no model execution)."""

import importlib.util
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
