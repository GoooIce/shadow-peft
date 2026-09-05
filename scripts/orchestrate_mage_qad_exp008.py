#!/usr/bin/env python3
"""Monitor EXP-008 screening and run its gated QAD chain through the human-dev stop."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

from shadow_lab.mage_qad_campaign import atomic_json
from shadow_lab.mage_qad_data import prompt_identity


def read_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


class ExperimentFailed(RuntimeError):
    """A measured stage failure, distinct from recoverable monitor transport errors."""


def process_alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.split(") ", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False


def retry_transport(command, **kwargs):
    for attempt in range(3):
        try:
            return subprocess.run(command, check=True, timeout=30, **kwargs)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def worker(path: Path):
    record = read_json(path)
    process = subprocess.Popen(record["command"], cwd=record["repository"])
    atomic_json(path.with_suffix(".child.json"), {"pid": process.pid})
    result = process.wait()
    atomic_json(
        path.with_suffix(".result.json"), {"returncode": result, "finished_at": time.time()}
    )


def monitored_command(command, phase, root, repository, python, publish, poll_seconds, campaign):
    path = root / f"{phase}.worker.json"
    record = {"command": command, "repository": repository}
    if path.exists():
        if read_json(path) != record:
            raise ExperimentFailed(f"{phase} command identity changed")
    else:
        atomic_json(path, record)
        with (root / f"{phase}.log").open("x") as log:
            process = subprocess.Popen(
                [python, "-u", str(Path(__file__).resolve()), "--worker", str(path)],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        atomic_json(path.with_suffix(".process.json"), {"pid": process.pid})
    process_record = read_json(path.with_suffix(".process.json"))
    if not process_record:
        raise ExperimentFailed(
            f"{phase} launch interrupted before PID commit; inspect before retry"
        )
    pid = process_record["pid"]
    publish(phase=phase, child_pid=pid)
    while True:
        result = read_json(path.with_suffix(".result.json"))
        if result is not None:
            if result["returncode"]:
                raise ExperimentFailed(
                    f"{phase} exited {result['returncode']}; see {root / f'{phase}.log'}"
                )
            return
        if not process_alive(pid):
            # The worker may have committed its result between our first read
            # and the liveness check. Consume it on the next iteration.
            if read_json(path.with_suffix(".result.json")) is not None:
                continue
            child = read_json(path.with_suffix(".child.json"))
            if child and process_alive(child["pid"]):
                os.kill(child["pid"], 15)
            raise ExperimentFailed(f"{phase} worker exited without a terminal report")
        report = read_json(Path(campaign) / "campaign.json")
        summary = (
            None
            if report is None
            else {k: report.get(k) for k in ("status", "stage", "progress", "error")}
        )
        publish(training=summary, live=read_json(Path(campaign) / "live.json"))
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", type=Path)
    mode.add_argument("--worker", type=Path)
    parser.add_argument(
        "--observe-only",
        action="store_true",
        help="Monitor preparation without authorizing any downstream stage",
    )
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    plan = json.loads(args.plan.read_text())
    root = Path(plan["monitor_directory"])
    root.mkdir(parents=True, exist_ok=True)
    # Advisory lock is released on process death. A replacement coordinator can
    # adopt durable stage workers; it never needs to delete a one-shot gate claim.
    lock = (root / "coordinator.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = read_json(root / "status.json")
    if state is not None:
        if state["plan"] != plan:
            raise ValueError("monitor plan identity changed")
        if state["status"] in {"failed", "awaiting_human"}:
            raise ExperimentFailed("coordinator is terminal; do not replay failed stages")
        state["status"] = "running"
    else:
        state = {
            "started_at": time.time(),
            "phase": "screening",
            "status": "running",
            "human_approval_required": True,
            "plan": plan,
        }

    def publish(**updates):
        state.update(updates, updated_at=time.time())
        atomic_json(root / "status.json", state)
        print(json.dumps({k: v for k, v in state.items() if k != "plan"}), flush=True)

    def child_run(command, phase):
        monitored_command(
            command,
            phase,
            root,
            plan["repository"],
            plan["python"],
            publish,
            plan.get("poll_seconds", 30),
            plan["campaign"],
        )

    try:
        while True:
            summaries = []
            for shard in plan["screening"]:
                code = """import json, os
from pathlib import Path
p=Path(INPUT_PATH)
a=p/'accepted.jsonl'
b=p/'audit.jsonl'
result={'accepted': len(a.read_text().splitlines()) if a.exists() else 0,
'screened':len(b.read_text().splitlines()) if b.exists() else 0,
'finished':(p/'run.json').exists()}
try: os.kill(INPUT_PID, 0); result['alive']=True
except ProcessLookupError: result['alive']=False
if result['finished']: result['report']=json.loads((p/'run.json').read_text())
print(json.dumps(result))
""".replace("INPUT_PATH", repr(shard["directory"])).replace("INPUT_PID", str(shard["pid"]))
                if shard["host"] == "local":
                    result = subprocess.run(
                        ["python3", "-c", code], capture_output=True, text=True, check=True
                    )
                else:
                    result = retry_transport(
                        ["ssh", "-o", "ConnectTimeout=10", shard["host"], "python3", "-"],
                        input=code,
                        capture_output=True,
                        text=True,
                    )
                summary = json.loads(result.stdout)
                summary["host"] = shard["host"]
                summaries.append(summary)
                if not summary["finished"] and not summary["alive"]:
                    raise ExperimentFailed(
                        f"screening process exited without completion: {shard['host']}"
                    )
            probe = read_json(Path(plan["probe_report"]))
            publish(screening=summaries, probe=probe)
            if args.observe_only:
                publish(
                    status="observing",
                    phase="data_ready" if all(s["finished"] for s in summaries) else "screening",
                    downstream_launch_disabled=True,
                )
                time.sleep(plan.get("poll_seconds", 30))
                continue
            if (not probe or probe.get("status") not in {"passed", "failed"}) and not process_alive(
                plan["probe_pid"]
            ):
                probe = read_json(Path(plan["probe_report"]))
                if not probe or probe.get("status") not in {"passed", "failed"}:
                    raise ExperimentFailed("interface probe exited without a terminal report")
            if probe and probe["status"] == "failed":
                raise ExperimentFailed(
                    "four-position interface probe failed; QAD launch prohibited"
                )
            if all(s["finished"] for s in summaries) and probe and probe["status"] == "passed":
                break
            time.sleep(plan.get("poll_seconds", 30))
        publish(phase="freeze_data")
        combined = []
        for index, shard in enumerate(plan["screening"]):
            if shard["host"] == "local":
                source = Path(shard["directory"]) / "accepted.jsonl"
            else:
                source = root / f"accepted-{index}.jsonl"
                subprocess.run(
                    [
                        "rsync",
                        "-a",
                        f"{shard['host']}:{shard['directory']}/accepted.jsonl",
                        str(source),
                    ],
                    check=True,
                )
            combined.extend(
                json.loads(line) for line in source.read_text().splitlines() if line.strip()
            )
        probe_sample = json.loads(Path(plan["probe_sample"]).read_text())
        combined = [
            r
            for r in combined
            if prompt_identity(r["prompt"]) != prompt_identity(probe_sample["prompt"])
        ]
        sources = root / "accepted-sources.jsonl"
        sources.write_text("".join(json.dumps(row) + "\n" for row in combined))
        cli = [plan["python"], "-u", "scripts/run_mage_flow_healing.py"]
        child_run(
            [
                *cli,
                "build-data",
                "--sources",
                str(sources),
                "--eval-pool",
                plan["evaluation_pool"],
                "--out",
                plan["qad_data"],
            ],
            "build_data",
        )
        child_run(
            [
                *cli,
                "train",
                "--model-path",
                plan["model"],
                "--data",
                plan["qad_data"],
                "--config",
                plan["config"],
                "--campaign",
                plan["campaign"],
            ],
            "qad",
        )
        common = [
            "--campaign",
            plan["campaign"],
            "--model-path",
            plan["model"],
            "--eval-pool",
            plan["evaluation_pool"],
            "--calibration",
            plan["calibration"],
            "--hps-checkpoint",
            plan["hps"],
            "--clip-checkpoint",
            plan["clip"],
        ]
        child_run([*cli, "checkpoint-gate", *common], "checkpoint_gate")
        child_run([*cli, "human-dev", *common], "human_dev")
        publish(
            status="awaiting_human",
            phase="human_dev",
            blind_directory=str(Path(plan["campaign"]) / "healing/human_dev/blind/public"),
        )
    except Exception as error:
        publish(
            status="failed" if isinstance(error, ExperimentFailed) else "monitor_error",
            error=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    main()
