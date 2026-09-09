#!/usr/bin/env python3
"""Coordinator-side authenticated GB10 launch/recovery with one pilot ledger."""

import argparse
import json
import re
import shlex
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

from shadow_lab.mage_w4_search.budget import BudgetLedger
from shadow_lab.mage_w4_search.contracts import preflight
from shadow_lab.mage_w4_search.execution import read_manifest, read_request

REMOTE_ROOT = "/home/goooice/runs/mage-flow/direct-w4a4-20260909"
REPO = "/home/goooice/Code/shadow-peft"
PYTHON = "/home/goooice/Code/Mage/mage_flow/.venv/bin/python"
PILOT_LEDGER = (
    Path(__file__).resolve().parents[1]
    / "runs-local/mage-direct-w4a4-20260909/pilot-budget.sqlite3"
)


def open_pilot_ledger():
    """An output directory is never authority to initialize a new allowance."""
    if not PILOT_LEDGER.is_file() or PILOT_LEDGER.stat().st_size == 0:
        raise ValueError(
            "existing ticket 01 pilot ledger is required; launch cannot initialize a budget"
        )
    return BudgetLedger(PILOT_LEDGER)


def remote(host, command, *, check=True):
    return subprocess.run(
        [
            "ssh",
            "-n",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=60",
            "-o",
            "ControlPath=/tmp/mage-w4-ssh-%C",
            host,
            command,
        ],
        text=True,
        capture_output=True,
        check=check,
        timeout=45,
    )


def worker_command(kind, request):
    """Choose a worker before reserving; complete scoring has an explicit entry."""
    if kind == "evaluation":
        if (
            request.ticket_id != "05"
            or request.evaluation_mode != "complete"
            or request.purpose not in {"scale_calibration", "candidate_development"}
            or not request.request_id.startswith("evaluation-")
        ):
            raise ValueError("evaluation worker requires a complete ticket 05 frozen request")
        return [PYTHON, f"{REPO}/scripts/mage_w4_evaluate.py"]
    if kind == "native":
        return [PYTHON, f"{REPO}/scripts/mage_w4_native_probe.py"]
    if kind == "ocr":
        return [PYTHON, "-m", "shadow_lab.mage_w4_search.ocr_worker"]
    raise ValueError("unknown worker kind")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("gooo", "goooinext"), required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--worker-kind", choices=("native", "ocr", "evaluation"), default="native")
    parser.add_argument(
        "--evidence-root", type=Path, default=Path("runs-local/mage-direct-w4a4-20260909")
    )
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]+", args.job_id):
        parser.error("job-id must contain lowercase letters, numbers and hyphens")
    local = args.evidence_root / args.host / "jobs" / args.job_id
    local.mkdir(parents=True, exist_ok=True)
    remote_job = f"{REMOTE_ROOT}/jobs/{args.job_id}"
    ledger = open_pilot_ledger()
    manifest = read_manifest(json.loads((args.evidence_root / "data-v1/manifest.json").read_text()))
    if not args.recover:
        if (local / "lease.json").exists():
            parser.error("lease already exists; recover instead of relaunching")
        value = json.loads(
            remote(args.host, shlex.join(["cat", f"{remote_job}/prepared.json"])).stdout
        )
        (local / "prepared.json").write_text(json.dumps(value, indent=2) + "\n")
        request = read_request(value["request"])
        worker = worker_command(args.worker_kind, request)
        if args.worker_kind == "ocr" and (
            request.ticket_id != "04"
            or request.purpose != "ocr_selection"
            or not request.runtime_identity.get("ocr_comparison_job_sha256")
        ):
            raise ValueError("OCR worker needs a frozen ticket 04 comparison job")
        check = preflight(request, manifest, expected_center_version=0)
        if check.status != "ready_for_reservation" or request.device_id != args.host:
            raise ValueError(check.to_dict())
        before = time.time()
        worker_time = float(
            remote(args.host, shlex.join([PYTHON, "-c", "import time; print(time.time())"])).stdout
        )
        after = time.time()
        offset = worker_time - (before + after) / 2
        uncertainty = (after - before) / 2
        if abs(offset) + uncertainty > 2:
            raise ValueError("clock alignment exceeds the predeclared two-second bound")
        clock = {
            "offset_seconds": offset,
            "uncertainty_seconds": uncertainty,
            "cleanup_reserve_seconds": 30,
            "at": after,
        }
        (local / "clock.json").write_text(json.dumps(clock, indent=2) + "\n")
        lease = ledger.reserve(
            args.job_id + "-occupancy",
            request.ticket_id,
            args.host,
            args.seconds,
            run_id=request.run_id,
        )
        (local / "lease.json").write_text(json.dumps(lease.to_dict(), indent=2) + "\n")
        request = replace(request, lease_id=lease.lease_id)
        grant = preflight(request, manifest, expected_center_version=0, ledger=ledger)
        if grant.status != "authorized":
            # No worker command has been sent; an authenticated launch was never attempted.
            ledger.release(
                lease.lease_id, reason="pre-launch-authorization-rejected-no-process-created"
            )
            raise ValueError(grant.to_dict())
        payload = {**grant.to_dict(), "request": asdict(request)}
        (local / "grant.json").write_text(json.dumps(payload, indent=2) + "\n")
        subprocess.run(
            [
                "rsync",
                "-a",
                str(local / "lease.json"),
                str(local / "grant.json"),
                f"{args.host}:{remote_job}/",
            ],
            check=True,
        )
        command = [
            "env",
            f"PYTHONPATH={REPO}/src",
            "PYTHONUNBUFFERED=1",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "OMP_NUM_THREADS=8",
            PYTHON,
            "-m",
            "shadow_lab.mage_w4_search.budget",
            "--lease",
            f"{remote_job}/lease.json",
            "--result",
            f"{remote_job}/stopped.json",
            "--cleanup-seconds",
            "30",
            "--",
            "strace",
            "-f",
            "-e",
            "trace=openat,openat2",
            "-o",
            f"{remote_job}/file-access.trace",
            *worker,
            "--run-root",
            REMOTE_ROOT,
            "--job-id",
            args.job_id,
            "--device-id",
            args.host,
        ]
        launch = f"nohup setsid {shlex.join(command)} > {shlex.quote(remote_job + '/worker.log')} 2>&1 < /dev/null &"
        # After this send, any error is uncertain. Leave the lease occupied for recovery.
        remote(args.host, launch)
        print(json.dumps({"launched": args.job_id, "lease": lease.to_dict()}), flush=True)
    lease_value = json.loads((local / "lease.json").read_text())
    while True:
        response = remote(args.host, shlex.join(["cat", f"{remote_job}/stopped.json"]), check=False)
        if response.returncode == 0:
            stopped = json.loads(response.stdout)
            if stopped.get("lease_id") != lease_value["lease_id"] or not stopped.get(
                "stop_confirmed"
            ):
                raise RuntimeError("worker group stop unconfirmed; lease remains occupied")
            (local / "stopped.json").write_text(json.dumps(stopped, indent=2) + "\n")
            snapshot = ledger.release(
                stopped["lease_id"], reason="authenticated-owned-process-group-stop-receipt"
            )
            (args.evidence_root / "budget-snapshot.json").write_text(
                json.dumps(snapshot, indent=2) + "\n"
            )
            transfer_options = ["--exclude", "*.pt"]
            # Complete batches keep large per-image profiles on their GB10. The
            # worker has verified every trace; reports retain original paths/SHA.
            recorded_preparation = json.loads((local / "prepared.json").read_text())
            if "evaluation_job" in recorded_preparation:
                transfer_options += ["--exclude", "*.trace.json"]
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    *transfer_options,
                    f"{args.host}:{remote_job}/",
                    str(local) + "/",
                ],
                check=True,
            )
            print(json.dumps({"stopped": stopped, "budget": snapshot}), flush=True)
            return
        if time.time() > lease_value["hard_deadline"]:
            raise RuntimeError(
                "no stop acknowledgement by deadline; recover lease and inspect owned process group"
            )
        time.sleep(2)


if __name__ == "__main__":
    main()
