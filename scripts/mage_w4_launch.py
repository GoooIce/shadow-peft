#!/usr/bin/env python3
"""Coordinator-side authenticated GB10 launch/recovery with one pilot ledger."""

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import time
from contextlib import ExitStack
from dataclasses import asdict, fields, replace
from pathlib import Path, PurePosixPath

from shadow_lab.mage_w4_search.budget import BudgetLedger, DeviceLease
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


def bind_launch_target(local, *, host, job_id, remote_root, recover):
    """Keep recovery on the original host/directory before any transport or budget action."""
    target = {
        "schema": "mage-w4-launch-target/v1",
        "host": host,
        "job_id": job_id,
        "remote_root": remote_root,
    }
    path = local / "launch-target.json"
    if path.exists():
        if json.loads(path.read_text()) != target:
            raise ValueError("launch target differs from the original recorded host/root/job")
        return
    if recover and remote_root != REMOTE_ROOT:
        prepared_path = local / "prepared.json"
        prepared = json.loads(prepared_path.read_text()) if prepared_path.exists() else {}
        recorded_root = prepared.get("evaluation_job", {}).get("shared", {}).get("run_root")
        if recorded_root != remote_root:
            raise ValueError("launch target missing: nonlegacy recovery needs frozen preparation")
    with path.open("x") as stream:
        json.dump(target, stream, sort_keys=True)


def verify_prepared_target(value, *, remote_root, job_id):
    if value["request"]["run_id"] != job_id:
        raise ValueError("launch target job differs from its frozen preparation")
    if "evaluation_job" in value and value["evaluation_job"]["shared"]["run_root"] != remote_root:
        raise ValueError("launch target root differs from its frozen preparation")


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
    replay = request.request_id.startswith("evaluation-replay-")
    if replay != (kind == "development-replay"):
        raise ValueError("development replay requires its explicit worker entry")
    if kind == "development-replay":
        if (
            request.ticket_id != "06"
            or request.evaluation_mode != "complete"
            or request.purpose != "candidate_development"
            or request.effective_edits <= 0
        ):
            raise ValueError("development replay requires a complete ticket06 candidate")
        return [PYTHON, f"{REPO}/scripts/mage_w4_replay.py", "run"]
    if kind == "evaluation":
        if (
            request.ticket_id not in {"05", "06", "08"}
            or request.evaluation_mode != "complete"
            or request.purpose not in {"scale_calibration", "candidate_development"}
            or not request.request_id.startswith("evaluation-")
        ):
            raise ValueError("evaluation worker requires a complete ticket 05/06/08 frozen request")
        if request.ticket_id != "05" and request.purpose != "candidate_development":
            raise ValueError("ticket 06/08 cannot recalibrate the fixed q0 scale")
        return [PYTHON, f"{REPO}/scripts/mage_w4_evaluate.py"]
    if kind == "native":
        return [PYTHON, f"{REPO}/scripts/mage_w4_native_probe.py"]
    if kind == "ocr":
        return [PYTHON, "-m", "shadow_lab.mage_w4_search.ocr_worker"]
    raise ValueError("unknown worker kind")


def replay_launch_metadata(
    prepared,
    manifest,
    *,
    receipt_text=None,
    snapshot=None,
    registration=None,
    execution_kind="gb10_measured",
):
    """Authenticate replay metadata before occupancy; never interpret scores on this host."""
    from shadow_lab.mage_w4_search.development_replay import (
        authorize_replay_control,
        replay_registration_binding,
    )

    request = read_request(prepared["request"])
    if request.request_id.startswith("evaluation-replay-"):
        if registration is not None or receipt_text is None or snapshot is None:
            raise ValueError(
                "replay launch needs its source verification and original ledger snapshot"
            )
        binding = authorize_replay_control(
            prepared, manifest, json.loads(receipt_text), snapshot, execution_kind=execution_kind
        )
        binding["source_verification_file_sha256"] = hashlib.sha256(
            receipt_text.encode()
        ).hexdigest()
        return {"development_replay": binding}
    if receipt_text is not None:
        raise ValueError("ordinary launch cannot consume a replay source verification")
    if registration is None:
        return {}
    return {
        "development_replay_registration": replay_registration_binding(
            registration, manifest, request
        )
    }


def authorize_existing_claim(ledger, request, manifest):
    """Turn one already claimed occupancy into a worker grant; never reserve again."""
    if request.lease_id is None:
        raise ValueError("an existing coordinator claim is required")
    lease = ledger.assert_active(request.lease_id)
    grant = preflight(
        request, manifest, expected_center_version=request.center_version, ledger=ledger
    )
    if grant.status != "authorized":
        raise ValueError(grant.to_dict())
    return lease, {**grant.to_dict(), "request": asdict(request)}


def coordinator_command(host):
    if host not in {"gooo", "goooinext"}:
        raise ValueError("only the existing LAN GB10 SSH identities are allowed")
    return [
        "ssh",
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
        shlex.join(
            ["env", f"PYTHONPATH={REPO}/src", PYTHON, "-m", "shadow_lab.mage_w4_search.budget_rpc"]
        ),
    ]


def find_recorded_lease(ledger, request):
    """Recover original, possibly expired, occupancy after a lost claim reply; never allocate."""
    matches = [
        row
        for row in ledger.snapshot()["leases"]
        if (row["run_id"], row["ticket_id"], row["device_id"])
        == (request.run_id, request.ticket_id, request.device_id)
        and row["request_id"] in {request.request_id, request.run_id + "-occupancy"}
    ]
    if len(matches) > 1:
        raise ValueError("ambiguous original occupancy; keep all leases for investigation")
    return (
        DeviceLease(**{field.name: matches[0][field.name] for field in fields(DeviceLease)})
        if matches
        else None
    )


def acquire_launch_lease(
    request,
    *,
    host,
    job_id,
    seconds,
    ledger,
    clock_path,
    coordinator_config=None,
    coordinator_host="gooo",
    round_id=None,
):
    """Prepare GB10 state first, align clocks, then claim exactly one Mac-owned occupancy."""
    with ExitStack() as cleanup:
        coordinator = None
        if coordinator_config is not None:
            from shadow_lab.mage_w4_search.budget_rpc import RemoteCoordinator

            coordinator = cleanup.enter_context(
                RemoteCoordinator(coordinator_command(coordinator_host), coordinator_config, ledger)
            )
        before = time.time()
        worker_time = float(
            remote(host, shlex.join([PYTHON, "-c", "import time; print(time.time())"])).stdout
        )
        after = time.time()
        offset, uncertainty = worker_time - (before + after) / 2, (after - before) / 2
        if abs(offset) + uncertainty > 2:
            raise ValueError("clock alignment exceeds the predeclared two-second bound")
        clock = {
            "offset_seconds": offset,
            "uncertainty_seconds": uncertainty,
            "cleanup_reserve_seconds": 30,
            "at": after,
        }
        Path(clock_path).write_text(json.dumps(clock, indent=2) + "\n")
        if coordinator is None:
            return ledger.reserve(
                job_id + "-occupancy", request.ticket_id, host, seconds, run_id=request.run_id
            )
        try:
            claimed_value = coordinator.call(
                "claim",
                dict(
                    round_id=round_id,
                    candidate_id=request.candidate_id,
                    device_id=host,
                    reserve_seconds=seconds,
                    expected_request_digest=request.request_digest,
                    require_unclaimed=True,
                ),
            )
        finally:
            Path(clock_path).with_name("coordinator-resources.json").write_text(
                json.dumps(
                    {
                        "initialization": coordinator.initialization_timing,
                        "operation": coordinator.last_operation_timing,
                        "remote_artifact_path": coordinator.timing_artifact_path,
                        "budget_authority": coordinator.authority,
                    },
                    sort_keys=True,
                    allow_nan=False,
                )
            )
        claimed = read_request(claimed_value)
        if claimed.request_digest != request.request_digest:
            raise ValueError(
                "coordinator returned another request; occupied lease requires recovery"
            )
        return ledger.assert_active(claimed.lease_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("gooo", "goooinext"), required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument(
        "--worker-kind",
        choices=("native", "ocr", "evaluation", "development-replay"),
        default="native",
    )
    parser.add_argument(
        "--replay-registration",
        help="remote frozen replay prepared.json to bind before source launch",
    )
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    parser.add_argument(
        "--evidence-root", type=Path, default=Path("runs-local/mage-direct-w4a4-20260909")
    )
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--coordinator-config", type=Path)
    parser.add_argument("--coordinator-host", choices=("gooo", "goooinext"), default="gooo")
    parser.add_argument("--round-id")
    args = parser.parse_args()
    if args.replay_registration is not None and (
        args.worker_kind != "evaluation"
        or args.recover
        or not PurePosixPath(args.replay_registration).is_absolute()
        or ".." in PurePosixPath(args.replay_registration).parts
    ):
        parser.error(
            "replay-registration is an absolute frozen source-launch input, never recovery"
        )
    remote_root = PurePosixPath(args.remote_root)
    if (
        not remote_root.is_absolute()
        or ".." in remote_root.parts
        or str(remote_root) != args.remote_root
        or any(character in args.remote_root for character in "\x00\n\r")
    ):
        parser.error("remote-root must be an explicit absolute canonical POSIX path")
    if not re.fullmatch(r"[a-z0-9-]+", args.job_id):
        parser.error("job-id must contain lowercase letters, numbers and hyphens")
    if (args.coordinator_config is None) != (args.round_id is None):
        parser.error("coordinator-config and round-id must be supplied together")
    local = args.evidence_root / args.host / "jobs" / args.job_id
    local.mkdir(parents=True, exist_ok=True)
    bind_launch_target(
        local,
        host=args.host,
        job_id=args.job_id,
        remote_root=args.remote_root,
        recover=args.recover,
    )
    remote_job = f"{args.remote_root}/jobs/{args.job_id}"
    ledger = open_pilot_ledger()
    coordinator_config = (
        json.loads(args.coordinator_config.read_text()) if args.coordinator_config else None
    )
    manifest = read_manifest(json.loads((args.evidence_root / "data-v1/manifest.json").read_text()))
    if not args.recover:
        if (local / "lease.json").exists():
            parser.error("lease already exists; recover instead of relaunching")
        value = json.loads(
            remote(args.host, shlex.join(["cat", f"{remote_job}/prepared.json"])).stdout
        )
        verify_prepared_target(value, remote_root=args.remote_root, job_id=args.job_id)
        (local / "prepared.json").write_text(json.dumps(value, indent=2) + "\n")
        request = read_request(value["request"])
        worker = worker_command(args.worker_kind, request)
        replay = args.worker_kind == "development-replay"
        if replay and coordinator_config is not None:
            raise ValueError("development replay cannot claim optimizer assignment")
        registration = (
            json.loads(remote(args.host, shlex.join(["cat", args.replay_registration])).stdout)
            if args.replay_registration
            else None
        )
        receipt_text = (
            remote(
                args.host, shlex.join(["cat", f"{remote_job}/replay-source-verification.json"])
            ).stdout
            if replay
            else None
        )
        replay_metadata = replay_launch_metadata(
            value,
            manifest,
            registration=registration,
            receipt_text=receipt_text,
            snapshot=ledger.snapshot(),
        )
        if receipt_text is not None:
            (local / "replay-source-verification.json").write_text(receipt_text)
        if registration is not None:
            (local / "replay-registration.json").write_text(json.dumps(registration))
        if args.worker_kind == "ocr" and (
            request.ticket_id != "04"
            or request.purpose != "ocr_selection"
            or not request.runtime_identity.get("ocr_comparison_job_sha256")
        ):
            raise ValueError("OCR worker needs a frozen ticket 04 comparison job")
        if coordinator_config is not None and (
            request.ticket_id not in {"06", "08"} or request.effective_edits <= 0
        ):
            raise ValueError("coordinator claims only registered ticket 06/08 candidate work")
        if (
            request.ticket_id == "06"
            and request.effective_edits > 0
            and coordinator_config is None
            and not replay
        ):
            raise ValueError("ticket 06 candidate requires its registered coordinator round")
        check = preflight(request, manifest, expected_center_version=request.center_version)
        if check.status != "ready_for_reservation" or request.device_id != args.host:
            raise ValueError(check.to_dict())
        if find_recorded_lease(ledger, request) is not None:
            raise ValueError("original request already has occupancy; recover without relaunching")
        lease = acquire_launch_lease(
            request,
            host=args.host,
            job_id=args.job_id,
            seconds=args.seconds,
            ledger=ledger,
            clock_path=local / "clock.json",
            coordinator_config=coordinator_config,
            coordinator_host=args.coordinator_host,
            round_id=args.round_id,
        )
        (local / "lease.json").write_text(json.dumps(lease.to_dict(), indent=2) + "\n")
        request = replace(request, lease_id=lease.lease_id)
        try:
            lease, payload = authorize_existing_claim(ledger, request, manifest)
            payload.update(replay_metadata)
        except ValueError:
            # No worker command has been sent; an authenticated launch was never attempted.
            ledger.release(
                lease.lease_id, reason="pre-launch-authorization-rejected-no-process-created"
            )
            raise
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
            args.remote_root,
            "--job-id",
            args.job_id,
            "--device-id",
            args.host,
        ]
        launch = f"nohup setsid {shlex.join(command)} > {shlex.quote(remote_job + '/worker.log')} 2>&1 < /dev/null &"
        # After this send, any error is uncertain. Leave the lease occupied for recovery.
        remote(args.host, launch)
        print(json.dumps({"launched": args.job_id, "lease": lease.to_dict()}), flush=True)
    if args.recover:
        if not (local / "prepared.json").exists():
            value = json.loads(
                remote(args.host, shlex.join(["cat", f"{remote_job}/prepared.json"])).stdout
            )
            (local / "prepared.json").write_text(json.dumps(value, indent=2) + "\n")
        verify_prepared_target(
            json.loads((local / "prepared.json").read_text()),
            remote_root=args.remote_root,
            job_id=args.job_id,
        )
    if args.recover and not (local / "lease.json").exists():
        request = read_request(json.loads((local / "prepared.json").read_text())["request"])
        lease = find_recorded_lease(ledger, request)
        if lease is None:
            raise ValueError("no original occupancy exists to recover")
        (local / "lease.json").write_text(json.dumps(asdict(lease), indent=2) + "\n")
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
