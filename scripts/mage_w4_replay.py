#!/usr/bin/env python3
"""Prepare, validate and close one explicit ticket06 development replay on GB10."""

import argparse
import json
import platform
import re
import resource
import socket
import subprocess
import time
from pathlib import Path

from shadow_lab.mage_w4_search.contracts import DataManifest
from shadow_lab.mage_w4_search.development_replay import (
    close_development_replay,
    prepare_development_replay,
    verify_replay_prepared,
    verify_replay_source,
)
from shadow_lab.mage_w4_search.evaluation_worker import run_evaluation_job, verify_evaluation_inputs
from shadow_lab.mage_w4_search.human_transcription import read_json, read_json_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify-source", "run", "close"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--device-id", choices=("gooo", "goooinext"), required=True)
    parser.add_argument("--source-prepared", type=Path)
    parser.add_argument("--rule", type=Path)
    parser.add_argument("--registered-at-utc")
    parser.add_argument("--budget-snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.run_root.is_absolute() or not re.fullmatch(r"[a-z0-9-]+", args.job_id):
        parser.error("absolute run-root and lowercase alphanumeric/hyphen job-id required")
    if args.command == "prepare":
        if any(
            value is None for value in (args.source_prepared, args.rule, args.registered_at_utc)
        ):
            parser.error("prepare requires source-prepared, rule and registered-at-utc")
        if args.budget_snapshot is not None or args.output is not None:
            parser.error("prepare writes only a new job; no budget snapshot or output override")
    elif any(
        value is not None for value in (args.source_prepared, args.rule, args.registered_at_utc)
    ):
        parser.error("later operations use only the frozen replay preparation")
    if args.command in {"verify-source", "close"}:
        if args.budget_snapshot is None or args.output is None:
            parser.error(
                "verification/close requires the original authority snapshot and new output"
            )
    elif args.budget_snapshot is not None or args.output is not None:
        parser.error("run cannot access a budget snapshot or change output")
    directory = args.run_root / "jobs" / args.job_id
    if args.command == "run":
        prepared = read_json(directory / "prepared.json")
        manifest = DataManifest.from_dict(read_json(args.run_root / "data-v1/manifest.json"))
        verify_replay_prepared(prepared, manifest)
        if prepared["request"]["run_id"] != args.job_id:
            parser.error("replay run identity differs from its job directory")
        print(json.dumps(run_evaluation_job(directory, manifest, device_id=args.device_id)))
        return
    output = directory if args.command == "prepare" else args.output
    output.mkdir(parents=True, exist_ok=False)
    wall, cpu = time.monotonic(), time.process_time()
    timing = {
        "schema": "mage-w4-replay-host-timing/v1",
        "command": args.command,
        "job_id": args.job_id,
        "device_id": args.device_id,
        "hostname": socket.gethostname(),
        "started_at": time.time(),
        "status": "failed",
        "cpu_scope": "current process",
        "accounting": "host work only; no lease; overlap never adds duplicate device occupancy",
    }
    try:
        if platform.system() != "Linux" or platform.machine() != "aarch64":
            raise ValueError("real replay preparation/validation requires GB10 Linux ARM")
        manifest = DataManifest.from_dict(read_json(args.run_root / "data-v1/manifest.json"))
        if args.command == "prepare":
            source, sha, size = read_json_artifact(args.source_prepared)
            if source["evaluation_job"]["shared"]["run_root"] != str(args.run_root):
                raise ValueError("replay source and destination must retain one frozen run-root")
            hardware = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
                text=True,
            ).strip()
            if "NVIDIA GB10" not in hardware:
                raise ValueError("replay preparation requires the known GB10 device")
            prepared = prepare_development_replay(
                source,
                manifest,
                source_artifact={
                    "path": str(args.source_prepared.resolve()),
                    "sha256": sha,
                    "size_bytes": size,
                },
                run_id=args.job_id,
                device={"id": args.device_id, "nvidia_identity": hardware},
                registered_at_utc=args.registered_at_utc,
                rule=read_json(args.rule),
            )
            verify_evaluation_inputs(prepared["evaluation_job"]["shared"])
            from shadow_lab.mage_w4_search.center_execution import decode_full_center

            for value in prepared["evaluation_job"]["centers"].values():
                decode_full_center(value).verify_files()
            artifacts = {
                "prepared.json": prepared,
                "evaluation-job.json": prepared["evaluation_job"],
            }
            result = {
                "status": "prepared",
                "evaluation_job_sha256": prepared["evaluation_job_sha256"],
            }
        else:
            prepared = read_json(directory / "prepared.json")
            if prepared["request"]["run_id"] != args.job_id:
                raise ValueError("replay job directory identity mismatch")
            expected_device = (
                prepared["evaluation_job"]["source_prepared"]["request"]["device_id"]
                if args.command == "verify-source"
                else prepared["request"]["device_id"]
            )
            if args.device_id != expected_device:
                raise ValueError("source validation or replay closeout device identity mismatch")
            snapshot = read_json(args.budget_snapshot)
            if args.command == "verify-source":
                result = verify_replay_source(prepared, manifest, snapshot)
                artifacts = {"replay-source-verification.json": result}
            else:
                result = close_development_replay(prepared, manifest, snapshot, directory)
                artifacts = {"replay-closeout.json": result}
        for name, value in artifacts.items():
            with (output / name).open("x") as stream:
                json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
        timing["status"] = result["status"]
        print(json.dumps(result, ensure_ascii=False))
    except BaseException as error:
        timing["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        timing.update(
            finished_at=time.time(),
            wall_seconds=time.monotonic() - wall,
            host_cpu_seconds=time.process_time() - cpu,
            max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            max_rss_unit="bytes" if platform.system() == "Darwin" else "KiB",
        )
        with (output / "host-timing.json").open("x") as stream:
            json.dump(timing, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    main()
