#!/usr/bin/env python3
"""Prepare one explicit complete batch on GB10; execute only its coordinator grant."""

import argparse
import json
import re
import socket
import time
from pathlib import Path

from shadow_lab.mage_w4_search.contracts import DataManifest
from shadow_lab.mage_w4_search.evaluation_worker import prepare_on_gb10, run_evaluation_job
from shadow_lab.mage_w4_search.human_transcription import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--device-id", choices=("gooo", "goooinext"), required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--inputs", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]+", args.job_id):
        parser.error("job-id must contain lowercase letters, numbers and hyphens")
    job = args.run_root / "jobs" / args.job_id
    if args.prepare:
        if args.plan is None or args.inputs is None:
            parser.error("prepare requires explicit --plan and --inputs")
        if job.exists():
            parser.error("job directory already exists; never overwrite frozen preparation")
        job.mkdir(parents=True, exist_ok=False)
        started_monotonic, started_cpu = time.monotonic(), time.process_time()
        timing = {
            "schema": "mage-w4-evaluation-preparation-timing/v1",
            "job_id": args.job_id,
            "device_id": args.device_id,
            "hostname": socket.gethostname(),
            "status": "failed",
            "started_at": time.time(),
            "cpu_scope": "current preparation process",
            "accounting": (
                "host preparation only; no new lease; any overlap with an existing lease "
                "is counted once by the authoritative ledger, never added as device occupancy"
            ),
        }
        try:
            prepared = prepare_on_gb10(
                args.run_root,
                args.job_id,
                args.device_id,
                read_json(args.plan),
                read_json(args.inputs),
            )
            for name, value in (
                ("prepared.json", prepared),
                ("evaluation-job.json", prepared["evaluation_job"]),
                ("candidate.json", prepared["evaluation_job"]["candidate_receipt"]),
            ):
                with (job / name).open("x") as stream:
                    json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
            timing["status"] = "prepared"
        except BaseException as error:
            timing["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            timing.update(
                finished_at=time.time(),
                wall_seconds=time.monotonic() - started_monotonic,
                host_cpu_seconds=time.process_time() - started_cpu,
            )
            with (job / "preparation-timing.json").open("x") as stream:
                json.dump(timing, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "evaluation_job_sha256": prepared["evaluation_job_sha256"],
                    "request_digest": prepared["preflight"]["request_digest"],
                }
            )
        )
        return
    if args.plan is not None or args.inputs is not None:
        parser.error("execution follows only the frozen prepared job and coordinator grant")
    if read_json(job / "prepared.json")["request"]["run_id"] != args.job_id:
        parser.error("worker job directory differs from its frozen run identity")
    manifest = DataManifest.from_dict(read_json(args.run_root / "data-v1/manifest.json"))
    result = run_evaluation_job(job, manifest, device_id=args.device_id)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
