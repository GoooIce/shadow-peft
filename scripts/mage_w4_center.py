#!/usr/bin/env python3
"""Register q0 or prepare a versioned complete center job on GB10, without a lease."""

import argparse
import json
import platform
import re
import socket
import time
from dataclasses import asdict
from pathlib import Path

from shadow_lab.mage_w4_search.center_execution import prepare_center_on_gb10, register_q0_center
from shadow_lab.mage_w4_search.human_transcription import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register-q0", "prepare"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--device-id", choices=("gooo", "goooinext"), required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--q0", type=Path)
    parser.add_argument("--current", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]+", args.job_id):
        parser.error("job-id must contain lowercase letters, numbers and hyphens")
    if args.command == "prepare" and any(
        value is None for value in (args.plan, args.q0, args.current)
    ):
        parser.error("prepare requires explicit plan, fixed q0 and current manifests")
    if args.command == "register-q0" and any(
        value is not None for value in (args.plan, args.q0, args.current)
    ):
        parser.error("q0 registration follows only its fixed template")
    job = args.run_root / "jobs" / args.job_id
    job.mkdir(parents=True, exist_ok=False)
    wall, cpu = time.monotonic(), time.process_time()
    timing = {
        "schema": "mage-w4-center-preparation-timing/v1",
        "started_at": time.time(),
        "status": "failed",
        "command": args.command,
        "job_id": args.job_id,
        "device_id": args.device_id,
        "hostname": socket.gethostname(),
        "cpu_scope": "current preparation process",
        "accounting": "host preparation; no new lease; overlaps never add duplicate device occupancy",
    }
    try:
        if platform.system() != "Linux" or platform.machine() != "aarch64":
            raise ValueError("real center preparation requires a GB10 Linux ARM host")
        if args.command == "register-q0":
            shared = read_json(args.template)["evaluation_job"]["shared"]
            root = Path(shared["run_root"])
            center = register_q0_center(
                root / "input/qad-nvfp4-best-673.pt",
                root / "input/qad-transformer-remainder-best-673.pt",
                job / "center",
                allowed_assets=shared["inventory"]["allowed_assets"],
                expected_native_sha256=shared["q0_digest"],
            )
            result = {"full_center": asdict(center)}
        else:
            result = prepare_center_on_gb10(
                args.job_id,
                args.device_id,
                read_json(args.plan),
                template_path=args.template,
                q0_path=args.q0,
                current_path=args.current,
                candidate_directory=job / "candidate-center",
            )
            for name, value in (
                ("prepared.json", result),
                ("evaluation-job.json", result["evaluation_job"]),
                ("candidate.json", result["evaluation_job"]["candidate_receipt"]),
                ("full-center.json", result["evaluation_job"]["centers"]["evaluated"]),
            ):
                with (job / name).open("x") as stream:
                    json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
        timing["status"] = "prepared"
        print(json.dumps(result, ensure_ascii=False))
    except BaseException as error:
        timing["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        timing.update(
            finished_at=time.time(),
            wall_seconds=time.monotonic() - wall,
            host_cpu_seconds=time.process_time() - cpu,
        )
        with (job / "preparation-timing.json").open("x") as stream:
            json.dump(timing, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    main()
