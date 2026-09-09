#!/usr/bin/env python3
"""Register or close a finite group on GB10; never launch workers or open a budget DB."""

import argparse
import json
import platform
import resource
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

from shadow_lab.mage_w4_search.contracts import DataManifest
from shadow_lab.mage_w4_search.group_execution import (
    close_group,
    load_fixed_scale,
    read,
    register_execution,
)
from shadow_lab.mage_w4_search.pilot import artifact_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "close-group"))
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = read(args.inputs)
    kind = inputs["execution_kind"]
    if (
        kind not in {"fixture", "gb10_measured"}
        or kind == "gb10_measured"
        and (platform.system() != "Linux" or platform.machine() != "aarch64")
    ):
        raise ValueError("real grid registration and numerical closeout require GB10 Linux ARM")
    args.output.mkdir(parents=True, exist_ok=False)
    wall, cpu = time.monotonic(), time.process_time()
    timing = dict(
        schema="mage-w4-group-execution-timing/v1",
        started_at=time.time(),
        hostname=socket.gethostname(),
        command=args.command,
        status="failed",
        execution_kind=kind,
        input=artifact_reference(args.inputs),
        creates_lease=False,
    )

    def save(name, value):
        with (args.output / name).open("x") as stream:
            json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")

    try:
        manifest = DataManifest.from_dict(read(inputs["manifest_path"]))
        snapshot = read(inputs["budget_snapshot_path"])
        scales, scale_proof = load_fixed_scale(inputs, manifest, snapshot)
        if args.command == "register":
            if inputs["config"]["execution_kind"] != kind:
                raise ValueError("grid execution kind differs from source validation")
            result = register_execution(
                manifest,
                scales,
                config=inputs["config"],
                registered_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            save("registration.json", result)
            for row in result["jobs"]:
                save(row["candidate_id"] + ".plan.json", row["plan"])
            timing["status"] = "grid-preregistered"
        else:
            registration = read(inputs["registration_path"])
            if registration["config"]["execution_kind"] != kind:
                raise ValueError("grid execution kind differs from source validation")
            result, proof = close_group(
                registration,
                manifest,
                scales,
                snapshot,
                group_id=inputs["group_id"],
                candidate_jobs=inputs["candidate_job_directories"],
                reference_jobs=inputs["reference_job_directories"],
            )
            save("group-receipt.json", result)
            save("group-evidence.json", proof)
            timing["status"] = result["status"]
        save("scale-evidence.json", scale_proof)
    except BaseException as error:
        timing["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        timing.update(
            finished_at=time.time(),
            wall_seconds=time.monotonic() - wall,
            host_cpu_seconds=time.process_time() - cpu,
            max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            max_rss_unit="KiB" if platform.system() == "Linux" else "bytes",
        )
        save("summary.json", timing)
    print(json.dumps(timing, ensure_ascii=False))


if __name__ == "__main__":
    main()
