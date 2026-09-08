#!/usr/bin/env python3
"""Prepare the frozen development/sealed metadata on a GB10 CPU (no inference)."""

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

from shadow_lab.mage_w4_search.data import freeze_pilot_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        parser.error("real data preparation belongs on a GB10 host")
    start, cpu = time.monotonic(), time.process_time()
    source = args.old_development.read_bytes()
    rows = [json.loads(line) for line in source.splitlines() if line.strip()]
    manifest, audit = freeze_pilot_manifest(rows)
    audit.update(
        {
            "host": platform.node(),
            "old_development_sha256": hashlib.sha256(source).hexdigest(),
            "wall_seconds": time.monotonic() - start,
            "host_cpu_seconds": time.process_time() - cpu,
            "device_occupied": False,
            "history": "Old es_train has prior ES/development exposure; no claim of fresh baseline data.",
        }
    )
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("manifest.json", asdict(manifest)), ("data-audit.json", audit)):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in audit.items()
                if k not in {"source_links", "derived_exhaustive_target_ids"}
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
