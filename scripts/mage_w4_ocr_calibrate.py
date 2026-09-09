#!/usr/bin/env python3
"""Import human transcription or close recorded OCR jobs without running models.

closeout --inputs accepts a JSON object of keyword paths for close_comparison.
All paths are explicit. The budget input is a read-only snapshot, never a DB.
Real comparison/freeze must execute on the GB10 Linux ARM host.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from shadow_lab.mage_w4_search.human_transcription import (
    close_comparison,
    decode_plan,
    import_human_work,
    read_json,
)


def write_json(path, value):
    path.write_text(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    intake = commands.add_parser("import", help="Validate the full human inventory; do not freeze")
    intake.add_argument("--source", type=Path, required=True)
    intake.add_argument("--inventory", type=Path, required=True)
    intake.add_argument("--plan", type=Path)
    intake.add_argument("--output", type=Path, required=True)
    closeout = commands.add_parser(
        "closeout", help="Verify preregistered observations and closed leases"
    )
    closeout.add_argument("--inputs", type=Path, required=True)
    closeout.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("output directory already exists; recorded artifacts are immutable")
    if args.command == "import":
        work = import_human_work(args.source, args.inventory)
        result = {"schema": "mage-ocr-human-import/v1", **asdict(work)}
        if args.plan is not None:
            plan = decode_plan(read_json(args.plan))
            result["plan_typed_sha256"] = plan.sha256
            result["selected_transcriptions"] = [asdict(row) for row in work.for_plan(plan)]
            result["exclusions"] = list(work.exclusions_for(plan))
        original = args.source.read_bytes()
        if hashlib.sha256(original).hexdigest() != work.artifact_sha256:
            raise ValueError("human artifact changed during import")
        args.output.mkdir(parents=True)
        (args.output / "human-original.json").write_bytes(original)
        write_json(args.output / "import.json", result)
        print(
            json.dumps({"status": work.status, "rows": len(work.rows), "output": str(args.output)})
        )
        return
    inputs = read_json(args.inputs)
    result = close_comparison(**inputs)
    args.output.mkdir(parents=True)
    write_json(args.output / "calibration-report.json", asdict(result.report))
    write_json(args.output / "device-evidence.json", result.evidence)
    write_json(args.output / "accounting.json", result.accounting)
    if result.scorer_lock is not None:
        write_json(args.output / "scorer-lock.json", result.scorer_lock)
    status = (
        "fixture_only"
        if result.report.execution_kind == "fixture"
        else "scorer_frozen"
        if result.scorer_lock is not None
        else "no_engine_qualified"
    )
    summary = {
        "schema": "mage-w4-ocr-closeout/v1",
        "status": status,
        "execution_kind": result.report.execution_kind,
        "selected_engine": result.report.selected_engine,
        "calibration_report_typed_sha256": result.report.sha256,
        "plan_typed_sha256": result.report.plan.sha256,
        "comparison_total_seconds": result.accounting["comparison_total_seconds"],
        "artifacts": {
            path.name: {
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(args.output.iterdir())
        },
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
