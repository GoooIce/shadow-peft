#!/usr/bin/env python3
"""Run an explicit coordinator operation on GB10 through the unique Mac budget authority."""

import argparse
import json
from pathlib import Path

from mage_w4_launch import coordinator_command, open_pilot_ledger

from shadow_lab.mage_w4_search.budget_rpc import RemoteCoordinator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=(
            "initialize",
            "snapshot",
            "next_mutation_seeds",
            "register",
            "work_request",
            "reconcile_claim",
            "submit",
            "screen",
            "finish",
            "abort",
        ),
    )
    parser.add_argument("--host", choices=("gooo", "goooinext"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Reserve the receipt path before an operation so a replay cannot overwrite evidence.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        result = {
            "schema": "mage-w4-coordinator-command/v1",
            "status": "failed",
            "host": args.host,
            "operation": args.operation,
        }
        remote = None
        try:
            remote = RemoteCoordinator(
                coordinator_command(args.host),
                json.loads(args.config.read_text()),
                open_pilot_ledger(),
                initialize=args.operation == "initialize",
            )
            with remote:
                operation = "snapshot" if args.operation == "initialize" else args.operation
                result.update(
                    status="ok",
                    result=remote.call(operation, json.loads(args.inputs.read_text())),
                    budget_authority=remote.authority,
                )
            print(json.dumps(result, ensure_ascii=False))
        except BaseException as error:
            result["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            if remote is not None:
                result["resources"] = {
                    "initialization": remote.initialization_timing,
                    "operation": remote.last_operation_timing,
                    "remote_artifact_path": remote.timing_artifact_path,
                }
            json.dump(result, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    main()
