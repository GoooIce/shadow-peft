"""Command-line administration for the local annotation application."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import uuid
from pathlib import Path

from .analysis import summarize_annotations
from .exports import to_csv, to_jsonl
from .store import AnnotationStore
from .web import serve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--lease-seconds", type=int, default=900)
    commands = parser.add_subparsers(dest="command", required=True)

    user = commands.add_parser("create-user")
    user.add_argument("--user-id", default=None)
    user.add_argument("--name", required=True)
    user.add_argument(
        "--role",
        required=True,
        choices=("annotator", "reviewer", "experiment_owner"),
    )

    import_command = commands.add_parser("import-manifest")
    import_command.add_argument("manifest", type=Path)

    transition = commands.add_parser("transition")
    transition.add_argument("experiment_id")
    transition.add_argument(
        "status",
        choices=("frozen", "collecting", "answers_frozen", "revealed", "archived"),
    )

    server = commands.add_parser("serve")
    server.add_argument("experiment_id")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", default=8765, type=int)

    export = commands.add_parser("export")
    export.add_argument("experiment_id")
    export.add_argument("--revealed", action="store_true")
    export.add_argument("--format", choices=("json", "jsonl", "csv"), default="json")
    export.add_argument("--out", required=True, type=Path)

    analyze = commands.add_parser("analyze")
    analyze.add_argument("experiment_id")
    analyze.add_argument("--revealed", action="store_true")
    analyze.add_argument("--bootstrap-samples", type=int, default=2000)
    analyze.add_argument("--bootstrap-seed", type=int, default=20260904)
    analyze.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def _owner(store: AnnotationStore):
    token = os.environ.get("SHADOW_ANNOTATION_TOKEN")
    if not token:
        token = getpass.getpass("Experiment-owner token: ")
    return store.authenticate(token)


def _atomic_write(path: Path, value: str) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    store = AnnotationStore(
        args.database,
        args.assets,
        lease_seconds=args.lease_seconds,
    )
    if args.command == "create-user":
        user_id = args.user_id or str(uuid.uuid4())
        token = store.create_user(user_id, args.name, args.role)
        print(json.dumps({"user_id": user_id, "role": args.role, "token": token}))
        return
    if args.command == "import-manifest":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        experiment_id = store.import_manifest(manifest)
        print(json.dumps({"experiment_id": experiment_id, "status": "draft"}))
        return
    if args.command == "transition":
        store.transition_experiment(_owner(store), args.experiment_id, args.status)
        print(json.dumps({"experiment_id": args.experiment_id, "status": args.status}))
        return
    if args.command == "serve":
        serve(store, args.experiment_id, host=args.host, port=args.port)
        return
    if args.command == "export":
        export = store.export_experiment(
            _owner(store),
            args.experiment_id,
            revealed=args.revealed,
        )
        if args.format == "jsonl":
            encoded = to_jsonl(export)
        elif args.format == "csv":
            encoded = to_csv(export)
        else:
            encoded = json.dumps(export, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        _atomic_write(args.out, encoded)
        print(json.dumps({"out": str(args.out), "format": args.format}))
        return
    if args.command == "analyze":
        export = store.export_experiment(
            _owner(store),
            args.experiment_id,
            revealed=args.revealed,
        )
        report = summarize_annotations(
            export,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        _atomic_write(
            args.out,
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        print(json.dumps({"out": str(args.out), "analysis": True}))
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
