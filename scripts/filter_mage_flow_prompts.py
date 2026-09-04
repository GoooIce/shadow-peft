#!/usr/bin/env python3
"""Freeze a Mage-Flow prompt subset that passes the model's own content gate."""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import json
import platform
import socket
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--candidates-jsonl", required=True, type=Path)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument(
        "--screen-all",
        action="store_true",
        help="screen the full candidate file instead of stopping at --limit accepted rows",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--determinism-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="screen the first candidate twice and require an identical verdict",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def verdict_record(verdict: Any) -> dict[str, Any]:
    return {
        "violates": bool(verdict.violates),
        "categories": list(verdict.categories),
        "reason": verdict.reason,
        "raw": verdict.raw,
    }


def verdict_identity(verdict: Any) -> tuple[bool, tuple[str, ...], str]:
    return bool(verdict.violates), tuple(verdict.categories), verdict.reason


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    model_path = args.model_path.expanduser().resolve()
    candidates_path = args.candidates_jsonl.expanduser().resolve()
    if not (model_path / "model_index.json").is_file():
        raise FileNotFoundError(f"not a Mage diffusers repository: {model_path}")
    if not candidates_path.is_file():
        raise FileNotFoundError(candidates_path)

    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to mix filter runs in non-empty directory: {out_dir}")

    candidates: list[dict[str, Any]] = []
    with candidates_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get(args.prompt_field) if isinstance(row, dict) else None
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"{candidates_path}:{line_number}: "
                    f"missing non-empty string field {args.prompt_field!r}"
                )
            candidates.append(row)
    if not args.screen_all and len(candidates) < args.limit:
        raise ValueError(f"only {len(candidates)} candidates for requested limit {args.limit}")

    import torch
    from mage_flow import MageFlowPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("prompt filtering must run on a CUDA GB10")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 1):
        raise RuntimeError(f"expected GB10 compute capability (12, 1), got {capability}")

    started_at = time.time()
    load_started = time.perf_counter()
    pipeline = MageFlowPipeline.from_pretrained(str(model_path), "cuda")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    accepted_path = out_dir / "accepted.jsonl"
    audit_path = out_dir / "audit.jsonl"
    accepted = 0
    screened = 0
    rejected = 0
    category_counts: collections.Counter[str] = collections.Counter()
    first_identity: tuple[bool, tuple[str, ...], str] | None = None

    for candidate_index, row in enumerate(candidates):
        prompt = row[args.prompt_field]
        screen_started = time.perf_counter()
        verdict = pipeline.model.txt_enc.screen_text(prompt)
        torch.cuda.synchronize()
        screen_seconds = time.perf_counter() - screen_started
        identity = verdict_identity(verdict)
        repeat_identity = None
        if candidate_index == 0 and args.determinism_check:
            repeat = pipeline.model.txt_enc.screen_text(prompt)
            torch.cuda.synchronize()
            repeat_identity = verdict_identity(repeat)
            if identity != repeat_identity:
                raise RuntimeError(
                    f"content-gate verdict changed for the same prompt: {identity!r} != {repeat_identity!r}"
                )
            first_identity = repeat_identity

        screened += 1
        if verdict.violates:
            rejected += 1
            category_counts.update(verdict.categories)
        else:
            accepted += 1

        audit = {
            "candidate_index": candidate_index,
            "prompt": prompt,
            "prompt_metadata": {
                key: value for key, value in row.items() if key != args.prompt_field
            },
            "verdict": verdict_record(verdict),
            "repeat_identity": repeat_identity,
            "screen_seconds": screen_seconds,
        }
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
        print(json.dumps(audit, ensure_ascii=False), flush=True)

        if not verdict.violates:
            output_row = dict(row)
            output_row["content_gate"] = {
                "model": "Mage-Flow-Turbo bundled Qwen3-VL gate",
                "passed": True,
                "reason": verdict.reason,
            }
            with accepted_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
        if not args.screen_all and accepted >= args.limit:
            break

    if not args.screen_all and accepted < args.limit:
        raise RuntimeError(
            f"candidate pool exhausted after {screened} rows with only {accepted} accepted"
        )

    finished_at = time.time()
    run = {
        "schema_version": 1,
        "started_at_unix": started_at,
        "finished_at_unix": finished_at,
        "elapsed_seconds": finished_at - started_at,
        "load_seconds": load_seconds,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(capability),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "mage-flow": package_version("mage-flow"),
            "transformers": package_version("transformers"),
        },
        "model_path": str(model_path),
        "candidate_source": {
            "path": str(candidates_path),
            "sha256": file_sha256(candidates_path),
            "rows": len(candidates),
            "prompt_field": args.prompt_field,
        },
        "screened": screened,
        "accepted": accepted,
        "rejected": rejected,
        "rejection_categories": dict(sorted(category_counts.items())),
        "screen_all": args.screen_all,
        "determinism_check": {
            "enabled": args.determinism_check,
            "first_verdict_identity": first_identity,
        },
        "accepted_sha256": file_sha256(accepted_path),
        "audit_sha256": file_sha256(audit_path),
    }
    (out_dir / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"filter manifest: {out_dir / 'run.json'}")


if __name__ == "__main__":
    main()
