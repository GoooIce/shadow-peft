#!/usr/bin/env python3
"""Build the frozen 500-prompt Mage ES/evaluation pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from shadow_lab.image_eval import (
    EVAL_SPLITS,
    SourcePrompt,
    build_frozen_eval_pool,
    canonical_json,
)

GENEVAL_REVISION = (
    "modelscope:JunweiXi/EvalGEN@master;"
    "spec:github:djghosh13/geneval@af4902f24d3ca90ebbb446dd9891a59e0f82725f"
)
TEXT_REVISION = "modelscope:Qwen/Qwen-Image-Bench@master"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geneval-jsonl", required=True, type=Path)
    parser.add_argument("--text-jsonl", required=True, type=Path)
    parser.add_argument("--geneval-revision", default=GENEVAL_REVISION)
    parser.add_argument("--text-revision", default=TEXT_REVISION)
    parser.add_argument("--selection-seed", type=int, default=20260904)
    parser.add_argument("--sample-seed-base", type=int, default=7_300_000)
    parser.add_argument("--content-gate-accepted", type=Path)
    parser.add_argument(
        "--candidates-only",
        action="store_true",
        help="write normalized candidates.jsonl for Qwen3-VL screening and stop",
    )
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_geneval(path: Path, revision: str) -> list[SourcePrompt]:
    rows: list[SourcePrompt] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            tag = row.get("tag")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{index + 1}: missing prompt")
            if not isinstance(tag, str) or not tag.strip():
                raise ValueError(f"{path}:{index + 1}: missing tag")
            metadata = {key: value for key, value in row.items() if key != "prompt"}
            rows.append(
                SourcePrompt(
                    source="geneval",
                    revision=revision,
                    source_id=str(index),
                    prompt=prompt,
                    stratum=tag,
                    metadata_json=canonical_json(metadata),
                )
            )
    if len(rows) != 553:
        raise ValueError(f"official GenEval metadata must contain 553 prompts, got {len(rows)}")
    return rows


def _text_rendering_stratum(dims: str) -> str | None:
    leaves = []
    for item in dims.split(";"):
        parts = [part.strip() for part in item.split("/")]
        if len(parts) >= 3 and parts[1].casefold() == "text rendering":
            leaves.append(parts[2].casefold().replace(" ", "_"))
    return "text_rendering:" + "+".join(sorted(set(leaves))) if leaves else None


def load_text_rendering(path: Path, revision: str) -> list[SourcePrompt]:
    rows: list[SourcePrompt] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            dims = row.get("dims_en")
            if not isinstance(dims, str):
                continue
            stratum = _text_rendering_stratum(dims)
            if stratum is None:
                continue
            prompt = row.get("prompt_en")
            source_id = row.get("ID")
            if not isinstance(prompt, str) or not prompt.strip() or source_id is None:
                raise ValueError(f"{path}:{line_number}: invalid text-rendering row")
            rows.append(
                SourcePrompt(
                    source="text_rendering",
                    revision=revision,
                    source_id=str(source_id),
                    prompt=prompt,
                    stratum=stratum,
                    metadata_json=canonical_json(
                        {
                            "ID": source_id,
                            "dims_en": dims,
                            "dims_cn": row.get("dims_cn"),
                            "prompt_cn": row.get("prompt_cn"),
                        }
                    ),
                )
            )
    if len(rows) < 100:
        raise ValueError(f"need at least 100 text-rendering prompts, got {len(rows)}")
    return rows


def candidate_record(row: SourcePrompt) -> dict:
    return {
        "source": row.source,
        "source_revision": row.revision,
        "source_id": row.source_id,
        "prompt": row.prompt,
        "stratum": row.stratum,
        "metadata": json.loads(row.metadata_json),
    }


def apply_content_gate(
    candidates: list[SourcePrompt], accepted_path: Path
) -> list[SourcePrompt]:
    accepted: dict[tuple[str, str], dict] = {}
    with accepted_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("source", "")), str(row.get("source_id", "")))
            gate = row.get("content_gate")
            if not all(key) or not isinstance(gate, dict) or gate.get("passed") is not True:
                raise ValueError(f"{accepted_path}:{line_number}: invalid accepted gate row")
            if key in accepted:
                raise ValueError(
                    f"{accepted_path}:{line_number}: duplicate accepted prompt identity {key}"
                )
            accepted[key] = gate
    filtered = []
    candidate_keys = {(row.source, row.source_id) for row in candidates}
    unknown = sorted(set(accepted) - candidate_keys)
    if unknown:
        raise ValueError(f"content gate contains unknown prompt identities: {unknown[:3]}")
    for row in candidates:
        gate = accepted.get((row.source, row.source_id))
        if gate is None:
            continue
        metadata = json.loads(row.metadata_json)
        metadata["content_gate"] = gate
        filtered.append(replace(row, metadata_json=canonical_json(metadata)))
    return filtered


def validate_content_gate_run(
    *, candidates_path: Path, candidate_count: int, accepted_path: Path
) -> tuple[Path, dict]:
    """Require evidence that every candidate was screened before pool freezing."""

    run_path = accepted_path.parent / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(
            f"content-gate run manifest is required beside accepted rows: {run_path}"
        )
    run = json.loads(run_path.read_text(encoding="utf-8"))
    source = run.get("candidate_source")
    if not isinstance(source, dict):
        raise ValueError("content-gate run has no candidate_source object")
    checks = {
        "screen_all": run.get("screen_all") is True,
        "screened_all_candidates": run.get("screened") == candidate_count,
        "source_row_count": source.get("rows") == candidate_count,
        "candidate_sha256": source.get("sha256") == file_sha256(candidates_path),
        "accepted_sha256": run.get("accepted_sha256") == file_sha256(accepted_path),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"content-gate run is incomplete or mismatched: {failed}")

    accepted = run.get("accepted")
    rejected = run.get("rejected")
    screened = run.get("screened")
    if not all(isinstance(value, int) and value >= 0 for value in (accepted, rejected, screened)):
        raise ValueError("content-gate run counts must be non-negative integers")
    if accepted + rejected != screened:
        raise ValueError("content-gate accepted/rejected counts do not sum to screened")

    audit_path = accepted_path.parent / "audit.jsonl"
    if not audit_path.is_file() or run.get("audit_sha256") != file_sha256(audit_path):
        raise ValueError("content-gate audit is missing or its SHA-256 does not match")
    audit_rows = sum(1 for line in audit_path.read_text(encoding="utf-8").splitlines() if line)
    if audit_rows != screened:
        raise ValueError(
            f"content-gate audit row count differs: expected {screened}, got {audit_rows}"
        )
    return run_path, run


def main() -> None:
    args = parse_args()
    if args.selection_seed < 0 or args.sample_seed_base < 0:
        raise ValueError("seeds must be non-negative")
    geneval_path = args.geneval_jsonl.expanduser().resolve()
    text_path = args.text_jsonl.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {out_dir}")

    candidates = load_geneval(geneval_path, args.geneval_revision)
    candidates.extend(load_text_rendering(text_path, args.text_revision))
    candidates_path = out_dir / "candidates.jsonl"
    candidates_path.write_text(
        "".join(
            json.dumps(candidate_record(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in candidates
        ),
        encoding="utf-8",
    )
    candidate_count = len(candidates)
    if args.candidates_only:
        print(
            json.dumps(
                {
                    "candidates": str(candidates_path),
                    "rows": candidate_count,
                    "sha256": file_sha256(candidates_path),
                },
                ensure_ascii=False,
            )
        )
        return
    gate_path = (
        args.content_gate_accepted.expanduser().resolve()
        if args.content_gate_accepted is not None
        else None
    )
    if gate_path is None:
        raise ValueError("final pool build requires --content-gate-accepted")
    gate_run_path, gate_run = validate_content_gate_run(
        candidates_path=candidates_path,
        candidate_count=candidate_count,
        accepted_path=gate_path,
    )
    candidates = apply_content_gate(candidates, gate_path)
    if gate_run["accepted"] != len(candidates):
        raise ValueError(
            "content-gate accepted count differs from the accepted prompt identities"
        )
    pool = build_frozen_eval_pool(
        candidates,
        selection_seed=args.selection_seed,
        sample_seed_base=args.sample_seed_base,
    )
    records = [sample.as_record() for sample in pool]
    pool_path = out_dir / "pool.jsonl"
    pool_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    split_files = {}
    for split in EVAL_SPLITS:
        split_path = out_dir / f"{split}.jsonl"
        split_rows = [row for row in records if row["split"] == split]
        split_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in split_rows
            ),
            encoding="utf-8",
        )
        split_files[split] = {
            "path": split_path.name,
            "rows": len(split_rows),
            "sha256": file_sha256(split_path),
        }

    manifest = {
        "schema_version": 1,
        "selection_seed": args.selection_seed,
        "sample_seed_base": args.sample_seed_base,
        "inputs": {
            "geneval": {
                "path": str(geneval_path),
                "revision": args.geneval_revision,
                "sha256": file_sha256(geneval_path),
            },
            "text_rendering": {
                "path": str(text_path),
                "revision": args.text_revision,
                "sha256": file_sha256(text_path),
            },
            "content_gate_accepted": {
                "path": str(gate_path),
                "sha256": file_sha256(gate_path),
                "passed_rows": len(candidates),
            },
            "content_gate_run": {
                "path": str(gate_run_path),
                "sha256": file_sha256(gate_run_path),
                "screened": gate_run["screened"],
                "accepted": gate_run["accepted"],
                "rejected": gate_run["rejected"],
            },
        },
        "candidates": {
            "path": candidates_path.name,
            "rows": candidate_count,
            "sha256": file_sha256(candidates_path),
        },
        "pool": {"path": pool_path.name, "rows": len(pool), "sha256": file_sha256(pool_path)},
        "splits": split_files,
        "policy": {
            "primary_reward_axes": ["hps_v2_1", "clip_h14"],
            "geneval_role": "checkpoint/test diagnostic; never an episode hard-zero gate",
            "paired_seed": "each sample seed is immutable and shared by every candidate",
            "human_test": "one-time holdout; never expose during ES or annotation-app development",
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path), **manifest["pool"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
