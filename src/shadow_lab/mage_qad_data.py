"""Separately sourced, stratified QAD prompts; evaluation identities are exclusion-only."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict, deque

from .image_eval import canonical_json, sha256_text

QAD_COUNTS = {"qad_train": 1024, "qad_val": 128}


def prompt_identity(prompt: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())


def data_hash(rows: list[dict]) -> str:
    return sha256_text(canonical_json(rows))


def exclusion_identity(field: str, value) -> str:
    value = prompt_identity(value) if field == "prompt" else value
    return sha256_text(canonical_json(value))


def validate_qad_manifest(rows: list[dict], manifest: dict) -> None:
    """Validate QAD without opening any held-out prompt file during training."""
    if Counter(row["split"] for row in rows) != QAD_COUNTS:
        raise ValueError("QAD requires 1024 train and 128 validation samples")
    seen = {
        field: set(manifest["exclusion_hashes"][field]) for field in ("prompt", "sample_id", "seed")
    }
    if any(len(seen[field]) != 500 for field in ("sample_id", "seed")) or any(
        not entries or any(not re.fullmatch(r"[0-9a-f]{64}", entry) for entry in entries)
        for entries in seen.values()
    ):
        raise ValueError("QAD manifest lacks complete evaluation exclusion identities")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest.get("evaluation_sha256", "")):
        raise ValueError("QAD manifest lacks a valid evaluation pool hash")
    for row in rows:
        if any(
            not row.get(k)
            for k in ("prompt", "sample_id", "source", "source_revision", "source_id", "stratum")
        ):
            raise ValueError("QAD source identity must be complete")
        if type(row["seed"]) is not int or row["seed"] < 0:
            raise ValueError("invalid QAD seed")
        for field in seen:
            value = exclusion_identity(field, row[field])
            if value in seen[field]:
                raise ValueError(f"QAD {field} leak or duplicate")
            seen[field].add(value)
    if data_hash(rows) != manifest["data_sha256"]:
        raise ValueError("QAD data identity mismatch")
    actual_sources = sorted({(r["source"], r["source_revision"]) for r in rows})
    actual_strata = {
        split: dict(Counter(r["stratum"] for r in rows if r["split"] == split))
        for split in QAD_COUNTS
    }
    if (
        canonical_json(actual_sources) != canonical_json(manifest["sources"])
        or actual_strata != manifest["strata"]
        or manifest["counts"] != QAD_COUNTS
    ):
        raise ValueError("QAD manifest source/stratum/count mismatch")


def validate_qad_pool(rows: list[dict], evaluation: list[dict]) -> dict:
    if len(evaluation) != 500 or len({r["sample_id"] for r in evaluation}) != 500:
        raise ValueError("QAD exclusion requires the complete unique 500-sample evaluation pool")
    manifest = {
        "schema_version": 1,
        "counts": dict(QAD_COUNTS),
        "data_sha256": data_hash(rows),
        "evaluation_sha256": data_hash(evaluation),
        "exclusion_hashes": {
            field: sorted({exclusion_identity(field, row[field]) for row in evaluation})
            for field in ("prompt", "sample_id", "seed")
        },
        "sources": sorted({(r["source"], r["source_revision"]) for r in rows}),
        "strata": {
            split: dict(Counter(r["stratum"] for r in rows if r["split"] == split))
            for split in QAD_COUNTS
        },
    }
    validate_qad_manifest(rows, manifest)
    return manifest


def build_qad_pool(sources: list[dict], evaluation: list[dict]) -> tuple[list[dict], dict]:
    excluded = {prompt_identity(r["prompt"]) for r in evaluation}
    groups = defaultdict(list)
    for source in sorted(sources, key=lambda r: data_hash([r])):
        identity = prompt_identity(source["prompt"])
        if identity in excluded:
            continue
        excluded.add(identity)
        groups[source["stratum"]].append(source)
    queues = [deque(groups[k]) for k in sorted(groups)]
    selected = []
    while any(queues):
        for queue in queues:
            if queue:
                selected.append(queue.popleft())
    if len(selected) < 1152:
        raise ValueError("insufficient distinct non-evaluation prompts for QAD")
    rows = []
    used_seeds = {r["seed"] for r in evaluation}
    seed = 0
    for split, count in QAD_COUNTS.items():
        for _ in range(count):
            source = selected[len(rows)]
            while seed in used_seeds:
                seed += 1
            row = dict(source, split=split, seed=seed, sample_id="qad-" + data_hash([source])[:24])
            rows.append(row)
            used_seeds.add(seed)
    return rows, validate_qad_pool(rows, evaluation)
