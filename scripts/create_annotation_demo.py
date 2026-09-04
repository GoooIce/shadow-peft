#!/usr/bin/env python3
"""Create a self-contained local Shadow blind-annotation demo."""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import os
import struct
import uuid
import zlib
from pathlib import Path

from shadow_lab.annotation import AnnotationStore, AnswerCommand


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--review-ready",
        action="store_true",
        help="seed one flagged answer and freeze answers for reviewer UI testing",
    )
    return parser.parse_args()


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
    )


def gradient_png(width: int, height: int, palette: tuple[int, int, int]) -> bytes:
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            glow = int(85 * (x / width) + 45 * (1 - y / height))
            row.extend(min(channel + glow, 255) for channel in palette)
        rows.append(bytes(row))
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            png_chunk(b"IDAT", zlib.compress(b"".join(rows), level=6)),
            png_chunk(b"IEND", b""),
        )
    )


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty demo directory: {output_dir}")
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    asset_records = []
    for palette, arm_id in (((28, 56, 104), "demo-control"), ((91, 41, 99), "demo-candidate")):
        path = assets_dir / f"{uuid.uuid4()}.png"
        payload = gradient_png(768, 512, palette)
        path.write_bytes(payload)
        asset_records.append(
            {
                "asset_id": str(uuid.uuid4()),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "mime_type": "image/png",
                "width": 768,
                "height": 512,
                "byte_length": len(payload),
                "storage_key": path.name,
                "arm_id": arm_id,
            }
        )
    experiment_id = str(uuid.uuid4())
    manifest = {
        "schema_version": "shadow-ab/0.1",
        "experiment": {
            "experiment_id": experiment_id,
            "protocol_sha256": "",
            "dataset_id": "shadow-annotation-demo-v1",
            "split": "human_dev",
            "guideline_version": "1.0.0",
            "target_votes": 1,
        },
        "assets": asset_records,
        "comparisons": [
            {
                "comparison_id": str(uuid.uuid4()),
                "prompt_id": str(uuid.uuid4()),
                "prompt_text": prompt,
                "seed": 7300000 + index,
                "width": 768,
                "height": 512,
                "generation_protocol_sha256": "d" * 64,
                "candidate_asset_ids_unordered": [
                    asset_records[1]["asset_id"],
                    asset_records[0]["asset_id"],
                ],
                "qc_kind": "none",
                "has_text_target": index == 1,
            }
            for index, prompt in enumerate(
                (
                    "A cobalt glass sculpture on a walnut desk, soft afternoon light",
                    'A poster reading "FLOW" in crisp white letters above a river',
                    "Two red cubes beside one blue sphere, studio photograph",
                )
            )
        ],
    }
    protocol = json.loads(json.dumps(manifest))
    protocol["experiment"].pop("protocol_sha256")
    manifest["experiment"]["protocol_sha256"] = canonical_sha256(protocol)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    store = AnnotationStore(output_dir / "annotation.sqlite3", assets_dir)
    owner_token = store.create_user(str(uuid.uuid4()), "Demo Owner", "experiment_owner")
    annotator_token = store.create_user(str(uuid.uuid4()), "Demo Annotator", "annotator")
    reviewer_token = store.create_user(str(uuid.uuid4()), "Demo Reviewer", "reviewer")
    store.import_manifest(manifest)
    store.transition_experiment(
        store.authenticate(owner_token),
        experiment_id,
        "frozen",
    )
    status = "frozen"
    if args.review_ready:
        annotator = store.authenticate(annotator_token)
        assignment = store.claim_next(annotator, experiment_id)
        if assignment is None:
            raise RuntimeError("demo comparison was not claimable")
        store.submit_answer(
            annotator,
            AnswerCommand(
                assignment_id=assignment.assignment_id,
                lease_token=assignment.lease_token,
                idempotency_key=str(uuid.uuid4()),
                decision="tie",
                flagged=True,
                client_version="demo-seed",
            ),
            session_id=str(uuid.uuid4()),
        )
        store.transition_experiment(
            store.authenticate(owner_token),
            experiment_id,
            "answers_frozen",
        )
        status = "answers_frozen"
    access_path = output_dir / "demo-access.json"
    access_path.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "owner_token": owner_token,
                "annotator_token": annotator_token,
                "reviewer_token": reviewer_token,
                "database": str(output_dir / "annotation.sqlite3"),
                "assets": str(assets_dir),
                "status": status,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(access_path, 0o600)
    print(json.dumps({"demo": str(output_dir), "access": str(access_path)}))


if __name__ == "__main__":
    main()
