#!/usr/bin/env python3
"""Select, preregister and close development work from explicit JSON inputs.

Commands: select, register-scale, close-scale, register-screen, close-screen.
Each receives --inputs JSON and creates a new --output directory. Every input
requires execution_kind (fixture or gb10_measured). Real data selection and all
numerical closeout run only on the GB10 Linux ARM host, with no budget database.
"""

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from shadow_lab.mage_w4_search.contracts import CandidateRequest, DataManifest
from shadow_lab.mage_w4_search.development import (
    close_scale_registration,
    close_screen_registration,
    decode_scale_receipt,
    register_scale_plan,
    register_screen_plan,
    select_development_samples,
    verify_evaluation_job,
)
from shadow_lab.mage_w4_search.human_transcription import read_json_artifact


def read(path):
    return read_json_artifact(path)[0]


def reference(path):
    data = Path(path).read_bytes()
    return {
        "path": str(Path(path).resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def prepared_request(path):
    return CandidateRequest.from_dict(read(path)["request"])


def load_verified(directory, request, manifest, snapshot, kind, registered_at_utc):
    lease = read(Path(directory) / "lease.json")
    registered = datetime.fromisoformat(registered_at_utc.replace("Z", "+00:00")).timestamp()
    if lease["started_at"] < registered:
        raise ValueError("worker lease precedes development preregistration")
    return verify_evaluation_job(directory, request, manifest, snapshot, execution_kind=kind)


def load_scale(inputs, manifest, snapshot):
    registration = read(inputs["scale_registration_path"])
    request = CandidateRequest.from_dict(registration["request"])
    batch, proof = load_verified(
        inputs["scale_job_directory"],
        request,
        manifest,
        snapshot,
        inputs["execution_kind"],
        registration["plan"]["registered_at_utc"],
    )
    receipt = close_scale_registration(registration, batch)
    if "scale_receipt_path" in inputs:
        saved = decode_scale_receipt(read(inputs["scale_receipt_path"]))
        if saved.sha256 != receipt.sha256:
            raise ValueError("saved scales do not replay from their registered q0 job")
    return receipt, proof, batch


def execute(command, inputs):
    kind = inputs["execution_kind"]
    if kind not in {"fixture", "gb10_measured"}:
        raise ValueError("explicit fixture or gb10_measured execution kind required")
    if kind == "gb10_measured" and (
        platform.system() != "Linux" or platform.machine() not in {"aarch64", "arm64"}
    ):
        raise ValueError(
            "real development selection/registration/closeout requires the GB10 Linux ARM host"
        )
    manifest = DataManifest.from_dict(read(inputs["manifest_path"]))
    registered = inputs.get("registered_at_utc", datetime.now(timezone.utc).isoformat())
    if command == "select":
        samples = select_development_samples(manifest, inputs["purpose"], **inputs["selection"])
        return {
            "selection.json": {
                "schema": "mage-w4-development-selection/v1",
                "purpose": inputs["purpose"],
                "manifest_digest": manifest.digest,
                "selection": inputs["selection"],
                "samples": [asdict(row) for row in samples],
                "execution_kind": kind,
                "formal_execution_authorized": False,
            }
        }, "samples-selected"
    if command == "register-scale":
        request = prepared_request(inputs["prepared_path"])
        registration = register_scale_plan(
            request,
            manifest,
            selection=inputs["selection"],
            reliability=inputs["reliability"],
            registered_at_utc=registered,
        )
        return {"registration.json": registration}, "scale-preregistered"
    snapshot = read(inputs["budget_snapshot_path"])
    scales, scale_proof, baseline = load_scale(inputs, manifest, snapshot)
    if command == "close-scale":
        return {
            "scale-receipt.json": asdict(scales),
            "evidence.json": scale_proof,
            "closeout.json": {
                "status": scales.status,
                "scale_receipt_sha256": scales.sha256,
                "reasons": list(scales.reasons),
                "execution_kind": kind,
                "baseline": reference(Path(inputs["scale_job_directory"]) / "scores/batch.json"),
                "registration": reference(inputs["scale_registration_path"]),
                "budget_snapshot": reference(inputs["budget_snapshot_path"]),
                "occupied_seconds": scale_proof["occupied_seconds"],
                "formal_execution_authorized": False,
            },
        }, scales.status
    if scales.status != "ready":
        raise ValueError("unreliable three-axis scales cannot register or screen candidates")
    if command == "register-screen":
        candidates = [prepared_request(path) for path in inputs["candidate_prepared_paths"]]
        refs = {
            device: prepared_request(path)
            for device, path in inputs["reference_prepared_paths"].items()
        }
        registration = register_screen_plan(
            scales,
            candidates,
            refs,
            manifest,
            selection=inputs["selection"],
            candidate_order=tuple(inputs["candidate_order"]),
            registered_at_utc=registered,
        )
        return {
            "registration.json": registration,
            "scale-evidence.json": scale_proof,
        }, "screen-preregistered"
    registration = read(inputs["screen_registration_path"])
    expected = {
        row["candidate_id"]: CandidateRequest.from_dict(row) for row in registration["candidates"]
    }
    refs = {
        key: CandidateRequest.from_dict(row)
        for key, row in registration["references_by_device"].items()
    }
    if set(inputs["candidate_job_directories"]) != set(expected) or set(
        inputs["reference_job_directories"]
    ) != set(refs):
        raise ValueError("job directories must exactly cover the preregistered screen")
    batches, reference_batches, proofs = [], {}, []
    for key, request in expected.items():
        batch, proof = load_verified(
            inputs["candidate_job_directories"][key],
            request,
            manifest,
            snapshot,
            kind,
            registration["registered_at_utc"],
        )
        batches.append(batch)
        proofs.append(proof)
    for device, request in refs.items():
        batch, proof = load_verified(
            inputs["reference_job_directories"][device],
            request,
            manifest,
            snapshot,
            kind,
            registration["registered_at_utc"],
        )
        reference_batches[device] = batch
        proofs.append(proof)
    all_proofs = [scale_proof, *proofs]
    if len({row["lease_id"] for row in all_proofs}) != len(all_proofs):
        raise ValueError("separate complete worker batches require distinct accounted leases")
    screened = close_screen_registration(registration, scales, batches, reference_batches)
    return {
        "screen-receipt.json": asdict(screened),
        "evidence.json": {"scale": scale_proof, "screen": proofs},
        "closeout.json": {
            "status": screened.status,
            "selected_candidate_id": screened.selected_candidate_id,
            "screen_receipt_sha256": screened.sha256,
            "execution_kind": kind,
            "screen_occupied_seconds": sum(row["occupied_seconds"] for row in proofs),
            "scale_occupied_seconds": scale_proof["occupied_seconds"],
            "combined_unique_seconds": sum(row["occupied_seconds"] for row in all_proofs),
            "scale_batch_sha256": baseline.sha256,
            "registration": reference(inputs["screen_registration_path"]),
            "budget_snapshot": reference(inputs["budget_snapshot_path"]),
            "center_updated": False,
            "independently_confirmed": False,
            "formal_density_frozen": False,
            "formal_execution_authorized": False,
        },
    }, screened.status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("select", "register-scale", "close-scale", "register-screen", "close-screen"),
    )
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("output directory exists; append a new immutable development artifact")
    started, cpu = time.monotonic(), time.process_time()
    inputs, inputs_sha, inputs_size = read_json_artifact(args.inputs)
    files, status = execute(args.command, inputs)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in files.items():
        (args.output / name).write_text(
            json.dumps(
                value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
        )
    summary = {
        "schema": "mage-w4-development-cli/v1",
        "command": args.command,
        "status": status,
        "execution_kind": inputs["execution_kind"],
        "input_file_sha256": inputs_sha,
        "input_size_bytes": inputs_size,
        "host_cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.monotonic() - started,
        "timing_scope": "host computation; this CLI creates no lease, and any overlap with an existing reservation is charged once",
        "artifacts": {name: reference(args.output / name) for name in files},
        "formal_execution_authorized": False,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
