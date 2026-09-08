#!/usr/bin/env python3
"""Prepare on GB10 CPU, then execute an authorized native replay under a lease."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from shadow_lab.mage_w4_search.budget import DeviceLease
from shadow_lab.mage_w4_search.contracts import (
    ArtifactIdentity,
    CandidateRequest,
    RequestedSample,
    digest_json,
    preflight,
)
from shadow_lab.mage_w4_search.execution import read_manifest, validate_worker_grant
from shadow_lab.mage_w4_search.mutation import (
    audit_movable_coordinates,
    build_layer_groups,
    propose_candidate,
    replay_candidate,
)
from shadow_lab.mage_w4_search.native import (
    GenerationConfig,
    build_native_pipeline,
    install_quantized_assets,
    load_quantized_assets,
)


def implementation_hashes():
    repo = Path(__file__).resolve().parents[1]
    files = [
        Path(__file__).resolve(),
        *(
            repo / "src/shadow_lab/mage_w4_search" / name
            for name in (
                "__init__.py",
                "native.py",
                "mutation.py",
                "contracts.py",
                "execution.py",
                "budget.py",
            )
        ),
    ]
    files.extend(
        repo / "src/shadow_lab" / name
        for name in (
            "mage_qad_checkpoint.py",
            "nvfp4_runtime.py",
            "quasar.py",
            "es_quant.py",
            "image_eval.py",
        )
    )
    return {
        str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }


def select_probe_records(manifest, action):
    if action == "generate":
        selected = [
            r
            for r in manifest.records
            if r.development_uses == ("ocr_selection",)
            and r.target_text
            and r.complete_target_inventory
        ]
        if len(selected) != 5:
            raise ValueError(
                "OCR selection generation requires exactly five complete-target prompts"
            )
        return selected
    eligible = [r for r in manifest.records if r.development_uses == ("candidate_development",)]
    return [
        next(r for r in eligible if not r.target_text),
        next(r for r in eligible if r.target_text),
    ]


def pngs_equal(first, second):
    return (
        first["qualified"]
        and second["qualified"]
        and len(first["rows"]) == len(second["rows"])
        and all(
            a["image_file_sha256"] == b["image_file_sha256"]
            for a, b in zip(first["rows"], second["rows"], strict=True)
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--device-id", choices=("gooo", "goooinext"), required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument(
        "--action",
        choices=("native_replay", "mutation_replay", "generate"),
        default="native_replay",
        help="Preparation only; execution always follows its coordinator grant",
    )
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        parser.error("real payload preparation/execution requires a GB10 host")
    root = args.run_root
    inventory = json.loads((root / "inventory.json").read_text())
    manifest = read_manifest(json.loads((root / "data-v1/manifest.json").read_text()))
    job = root / "jobs" / args.job_id
    generation = GenerationConfig(watermark_key=20260909)
    started = time.time()
    cpu = time.process_time()
    if not args.prepare:
        grant = json.loads((job / "grant.json").read_text())
        lease = DeviceLease(**json.loads((job / "lease.json").read_text()))
        request = validate_worker_grant(
            grant, lease, manifest, device_id=args.device_id, now=time.time()
        )
        if (
            digest_json(inventory["allowed_assets"])
            != request.runtime_identity["input_inventory_digest"]
            or generation.identity() != request.runtime_identity["generation"]
            or inventory["runtime"] != request.runtime_identity["software"]
        ):
            raise ValueError("runtime inputs changed after request freeze")
        if request.runtime_identity["implementation_sha256"] != implementation_hashes():
            raise ValueError("worker implementation changed after request freeze")
    assets = load_quantized_assets(
        root / "input/qad-nvfp4-best-673.pt",
        root / "input/qad-transformer-remainder-best-673.pt",
        allowed_assets=inventory["allowed_assets"],
    )
    if args.prepare:
        job.mkdir(parents=True, exist_ok=False)
        hardware = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        selected = select_probe_records(manifest, args.action)
        samples = tuple(
            RequestedSample(r.prompt_id, digest_json(r.prompt), r.source_group, 20260909)
            for r in selected
        )
        paths = inventory["allowed_assets"]
        dependencies = []
        for role, path in (
            ("quantized_packed", root / "input/qad-nvfp4-best-673.pt"),
            ("quantized_remainder", root / "input/qad-transformer-remainder-best-673.pt"),
        ):
            dependencies.append(ArtifactIdentity(role, paths[str(path)], path.stat().st_size))
        descriptions = {
            "weight_scales": {"fixed_state_sha256": assets.fixed_sha256},
            "layer_manifest": {n: layer.shape for n, layer in assets.layers.items()},
            "text_encoder": {p: h for p, h in paths.items() if "/auxiliary/text_encoder/" in p},
            "vae": {p: h for p, h in paths.items() if "/auxiliary/vae/" in p},
        }
        for role, value in descriptions.items():
            dependencies.append(
                ArtifactIdentity(role, digest_json(value), len(json.dumps(value).encode()))
            )
        proposal = None
        if args.action == "mutation_replay":
            proposal = propose_candidate(
                assets,
                build_layer_groups(assets),
                group_id="outside",
                mutation_count=1,
                seed=20260909,
                candidate_index=0,
            )
            if proposal.assets is None:
                raise ValueError(f"engineering smoke proposal failed: {proposal.receipt}")
            (job / "candidate.json").write_text(json.dumps(proposal.receipt, indent=2) + "\n")
            (job / "movable-audit.json").write_text(
                json.dumps(audit_movable_coordinates(assets), indent=2) + "\n"
            )
        request = CandidateRequest(
            args.job_id,
            args.job_id,
            "04" if args.action == "generate" else "03" if proposal else "02",
            0,
            assets.sha256,
            assets.sha256,
            "engineering-smoke-1" if proposal else "q0",
            proposal.assets.sha256 if proposal else assets.sha256,
            12 if proposal else 0,
            20260909 if proposal else 0,
            1 if proposal else 0,
            samples,
            "ocr_selection" if args.action == "generate" else "candidate_development",
            args.device_id,
            tuple(dependencies),
            {
                "software": inventory["runtime"],
                "kernel": "torchao-native-dynamic-a4-block16",
                "sampler": "Mage-static-shift6-euler",
                "generation": generation.identity(),
                "device": {"id": args.device_id, "nvidia_identity": hardware},
                "input_inventory_digest": digest_json(paths),
                "implementation_sha256": implementation_hashes(),
                "execution_action": args.action,
                "candidate_receipt_digest": digest_json(proposal.receipt) if proposal else None,
                "mutation_count_purpose": "engineering-smoke-only" if proposal else None,
            },
            manifest.digest,
            evaluation_mode="generation" if args.action == "generate" else "replay",
        )
        receipt = preflight(request, manifest, expected_center_version=0)
        payload = {
            "request": asdict(request),
            "preflight": receipt.to_dict(),
            "fixed_state_sha256": assets.fixed_sha256,
            "layers": len(assets.layers),
            "remainder_tensors": len(assets.remainder),
            "host_cpu_seconds": time.process_time() - cpu,
            "wall_seconds": time.time() - started,
            "device_occupied": False,
            "comparison": {
                "same_device_max_abs_pixel_error": 0,
                "cross_device_max_abs_pixel_error": 0,
                "require_identical_png": True,
            },
            "grant_transport": "authenticated LAN SSH; worker consumes coordinator lease, never a second ledger",
        }
        (job / "prepared.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({k: v for k, v in payload.items() if k not in {"request", "comparison"}}))
        return
    if assets.sha256 != request.q0_digest:
        raise ValueError("q0 changed after preflight")
    by_id = {r.prompt_id: r for r in manifest.records}
    samples = [
        {"sample_id": s.sample_id, "prompt": by_id[s.prompt_id].prompt, "seed": s.seed}
        for s in request.samples
    ]

    def stage(name):
        if time.time() >= lease.hard_deadline - 30:
            raise TimeoutError("device hard deadline cleanup reserve")
        with (job / "stages.jsonl").open("a") as stream:
            stream.write(
                json.dumps({"stage": name, "at": time.time(), "lease_id": lease.lease_id}) + "\n"
            )
        print(name, flush=True)

    runtime = build_native_pipeline(
        assets,
        auxiliary_root=inventory["auxiliary_root"],
        allowed_assets=inventory["allowed_assets"],
        generation_config=generation,
        expected_runtime=inventory["runtime"],
        before_stage=stage,
    )
    first = runtime.generate(samples, job / "baseline")
    if not first["qualified"]:
        raise RuntimeError("native baseline did not qualify; inspect baseline/report.json")
    if request.runtime_identity["execution_action"] == "generate":
        if (
            request.purpose != "ocr_selection"
            or request.ticket_id != "04"
            or request.candidate_digest != assets.sha256
        ):
            raise ValueError("generation-only probe requires a q0 OCR selection grant")
        verdict = {
            "schema": "mage-ocr-selection-generation/v1",
            "scope": "ocr-selection-images-only",
            "lease_id": lease.lease_id,
            "q0_sha256": assets.sha256,
            "fixed_state_sha256": assets.fixed_sha256,
            "qualified": first["qualified"],
            "generation_report": str(job / "baseline/report.json"),
            "image_count": len(first["rows"]),
            "recognition_performed": False,
            "quality_improvement_claimed": False,
            "pid": os.getpid(),
            "wall_seconds": time.time() - started,
        }
        (job / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
        print(json.dumps(verdict), flush=True)
        return
    if request.runtime_identity["execution_action"] == "mutation_replay":
        receipt = json.loads((job / "candidate.json").read_text())
        if digest_json(receipt) != request.runtime_identity["candidate_receipt_digest"]:
            raise ValueError("candidate receipt changed after request freeze")
        candidate = replay_candidate(assets, receipt)
        if (
            candidate.sha256 != request.candidate_digest
            or receipt["actual_changes"] != request.effective_edits
        ):
            raise ValueError("candidate differs from granted request")
        stage("candidate_install")
        runtime.installation = install_quantized_assets(
            runtime.pipeline.model.transformer, candidate, device="cuda"
        )
        runtime.assets = candidate
        replay = runtime.replay(samples, job / "candidate-replay")
        stage("q0_restore_install")
        runtime.installation = install_quantized_assets(
            runtime.pipeline.model.transformer, assets, device="cuda"
        )
        runtime.assets = assets
        restored = runtime.generate(samples, job / "q0-restored")
        verdict = {
            "schema": "mage-mutation-probe/v1",
            "scope": "single-code-engineering-smoke-only",
            "lease_id": lease.lease_id,
            "q0_sha256": assets.sha256,
            "candidate_sha256": candidate.sha256,
            "actual_changes": receipt["actual_changes"],
            "fixed_state_sha256": assets.fixed_sha256,
            "candidate_replay": replay,
            "q0_restore_images_equal": pngs_equal(first, restored),
            "qualified": replay["qualified"] and pngs_equal(first, restored),
            "quality_improvement_claimed": False,
            "pid": os.getpid(),
            "wall_seconds": time.time() - started,
        }
        (job / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
        print(json.dumps({k: v for k, v in verdict.items() if k != "candidate_replay"}), flush=True)
        return
    if request.runtime_identity["execution_action"] != "native_replay":
        raise ValueError("unknown granted execution action")
    stage("zero_update_install")
    runtime.installation = install_quantized_assets(
        runtime.pipeline.model.transformer, assets, device="cuda"
    )
    replay = runtime.replay(samples, job / "replay")
    zero_equal = pngs_equal(first, json.loads((job / "replay/before/report.json").read_text()))
    verdict = {
        "schema": "mage-native-probe/v1",
        "scope": "development-replay-only",
        "lease_id": lease.lease_id,
        "q0_sha256": assets.sha256,
        "fixed_state_sha256": assets.fixed_sha256,
        "zero_update_images_equal": zero_equal,
        "replay": replay,
        "qualified": zero_equal and replay["qualified"],
        "pid": os.getpid(),
        "wall_seconds": time.time() - started,
    }
    (job / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps({k: v for k, v in verdict.items() if k != "replay"}), flush=True)


if __name__ == "__main__":
    main()
