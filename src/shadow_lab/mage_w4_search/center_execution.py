"""Full-center file wiring for development jobs; numerical code remains frozen.

GB10 owns packed preparation, coordinator state and all numerical evaluation.
The Mac only serves the unique occupancy ledger and may verify byte archives.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

from .contracts import ArtifactIdentity, CandidateRequest, DataManifest, digest_json, preflight
from .coordinator import CandidateAssignment, DevelopmentCoordinator, FullCenter


def decode_full_center(value):
    return FullCenter(
        value["native_payload_sha256"], tuple(ArtifactIdentity(**row) for row in value["artifacts"])
    )


def _artifact(role, path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    after = path.stat()
    if any(
        getattr(before, key) != getattr(after, key)
        for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise ValueError("center artifact changed while hashing")
    return ArtifactIdentity(role, sha.hexdigest(), after.st_size, str(path))


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(
            value,
            stream,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )


def _fixed_descriptions(assets):
    return {
        "weight_scales": {"fixed_state_sha256": assets.fixed_sha256},
        "layer_manifest": {name: list(layer.shape) for name, layer in assets.layers.items()},
    }


def _check_fixed(assets, center):
    artifacts = {row.role: row for row in center.artifacts}
    for role, expected in _fixed_descriptions(assets).items():
        if json.loads(Path(artifacts[role].path).read_text()) != expected:
            raise ValueError(f"center changed frozen {role}")


def register_q0_center(
    packed_path, remainder_path, directory, *, allowed_assets, expected_native_sha256
):
    """Register existing q0 bytes and write the exact fixed descriptors used by 05."""
    from .native import load_quantized_assets

    assets = load_quantized_assets(packed_path, remainder_path, allowed_assets=allowed_assets)
    if assets.sha256 != expected_native_sha256:
        raise ValueError("q0 canonical identity differs from the fixed scale origin")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    artifacts = [
        _artifact("quantized_packed", packed_path),
        _artifact("quantized_remainder", remainder_path),
    ]
    for role, value in _fixed_descriptions(assets).items():
        path = directory / f"{role}.json"
        _write(path, value)
        artifacts.append(_artifact(role, path))
    center = FullCenter(assets.sha256, tuple(artifacts))
    center.verify_files()
    _write(directory / "full-center.json", asdict(center))
    return center


def load_full_center(center):
    """Deserialize verified packed/remainder only; no teacher, model or CUDA import."""
    from .native import load_quantized_assets

    center.verify_files()
    artifacts = {row.role: row for row in center.artifacts}
    assets = load_quantized_assets(
        artifacts["quantized_packed"].path,
        artifacts["quantized_remainder"].path,
        allowed_assets={row.path: row.sha256 for row in center.artifacts},
        expected_shapes=json.loads(Path(artifacts["layer_manifest"].path).read_text()),
    )
    if assets.sha256 != center.digest:
        raise ValueError("full-center canonical identity differs from its actual packed payload")
    _check_fixed(assets, center)
    return assets


def materialize_candidate(assets, directory, *, fixed_center):
    """Write complete packed bytes; preserve original fixed-file bytes without reserialization."""
    from ..mage_qad_checkpoint import atomic_save, content_hash

    fixed_center.verify_files()
    assets.verify()
    _check_fixed(assets, fixed_center)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    payload = {"schema_version": 1, "packed": assets.record()["packed"]}
    packed = directory / "packed.pt"
    atomic_save({**payload, "sha256": content_hash(payload)}, packed)
    center = FullCenter(
        assets.sha256,
        tuple(
            _artifact("quantized_packed", packed) if row.role == "quantized_packed" else row
            for row in fixed_center.artifacts
        ),
    )
    if load_full_center(center).sha256 != assets.sha256:
        raise ValueError("saved candidate failed native canonical reload")
    _write(directory / "full-center.json", asdict(center))
    return center


def prepare_center_job(
    run_id, plan, manifest, *, shared, q0, current, evaluated, template, candidate_receipt=None
):
    """Bind versioned real centers to one complete ticket 06/08 request.

    File locations belong to the job's center manifests. Request dependencies
    declare their bytes without host paths, allowing coordinator-local archives.
    """
    from .evaluation_worker import prepare_evaluation_job

    if (
        set(plan) != {"schema", "ticket_id", "center_version", "purpose", "samples", "candidate"}
        or plan["schema"] != "mage-w4-center-plan/v1"
        or plan["ticket_id"] not in {"06", "08"}
        or plan["purpose"] != "candidate_development"
        or type(plan["center_version"]) is not int
        or plan["center_version"] < 0
    ):
        raise ValueError("explicit ticket 06/08 development center plan required")
    if q0.digest != shared["q0_digest"] or (
        plan["center_version"] == 0 and current.digest != q0.digest
    ):
        raise ValueError("center version/q0 origin mismatch")
    fixed = {
        (row.role, row.sha256, row.size_bytes)
        for row in q0.artifacts
        if row.role != "quantized_packed"
    }
    for center in (q0, current, evaluated):
        if (
            len(center.artifacts) != 4
            or {row.role for row in center.artifacts}
            != {"quantized_packed", "quantized_remainder", "weight_scales", "layer_manifest"}
            or {
                (row.role, row.sha256, row.size_bytes)
                for row in center.artifacts
                if row.role != "quantized_packed"
            }
            != fixed
        ):
            raise ValueError("complete centers must preserve frozen q0 state artifacts")
    kind = plan["candidate"]["kind"]
    if kind not in {"q0", "current", "adjacent"}:
        raise ValueError("center job requires q0/current/adjacent payload")
    expected = q0.digest if kind == "q0" else current.digest
    if kind == "adjacent":
        if not candidate_receipt:
            raise ValueError("adjacent payload receipt required")
        expected = candidate_receipt["candidate_sha256"]
    if evaluated.digest != expected:
        raise ValueError("evaluated payload differs from reference or candidate receipt")
    artifacts = {row.role: row for row in evaluated.artifacts}
    dependencies = [
        asdict(replace(artifacts.get(row["role"], ArtifactIdentity(**row)), path=None))
        for row in shared["dependencies"]
    ]
    shared = {**shared, "dependencies": dependencies}
    # Reuse the original strict candidate/receipt validation, with the actual
    # current center as its mutation base; then bind the independent fixed q0.
    mapped = {key: plan[key] for key in ("purpose", "samples", "candidate")}
    mapped["schema"] = "mage-w4-evaluation-plan/v1"
    if kind == "current":
        mapped["candidate"] = {**plan["candidate"], "kind": "q0"}
    validated = prepare_evaluation_job(
        run_id,
        mapped,
        manifest,
        shared={**shared, "q0_digest": current.digest},
        candidate_receipt=candidate_receipt,
    )
    job = json.loads(
        json.dumps(
            {
                "schema": "mage-w4-center-job/v1",
                "run_id": run_id,
                "plan": plan,
                "shared": shared,
                "candidate_receipt": candidate_receipt,
                "template": template,
                "centers": {
                    "q0": asdict(q0),
                    "current": asdict(current),
                    "evaluated": asdict(evaluated),
                },
            },
            allow_nan=False,
        )
    )
    request = replace(
        CandidateRequest.from_dict(validated["request"]),
        request_id="evaluation-" + digest_json(job),
        ticket_id=plan["ticket_id"],
        center_version=plan["center_version"],
        center_digest=current.digest,
        q0_digest=q0.digest,
        candidate_digest=evaluated.digest,
    )
    checked = preflight(request, manifest, expected_center_version=plan["center_version"])
    if checked.status != "ready_for_reservation":
        raise ValueError(checked.to_dict())
    return {
        "request": asdict(request),
        "preflight": checked.to_dict(),
        "evaluation_job": job,
        "evaluation_job_sha256": digest_json(job),
    }


def prepare_center_payload(plan, *, q0, current, directory):
    """Create one finite proposal from the actual current packed checkpoint on CPU."""
    from .mutation import build_layer_groups, propose_candidate

    candidate = plan["candidate"]
    if candidate["kind"] in {"q0", "current"}:
        selected = q0 if candidate["kind"] == "q0" else current
        load_full_center(selected)
        return selected, None
    if candidate["kind"] != "adjacent":
        raise ValueError("explicit q0/current/adjacent candidate required")
    assets = load_full_center(current)
    _check_fixed(assets, q0)
    proposal = propose_candidate(
        assets,
        build_layer_groups(assets),
        group_id="outside" if candidate["group_id"] == 12 else f"block_{candidate['group_id']:02d}",
        mutation_count=candidate["mutation_count"],
        seed=candidate["seed"],
        candidate_index=candidate["candidate_index"],
        max_attempts=candidate["max_attempts"],
    )
    if proposal.assets is None:
        raise ValueError(f"finite candidate preparation failed: {proposal.receipt['status']}")
    return materialize_candidate(proposal.assets, directory, fixed_center=q0), proposal.receipt


def load_execution_center(job, request, stage):
    """Load the authorized full candidate and independently replay its current-center edits."""
    from .mutation import replay_candidate

    template = job["template"]
    if _artifact("template", template["path"]).sha256 != template["sha256"]:
        raise ValueError("frozen preparation template changed")
    centers = {name: decode_full_center(value) for name, value in job["centers"].items()}
    for center in centers.values():
        center.verify_files()
    if (
        centers["q0"].digest != request.q0_digest
        or centers["current"].digest != request.center_digest
        or centers["evaluated"].digest != request.candidate_digest
    ):
        raise ValueError("execution center differs from the granted request")
    stage("quantized_asset_load")
    assets = load_full_center(centers["evaluated"])
    if assets.fixed_sha256 != job["shared"]["fixed_state_sha256"]:
        raise ValueError("quantized center changed the frozen state")
    if job["candidate_receipt"] is not None:
        stage("candidate_replay")
        current = load_full_center(centers["current"])
        replayed = replay_candidate(current, job["candidate_receipt"])
        if replayed.sha256 != assets.sha256:
            raise ValueError("saved candidate differs from current-center receipt replay")
    return assets


def prepare_center_on_gb10(
    run_id, device_id, plan, *, template_path, q0_path, current_path, candidate_directory
):
    """Prepare against an existing fixed-runtime job; keep current-specific files out of runtime."""
    from .evaluation_worker import implementation_hashes, verify_evaluation_inputs

    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise ValueError("real center preparation requires a GB10 Linux ARM host")
    if device_id not in {"gooo", "goooinext"}:
        raise ValueError("unsupported pilot device")
    template_artifact = _artifact("template", template_path)
    source = json.loads(Path(template_path).read_text())
    if _artifact("template", template_path) != template_artifact:
        raise ValueError("preparation template changed while reading")
    shared = source["evaluation_job"]["shared"]
    shared["implementation_sha256"] = implementation_hashes()
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"], text=True
    ).strip()
    if "NVIDIA GB10" not in hardware:
        raise ValueError("preparation requires the known GB10 device")
    shared["runtime_identity"]["device"] = {"id": device_id, "nvidia_identity": hardware}
    verify_evaluation_inputs(shared)
    q0 = decode_full_center(json.loads(Path(q0_path).read_text()))
    current = decode_full_center(json.loads(Path(current_path).read_text()))
    q0_assets = load_full_center(q0)
    if q0.digest != shared["q0_digest"] or q0_assets.fixed_sha256 != shared["fixed_state_sha256"]:
        raise ValueError("q0 differs from the fixed numerical template")
    declared = {row["role"]: (row["sha256"], row["size_bytes"]) for row in shared["dependencies"]}
    for row in q0.artifacts:
        if row.role != "quantized_packed" and declared.get(row.role) != (
            row.sha256,
            row.size_bytes,
        ):
            raise ValueError("fixed state bytes differ from the calibrated template")
    del q0_assets
    evaluated, receipt = prepare_center_payload(
        plan, q0=q0, current=current, directory=candidate_directory
    )
    manifest = DataManifest.from_dict(
        json.loads((Path(shared["run_root"]) / "data-v1/manifest.json").read_text())
    )
    return prepare_center_job(
        run_id,
        plan,
        manifest,
        shared=shared,
        q0=q0,
        current=current,
        evaluated=evaluated,
        candidate_receipt=receipt,
        template={"path": template_artifact.path, "sha256": template_artifact.sha256},
    )


def map_center_archive(center, paths):
    """Map already downloaded byte-identical files; never deserialize tensors on the Mac."""
    if set(paths) != {row.role for row in center.artifacts}:
        raise ValueError("archive mapping must cover exactly the four center roles")
    archived = FullCenter(
        center.digest,
        tuple(replace(row, path=str(Path(paths[row.role]).resolve())) for row in center.artifacts),
    )
    archived.verify_files()
    return archived


def assignment_from_prepared(prepared, manifest, archived_center):
    """Validate a frozen remote request against a local byte archive without rewriting it."""
    from .evaluation_worker import reconstruct_evaluation_job

    if digest_json(reconstruct_evaluation_job(prepared["evaluation_job"], manifest)) != digest_json(
        prepared
    ):
        raise ValueError("prepared center job identity mismatch")
    request = CandidateRequest.from_dict(prepared["request"])
    archived_center.verify_files()
    if request.candidate_digest != archived_center.digest:
        raise ValueError("archived center is not the frozen candidate")
    declared = {row.role: (row.sha256, row.size_bytes) for row in request.dependencies}
    if any(
        declared.get(row.role) != (row.sha256, row.size_bytes) for row in archived_center.artifacts
    ):
        raise ValueError("request state artifacts differ from the local archive")
    return CandidateAssignment(request, manifest, archived_center)


def open_coordinator_config(config, *, budget, require_existing=True):
    """Open coordinator state with the caller's one authoritative ledger; no budget constructor."""
    from .development import decode_scale_receipt

    if set(config) != {"schema", "state_path", "q0_path", "scale_receipt_path", "rng_seed"} or (
        config["schema"] != "mage-w4-coordinator-config/v1"
    ):
        raise ValueError("explicit development coordinator configuration required")
    if require_existing and not Path(config["state_path"]).is_file():
        raise ValueError("coordinator must be initialized before a candidate launch")
    q0 = decode_full_center(json.loads(Path(config["q0_path"]).read_text()))
    scales = decode_scale_receipt(json.loads(Path(config["scale_receipt_path"]).read_text()))
    return DevelopmentCoordinator(
        config["state_path"], q0=q0, scales=scales, budget=budget, rng_seed=config["rng_seed"]
    )
