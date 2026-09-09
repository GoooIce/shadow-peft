"""One authenticated request produces one complete native development score batch.

Preparation freezes metadata; only a coordinator lease can run models. Job
identity belongs to request_id, leaving the fixed runtime identical across q0
scale calibration, q0 references and candidate batches.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import resource
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

from .budget import DeviceLease
from .contracts import ArtifactIdentity, CandidateRequest, RequestedSample, digest_json, preflight
from .evaluation import GenerationBinding, evaluate_generated_candidate
from .execution import validate_worker_grant
from .human_transcription import read_json
from .scorers import decode_ocr_config


def prepare_evaluation_job(run_id, plan, manifest, *, shared, candidate_receipt=None):
    """Freeze one explicit sample plan and a q0/adjacent candidate into a request."""
    if not re.fullmatch(r"[a-z0-9-]+", run_id):
        raise ValueError("evaluation run ID must contain lowercase letters, digits and hyphens")
    if set(plan) != {"schema", "purpose", "samples", "candidate"} or (
        plan["schema"] != "mage-w4-evaluation-plan/v1"
        or plan["purpose"] not in {"scale_calibration", "candidate_development"}
    ):
        raise ValueError("explicit evaluation plan schema/purpose required")
    candidate = plan["candidate"]
    identity, seed, edits = shared["q0_digest"], 0, 0
    if candidate["kind"] == "q0":
        if set(candidate) != {"kind", "candidate_id", "group_id"} or candidate_receipt is not None:
            raise ValueError("q0 cannot carry candidate edits")
    else:
        if plan["purpose"] == "scale_calibration":
            raise ValueError("scale calibration requires q0")
        if (
            candidate["kind"] != "adjacent"
            or set(candidate)
            != {
                "kind",
                "candidate_id",
                "group_id",
                "mutation_count",
                "seed",
                "candidate_index",
                "max_attempts",
            }
            or not isinstance(candidate_receipt, dict)
        ):
            raise ValueError("adjacent candidate specification and receipt required")
        if any(
            type(candidate[name]) is not int or candidate[name] < minimum
            for name, minimum in (
                ("mutation_count", 1),
                ("seed", 0),
                ("candidate_index", 0),
                ("max_attempts", 1),
            )
        ):
            raise ValueError("candidate requires explicit positive edits and bounded attempts")
        expected = {
            "status": "candidate",
            "base_sha256": shared["q0_digest"],
            "fixed_sha256": shared["fixed_state_sha256"],
            "group_id": "outside"
            if candidate["group_id"] == 12
            else f"block_{candidate['group_id']:02d}",
            "actual_changes": candidate["mutation_count"],
            "requested_changes": candidate["mutation_count"],
            "seed": candidate["seed"],
            "candidate_index": candidate["candidate_index"],
            "max_attempts": candidate["max_attempts"],
        }
        if any(candidate_receipt.get(key) != value for key, value in expected.items()):
            raise ValueError("candidate receipt differs from explicit plan")
        identity, seed, edits = (
            candidate_receipt["candidate_sha256"],
            candidate["seed"],
            candidate["mutation_count"],
        )
        if identity == shared["q0_digest"]:
            raise ValueError("candidate cannot be a q0 no-op")
    job = json.loads(
        json.dumps(
            {
                "schema": "mage-w4-evaluation-job/v1",
                "run_id": run_id,
                "plan": plan,
                "shared": shared,
                "candidate_receipt": candidate_receipt,
            },
            allow_nan=False,
        )
    )
    request = CandidateRequest(
        request_id="evaluation-" + digest_json(job),
        run_id=run_id,
        ticket_id="05",
        center_version=0,
        center_digest=shared["q0_digest"],
        q0_digest=shared["q0_digest"],
        candidate_id=candidate["candidate_id"],
        candidate_digest=identity,
        group_id=candidate["group_id"],
        mutation_seed=seed,
        effective_edits=edits,
        samples=tuple(RequestedSample(**row) for row in plan["samples"]),
        purpose=plan["purpose"],
        device_id=shared["runtime_identity"]["device"]["id"],
        dependencies=tuple(ArtifactIdentity(**row) for row in shared["dependencies"]),
        runtime_identity=job["shared"]["runtime_identity"],
        manifest_digest=manifest.digest,
        evaluation_mode="complete",
    )
    checked = preflight(request, manifest, expected_center_version=0)
    if checked.status != "ready_for_reservation":
        raise ValueError(checked.to_dict())
    return {
        "request": asdict(request),
        "preflight": checked.to_dict(),
        "evaluation_job": job,
        "evaluation_job_sha256": digest_json(job),
    }


def _write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)


def reconstruct_evaluation_job(job, manifest):
    """Replay the original schema's authenticated preparation without reading model files."""
    if job["schema"] == "mage-w4-center-job/v1":
        from .center_execution import decode_full_center, prepare_center_job

        return prepare_center_job(
            job["run_id"],
            job["plan"],
            manifest,
            shared=job["shared"],
            **{name: decode_full_center(value) for name, value in job["centers"].items()},
            template=job["template"],
            candidate_receipt=job["candidate_receipt"],
        )
    if job["schema"] != "mage-w4-evaluation-job/v1":
        raise ValueError("unsupported complete evaluation job schema")
    return prepare_evaluation_job(
        job["run_id"],
        job["plan"],
        manifest,
        shared=job["shared"],
        candidate_receipt=job["candidate_receipt"],
    )


def _file_sha(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verify_scoring_inputs(inputs):
    """Read only explicit frozen local artifacts; this never imports a model or downloads."""
    required = {
        "schema",
        "clip_checkpoint",
        "hps_checkpoint",
        "ocr_lock",
        "ocr_calibration",
        "ocr_model_root",
        "ocr_python_executable",
        "visual_batch_size",
        "generation",
    }
    if set(inputs) != required or inputs["schema"] != "mage-w4-evaluation-inputs/v1":
        raise ValueError("explicit scoring input schema required")
    if type(inputs["visual_batch_size"]) is not int or inputs["visual_batch_size"] < 1:
        raise ValueError("positive frozen visual batch size required")
    paths = {}
    for role in ("clip_checkpoint", "hps_checkpoint", "ocr_lock", "ocr_calibration"):
        reference = inputs[role]
        if set(reference) != {"path", "sha256", "size_bytes"}:
            raise ValueError("scoring artifact requires explicit path/byte SHA/size")
        path = Path(reference["path"]).resolve(strict=True)
        before = path.stat()
        sha = _file_sha(path)
        after = path.stat()
        if (
            sha != reference["sha256"]
            or after.st_size != reference["size_bytes"]
            or any(
                getattr(before, key) != getattr(after, key)
                for key in (
                    "st_ino",
                    "st_dev",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
            )
        ):
            raise ValueError(f"scoring artifact changed: {role}")
        paths[role] = path
    from .development import decode_calibration_report
    from .ocr import freeze_scorer

    lock = read_json(paths["ocr_lock"])
    calibration = decode_calibration_report(read_json(paths["ocr_calibration"]))
    if digest_json(freeze_scorer(calibration)) != digest_json(lock):
        raise ValueError("frozen OCR lock/calibration mismatch")
    config = decode_ocr_config(lock["config"])
    if config.sha256 != lock["config_sha256"]:
        raise ValueError("OCR selected config identity mismatch")
    for role in ("ocr_lock", "ocr_calibration"):
        if _file_sha(paths[role]) != inputs[role]["sha256"]:
            raise ValueError(f"scoring artifact changed during decode: {role}")
    ocr_python = Path(inputs["ocr_python_executable"]).absolute()
    if not ocr_python.is_file():
        raise ValueError("existing isolated OCR Python executable required")
    return {
        "ocr_config": config,
        "ocr_lock": lock,
        "ocr_calibration": calibration,
        "clip_checkpoint": paths["clip_checkpoint"],
        "hps_checkpoint": paths["hps_checkpoint"],
        "ocr_model_root": str(Path(inputs["ocr_model_root"]).resolve(strict=True)),
        # A venv's symlink path selects its environment; resolving it would run base Python.
        "ocr_python_executable": str(ocr_python),
    }


def implementation_hashes():
    """Freeze the explicit runner and numerical/validation implementation sources."""
    repo = Path(__file__).resolve().parents[3]
    paths = [
        repo / "scripts/mage_w4_evaluate.py",
        repo / "scripts/mage_w4_launch.py",
        repo / "scripts/mage_w4_center.py",
        repo / "scripts/mage_w4_coordinate.py",
    ]
    paths += [
        Path(__file__).parent / name
        for name in (
            "__init__.py",
            "evaluation_worker.py",
            "center_execution.py",
            "budget_rpc.py",
            "evaluation.py",
            "native.py",
            "mutation.py",
            "contracts.py",
            "execution.py",
            "budget.py",
            "rewards.py",
            "ocr.py",
            "scorers.py",
            "development.py",
            "human_transcription.py",
            "coordinator.py",
            "pilot.py",
        )
    ]
    paths += [
        Path(__file__).parent.parent / name
        for name in (
            "__init__.py",
            "mage_qad_checkpoint.py",
            "nvfp4_runtime.py",
            "quasar.py",
            "es_quant.py",
            "image_eval.py",
            "image_reward_models.py",
        )
    ]
    return {str(path.relative_to(repo)): _file_sha(path) for path in paths}


def numerical_implementation_hashes():
    """Only these fixed numerical implementations determine cross-batch compatibility.

    Every other runner/controller source remains frozen per authenticated job;
    evolving orchestration cannot silently change this explicit numerical set.
    """
    sources = implementation_hashes()
    paths = [
        f"src/shadow_lab/mage_w4_search/{name}"
        for name in (
            "native.py",
            "mutation.py",
            "evaluation.py",
            "ocr.py",
            "scorers.py",
            "rewards.py",
        )
    ]
    paths += [
        f"src/shadow_lab/{name}"
        for name in (
            "mage_qad_checkpoint.py",
            "nvfp4_runtime.py",
            "quasar.py",
            "es_quant.py",
            "image_eval.py",
            "image_reward_models.py",
        )
    ]
    return {path: sources[path] for path in paths}


def _visual_versions():
    from importlib.metadata import version

    return {name: version(name) for name in ("torch", "open-clip-torch", "hpsv2")}


def package_runtime_sources(package_roots):
    """Fingerprint package code and inference configuration/vocabulary with portable keys."""
    result = {}
    for package, root in sorted(package_roots.items()):
        root = Path(root).resolve(strict=True)
        files = {}
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if not path.is_file() or any(
                part.startswith(".") or part in {"__pycache__", "cache", "caches"}
                for part in relative.parts
            ):
                continue
            configuration = path.suffix == ".json" and "model_configs" in relative.parts
            vocabulary = (
                path.name.startswith(("bpe_", "vocab", "tokenizer", "merges"))
                and path.suffix in {".json", ".txt", ".gz", ".model"}
            ) or path.name == "special_tokens_map.json"
            if path.suffix != ".py" and not configuration and not vocabulary:
                continue
            if not path.resolve(strict=True).is_relative_to(root):
                raise ValueError("external package source escapes its declared package root")
            files[relative.as_posix()] = _file_sha(path)
        if not files:
            raise ValueError(
                f"installed scoring package has no auditable runtime sources: {package}"
            )
        result[package] = files
    return result


def _installed_external_sources():
    # Looking up a top-level package spec locates editable sources without
    # importing its __init__, model implementations or optional GUI modules.
    from importlib.util import find_spec

    roots = {}
    for name in ("hpsv2", "open_clip"):
        spec = find_spec(name)
        if (
            spec is None
            or not spec.submodule_search_locations
            or len(spec.submodule_search_locations) != 1
        ):
            raise ValueError(f"one explicit installed scoring package root required: {name}")
        roots[name] = next(iter(spec.submodule_search_locations))
    return package_runtime_sources(roots)


def prepare_on_gb10(run_root, run_id, device_id, plan, inputs):
    """CPU preparation uses existing packed assets; it neither loads Mage nor starts CUDA."""
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise ValueError("real payload preparation requires a GB10 Linux ARM host")
    if device_id not in {"gooo", "goooinext"}:
        raise ValueError("unsupported pilot device")
    from .contracts import DataManifest
    from .mutation import build_layer_groups, propose_candidate
    from .native import GenerationConfig, load_quantized_assets

    root = Path(run_root).resolve(strict=True)
    inventory = read_json(root / "inventory.json")
    manifest = DataManifest.from_dict(read_json(root / "data-v1/manifest.json"))
    scoring = verify_scoring_inputs(inputs)
    generation = GenerationConfig(**inputs["generation"])
    if generation != GenerationConfig(watermark_key=20260909):
        raise ValueError("05 must preserve the proven native q0 generation configuration")
    packed = (root / "input/qad-nvfp4-best-673.pt").resolve(strict=True)
    remainder = (root / "input/qad-transformer-remainder-best-673.pt").resolve(strict=True)
    assets = load_quantized_assets(packed, remainder, allowed_assets=inventory["allowed_assets"])
    dependencies = [
        ArtifactIdentity(role, inventory["allowed_assets"][str(path)], path.stat().st_size)
        for role, path in (("quantized_packed", packed), ("quantized_remainder", remainder))
    ]
    paths = inventory["allowed_assets"]
    descriptions = {
        "weight_scales": {"fixed_state_sha256": assets.fixed_sha256},
        "layer_manifest": {name: layer.shape for name, layer in assets.layers.items()},
        "text_encoder": {p: h for p, h in paths.items() if "/auxiliary/text_encoder/" in p},
        "vae": {p: h for p, h in paths.items() if "/auxiliary/vae/" in p},
    }
    dependencies += [
        ArtifactIdentity(
            role,
            digest_json(value),
            len(
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode()
            ),
        )
        for role, value in descriptions.items()
    ]
    for axis in ("clip", "hps"):
        reference = inputs[f"{axis}_checkpoint"]
        dependencies.append(
            ArtifactIdentity(f"{axis}_scorer", reference["sha256"], reference["size_bytes"])
        )
    dependencies.append(
        ArtifactIdentity(
            "ocr_scorer",
            scoring["ocr_config"].sha256,
            # OCR's existing typed identity intentionally uses the default JSON separators.
            len(
                json.dumps(
                    asdict(scoring["ocr_config"]),
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode()
            ),
        )
    )
    hardware = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if "NVIDIA GB10" not in hardware:
        raise ValueError("preparation requires the known GB10 device")
    runtime = {
        "software": inventory["runtime"],
        "kernel": "torchao-native-dynamic-a4-block16",
        "sampler": "Mage-static-shift6-euler",
        "generation": generation.identity(),
        "device": {"id": device_id, "nvidia_identity": hardware},
        "input_inventory_digest": digest_json(inventory["allowed_assets"]),
        "numerical_implementation_sha256": numerical_implementation_hashes(),
        "external_source_sha256": _installed_external_sources(),
        "visual_scoring": {
            "package_versions": _visual_versions(),
            "adapter_source_sha256": _file_sha(
                Path(__file__).parent.parent / "image_reward_models.py"
            ),
            "batch_size": inputs["visual_batch_size"],
            "numerical_mode": "existing-bfloat16-dot-v1",
        },
        "scoring_artifacts": {
            role: {key: inputs[role][key] for key in ("sha256", "size_bytes")}
            for role in ("clip_checkpoint", "hps_checkpoint", "ocr_lock", "ocr_calibration")
        },
    }
    shared = {
        "q0_digest": assets.sha256,
        "fixed_state_sha256": assets.fixed_sha256,
        "runtime_identity": runtime,
        "implementation_sha256": implementation_hashes(),
        "dependencies": [asdict(row) for row in dependencies],
        "inputs": inputs,
        "inventory": inventory,
        "run_root": str(root),
    }
    receipt = None
    candidate = plan["candidate"]
    if candidate["kind"] == "adjacent":
        proposal = propose_candidate(
            assets,
            build_layer_groups(assets),
            group_id="outside"
            if candidate["group_id"] == 12
            else f"block_{candidate['group_id']:02d}",
            mutation_count=candidate["mutation_count"],
            seed=candidate["seed"],
            candidate_index=candidate["candidate_index"],
            max_attempts=candidate["max_attempts"],
        )
        receipt = proposal.receipt
        if proposal.assets is None:
            raise ValueError(f"finite candidate preparation failed: {receipt['status']}")
    return prepare_evaluation_job(run_id, plan, manifest, shared=shared, candidate_receipt=receipt)


def verify_evaluation_inputs(shared):
    """Recheck authenticated implementation, fixed runtime and scoring inputs without models."""
    runtime, inputs, inventory = shared["runtime_identity"], shared["inputs"], shared["inventory"]
    if shared["implementation_sha256"] != implementation_hashes():
        raise ValueError("evaluation implementation changed after preparation")
    if runtime["numerical_implementation_sha256"] != numerical_implementation_hashes():
        raise ValueError("frozen numerical implementation changed")
    if runtime["external_source_sha256"] != _installed_external_sources():
        raise ValueError("installed scoring package source changed after preparation")
    if runtime["visual_scoring"]["package_versions"] != _visual_versions():
        raise ValueError("visual scorer runtime changed after preparation")
    if (
        runtime["input_inventory_digest"] != digest_json(inventory["allowed_assets"])
        or runtime["software"] != inventory["runtime"]
    ):
        raise ValueError("frozen native inventory identity mismatch")
    return verify_scoring_inputs(inputs)


def _generate_real(job, request, manifest, directory, stage):
    import gc

    from .mutation import replay_candidate
    from .native import GenerationConfig, build_native_pipeline, load_quantized_assets

    shared = job["shared"]
    root, inventory = Path(shared["run_root"]), shared["inventory"]
    if job["schema"] == "mage-w4-center-job/v1":
        from .center_execution import load_execution_center

        assets = load_execution_center(job, request, stage)
    else:
        stage("quantized_asset_load")
        assets = load_quantized_assets(
            root / "input/qad-nvfp4-best-673.pt",
            root / "input/qad-transformer-remainder-best-673.pt",
            allowed_assets=inventory["allowed_assets"],
        )
        if (
            assets.sha256 != request.q0_digest
            or assets.fixed_sha256 != shared["fixed_state_sha256"]
        ):
            raise ValueError("quantized q0/fixed state changed after preparation")
        if job["candidate_receipt"] is not None:
            stage("candidate_replay")
            assets = replay_candidate(assets, job["candidate_receipt"])
    if assets.sha256 != request.candidate_digest:
        raise ValueError("quantized candidate differs from authorized request")
    generation = GenerationConfig(**shared["inputs"]["generation"])
    if generation.identity() != request.runtime_identity["generation"]:
        raise ValueError("generation configuration differs from authorized request")
    pipeline = None
    try:
        pipeline = build_native_pipeline(
            assets,
            auxiliary_root=inventory["auxiliary_root"],
            allowed_assets=inventory["allowed_assets"],
            generation_config=generation,
            expected_runtime=inventory["runtime"],
            before_stage=stage,
        )
        by_id = {row.prompt_id: row for row in manifest.records}
        return pipeline.generate(
            [
                {
                    "sample_id": sample.sample_id,
                    "prompt": by_id[sample.prompt_id].prompt,
                    "seed": sample.seed,
                }
                for sample in request.samples
            ],
            directory,
        )
    finally:
        del pipeline
        gc.collect()
        import torch

        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def run_evaluation_job(
    job_directory,
    manifest,
    *,
    device_id,
    fixture_generation=None,
    fixture_scorers=None,
):
    """Execute one frozen grant; fixture external boundaries cannot emit measured evidence.

    Real callers cannot pass a claimed execution kind. A killed process can leave
    partial artifacts; only a successful result plus supervisor stop closes a job.
    No ledger is opened or initialized here.
    """
    job_directory = Path(job_directory).resolve()
    if (job_directory / "started.json").exists() or (job_directory / "result.json").exists():
        raise ValueError("evaluation already started; recover evidence, never relaunch this job")
    started, cpu = time.time(), time.process_time()
    _write_json(
        job_directory / "started.json",
        {
            "at": started,
            "pid": os.getpid(),
            "process_group": os.getpgrp(),
        },
    )
    request, binding = None, None
    measured = fixture_generation is None
    markers = []
    result = {
        "schema": "mage-w4-evaluation-result/v1",
        "status": "failed",
        "execution_kind": "gb10_measured" if measured else "fixture",
    }
    try:
        if measured and (platform.system() != "Linux" or platform.machine() != "aarch64"):
            raise ValueError("real complete evaluation requires a GB10 Linux ARM host")
        if (fixture_generation is None) != (fixture_scorers is None):
            raise ValueError("fixture generation and all fixture scorers must be supplied together")
        prepared = read_json(job_directory / "prepared.json")
        job = prepared["evaluation_job"]
        shared = job["shared"]
        reconstructed = reconstruct_evaluation_job(job, manifest)
        if digest_json(reconstructed) != digest_json(prepared):
            raise ValueError("frozen evaluation job/prepared identity mismatch")
        lease = DeviceLease(**read_json(job_directory / "lease.json"))
        grant = read_json(job_directory / "grant.json")
        request = validate_worker_grant(
            grant, lease, manifest, device_id=device_id, now=time.time()
        )
        if (
            grant.get("lease") != asdict(lease)
            or digest_json(asdict(replace(request, lease_id=None)))
            != digest_json(prepared["request"])
            or request.evaluation_mode != "complete"
            or request.ticket_id not in {"05", "06", "08"}
        ):
            raise ValueError("grant does not authorize this one complete evaluation job")
        result.update(
            request_digest=request.request_digest,
            lease_id=lease.lease_id,
            device_id=device_id,
            evaluation_job_sha256=digest_json(job),
        )

        def stage(name):
            now = time.time()
            if now >= lease.hard_deadline - 30:
                raise TimeoutError("evaluation reached its hard-deadline cleanup reserve")
            entry = {"stage": name, "at": now, "lease_id": lease.lease_id}
            markers.append(entry)
            with (job_directory / "stages.jsonl").open("a") as stream:
                stream.write(json.dumps(entry) + "\n")
            print(name, flush=True)

        stage("evaluation_start")
        if measured:
            scoring = verify_evaluation_inputs(shared)
            report = _generate_real(job, request, manifest, job_directory / "generation", stage)
        else:
            scoring = {"ocr_config": decode_ocr_config(shared["inputs"]["ocr_config"])}
            report = fixture_generation(request, manifest, job_directory / "generation")
        if report.get("fixed_sha256") != shared["fixed_state_sha256"]:
            raise ValueError("generation changed the frozen scales/remainder")
        if digest_json(read_json(job_directory / "generation/report.json")) != digest_json(report):
            raise ValueError("generation report file differs from returned native receipt")
        binding = GenerationBinding(
            request.request_digest,
            digest_json(report),
            shared["fixed_state_sha256"],
            digest_json(report["runtime"]),
            digest_json(report["generation"]),
            result["execution_kind"],
        )
        _write_json(job_directory / "generation-binding.json", asdict(binding))
        stage("generation_released_before_scoring")
        batch = evaluate_generated_candidate(
            request,
            manifest,
            report,
            binding=binding,
            expected_center_version=request.center_version,
            image_root=job_directory,
            receipt_directory=job_directory / "scores",
            fixture_scorers=fixture_scorers,
            before_stage=stage,
            **scoring,
        )
        stage("complete_batch_saved")
        result.update(
            status="ok",
            generation_binding=asdict(binding),
            generation_binding_file_sha256=_file_sha(job_directory / "generation-binding.json"),
            native_report_sha256=digest_json(report),
            native_report_file_sha256=_file_sha(job_directory / "generation/report.json"),
            scored_batch_sha256=batch.sha256,
            scored_batch_file_sha256=_file_sha(job_directory / "scores/batch.json"),
            image_count=len(batch.samples),
            quality_improvement_claimed=False,
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        end = time.time()
        result.update(
            started_at=started,
            finished_at=end,
            wall_seconds=end - started,
            host_cpu_seconds=time.process_time() - cpu,
            max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            max_rss_unit="bytes" if platform.system() == "Darwin" else "KiB",
            pid=os.getpid(),
            process_group=os.getpgrp(),
            stages=[
                {
                    **marker,
                    "duration_seconds": (
                        markers[index + 1]["at"] if index + 1 < len(markers) else end
                    )
                    - marker["at"],
                }
                for index, marker in enumerate(markers)
            ],
            accounting="raw remote stage times; occupancy belongs only to the coordinator lease",
        )
        _write_json(job_directory / "result.json", result)
    return result
