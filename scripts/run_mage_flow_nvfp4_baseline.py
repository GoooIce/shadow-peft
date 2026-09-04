#!/usr/bin/env python3
"""Run a profiled Mage MMDiT native-NVFP4 baseline on a GB10."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import resource
import socket
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--scheme", required=True, choices=("rtn", "quasar-grid"))
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompts-jsonl", type=Path)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=525200)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument(
        "--target-module",
        action="append",
        dest="target_modules",
        help="quantize only this exact/suffix Linear name; repeatable (default: all eligible)",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--strict-native",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fail unless profiler sees an SM120/121 block-scaled E2M1 GEMM",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_prompt(args: argparse.Namespace) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if args.prompt is not None:
        return args.prompt, {}, {"kind": "explicit"}
    path = args.prompts_jsonl.expanduser().resolve()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not 0 <= args.prompt_index < len(rows):
        raise IndexError(f"prompt index {args.prompt_index} outside {len(rows)} rows")
    row = rows[args.prompt_index]
    prompt = row.get(args.prompt_field) if isinstance(row, dict) else None
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"row {args.prompt_index} has no {args.prompt_field!r} string")
    metadata = {key: value for key, value in row.items() if key != args.prompt_field}
    source = {
        "kind": "jsonl",
        "path": str(path),
        "sha256": file_sha256(path),
        "row": args.prompt_index,
        "field": args.prompt_field,
    }
    return prompt, metadata, source


def image_payload(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def reference_metrics(payload: bytes, reference_path: Path | None) -> dict | None:
    if reference_path is None:
        return None
    import math

    import numpy as np
    from PIL import Image

    reference_path = reference_path.expanduser().resolve()
    generated = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"), dtype=np.float32)
    reference = np.asarray(Image.open(reference_path).convert("RGB"), dtype=np.float32)
    if generated.shape != reference.shape:
        raise ValueError(
            f"reference shape {reference.shape} differs from generated shape {generated.shape}"
        )
    mse = float(np.square(generated - reference).mean())
    psnr = float("inf") if mse == 0 else 10.0 * math.log10((255.0**2) / mse)
    return {
        "path": str(reference_path),
        "sha256": file_sha256(reference_path),
        "pixel_mse": mse,
        "pixel_psnr_db": psnr,
    }


def main() -> None:
    args = parse_args()
    if args.prompt_index < 0:
        raise ValueError("--prompt-index must be non-negative")
    if args.height % 16 or args.width % 16:
        raise ValueError("Mage-Flow height and width must be multiples of 16")
    model_path = args.model_path.expanduser().resolve()
    if not (model_path / "model_index.json").is_file():
        raise FileNotFoundError(f"not a Mage diffusers repository: {model_path}")
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to mix runs in non-empty directory: {out_dir}")
    prompt, prompt_metadata, prompt_source = load_prompt(args)

    import torch
    from mage_flow import MageFlowPipeline
    from torch.profiler import ProfilerActivity, profile

    from shadow_lab.nvfp4_runtime import (
        materialize_native_nvfp4,
        native_kernel_evidence,
    )
    from shadow_lab.quasar import QuasarNVFP4Config

    if not torch.cuda.is_available():
        raise RuntimeError("native NVFP4 baseline requires a CUDA GB10")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 1):
        raise RuntimeError(f"expected GB10 compute capability (12, 1), got {capability}")

    started_at = time.time()
    load_started = time.perf_counter()
    pipeline = MageFlowPipeline.from_pretrained(str(model_path), "cuda")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    torch.cuda.reset_peak_memory_stats()
    quant_started = time.perf_counter()
    materialization = materialize_native_nvfp4(
        pipeline.model.transformer,
        scheme=args.scheme,
        quasar_config=QuasarNVFP4Config(
            reconstruction_group_size=128,
            quantize_activations=True,
        ),
        target_modules=tuple(args.target_modules) if args.target_modules else None,
        use_triton_kernel=False,
    )
    torch.cuda.synchronize()
    quantize_seconds = time.perf_counter() - quant_started
    quantize_peak = {
        "allocated": torch.cuda.max_memory_allocated(),
        "reserved": torch.cuda.max_memory_reserved(),
        "process_peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }

    generation_kwargs = {
        "seeds": [args.seed],
        "heights": [args.height],
        "widths": [args.width],
        "steps": args.steps,
        "cfg": args.cfg,
    }
    torch.cuda.reset_peak_memory_stats()
    warm_started = time.perf_counter()
    warm_image = pipeline.generate([prompt], **generation_kwargs)[0]
    torch.cuda.synchronize()
    warm_seconds = time.perf_counter() - warm_started
    warm_payload = image_payload(warm_image)

    profile_started = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        profiled_image = pipeline.generate([prompt], **generation_kwargs)[0]
        torch.cuda.synchronize()
    profile_seconds = time.perf_counter() - profile_started
    profiled_payload = image_payload(profiled_image)
    if warm_payload != profiled_payload:
        raise RuntimeError("same prompt/seed changed between warm and profiled NVFP4 runs")

    gpu_kernel_names = [
        event.name
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    cpu_scaled_mm_calls = sum(event.name == "aten::_scaled_mm" for event in prof.events())
    kernel = native_kernel_evidence(gpu_kernel_names)
    materialization.update(kernel)
    materialization["aten_scaled_mm_calls"] = cpu_scaled_mm_calls

    image_name = f"{args.scheme}_prompt_{args.prompt_index:04d}_seed_{args.seed}.png"
    (out_dir / image_name).write_bytes(profiled_payload)
    finished_at = time.time()
    run = {
        "schema_version": 1,
        "started_at_unix": started_at,
        "finished_at_unix": finished_at,
        "elapsed_seconds": finished_at - started_at,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(capability),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "torchao": package_version("torchao"),
            "mage-flow": package_version("mage-flow"),
            "diffusers": package_version("diffusers"),
            "transformers": package_version("transformers"),
        },
        "model_path": str(model_path),
        "prompt": prompt,
        "prompt_metadata": prompt_metadata,
        "prompt_source": prompt_source,
        "generation": {
            "seed": args.seed,
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "cfg": args.cfg,
        },
        "timing_seconds": {
            "load": load_seconds,
            "quantize": quantize_seconds,
            "warm_generation": warm_seconds,
            "profiled_generation": profile_seconds,
        },
        "memory": {
            "quantize_peak_bytes": quantize_peak,
            "generation_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "generation_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
        },
        "materialization": materialization,
        "output": {
            "image": image_name,
            "sha256": payload_sha256(profiled_payload),
            "deterministic": True,
            "reference": reference_metrics(profiled_payload, args.reference_image),
        },
    }
    manifest_path = out_dir / "run.json"
    manifest_path.write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "image": image_name,
                "image_sha256": run["output"]["sha256"],
                "native_compute": kernel["native_compute"],
                "quantized_layers": materialization["quantized_layers"],
                "timing_seconds": run["timing_seconds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.strict_native and not kernel["native_compute"]:
        raise RuntimeError(f"native NVFP4 profiler gate failed; see {manifest_path}")


if __name__ == "__main__":
    main()
