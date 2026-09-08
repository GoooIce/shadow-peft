#!/usr/bin/env python3
"""Hash an isolated auxiliary snapshot and the independent quantized inputs."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import time
from pathlib import Path


def file_hash(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    started, cpu = time.monotonic(), time.process_time()
    snapshot = args.run_root / "auxiliary"
    paths = [
        args.model / name
        for name in (
            "model_index.json",
            "transformer/config.json",
            "scheduler/scheduler_config.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        )
    ]
    paths += sorted(p for p in (args.model / "text_encoder").rglob("*") if p.is_file())
    files = {}
    for source in paths:
        dest = snapshot / source.relative_to(args.model)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            os.link(source, dest)
        elif not os.path.samefile(source, dest):
            raise ValueError(f"existing snapshot is not the same input: {dest}")
        files[str(dest.resolve())] = file_hash(dest)
    spec = importlib.util.find_spec("mage_flow")
    if spec is None or not spec.submodule_search_locations:
        raise ValueError("Mage installation missing")
    source_root = Path(next(iter(spec.submodule_search_locations)))
    for path in sorted(source_root.rglob("*.py")):
        if any(
            part.startswith(".") or part == "__pycache__"
            for part in path.relative_to(source_root).parts
        ):
            continue
        files[str(path.resolve())] = file_hash(path)
    for name in (
        "qad-nvfp4-best-673.pt",
        "qad-transformer-remainder-best-673.pt",
        "qad-nvfp4-best-673.manifest.json",
    ):
        path = args.run_root / "input" / name
        files[str(path.resolve())] = file_hash(path)
    report = {
        "schema": "mage-w4-input-inventory/v1",
        "host": platform.node(),
        "auxiliary_root": str(snapshot),
        "allowed_assets": files,
        "runtime": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchao", "diffusers", "transformers")
        },
        "wall_seconds": time.monotonic() - started,
        "host_cpu_seconds": time.process_time() - cpu,
        "device_occupied": False,
        "isolation": "hardlink snapshot contains auxiliary weights and transformer config only",
    }
    dest = args.run_root / "inventory.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "allowed_assets"}))
    print(json.dumps({"file_count": len(files), "inventory_sha256": file_hash(dest)}))


if __name__ == "__main__":
    main()
