#!/usr/bin/env python3
"""Run a reproducible Mage-Flow-Turbo BF16 baseline on a GB10.

The script deliberately consumes either explicit prompts or line-oriented JSONL.
Converting a source parquet into the final train/validation/test split belongs to
ENG-003; this runner only records exactly what it was handed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt", action="append", dest="prompts")
    prompts.add_argument("--prompts-jsonl", type=Path)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seed-field",
        default="seed",
        help="use this per-row integer when present; otherwise use --seed + absolute index",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume a verified samples.jsonl prefix when no final run.json exists",
    )
    parser.add_argument(
        "--determinism-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="generate the first sample twice and require byte-identical PNGs",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompt_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.prompts is not None:
        values = [{args.prompt_field: value} for value in args.prompts]
    else:
        values = []
        with args.prompts_jsonl.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                value = row.get(args.prompt_field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"{args.prompts_jsonl}:{line_number}: "
                        f"missing non-empty string field {args.prompt_field!r}"
                    )
                values.append(row)
    selected = values[args.offset : args.offset + args.limit]
    if not selected:
        raise ValueError("prompt selection is empty")
    return selected


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def module_inventory(module: Any) -> dict[str, int]:
    parameters = list(module.parameters())
    return {
        "parameters": sum(item.numel() for item in parameters),
        "parameter_bytes": sum(item.numel() * item.element_size() for item in parameters),
    }


def png_bytes(image: Any) -> bytes:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def seed_for_row(
    row: dict[str, Any], *, seed_field: str, base_seed: int, absolute_index: int
) -> int:
    row_seed = row.get(seed_field)
    if row_seed is None:
        return base_seed + absolute_index
    if isinstance(row_seed, bool) or not isinstance(row_seed, int) or row_seed < 0:
        raise ValueError(
            f"prompt row {absolute_index} has invalid {seed_field!r}: {row_seed!r}"
        )
    return row_seed


def load_verified_resume_prefix(
    records_path: Path,
    out_dir: Path,
    prompt_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not records_path.is_file():
        return []
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(records) > len(prompt_rows):
        raise ValueError("resume record count exceeds selected prompts")
    for relative_index, record in enumerate(records):
        absolute_index = args.offset + relative_index
        row = prompt_rows[relative_index]
        expected_seed = seed_for_row(
            row,
            seed_field=args.seed_field,
            base_seed=args.seed,
            absolute_index=absolute_index,
        )
        expected_metadata = {
            key: value for key, value in row.items() if key != args.prompt_field
        }
        checks = {
            "index": record.get("index") == absolute_index,
            "prompt": record.get("prompt") == row[args.prompt_field],
            "prompt_metadata": record.get("prompt_metadata") == expected_metadata,
            "seed": record.get("seed") == expected_seed,
            "deterministic": record.get("deterministic") is True,
        }
        failed = sorted(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(
                f"resume record {relative_index} differs from frozen input: {failed}"
            )
        image_name = record.get("image")
        image_path = out_dir / image_name if isinstance(image_name, str) else out_dir
        if not image_path.is_file() or record.get("sha256") != sha256_file(image_path):
            raise ValueError(f"resume image {relative_index} is missing or hash-mismatched")
    return records


def main() -> None:
    args = parse_args()
    if args.limit <= 0 or args.offset < 0:
        raise ValueError("--limit must be positive and --offset non-negative")
    if args.height % 16 or args.width % 16:
        raise ValueError("Mage-Flow height and width must be multiples of 16")
    model_path = args.model_path.expanduser().resolve()
    if not (model_path / "model_index.json").is_file():
        raise FileNotFoundError(f"not a Mage diffusers repository: {model_path}")

    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to mix a baseline with existing files: {out_dir}")
    if args.resume and (out_dir / "run.json").exists():
        raise FileExistsError(f"baseline is already complete: {out_dir / 'run.json'}")
    prompt_rows = load_prompt_rows(args)
    prompts = [row[args.prompt_field] for row in prompt_rows]
    records_path = out_dir / "samples.jsonl"
    existing_records = (
        load_verified_resume_prefix(records_path, out_dir, prompt_rows, args)
        if args.resume
        else []
    )

    import torch
    from mage_flow import MageFlowPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("this baseline must run on a CUDA GB10")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 1):
        raise RuntimeError(f"expected GB10 compute capability (12, 1), got {capability}")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started_at = time.time()
    load_started = time.perf_counter()
    pipeline = MageFlowPipeline.from_pretrained(str(model_path), "cuda")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    components = {
        "transformer": module_inventory(pipeline.model.transformer),
        "text_encoder": module_inventory(pipeline.model.txt_enc),
        "vae": module_inventory(pipeline.model.vae),
    }
    run: dict[str, Any] = {
        "schema_version": 1,
        "started_at_unix": started_at,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "model_path": str(model_path),
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(capability),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "mage-flow": package_version("mage-flow"),
            "diffusers": package_version("diffusers"),
            "transformers": package_version("transformers"),
            "flash-attn": package_version("flash-attn"),
        },
        "generation": {
            "base_seed": args.seed,
            "seed_field": args.seed_field,
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "cfg": args.cfg,
            "prompt_offset": args.offset,
            "prompt_count": len(prompts),
        },
        "prompt_source": (
            {
                "kind": "jsonl",
                "path": str(args.prompts_jsonl.expanduser().resolve()),
                "sha256": sha256_file(args.prompts_jsonl.expanduser().resolve()),
                "field": args.prompt_field,
            }
            if args.prompts_jsonl is not None
            else {"kind": "explicit", "field": args.prompt_field}
        ),
        "components": components,
        "load_seconds": load_seconds,
        "resumed_samples": len(existing_records),
        "memory_after_load_bytes": {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "process_peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
        "samples": list(existing_records),
    }

    for index in range(len(existing_records), len(prompts)):
        prompt = prompts[index]
        seed = seed_for_row(
            prompt_rows[index],
            seed_field=args.seed_field,
            base_seed=args.seed,
            absolute_index=args.offset + index,
        )
        repetitions = 2 if index == 0 and args.determinism_check else 1
        hashes: list[str] = []
        latencies: list[float] = []
        first_payload: bytes | None = None
        torch.cuda.reset_peak_memory_stats()
        for _ in range(repetitions):
            sample_started = time.perf_counter()
            image = pipeline.generate(
                [prompt],
                seeds=[seed],
                heights=[args.height],
                widths=[args.width],
                steps=args.steps,
                cfg=args.cfg,
            )[0]
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - sample_started)
            payload = png_bytes(image)
            hashes.append(hashlib.sha256(payload).hexdigest())
            if first_payload is None:
                first_payload = payload

        deterministic = len(set(hashes)) == 1
        if index == 0 and args.determinism_check and not deterministic:
            raise RuntimeError(f"same prompt/seed produced different PNG hashes: {hashes}")
        image_name = f"gen_{args.offset + index:04d}_seed_{seed}.png"
        (out_dir / image_name).write_bytes(first_payload or b"")
        record = {
            "index": args.offset + index,
            "prompt": prompt,
            "prompt_metadata": {
                key: value
                for key, value in prompt_rows[index].items()
                if key != args.prompt_field
            },
            "seed": seed,
            "image": image_name,
            "sha256": hashes[0],
            "repeat_sha256": hashes[1:] or None,
            "deterministic": deterministic,
            "latency_seconds": latencies,
            "peak_memory_bytes": {
                "allocated": torch.cuda.max_memory_allocated(),
                "reserved": torch.cuda.max_memory_reserved(),
                "process_peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            },
        }
        run["samples"].append(record)
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    run["finished_at_unix"] = time.time()
    run["elapsed_seconds"] = run["finished_at_unix"] - started_at
    (out_dir / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"baseline manifest: {out_dir / 'run.json'}")


if __name__ == "__main__":
    main()
