"""Stream Qwen3.6-27B BF16 weights into a Bonsai-style MLX 2-bit checkpoint.

Unlike the model-object converter used for smaller checkpoints, this command
never materializes the full 52 GiB model. It reads one safetensors shard at a
time, applies ``quantize_2bit_bonsai`` to every 2-D language weight, and writes
the Prism-MLX-compatible ``U32 weight + FP16 scales/biases`` triplets.

The output is text-only: Qwen's vision tower and MTP draft layer are omitted,
matching what ``mlx_lm`` actually loads for the Qwen3.5 architecture used by
Qwen3.6.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors

from shadow_peft.quantization import quantize_2bit_bonsai

DEFAULT_MODEL = (
    Path.home()
    / ".cache/modelscope/models/Qwen--Qwen3.6-27B/snapshots/master"
)
DEFAULT_OUT = Path("./qwen3.6-27b-mlx-2bit-v2")
GROUP_SIZE = 128
REFINE_ITERS = 8
THR_RATIO = 0.5

AUX_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "configuration.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)


def _resolve_model_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "config.json").is_file():
        return path
    snapshot = path / "snapshots/master"
    if (snapshot / "config.json").is_file():
        return snapshot.resolve()
    raise FileNotFoundError(f"no config.json under {path}")


def _is_language_key(key: str) -> bool:
    return key.startswith("model.language_model.") or key.startswith("lm_head.")


def _to_mlx_key(key: str) -> str:
    if key.startswith("model.language_model."):
        return "language_model.model." + key.removeprefix("model.language_model.")
    if key.startswith("lm_head."):
        return "language_model." + key
    raise ValueError(f"not a Qwen language-model key: {key}")


def _module_name(key: str) -> str:
    if not key.endswith(".weight"):
        raise ValueError(f"quantized tensor is not a weight: {key}")
    return key.removesuffix(".weight")


def _peak_gib() -> float:
    # macOS reports ru_maxrss in bytes; Linux reports KiB.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if value < 1024**3:
        value *= 1024
    return value / 1024**3


def _load_index(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "model.safetensors.index.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing sharded checkpoint index: {path}")
    index = json.loads(path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{path} has no weight_map object")
    return index


def _build_plan(
    model_dir: Path, index: dict[str, Any], group_size: int
) -> tuple[list[tuple[str, list[tuple[str, bool]]]], int, int, int]:
    keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard_name in index["weight_map"].items():
        if _is_language_key(key):
            keys_by_shard[shard_name].append(key)

    plan: list[tuple[str, list[tuple[str, bool]]]] = []
    quantized_count = 0
    tensor_count = 0
    estimated_bytes = 0
    for shard_name in sorted(keys_by_shard):
        shard_path = model_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"missing source shard: {shard_path}")
        shard_plan: list[tuple[str, bool]] = []
        with safe_open(shard_path, framework="pt", device="cpu") as source:
            for key in sorted(keys_by_shard[shard_name]):
                shape = tuple(source.get_slice(key).get_shape())
                quantized = (
                    key.endswith(".weight")
                    and len(shape) == 2
                    and shape[-1] % group_size == 0
                )
                if len(shape) == 2 and not quantized:
                    raise ValueError(
                        f"2-D language weight is not g{group_size}-compatible: "
                        f"{key} {shape}"
                    )
                elements = 1
                for dim in shape:
                    elements *= dim
                if quantized:
                    rows, cols = shape
                    estimated_bytes += rows * (cols // 16) * 4
                    estimated_bytes += rows * (cols // group_size) * 2 * 2
                    quantized_count += 1
                else:
                    estimated_bytes += elements * 2
                tensor_count += 1
                shard_plan.append((key, quantized))
        plan.append((shard_name, shard_plan))
    return plan, tensor_count, quantized_count, estimated_bytes


def _write_config_and_metadata(
    model_dir: Path,
    staging_dir: Path,
    *,
    group_size: int,
    refine_iters: int,
    thr_ratio: float,
    quantized_names: list[str],
    weight_map: dict[str, str],
    total_size: int,
) -> None:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_5":
        raise ValueError(
            f"expected Qwen3.6's qwen3_5 implementation, got "
            f"{config.get('model_type')!r}"
        )
    quantization = {"bits": 2, "group_size": group_size, "mode": "affine"}
    config["quantization"] = dict(quantization)
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        text_config["quantization"] = dict(quantization)
    (staging_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (staging_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )

    manifest = {
        "source_model": str(model_dir),
        "text_only": True,
        "quantization": {
            "mode": "affine",
            "bits": 2,
            "ternary": True,
            "method": "bonsai",
            "group_size": group_size,
            "refine_iters": refine_iters,
            "thr_ratio": thr_ratio,
            "quantize_tied_weights": True,
            "trim_vocab_to": None,
        },
        "quantized": quantized_names,
        "skipped": {},
        "tied": [],
        "quantized_tied": [],
        "trimmed": {},
    }
    (staging_dir / "quantization.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    for filename in AUX_FILES:
        source = model_dir / filename
        if source.is_file():
            shutil.copy2(source, staging_dir / filename)


def convert(
    model_dir: Path,
    out_dir: Path,
    *,
    group_size: int,
    refine_iters: int,
    thr_ratio: float,
    plan_only: bool,
) -> None:
    model_dir = _resolve_model_dir(model_dir)
    out_dir = out_dir.expanduser().resolve()
    index = _load_index(model_dir)
    plan, tensor_count, quantized_count, estimated_bytes = _build_plan(
        model_dir, index, group_size
    )

    print(f"source:       {model_dir}", flush=True)
    print(f"output:       {out_dir}", flush=True)
    print(f"input shards: {len(plan)}", flush=True)
    print(f"language:     {tensor_count} tensors", flush=True)
    print(f"quantized:    {quantized_count} matrices", flush=True)
    print(f"estimated:    {estimated_bytes / 1024**3:.3f} GiB", flush=True)
    if quantized_count != 498:
        raise ValueError(
            f"expected 498 Qwen3.6 language matrices, found {quantized_count}"
        )
    if plan_only:
        return

    staging_dir = out_dir.with_name(out_dir.name + ".incomplete")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {out_dir}")
    if staging_dir.exists():
        raise FileExistsError(
            f"staging directory already exists; inspect or move it first: {staging_dir}"
        )
    staging_dir.mkdir(parents=True)

    weight_map: dict[str, str] = {}
    quantized_names: list[str] = []
    total_size = 0
    started = time.monotonic()
    shard_total = len(plan)

    with torch.inference_mode():
        for shard_number, (source_name, shard_plan) in enumerate(plan, start=1):
            output_name = (
                f"model-{shard_number:05d}-of-{shard_total:05d}.safetensors"
            )
            output_tensors: dict[str, torch.Tensor] = {}
            shard_started = time.monotonic()
            shard_quantized = 0
            print(
                f"[{shard_number:02d}/{shard_total:02d}] {source_name}: "
                f"{len(shard_plan)} tensors",
                flush=True,
            )
            with safe_open(
                model_dir / source_name, framework="pt", device="cpu"
            ) as source:
                for tensor_number, (source_key, quantized) in enumerate(
                    shard_plan, start=1
                ):
                    tensor = source.get_tensor(source_key)
                    mlx_key = _to_mlx_key(source_key)
                    if quantized:
                        packed, scales, biases = quantize_2bit_bonsai(
                            tensor,
                            group_size=group_size,
                            refine_iters=refine_iters,
                            thr_ratio=thr_ratio,
                        )
                        base = mlx_key.removesuffix(".weight")
                        output_tensors[base + ".weight"] = packed.view(
                            torch.uint32
                        ).contiguous()
                        output_tensors[base + ".scales"] = scales.to(
                            torch.float16
                        ).contiguous()
                        output_tensors[base + ".biases"] = biases.to(
                            torch.float16
                        ).contiguous()
                        quantized_names.append(_module_name(source_key))
                        shard_quantized += 1
                        if tensor_number == 1 or tensor_number % 10 == 0:
                            print(
                                f"  {tensor_number:3d}/{len(shard_plan)} "
                                f"{source_key} {tuple(tensor.shape)} "
                                f"peak={_peak_gib():.2f} GiB",
                                flush=True,
                            )
                    else:
                        output_tensors[mlx_key] = (
                            tensor.to(torch.float16).contiguous()
                            if tensor.is_floating_point()
                            else tensor.contiguous()
                        )

            save_safetensors(output_tensors, staging_dir / output_name)
            shard_bytes = 0
            for key, tensor in output_tensors.items():
                weight_map[key] = output_name
                size = tensor.numel() * tensor.element_size()
                total_size += size
                shard_bytes += size
            del output_tensors
            gc.collect()
            print(
                f"  wrote {output_name}: {shard_quantized} quantized, "
                f"{shard_bytes / 1024**3:.3f} GiB, "
                f"{time.monotonic() - shard_started:.1f}s",
                flush=True,
            )

    if len(quantized_names) != quantized_count:
        raise RuntimeError(
            f"converted {len(quantized_names)} matrices, expected {quantized_count}"
        )
    _write_config_and_metadata(
        model_dir,
        staging_dir,
        group_size=group_size,
        refine_iters=refine_iters,
        thr_ratio=thr_ratio,
        quantized_names=quantized_names,
        weight_map=weight_map,
        total_size=total_size,
    )
    staging_dir.rename(out_dir)
    print(
        f"DONE: {out_dir} ({total_size / 1024**3:.3f} GiB, "
        f"{time.monotonic() - started:.1f}s, peak={_peak_gib():.2f} GiB)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-mlx", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--refine-iters", type=int, default=REFINE_ITERS)
    parser.add_argument("--thr-ratio", type=float, default=THR_RATIO)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    if args.threads < 0:
        parser.error("--threads must be >= 0")
    if args.threads:
        torch.set_num_threads(args.threads)
    convert(
        args.model,
        args.out_mlx,
        group_size=args.group_size,
        refine_iters=args.refine_iters,
        thr_ratio=args.thr_ratio,
        plan_only=args.plan_only,
    )


if __name__ == "__main__":
    main()
