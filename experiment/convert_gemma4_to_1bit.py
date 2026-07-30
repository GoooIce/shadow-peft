#!/usr/bin/env python3
"""Convert google/gemma-4-12B-it to a Bonsai-style binary 1-bit checkpoint.

The language backbone is projected with the deterministic Q1_0-compatible
surrogate: binary {-d,+d}, group size 128, d=mean(abs(w)), 32 LSB-first bits
per word. The script always writes the repository's torch I32 container and,
unless ``--torch-only`` is selected, an MLX-style U32/F16 container.

The public representation is reproduced; Prism ML's migration/training solver
is not public and is not reproduced here. Executing the MLX checkpoint requires
an MLX build with bits=1 affine kernel support, such as the Prism ML fork.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import struct
from pathlib import Path

import torch

SKIP_PATTERNS = (
    "vision_embedder",
    "embed_vision",
    "embed_audio",
    "patch_dense",
    "patch_ln",
    "pos_embedding",
    "embedding_projection",
    "multi_modal_projector",
    "audio_encoder",
    "vision_tower",
)

AUX_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "generation_config.json",
    "configuration.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
    "spiece.model",
    "tokenizer.model",
    "preprocessor_config.json",
)

DEFAULT_MODEL = "google/gemma-4-12B-it"
GROUP_SIZE = 128


def _header_dtype(path: Path, key: str) -> str:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    return header[key]["dtype"]


def _copy_aux_files(source_dir: Path, target_dir: Path) -> list[str]:
    copied = []
    for filename in AUX_FILES:
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, target_dir / filename)
            copied.append(filename)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF/ModelScope id or path")
    parser.add_argument("--out-mlx", default="./gemma-4-12b-it-mlx-1bit")
    parser.add_argument("--out-torch", default="./gemma-4-12b-it-torch-1bit")
    parser.add_argument(
        "--torch-only",
        action="store_true",
        help="write and validate only the torch I32 checkpoint",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="download the model through ModelScope before conversion",
    )
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    args = parser.parse_args()

    model_path = args.model
    if args.download:
        from modelscope import snapshot_download

        model_path = snapshot_download(model_path)
        print(f"downloaded -> {model_path}", flush=True)

    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    print(f"loading {model_path} (bf16, CPU) ...", flush=True)
    try:
        config = AutoConfig.from_pretrained(model_path)
    except ValueError as exc:
        raise RuntimeError(
            "Gemma 4 config is unsupported by the installed Transformers "
            f"{transformers.__version__}. Install a build that registers "
            "the snapshot's gemma4_unified architecture (the current local "
            "snapshot declares Transformers 5.10.0.dev0) before conversion."
        ) from exc

    # Resolve all conversion/export dependencies before loading the large model.
    from shadow_peft.quantization import (
        quantize_model_1bit,
        save_quantized_checkpoint,
    )
    if not args.torch_only:
        from shadow_peft_mlx import save_1bit_mlx_checkpoint

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print(
        f"quantizing Bonsai-style binary weights (group_size={args.group_size}) ...",
        flush=True,
    )
    manifest = quantize_model_1bit(
        model,
        group_size=args.group_size,
        skip_patterns=SKIP_PATTERNS,
        method="bonsai",
    )
    print(f"  quantized layers: {len(manifest['quantized'])}", flush=True)
    print(f"  skipped layers:   {len(manifest['skipped'])}", flush=True)
    print(f"  quantized tied:   {len(manifest['quantized_tied'])}", flush=True)

    print(f"writing torch container -> {args.out_torch}", flush=True)
    save_quantized_checkpoint(model, manifest, args.out_torch)
    torch_dir = Path(args.out_torch)
    config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    config_dict["shadow_peft_quantization"] = dict(manifest["quantization"])
    (torch_dir / "config.json").write_text(
        json.dumps(config_dict, indent=2), encoding="utf-8"
    )

    info = None
    if not args.torch_only:
        print(f"writing MLX container -> {args.out_mlx}", flush=True)
        info = save_1bit_mlx_checkpoint(
            model,
            config_dict,
            manifest,
            args.out_mlx,
            group_size=args.group_size,
        )
        print(
            f"  MLX: {info['quantized_count']} tensors, {len(info['shards'])} shard(s)",
            flush=True,
        )

    source_dir = Path(model_path)
    target_dirs = [torch_dir]
    if not args.torch_only:
        target_dirs.append(Path(args.out_mlx))
    for target_dir in target_dirs:
        copied = _copy_aux_files(source_dir, target_dir)
        if copied:
            print(
                f"  copied {len(copied)} tokenizer/processor files -> {target_dir}",
                flush=True,
            )
        else:
            print(
                f"  WARNING: no local tokenizer/processor files copied -> {target_dir}",
                flush=True,
            )

    first_quantized = next(
        name for name in manifest["quantized"] if not name.startswith("lm_head")
    )
    weight_key = f"{first_quantized}.weight"
    torch_shard = torch_dir / "model.safetensors"
    torch_dtype = _header_dtype(torch_shard, weight_key)
    assert torch_dtype == "I32", f"expected torch I32, got {torch_dtype}"
    print(f"self-check: {weight_key} torch={torch_dtype}", flush=True)
    if info is not None:
        if info["index"] is not None:
            mlx_shard_name = info["index"]["weight_map"][weight_key]
        else:
            mlx_shard_name = info["shards"][0]
        mlx_shard = Path(args.out_mlx) / mlx_shard_name
        mlx_dtype = _header_dtype(mlx_shard, weight_key)
        assert mlx_dtype == "U32", f"expected MLX U32, got {mlx_dtype}"
        print(f"self-check: {weight_key} MLX={mlx_dtype}", flush=True)

    del model
    gc.collect()
    suffix = "" if args.torch_only else " MLX inference still requires bits=1 support."
    print(f"DONE.{suffix}", flush=True)


if __name__ == "__main__":
    main()
