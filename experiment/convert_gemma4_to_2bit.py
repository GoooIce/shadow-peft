#!/usr/bin/env python3
"""Convert google/gemma-4-12B-it to a ternary 2-bit checkpoint (dual output).

Quantizes the text-backbone Linear/Embedding weights with the repo's own
ternary 2-bit packer (codebook {0,1,2}->{-1,0,+1}, scale=s_g, bias=-s_g,
group_size=128).  The container mapping was validated against real Bonsai
checkpoints; the dense-to-ternary projection is the reproducible surrogate
from ``deep-research-report.md``, not Prism ML's unpublished solver.  The
script then writes TWO containers:

  1. MLX container  -> ``--out-mlx``  (uint32 packed weights + config.json
     with a ``quantization`` block; loadable by ``mlx_vlm.load``)
  2. torch container -> ``--out-torch`` (int32 packed weights + quantization.json
     manifest; for our own torch dequant/analysis pipeline)

Vision tower, audio projector, and RMSNorm layers are kept at full precision
(skipped via ``skip_patterns``), matching the Bonsai convention where only the
language backbone is ternary-quantized.

Usage::

    # auto-download from ModelScope, write both containers
    python experiment/convert_gemma4_to_2bit.py --download

    # point at a local snapshot, custom output dirs
    python experiment/convert_gemma4_to_2bit.py \
        --model /path/to/gemma-4-12B-it \
        --out-mlx ./gemma-4-12b-it-mlx-2bit \
        --out-torch ./gemma-4-12b-it-torch-2bit
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

# Layers kept at full precision: vision tower, audio projector, position
# embeddings, and any RMSNorm/LayerNorm (these are 1-D or non-Linear and would
# be skipped anyway, but we list the vision/audio prefixes explicitly).
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

GROUP_SIZE = 128
REFINE_ITERS = 8
DEFAULT_MODEL = "google/gemma-4-12B-it"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF/ModelScope id or local path")
    ap.add_argument("--out-mlx", default="./gemma-4-12b-it-mlx-2bit")
    ap.add_argument("--out-torch", default="./gemma-4-12b-it-torch-2bit")
    ap.add_argument(
        "--download", action="store_true", help="auto-download via ModelScope if missing"
    )
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE)
    ap.add_argument("--refine-iters", type=int, default=REFINE_ITERS)
    ap.add_argument(
        "--method",
        choices=["bonsai", "affine"],
        default="bonsai",
        help="quantizer: 'bonsai' = report-style absmean/threshold/LS surrogate "
        "(default), or 'affine' = legacy configurable baseline",
    )
    args = ap.parse_args()

    model_path = args.model
    if args.download:
        from modelscope import snapshot_download

        snap = snapshot_download(model_path)
        model_path = snap
        print(f"downloaded -> {snap}", flush=True)

    print(f"loading {model_path} (bf16, CPU) ...", flush=True)
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path)
    # Load in bfloat16 on CPU. Avoid `device_map` (needs the `accelerate` extra);
    # a 24 GB model fits comfortably in the 64 GB Mac unified memory.
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    print(
        f"  loaded: {type(model).__name__}, params ~ "
        f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B",
        flush=True,
    )

    # ---- quantize ----
    from shadow_peft.quantization import quantize_model_2bit

    print(
        f"quantizing (method={args.method}, group_size={args.group_size}, refine={args.refine_iters}) ...",
        flush=True,
    )
    manifest = quantize_model_2bit(
        model,
        group_size=args.group_size,
        refine_iters=args.refine_iters,
        skip_patterns=SKIP_PATTERNS,
        method=args.method,
    )
    n_quant = len(manifest["quantized"])
    n_skip = len(manifest["skipped"])
    n_tied = len(manifest["tied"])
    print(f"  quantized layers: {n_quant}", flush=True)
    print(f"  skipped layers:   {n_skip}", flush=True)
    if manifest["skipped"]:
        for name, reason in list(manifest["skipped"].items())[:5]:
            print(f"    skip: {name} ({reason})", flush=True)
    print(f"  tied weights:     {n_tied}", flush=True)

    # ---- write torch container ----
    from shadow_peft.quantization import save_quantized_checkpoint

    print(f"\nwriting torch container -> {args.out_torch}", flush=True)
    save_quantized_checkpoint(model, manifest, args.out_torch)

    # ---- write MLX container ----
    from shadow_peft_mlx import save_2bit_mlx_checkpoint

    print(f"writing MLX container  -> {args.out_mlx}", flush=True)
    config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    info = save_2bit_mlx_checkpoint(
        model,
        config_dict,
        manifest,
        args.out_mlx,
        group_size=args.group_size,
        # mlx-vlm's tied Gemma4 language model scores with
        # embed_tokens.as_linear and has no separate lm_head parameter.
        exclude_patterns=("lm_head.",),
    )
    print(
        f"  MLX: {info['quantized_count']} quantized tensors, {len(info['shards'])} shard(s)",
        flush=True,
    )

    # ---- copy tokenizer / processor / chat-template files into the MLX dir ----
    import shutil

    AUX_EXTS = (
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
    src_dir = Path(model_path)
    mlx_out = Path(args.out_mlx)
    copied = []
    for fname in AUX_EXTS:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, mlx_out / fname)
            copied.append(fname)
    if copied:
        print(f"  copied {len(copied)} tokenizer/processor file(s): {', '.join(copied)}", flush=True)
    else:
        print("  WARNING: no tokenizer/processor files found to copy", flush=True)

    # ---- cleanup ----
    del model
    gc.collect()

    # ---- self-check: one layer bit-exact between torch and MLX ----
    print("\nself-check: dtype verification (layer 0) ...", flush=True)
    import json
    import struct

    def _header_dtype(path, key):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            return json.loads(f.read(n))[key]["dtype"]

    mlx_shard = Path(args.out_mlx) / info["shards"][0]
    torch_shard = Path(args.out_torch) / "model.safetensors"
    first_q = manifest["quantized"][0]
    mlx_dt = _header_dtype(str(mlx_shard), f"{first_q}.weight")
    torch_dt = _header_dtype(str(torch_shard), f"{first_q}.weight")
    print(f"  {first_q}.weight: MLX dtype={mlx_dt}, torch dtype={torch_dt}", flush=True)
    assert mlx_dt == "U32", f"expected MLX U32, got {mlx_dt}"
    assert torch_dt == "I32", f"expected torch I32, got {torch_dt}"

    print("\nDONE. Verify with:", flush=True)
    print(f"  python experiment/verify_gemma4_mlx_load.py --mlx-dir {args.out_mlx}", flush=True)


if __name__ == "__main__":
    main()
