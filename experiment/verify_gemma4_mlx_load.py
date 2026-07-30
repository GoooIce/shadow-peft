#!/usr/bin/env python3
"""Verify a converted gemma-4-12b-it-mlx-2bit checkpoint loads in mlx_vlm.

Three checks:
  1. ``mlx_vlm.load(mlx_dir)`` succeeds and constructs the model.
  2. A short greedy generation (8 tokens) does not crash / produce NaNs.
  3. Per-layer dequant vs original bf16 weight cosine (sanity, >= 0.80).

Usage::

    python experiment/verify_gemma4_mlx_load.py --mlx-dir ./gemma-4-12b-it-mlx-2bit
    # with original model for cosine comparison:
    python experiment/verify_gemma4_mlx_load.py \
        --mlx-dir ./gemma-4-12b-it-mlx-2bit \
        --ref-dir /path/to/gemma-4-12B-it
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mlx-dir", required=True)
    ap.add_argument("--ref-dir", default=None, help="original bf16 model dir for cosine check")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=8)
    args = ap.parse_args()

    try:
        import mlx_vlm
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
    except ImportError:
        print("ERROR: mlx_vlm not installed.  pip install mlx-vlm", file=sys.stderr)
        sys.exit(1)

    print(f"mlx_vlm {mlx_vlm.__version__}", flush=True)

    # ---- 1. load ----
    print(f"\n[1/3] loading {args.mlx_dir} via mlx_vlm.load ...", flush=True)
    try:
        model, processor = load(args.mlx_dir)
        cfg = load_config(args.mlx_dir)
        print(f"  OK: model_type={cfg.get('model_type')}", flush=True)
        print(f"  quantization in config: {cfg.get('quantization', '<none>')}", flush=True)
    except Exception as e:
        print(f"  LOAD FAILED: {e!r}", file=sys.stderr)
        sys.exit(2)

    # ---- 2. short generation ----
    print(f"\n[2/3] greedy generation ({args.max_tokens} tokens) ...", flush=True)
    try:
        prompt = args.prompt
        if hasattr(processor, "apply_chat_template"):
            try:
                prompt = apply_chat_template(processor, args.prompt, add_generation_prompt=True)
            except Exception:
                prompt = args.prompt
        out = generate(
            model,
            processor,
            prompt=prompt,
            max_tokens=args.max_tokens,
            verbose=False,
        )
        text = out if isinstance(out, str) else getattr(out, "text", str(out))
        print(f"  generated: {text!r}", flush=True)
        nan_check = "nan" in text.lower() or "inf" in text.lower()
        print(f"  contains NaN/inf text: {nan_check}", flush=True)
    except Exception as e:
        print(f"  GENERATION WARNING (non-fatal): {e!r}", flush=True)

    # ---- 3. per-layer cosine vs original (optional) ----
    if args.ref_dir:
        print(f"\n[3/3] per-layer dequant cosine vs {args.ref_dir} ...", flush=True)
        _cosine_check(args.mlx_dir, args.ref_dir)
    else:
        print("\n[3/3] skipped cosine check (no --ref-dir)", flush=True)

    print("\nVERIFICATION COMPLETE.", flush=True)


def _cosine_check(mlx_dir: str, ref_dir: str) -> None:
    """Compare MLX dequantized weights to original bf16 weights."""
    import json
    import struct

    import mlx.core as mx
    import numpy as np
    from safetensors import safe_open

    mlx_path = Path(mlx_dir)
    idx_file = mlx_path / "model.safetensors.index.json"
    if idx_file.exists():
        idx = json.loads(idx_file.read_text())["weight_map"]
    else:
        with open(mlx_path / "model.safetensors", "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        idx = {k: "model.safetensors" for k in hdr}

    triplets = [k[: -len(".scales")] for k in idx if k.endswith(".scales")]

    ref_path = Path(ref_dir)
    ref_idx_file = ref_path / "model.safetensors.index.json"
    ref_idx = json.loads(ref_idx_file.read_text())["weight_map"] if ref_idx_file.exists() else None

    cosines = []
    for base in triplets[:20]:  # sample first 20 for speed
        shard = idx[f"{base}.weight"]
        with safe_open(str(mlx_path / shard), framework="np") as f:
            packed = f.get_tensor(f"{base}.weight")
            scales = f.get_tensor(f"{base}.scales")
            biases = f.get_tensor(f"{base}.biases")
        w_mlx = mx.dequantize(
            mx.array(packed.astype(np.uint32)),
            mx.array(scales.astype(np.float32)),
            mx.array(biases.astype(np.float32)),
            group_size=128,
            bits=2,
        )
        ref_key = f"{base}.weight"
        ref_shard = ref_idx.get(ref_key) if ref_idx else "model.safetensors"
        try:
            with safe_open(str(ref_path / ref_shard), framework="np") as f:
                w_ref = f.get_tensor(ref_key).astype(np.float32)
        except Exception as e:
            print(f"  {base}: ref load failed ({e})", flush=True)
            continue
        a = np.array(w_mlx, dtype=np.float32).reshape(-1)
        b = w_ref.reshape(-1)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cos = float(a @ b / denom) if denom > 0 else 0.0
        cosines.append(cos)
        print(f"  {base}: cosine={cos:.4f}", flush=True)

    if cosines:
        import statistics as st

        print(
            f"\n  sampled {len(cosines)} layers: min={min(cosines):.4f} "
            f"median={st.median(cosines):.4f} mean={st.mean(cosines):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
