#!/usr/bin/env python3
"""Generate the MiniCPM5-2B INT8 W8A8 start payload with llmcompressor.

Deliberate mirror of ``quantize_nvfp4_llmcompressor.py`` (ticket 10): only the
``--scheme`` default changed, from ``NVFP4`` (W4A4) to ``W8A8`` (INT8 W8A8,
the setting QES actually demonstrates).  Same tool versions, same
``targets="Linear"`` / ``ignore=["lm_head"]``, same calibration jsonl, same
``num_calibration_samples`` / ``max_seq_length`` / ``save_compressed``, same
sidecar copy list, same manifest writer.  Deviations are documented in
``docs/experiments/minicpm5-w8a8-payload-20260910.md``; the load-bearing one is
the *format*: W8A8 in compressed-tensors 0.18.0 is

    weights           int8, symmetric, strategy=channel, dynamic=False
    input_activations int8, symmetric, strategy=token,   dynamic=True

so the checkpoint is ``compressed-tensors/int-quantized`` (int8 weights +
per-output-channel FP32 ``weight_scale``) with **no** activation scale in the
file at all: activation quantisation is done per token at run time, and the 512
calibration sequences therefore only exercise the observers rather than
producing a static ``input_global_scale`` the way the W4A4 recipe did.

Calibration data is a neutral public corpus (wikitext-103 train); the model's
own post-training corpus (UltraData) is not used.

Usage:
    python quantize_int8_w8a8_llmcompressor.py --model ~/models/MiniCPM5-2B \
        --calib <calib.jsonl> --out-dir <payload dir> --num-samples 512
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--calib", required=True, help="calibration jsonl with a 'text' column")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scheme", default="W8A8")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--ignore", nargs="*", default=["lm_head"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print(f"[quant] loading {args.model} in bfloat16 on cuda", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dataset = load_dataset("json", data_files=args.calib, split="train")
    print(f"[quant] calibration rows={len(dataset)} samples={args.num_samples} "
          f"seq_len={args.max_seq_length}", flush=True)

    recipe = QuantizationModifier(
        targets="Linear", scheme=args.scheme, ignore=args.ignore
    )
    print(f"[quant] recipe {recipe}", flush=True)

    oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=recipe,
        dataset=dataset,
        num_calibration_samples=args.num_samples,
        max_seq_length=args.max_seq_length,
        save_compressed=True,
        output_dir=str(out_dir),
    )
    elapsed = time.time() - started
    print(f"[quant] oneshot done in {elapsed:.1f}s", flush=True)

    tokenizer.save_pretrained(out_dir)

    # Bring over the non-weight sidecar files so the payload is self-contained.
    for name in ("generation_config.json", "chat_template.jinja", "LICENSE", "README.md"):
        source = Path(args.model).expanduser() / name
        if source.exists() and not (out_dir / name).exists():
            (out_dir / name).write_bytes(source.read_bytes())

    # --- identity ------------------------------------------------------------
    config = json.loads((out_dir / "config.json").read_text())
    quant_config = config.get("quantization_config", {})
    files = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(out_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest = {
        "format": f"compressed-tensors/{quant_config.get('format', 'int-quantized')}",
        "tool": {
            "llmcompressor": "0.13.0",
            "compressed_tensors": "0.18.0",
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "source_model": str(Path(args.model).expanduser()),
        "scheme": args.scheme,
        "ignore": args.ignore,
        "calibration": {
            "file": str(Path(args.calib).expanduser()),
            "sha256": _sha256(Path(args.calib).expanduser()),
            "num_samples": args.num_samples,
            "max_seq_length": args.max_seq_length,
        },
        "quantization_config": quant_config,
        "wall_clock_seconds": round(elapsed, 1),
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
        "host": __import__("socket").gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability()),
    }
    (out_dir / "payload_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(
        {
            "format": manifest["format"],
            "quantization_config": quant_config,
            "total_bytes": manifest["total_bytes"],
            "files": [item["path"] for item in files],
            "wall_clock_seconds": manifest["wall_clock_seconds"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
