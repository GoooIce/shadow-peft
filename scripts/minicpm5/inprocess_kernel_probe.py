#!/usr/bin/env python3
"""In-process native NVFP4 kernel probe on the model's real layer shapes.

Ticket-11 A5: is a "cheap per-candidate scorer" built on in-process native
kernels (FlashInfer ``mm_fp4`` / TorchAO ``_scaled_mm``) viable, and what is its
throughput compared with the serving engine?

The probe runs a block-scaled NVFP4 GEMM per *fused* layer shape for a range of
M (tokens in flight) and reports per-GEMM latency plus an extrapolated
full-model forward cost.  It does not attempt to reproduce paged attention or
sampling; the number is a floor for scoring throughput.

Usage:
    python inprocess_kernel_probe.py --out <json> [--backends cutlass,b12x]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

# Fused layer shapes as the vLLM engine holds them: (out_features, in_features)
LAYERS = [
    ("self_attn.qkv_proj", 2560, 2048, 42),
    ("self_attn.o_proj", 2048, 2048, 42),
    ("mlp.gate_up_proj", 12288, 2048, 42),
    ("mlp.down_proj", 2048, 6144, 42),
]
M_VALUES = (1, 16, 128, 512, 2048)


def bench(fn, warmup=3, repeats=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--backends", default="cutlass,b12x")
    args = parser.parse_args()

    import flashinfer
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize

    backends = [item for item in args.backends.split(",") if item]
    report = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "flashinfer": getattr(flashinfer, "__version__", None),
        "backends": backends,
        "shapes": [],
    }

    for name, out_features, in_features, count in LAYERS:
        for m in M_VALUES:
            a = torch.randn(m, in_features, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)
            a_sf = (448 * 6) / a.float().abs().nan_to_num().max()
            b_sf = (448 * 6) / b.float().abs().nan_to_num().max()
            a_fp4, a_sc = nvfp4_quantize(
                a, a_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
            )
            b_fp4, b_sc = nvfp4_quantize(
                b, b_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
            )
            alpha = (1.0 / (a_sf * b_sf)).to(torch.float32)
            row = {"layer": name, "m": m, "k": in_features, "n": out_features}
            for backend in backends:
                try:
                    seconds = bench(
                        lambda: mm_fp4(
                            a_fp4,
                            b_fp4.T,
                            a_sc,
                            b_sc.T,
                            alpha,
                            torch.bfloat16,
                            backend=backend,
                        )
                    )
                    flops = 2 * m * out_features * in_features
                    row[backend] = {
                        "ms": seconds * 1e3,
                        "tflops": flops / seconds / 1e12,
                    }
                except Exception as exc:  # noqa: BLE001 - report, do not fail the probe
                    row[backend] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
            report["shapes"].append(row)
            print(row, flush=True)

    # Full-model forward estimate: sum over the 42 blocks.
    for backend in backends:
        estimate = {}
        for m in M_VALUES:
            total_ms = 0.0
            ok = True
            for name, out_features, in_features, count in LAYERS:
                row = next(
                    item
                    for item in report["shapes"]
                    if item["layer"] == name and item["m"] == m
                )
                entry = row.get(backend, {})
                if "ms" not in entry:
                    ok = False
                    break
                total_ms += entry["ms"] * count
            if ok:
                estimate[str(m)] = {
                    "gemm_ms_per_forward": total_ms,
                    "tokens_per_second": m / (total_ms / 1e3),
                }
        report[f"full_model_estimate_{backend}"] = estimate

    Path(args.out).expanduser().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(
        {key: value for key, value in report.items() if key.startswith("full_model")},
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
