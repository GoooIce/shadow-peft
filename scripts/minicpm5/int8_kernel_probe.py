#!/usr/bin/env python3
"""Bounded runtime probe: does a native INT8 W8A8 GEMM exist on this GPU?

vLLM's `cutlass_scaled_mm` requires column-major `b` (`b.stride(0) == 1`), i.e.
exactly the transposed weight view that `CutlassInt8ScaledMMLinearKernel`
installs.  Probe v1/v2 used `w.t().contiguous()` and therefore died in the
layout check at scaled_mm_entry.cu:209 -- that check must not be confused with
an architecture gate (the stable-ABI error text is empty).

This script probes, with correct operand layouts:

  A. vLLM CUTLASS INT8  (`ops.cutlass_scaled_mm` / `..._azp`) -- what the
     default `--linear-backend cutlass` selects on CUDA;
  B. vLLM CUTLASS FP8   -- the arch-controlled contrast: the wheel ships sm120a
     FP8 kernels, so a working FP8 call proves the entry point is reachable;
  C. vLLM Triton INT8   (`triton_scaled_mm`) -- the `triton` backend;
  D. PyTorch cuBLASLt   (`torch._scaled_mm`) -- generic fallback.

Every path that executes gets its CUDA kernel names captured with
torch.profiler; dtype of the running GEMM is what decides native vs emulated.
A few seconds of GPU time; writes only a JSON report.

Usage:
    python int8_kernel_probe.py --out <dir> [--m 512] [--k 2048] [--n 2048]
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path


def kernels_of(fn):
    import torch
    from torch.profiler import ProfilerActivity, profile

    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [
        e.name
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA and e.name.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    import vllm
    from vllm import _custom_ops as ops

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    capability = list(torch.cuda.get_device_capability())
    report: dict = {
        "host": os.uname().nodename,
        "gpu": torch.cuda.get_device_name(0),
        "capability": capability,
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "shape": {"m": args.m, "k": args.k, "n": args.n},
        "layout_note": (
            "b is passed as w.t() (row-major [N,K] viewed as column-major [K,N]) "
            "so that b.stride(0) == 1, which is what vLLM's INT8 linear kernel "
            "installs after process_weights_after_loading."
        ),
        "paths": {},
    }
    print(f"[probe] {report['gpu']} capability={capability} vllm={vllm.__version__}",
          flush=True)

    torch.manual_seed(args.seed)
    m, k, n = args.m, args.k, args.n
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)

    # INT8 operands exactly as vLLM produces them: dynamic per-token activations
    # (`ops.scaled_int8_quant`, input_scale=None) and per-channel weights.
    x_q, x_s, x_zp = ops.scaled_int8_quant(x.contiguous(), None, None, symmetric=True)
    w_scale = (w.abs().max(dim=1, keepdim=True).values.float() / 127.0).clamp(min=1e-8)
    w_q = (w / w_scale).round().clamp(-127, 127).to(torch.int8)   # [N, K] contiguous
    b_int8 = w_q.t()                                             # [K, N], stride(0)==1
    report["operand_layout"] = {
        "x_q": [list(x_q.shape), list(x_q.stride())],
        "b_int8": [list(b_int8.shape), list(b_int8.stride())],
        "x_s": [list(x_s.shape), str(x_s.dtype)],
        "w_s": [list(w_scale.shape), str(w_scale.dtype)],
    }
    print(f"[probe] x_q {tuple(x_q.shape)}{tuple(x_q.stride())} b {tuple(b_int8.shape)}"
          f"{tuple(b_int8.stride())}", flush=True)

    ref = ((x_q.float() * x_s.float()) @ w_q.float().t() * w_scale.t().float())
    ref = ref.to(torch.bfloat16)

    # FP8 siblings, same layout conventions.
    x_f8 = x.to(torch.float8_e4m3fn)
    w_f8 = w.t().contiguous().to(torch.float8_e4m3fn)  # [K, N] contiguous
    b_f8 = w_f8.t()                                    # [N, K] -> column-major [K,N]
    f8_tok = x_s.clone()
    f8_chan = w_scale.clone()
    f8_scalar_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    f8_scalar_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    ref_f8 = (x_f8.float() * f8_tok.float()) @ (b_f8.float() * f8_chan.float().t())
    ref_f8 = ref_f8.to(torch.bfloat16)

    def record(name, fn, *, extra=None, ref_t=None):
        entry: dict = {"name": name}
        try:
            out = fn()
            torch.cuda.synchronize()
            entry["status"] = "executed"
            entry["out_shape"] = list(out.shape)
            entry["out_dtype"] = str(out.dtype)
            entry["finite"] = bool(torch.isfinite(out).all().item())
            if ref_t is not None:
                entry["max_abs_err_vs_int8_reference"] = float(
                    (out.float() - ref_t.float()).abs().max().item()
                )
            try:
                entry["kernels"] = kernels_of(fn)
            except Exception as error:
                entry["kernels"] = None
                entry["profiler_error"] = f"{type(error).__name__}: {error}"
        except Exception as error:
            entry["status"] = "failed"
            entry["error_type"] = type(error).__name__
            entry["error"] = str(error)
            entry["traceback_tail"] = traceback.format_exc()[-700:]
        if extra:
            entry.update(extra)
        report["paths"][name] = entry
        print(f"[probe] {name}: {entry['status']} :: "
              f"{(entry.get('error') or entry.get('kernels') or '')}", flush=True)
        return entry

    # which class does vLLM select for a W8A8 int8 layer?
    try:
        from vllm.model_executor.kernels.linear import (
            _POSSIBLE_INT8_KERNELS,
            init_int8_linear_kernel,
        )
        from vllm.platforms import current_platform

        chosen = init_int8_linear_kernel(
            is_channelwise=True, is_static_input_scheme=False,
            input_symmetric=True, module_name="probe",
        )
        report["selected_int8_kernel"] = type(chosen).__name__
        report["possible_int8_kernels"] = [
            kth.__name__
            for kth in _POSSIBLE_INT8_KERNELS.get(current_platform._enum, [])
        ]
    except Exception as error:
        report["selected_int8_kernel_error"] = f"{type(error).__name__}: {error}"

    # --- A. vLLM CUTLASS INT8 -----------------------------------------------
    record("cutlass_int8_token_channel_scales",
           lambda: ops.cutlass_scaled_mm(x_q, b_int8, scale_a=x_s, scale_b=w_scale,
                                         out_dtype=torch.bfloat16),
           ref_t=ref)
    record("cutlass_int8_scalar_scales",
           lambda: ops.cutlass_scaled_mm(
               x_q, b_int8,
               scale_a=torch.tensor(1.0, device="cuda", dtype=torch.float32),
               scale_b=torch.tensor(1.0, device="cuda", dtype=torch.float32),
               out_dtype=torch.bfloat16))
    azp_adj = w_q.sum(dim=0, keepdim=True, dtype=torch.int32)
    record("cutlass_int8_azp",
           lambda: ops.cutlass_scaled_mm_azp(x_q, b_int8, scale_a=x_s, scale_b=w_scale,
                                             out_dtype=torch.bfloat16,
                                             azp_adj=azp_adj, azp=x_zp))

    # --- B. vLLM CUTLASS FP8 (arch-controlled contrast) ---------------------
    record("cutlass_fp8_scalar_scales",
           lambda: ops.cutlass_scaled_mm(x_f8, b_f8, scale_a=f8_scalar_a,
                                         scale_b=f8_scalar_b,
                                         out_dtype=torch.bfloat16))
    record("cutlass_fp8_token_channel_scales",
           lambda: ops.cutlass_scaled_mm(x_f8, b_f8, scale_a=f8_tok, scale_b=f8_chan,
                                         out_dtype=torch.bfloat16),
           ref_t=ref_f8)

    # --- C. vLLM Triton INT8 (--linear-backend triton) ----------------------
    def triton_path():
        from vllm.model_executor.layers.quantization.compressed_tensors.\
            triton_scaled_mm import triton_scaled_mm
        return triton_scaled_mm(x_q, b_int8, scale_a=x_s, scale_b=w_scale,
                                out_dtype=torch.bfloat16)

    record("triton_int8", triton_path, ref_t=ref)

    # --- D. PyTorch cuBLASLt -------------------------------------------------
    record("torch_scaled_mm_int8_out_i32",
           lambda: torch._scaled_mm(x_q, b_int8, scale_a=x_s, scale_b=w_scale,
                                    out_dtype=torch.int32))
    record("torch_scaled_mm_int8_scalar_out_i32",
           lambda: torch._scaled_mm(
               x_q, b_int8,
               scale_a=torch.tensor(1.0, device="cuda", dtype=torch.float32),
               scale_b=torch.tensor(1.0, device="cuda", dtype=torch.float32),
               out_dtype=torch.int32))

    # --- E. availability of the alternative int8 kernel backends ------------
    report["humming_installed"] = None
    try:
        import humming  # noqa: F401
        report["humming_installed"] = True
    except Exception as error:
        report["humming_installed"] = f"{type(error).__name__}: {error}"

    report_path = out_dir / "int8_kernel_probe.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[probe] wrote {report_path}")
    print(json.dumps({
        name: {"status": entry.get("status"),
               "error": (entry.get("error") or "")[:300],
               "kernels": (entry.get("kernels") or [None])[:4]}
        for name, entry in report["paths"].items()
    }, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
