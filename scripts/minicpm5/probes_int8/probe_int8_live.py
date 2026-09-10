#!/usr/bin/env python3
"""Live-layout probe for the INT8 W8A8 payload under TritonInt8ScaledMM (SM121).

Read-only. Answers the contract questions the runner adaptation needs:

  1. which module names hold the int8 linears (fusion naming, e.g. qkv_proj)?
  2. live weight: dtype / shape / stride (fused? transposed? contiguous?)
  3. weight_scale: dtype / shape (per-channel column vector?)
  4. do live weights equal the payload tensors row-concatenated per fusion?
  5. does `reload_weights(is_checkpoint_format=False)` accept a patched cache
     (bit-identical readback after a no-op reload)?

Usage (gooo, ~/venvs/minicpm5-vllm):
    python probe_int8_live.py --model <INT8 payload dir> --out <json> \
        [--linear-backend triton]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def probe_modules(self):
    import torch
    """Collect the int8 linear modules' live geometry."""
    model = self.model_runner.get_model()
    out = []
    for name, module in model.named_modules():
        w = getattr(module, "weight", None)
        if w is None or not hasattr(w, "dtype"):
            continue
        if w.dtype in (torch.int8, torch.uint8):
            entry = {
                "module": name,
                "class": type(module).__name__,
                "weight_dtype": str(w.dtype),
                "weight_shape": list(w.shape),
                "weight_stride": list(w.stride()),
                "weight_is_contiguous": bool(w.is_contiguous()),
                "weight_requires_grad": bool(w.requires_grad),
            }
            ws = getattr(module, "weight_scale", None)
            if ws is not None and hasattr(ws, "dtype"):
                entry["weight_scale_dtype"] = str(ws.dtype)
                entry["weight_scale_shape"] = list(ws.shape)
            bias = getattr(module, "bias", None)
            entry["has_bias"] = bias is not None
            for extra in ("weight_type", "alpha", "input_scale", "weight_zp"):
                v = getattr(module, extra, None)
                if v is not None:
                    entry[extra] = str(v)[:60]
            out.append(entry)
    return out


def probe_bytes(self, modules):
    import torch
    """sha256 of the raw bytes of the live weight tensors (payload comparison)."""
    model = self.model_runner.get_model()
    named = dict(model.named_modules())
    out = {}
    for name in modules:
        w = named[name].weight
        out[name] = hashlib.sha256(
            w.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
    return out


def probe_reload(self, cache_key):
    import torch
    """No-op reload from the cache: does the int8 path accept kernel-format?"""
    from vllm.config import set_current_vllm_config

    tensors = self._search_cache[cache_key]["tensors"]
    torch.cuda.synchronize()
    start = time.perf_counter()
    with set_current_vllm_config(self.vllm_config):
        self.model_runner.reload_weights(
            weights_iterator=iter(list(tensors.items())), is_checkpoint_format=False
        )
    torch.cuda.synchronize()
    return {"seconds": round(time.perf_counter() - start, 4),
            "tensors": len(tensors)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--linear-backend", default="triton")
    parser.add_argument("--quantization", default="compressed-tensors")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-mem", type=float, default=0.30)
    args = parser.parse_args()

    import torch  # noqa: F401  (needed inside the RPC lambdas)
    from vllm import LLM

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    llm = LLM(
        model=args.model, quantization=args.quantization, dtype="bfloat16",
        max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
        enable_prefix_caching=False, trust_remote_code=True, disable_log_stats=True,
        linear_backend=args.linear_backend,
    )
    report: dict = {"protocol": "minicpm5-int8-live-probe/v1",
                    "linear_backend": args.linear_backend}

    entries = llm.collective_rpc(probe_modules)[0]
    report["n_int8_modules"] = len(entries)
    report["sample_modules"] = entries[:8]
    fused = [e["module"] for e in entries if "qkv" in e["module"] or "gate_up" in e["module"]]
    unfused = [e["module"] for e in entries
               if any(s in e["module"] for s in ("q_proj", "k_proj", "gate_proj"))]
    report["fused_names_sample"] = fused[:4]
    report["unfused_names_sample"] = unfused[:4]

    # geometry census over shapes (fused row counts -> layout offsets)
    shapes: dict[str, int] = {}
    for e in entries:
        key = str(tuple(e["weight_shape"]))
        shapes[key] = shapes.get(key, 0) + 1
    report["shape_census"] = dict(sorted(shapes.items(), key=lambda kv: -kv[1])[:12])

    # byte-level comparison for layer 0 (fused + unfused) against the payload
    targets = [m for m in fused + unfused if ".layers.0." in m][:6]
    live_hashes = llm.collective_rpc(probe_bytes, kwargs={"modules": targets})[0]

    from safetensors.torch import load_file
    tensors = load_file(str(Path(args.model).expanduser() / "model.safetensors"))
    comparison = {}
    for module in targets:
        live = live_hashes.get(module)
        if module.endswith("qkv_proj"):
            parts = [tensors[f"{module[:-8]}{p}.weight"] for p in ("q_proj", "k_proj", "v_proj")]
            rebuilt = hashlib.sha256(torch.cat(parts, dim=0).contiguous().numpy().tobytes()).hexdigest()
        elif module.endswith("gate_up_proj"):
            parts = [tensors[f"{module[:-12]}{p}.weight"] for p in ("gate_proj", "up_proj")]
            rebuilt = hashlib.sha256(torch.cat(parts, dim=0).contiguous().numpy().tobytes()).hexdigest()
        else:
            t = tensors.get(f"{module}.weight")
            rebuilt = hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest() if t is not None else "missing"
        comparison[module] = {"live": live, "payload_concat": rebuilt,
                              "match": live == rebuilt}
    report["payload_comparison"] = comparison

    # no-op reload from a cloned cache (the refit contract)
    cache = {f"{e['module']}.weight": None for e in entries[:2]}
    # build a real cache: clone the live tensors through the worker
    def clone_cache(self):
        import torch
        model = self.model_runner.get_model()
        named = dict(model.named_modules())
        picked = targets[:2]
        tensors_out = {}
        for name in picked:
            w = named[name].weight
            tensors_out[f"{name}.weight"] = w.detach().cpu().clone().to(torch.int8) \
                if w.dtype != torch.int8 else w.detach().clone()
        cache = getattr(self, "_search_cache", None)
        if cache is None:
            cache = {}
            self._search_cache = cache
        cache["probe"] = {"tensors": tensors_out}
        return {k: str(t.dtype) for k, t in tensors_out.items()}
    report["cache_dtypes"] = llm.collective_rpc(clone_cache)[0]
    report["noop_reload"] = llm.collective_rpc(probe_reload, kwargs={"cache_key": "probe"})[0]
    post = llm.collective_rpc(probe_bytes, kwargs={"modules": targets[:2]})[0]
    report["reload_preserved_weights"] = {
        m: post.get(m) == live_hashes.get(m) for m in targets[:2]}

    report["seconds"] = round(time.time() - started, 1)
    (out / "int8_live_probe.json").write_text(
        json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("n_int8_modules", "fused_names_sample", "unfused_names_sample",
                       "shape_census", "payload_comparison", "noop_reload",
                       "reload_preserved_weights")}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
