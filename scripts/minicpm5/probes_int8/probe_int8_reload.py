#!/usr/bin/env python3
"""Probe 3: what does the live W8A8 module actually own, and what does
reload_weights need to accept a patched int8 weight?
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def dump_module_state(self, module_name):
    import torch
    model = self.model_runner.get_model()
    named = dict(model.named_modules())
    module = named[module_name]
    q = getattr(module, "quant_method", None)
    out = {
        "module_class": type(module).__name__,
        "quant_method_class": type(q).__name__ if q is not None else None,
        "parameters": {},
        "buffers": {},
        "plain_attrs": {},
    }
    for name, tensor in module.named_parameters(recurse=False):
        out["parameters"][name] = [str(tensor.dtype), list(tensor.shape),
                                   str(tensor.device)]
    for name, tensor in module.named_buffers(recurse=False):
        out["buffers"][name] = [str(tensor.dtype), list(tensor.shape)]
    for attr in dir(module):
        if attr.startswith("_") or callable(getattr(module, attr, None)):
            continue
        try:
            value = getattr(module, attr)
        except Exception:
            continue
        if isinstance(value, torch.Tensor):
            out["plain_attrs"][attr] = [str(value.dtype), list(value.shape)]
    if q is not None:
        for attr in dir(q):
            if attr.startswith("_") or callable(getattr(q, attr, None)):
                continue
            try:
                value = getattr(q, attr)
            except Exception:
                continue
            if isinstance(value, torch.Tensor):
                out["plain_attrs"][f"quant.{attr}"] = [str(value.dtype),
                                                       list(value.shape)]
    return out


def clone_state(self, module_name):
    """Cache weight+weight_scale of EVERY int8 module in the model."""
    import torch
    model = self.model_runner.get_model()
    tensors = {}
    for name, module in model.named_modules():
        w = getattr(module, "weight", None)
        if w is None or str(getattr(w, "dtype", "")) not in ("torch.int8", "torch.uint8"):
            continue
        tensors[f"{name}.weight"] = w.detach().clone()
        ws = getattr(module, "weight_scale", None)
        if isinstance(ws, torch.Tensor):
            tensors[f"{name}.weight_scale"] = ws.detach().clone()
    cache = getattr(self, "_search_cache", None)
    if cache is None:
        cache = {}
        self._search_cache = cache
    cache["probe3"] = {"tensors": tensors}
    return len(tensors)


def patch_reload_readback(self, module_name, edits):
    """Patch cached int8 weight, reload the FULL cached set, read back live."""
    import torch
    from vllm.config import set_current_vllm_config

    cache = self._search_cache["probe3"]
    tensors = cache["tensors"]
    key = f"{module_name}.weight"
    flat = tensors[key].view(torch.int8).reshape(-1)
    old = []
    for flat_index, new_code in edits:
        old.append(int(flat[int(flat_index)]))
        flat[int(flat_index)] = int(new_code)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with set_current_vllm_config(self.vllm_config):
        self.model_runner.reload_weights(
            weights_iterator=iter(list(tensors.items())),
            is_checkpoint_format=False,
        )
    torch.cuda.synchronize()
    model = self.model_runner.get_model()
    named = dict(model.named_modules())
    live = named[module_name].weight.detach().cpu().view(torch.int8).reshape(-1)
    readback = [int(live[int(i)]) for i, _c in edits]
    return {"reload_seconds": round(time.perf_counter() - start, 4),
            "old_codes": old, "readback": readback,
            "applied": readback == [c for _i, c in edits],
            "n_tensors_reloaded": len(tensors)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--linear-backend", default="triton")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-mem", type=float, default=0.30)
    args = parser.parse_args()

    from vllm import LLM
    llm = LLM(
        model=args.model, quantization="compressed-tensors", dtype="bfloat16",
        max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
        enable_prefix_caching=False, trust_remote_code=True, disable_log_stats=True,
        linear_backend=args.linear_backend,
    )
    module = "model.layers.0.self_attn.qkv_proj"
    report: dict = {"protocol": "minicpm5-int8-reload-probe/v1"}
    report["module_state"] = llm.collective_rpc(dump_module_state,
                                                kwargs={"module_name": module})[0]
    report["cached"] = llm.collective_rpc(clone_state,
                                          kwargs={"module_name": module})[0]
    edits = [[12345, 37], [543210, -22]]
    report["patch_reload"] = llm.collective_rpc(
        patch_reload_readback,
        kwargs={"module_name": module, "edits": edits})[0]

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "int8_reload_probe.json").write_text(
        json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=1)[:2600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
