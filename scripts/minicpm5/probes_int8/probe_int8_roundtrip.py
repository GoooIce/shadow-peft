#!/usr/bin/env python3
"""Contract test: live int8 weight == payload.T (bitwise), and a real code edit
survives patch -> in-place reload -> engine readback.

Follow-up to probe_int8_live.py (which established: fused module names,
weight int8 [in, out], scale fp32 [out, 1], no-op reload accepted).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def hash_bytes(t):
    import torch
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def transpose_check(self, modules):
    import torch
    model = self.model_runner.get_model()
    named = dict(model.named_modules())
    out = {}
    for module in modules:
        w = named[module].weight
        out[module] = {"live_hash": hash_bytes(w),
                       "live_shape": list(w.shape),
                       "live_dtype": str(w.dtype),
                       "live_stride": list(w.stride())}
    return out


def clone_all(self, modules):
    """Cache int8 clones of the live weights (the refit base)."""
    import torch
    model = self.model_runner.get_model()
    named = dict(model.named_modules())
    cache = getattr(self, "_search_cache", None)
    if cache is None:
        cache = {}
        self._search_cache = cache
    tensors = {}
    for module in modules:
        tensors[f"{module}.weight"] = named[module].weight.detach().clone()
    cache["probe"] = {"tensors": tensors}
    return sorted(tensors)


def patch_and_reload(self, edits):
    """{module: [[flat_index, new_code], ...]} on the cached int8 weights,
    then in-place reload -- the exact candidate-refit contract."""
    import torch
    from vllm.config import set_current_vllm_config

    cache = self._search_cache["probe"]
    written = {}
    for module, items in edits.items():
        flat = cache["tensors"][f"{module}.weight"].view(torch.int8).reshape(-1)
        for flat_index, new_code in items:
            old = int(flat[int(flat_index)])
            flat[int(flat_index)] = int(new_code)
            written.setdefault(module, []).append([int(flat_index), old])
    torch.cuda.synchronize()
    start = time.perf_counter()
    with set_current_vllm_config(self.vllm_config):
        self.model_runner.reload_weights(
            weights_iterator=iter(list(cache["tensors"].items())),
            is_checkpoint_format=False,
        )
    torch.cuda.synchronize()
    return {"reload_seconds": round(time.perf_counter() - start, 4),
            "written": written}


def engine_readback(self, queries):
    """{module: [flat_index, ...]} -> live engine int8 codes."""
    import torch
    model = self.model_runner.get_model()
    named = dict(model.named_modules())
    out = {}
    for module, indices in queries.items():
        flat = named[module].weight.detach().cpu().view(torch.int8).reshape(-1)
        out[module] = [int(flat[int(i)]) for i in indices]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--linear-backend", default="triton")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-mem", type=float, default=0.30)
    args = parser.parse_args()

    from vllm import LLM

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    llm = LLM(
        model=args.model, quantization="compressed-tensors", dtype="bfloat16",
        max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
        enable_prefix_caching=False, trust_remote_code=True, disable_log_stats=True,
        linear_backend=args.linear_backend,
    )
    report: dict = {"protocol": "minicpm5-int8-roundtrip/v1",
                    "linear_backend": args.linear_backend}

    # 1. bitwise transpose equality on layer 0 + layer 20 (fused + square)
    modules = ["model.layers.0.self_attn.qkv_proj",
               "model.layers.0.mlp.gate_up_proj",
               "model.layers.0.self_attn.o_proj",
               "model.layers.0.mlp.down_proj",
               "model.layers.20.mlp.gate_up_proj"]
    live = llm.collective_rpc(transpose_check, kwargs={"modules": modules})[0]

    import torch
    from safetensors.torch import load_file
    tensors = load_file(str(Path(args.model).expanduser() / "model.safetensors"))

    def payload_fused_transposed(module):
        if module.endswith("qkv_proj"):
            base = module[:-8]
            parts = [tensors[f"{base}{p}.weight"] for p in ("q_proj", "k_proj", "v_proj")]
            fused = torch.cat(parts, dim=0)          # [out_fused, in]
        elif module.endswith("gate_up_proj"):
            base = module[:-12]
            parts = [tensors[f"{base}{p}.weight"] for p in ("gate_proj", "up_proj")]
            fused = torch.cat(parts, dim=0)
        else:
            fused = tensors[f"{module}.weight"]
        return fused.t().contiguous()                 # [in, out_fused]

    comparison = {}
    for module in modules:
        payload_hash = hash_bytes(payload_fused_transposed(module))
        comparison[module] = {
            "live": live[module]["live_hash"][:16],
            "payload_T": payload_hash[:16],
            "match": live[module]["live_hash"] == payload_hash,
        }
    report["transpose_bitwise"] = comparison

    # 2. real code edit: patch -> reload -> engine readback
    llm.collective_rpc(clone_all, kwargs={"modules": modules})
    edits = {"model.layers.0.self_attn.qkv_proj": [[12345, 37], [543210, -22]],
             "model.layers.0.mlp.gate_up_proj": [[77777, -100]]}
    pr = llm.collective_rpc(patch_and_reload, kwargs={"edits": edits})[0]
    queries = {m: [i for i, _c in items] for m, items in edits.items()}
    readback = llm.collective_rpc(engine_readback, kwargs={"queries": queries})[0]
    report["patch_reload"] = pr
    report["engine_readback"] = readback
    report["readback_codes_match"] = {
        m: readback[m] == [c for _i, c in items] for m, items in edits.items()}

    # 3. score sanity after the edit (the engine still runs)
    report["seconds"] = round(time.time() - started, 1)
    (out / "int8_roundtrip.json").write_text(
        json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("transpose_bitwise", "patch_reload", "engine_readback",
                       "readback_codes_match")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
