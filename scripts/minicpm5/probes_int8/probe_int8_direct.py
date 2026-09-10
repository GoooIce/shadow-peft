#!/usr/bin/env python3
"""Probe 4: does a DIRECT in-place mutation of the live int8 parameter stick?
Isolates the failure: cache -> copy_ -> readback showed old codes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def direct_mutation(self, module_name, flat_index, new_code):
    import torch
    model = self.model_runner.get_model()
    named = dict(model.named_modules())
    module = named[module_name]
    weight = module.weight
    flat = weight.data.view(torch.int8).reshape(-1)
    before = int(flat[int(flat_index)])
    # 1. direct in-place write on the live parameter
    flat[int(flat_index)] = int(new_code)
    after_write = int(flat[int(flat_index)])
    # 2. read through a fresh named_modules lookup (different path)
    fresh = dict(model.named_modules())[module_name].weight
    after_fresh = int(fresh.data.view(torch.int8).reshape(-1)[int(flat_index)])
    # 3. get_parameter path (what reload uses)
    param = module.get_parameter("weight")
    after_getparam = int(param.data.view(torch.int8).reshape(-1)[int(flat_index)])
    # 4. same id?
    return {
        "before": before,
        "after_write": after_write,
        "after_fresh_lookup": after_fresh,
        "after_get_parameter": after_getparam,
        "same_storage_fresh": fresh.data_ptr() == weight.data_ptr(),
        "same_storage_getparam": param.data_ptr() == weight.data_ptr(),
        "id_module_fresh": id(dict(model.named_modules())[module_name]) == id(module),
    }


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
    report = llm.collective_rpc(
        direct_mutation, kwargs={"module_name": module,
                                 "flat_index": 12345, "new_code": 37})[0]
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "int8_direct_probe.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
