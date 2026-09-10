#!/usr/bin/env python3
"""Probe 5: the only test that matters -- does a weight mutation change what the
engine GENERATES?  Readback handles are unreliable (probe 4 showed write-visible-
on-one-handle, invisible-on-another with identical data_ptr).  Execution is the
ground truth the campaign needs.

Arms:
  A. baseline generation (fixed prompts, greedy);
  B. direct big mutation (layer-0 qkv: +1 to 8192 codes) -> generate;
  C. reload-based mutation (cache full set, zero 4096 codes, reload) -> generate.

If B or C changes outputs, the corresponding write path reaches the kernel.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

PROMPTS = [
    "What is 13 - 16 - 18? Reply with only the final answer.",
    "Anna is taller than Ben. Ben is taller than Cara. Who is tallest? Reply with only the name.",
    "What is 15% of 240? Reply with only the number.",
    "What is 47 modulo 9? Reply with only the number.",
]


def mutate_direct(self, module_name, count, delta):
    import torch
    model = self.model_runner.get_model()
    module = dict(model.named_modules())[module_name]
    flat = module.weight.data.view(torch.int8).reshape(-1)
    g = torch.Generator(device="cpu").manual_seed(20260910)
    idx = torch.randperm(flat.numel(), generator=g)[:count].to(flat.device)
    old = flat[idx].clone()
    flat[idx] = (flat[idx].to(torch.int16) + int(delta)).clamp(-127, 127).to(torch.int8)
    # read back through the SAME handle only (fresh handles are unreliable)
    return {"n": int(count), "sample_before": old[:4].tolist(),
            "sample_after": flat[idx][:4].tolist()}


def cache_all(self):
    import torch
    model = self.model_runner.get_model()
    tensors = {}
    for name, module in model.named_modules():
        w = getattr(module, "weight", None)
        if w is None or str(getattr(w, "dtype", "")) != "torch.int8":
            continue
        tensors[f"{name}.weight"] = w.detach().clone()
        ws = getattr(module, "weight_scale", None)
        if ws is not None:
            tensors[f"{name}.weight_scale"] = ws.detach().clone()
    cache = getattr(self, "_search_cache", None)
    if cache is None:
        cache = {}
        self._search_cache = cache
    cache["probe5"] = {"tensors": tensors}
    return len(tensors)


def mutate_cache_reload(self, module_name, count):
    import torch
    from vllm.config import set_current_vllm_config

    tensors = self._search_cache["probe5"]["tensors"]
    key = f"{module_name}.weight"
    flat = tensors[key].view(torch.int8).reshape(-1)
    g = torch.Generator(device="cpu").manual_seed(20260911)
    idx = torch.randperm(flat.numel(), generator=g)[:count].to(flat.device)
    flat[idx] = 0
    torch.cuda.synchronize()
    with set_current_vllm_config(self.vllm_config):
        self.model_runner.reload_weights(
            weights_iterator=iter(list(tensors.items())),
            is_checkpoint_format=False,
        )
    torch.cuda.synchronize()
    return {"n": int(count), "zeroed": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--linear-backend", default="triton")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-mem", type=float, default=0.30)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model, quantization="compressed-tensors", dtype="bfloat16",
        max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
        enable_prefix_caching=False, trust_remote_code=True, disable_log_stats=True,
        linear_backend=args.linear_backend,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=12)
    module = "model.layers.0.self_attn.qkv_proj"

    def gen(tag):
        outs = llm.generate(PROMPTS, sp)
        return {tag: [list(o.outputs[0].token_ids) for o in outs]}

    report: dict = {"protocol": "minicpm5-int8-execution-probe/v1"}
    report["A_baseline"] = gen("A")

    report["B_mutate"] = llm.collective_rpc(
        mutate_direct, kwargs={"module_name": module, "count": 8192,
                               "delta": 1})[0]
    report["B_generation"] = gen("B")

    n = llm.collective_rpc(cache_all)[0]
    report["C_cached_tensors"] = n
    report["C_mutate"] = llm.collective_rpc(
        mutate_cache_reload, kwargs={"module_name": module, "count": 4096})[0]
    report["C_generation"] = gen("C")

    report["B_changes_outputs"] = report["A_baseline"]["A"] != report["B_generation"]["B"]
    report["C_changes_outputs"] = report["A_baseline"]["A"] != report["C_generation"]["C"]
    report["B_equals_C"] = report["B_generation"]["B"] == report["C_generation"]["C"]

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "int8_execution_probe.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("A_baseline", "B_mutate", "B_generation", "C_mutate",
                       "C_generation", "B_changes_outputs", "C_changes_outputs",
                       "B_equals_C")}, indent=1)[:1800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
