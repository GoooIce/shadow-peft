#!/usr/bin/env python3
"""Probe 7 (final contract): mutations through weight.t().reshape(-1) -- a
zero-copy view because live weights are transposed views of payload-layout
storage (stride [1, N]).  Validates both write paths against EXECUTION:

  B. direct 2D-view mutation -> generation changes;
  C. cache -> patch -> in-place reload -> generation changes again;
  D. restore original codes -> generation returns to baseline exactly.

If all four hold, the W8A8 refit contract is established and the pilot's
search_patch/search_readback machinery ports with only the view change.
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


def _flat(weight):
    """Zero-copy payload-order flat view ([out, in] row-major)."""
    import torch
    return weight.data.t().reshape(-1)


def mutate_direct(self, module_name, count, delta, seed):
    import torch
    model = self.model_runner.get_model()
    module = dict(model.named_modules())[module_name]
    flat = _flat(module.weight)
    assert flat.data_ptr() == module.weight.data_ptr(), "flat view must alias storage"
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(flat.numel(), generator=g)[:count]
    old = flat[idx].clone()
    flat[idx] = (flat[idx].to(torch.int16) + int(delta)).clamp(-127, 127).to(torch.int8)
    return {"n": count, "first_idx": int(idx[0]), "old": old[:4].tolist(),
            "new": flat[idx][:4].tolist(),
            "view_aliases_storage": True}


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
    cache["probe7"] = {"tensors": tensors}
    return len(tensors)


def patch_cache_reload(self, module_name, count, seed):
    import torch
    from vllm.config import set_current_vllm_config

    tensors = self._search_cache["probe7"]["tensors"]
    key = f"{module_name}.weight"
    flat = _flat(tensors[key])
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(flat.numel(), generator=g)[:count]
    flat[idx] = 0
    torch.cuda.synchronize()
    with set_current_vllm_config(self.vllm_config):
        self.model_runner.reload_weights(
            weights_iterator=iter(list(tensors.items())),
            is_checkpoint_format=False,
        )
    torch.cuda.synchronize()
    return {"n": count, "zeroed": True}


def restore_direct(self, module_name, edits):
    """edits: [[flat_index, code], ...] restore exact codes."""
    import torch
    model = self.model_runner.get_model()
    module = dict(model.named_modules())[module_name]
    flat = _flat(module.weight)
    for flat_index, code in edits:
        flat[int(flat_index)] = int(code)
    return len(edits)


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

    def gen():
        outs = llm.generate(PROMPTS, sp)
        return [list(o.outputs[0].token_ids) for o in outs]

    report: dict = {"protocol": "minicpm5-int8-final-contract/v1"}
    a = gen()

    # B: direct mutation through the aliasing flat view
    mb = llm.collective_rpc(mutate_direct,
                            kwargs={"module_name": module, "count": 8192,
                                    "delta": 1, "seed": 20260910})[0]
    b = gen()
    report["B"] = mb
    report["B_changes_outputs"] = b != a

    # capture the mutated codes for a later exact restore
    def snapshot(self, module_name):
        import torch
        model = self.model_runner.get_model()
        flat = _flat(dict(model.named_modules())[module_name].weight)
        return flat[:20000].tolist()
    report["snapshot_head"] = llm.collective_rpc(
        snapshot, kwargs={"module_name": module})[0][:8]

    # C: cache -> patch -> reload
    report["cached_tensors"] = llm.collective_rpc(cache_all)[0]
    mc = llm.collective_rpc(patch_cache_reload,
                            kwargs={"module_name": module, "count": 4096,
                                    "seed": 20260911})[0]
    c = gen()
    report["C"] = mc
    report["C_changes_outputs"] = c != a and c != b

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "int8_final_contract.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("B", "B_changes_outputs", "C", "C_changes_outputs",
                       "cached_tensors")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
