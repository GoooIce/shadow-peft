#!/usr/bin/env python3
"""Worker-side RPC callables for ticket 11 (refit feasibility gate).

Every public function here is meant to be handed to ``LLM.collective_rpc`` as a
*callable*.  vLLM cloudpickles it, ships it to the worker process and invokes it
as ``func(worker, *args, **kwargs)`` (``vllm/v1/serial_utils.py:run_method``).
``worker.model_runner.get_model()`` is the live ``nn.Module``.

The functions deliberately return only summaries (pointers, shapes, hashes,
timings) -- bulk tensors stay on the machine and go to disk via the
``dump_*``/``capture_*`` helpers.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import torch

# Kernel-format NVFP4 parameters as they exist *after*
# ``CompressedTensorsW4A4Fp4.process_weights_after_loading``.
KERNEL_PARAMS = (
    "weight",
    "weight_scale",
    "weight_global_scale",
    "input_global_scale",
    "input_global_scale_inv",
    "alpha",
)


def _quant_modules(model):
    modules = {}
    for name, module in model.named_modules():
        if hasattr(module, "alpha") and hasattr(module, "weight_scale"):
            modules[name] = module
    return modules


def _data(obj):
    return obj.data if hasattr(obj, "data") else obj


def _sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().to("cpu").contiguous()
    if cpu.dim() == 0:
        cpu = cpu.reshape(1)
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def _select(modules, names, limit):
    selected = sorted(modules)
    if names:
        wanted = set(names)
        selected = [name for name in selected if name in wanted]
    if limit:
        selected = selected[:limit]
    return selected


def _timed_sync():
    torch.cuda.synchronize()
    return time.perf_counter()


# --------------------------------------------------------------------------- #
# census / kernel identity
# --------------------------------------------------------------------------- #


def kernel_census(self):
    """Which NVFP4 linear kernel class each quantised layer actually uses.

    ``module.quant_method`` is the vLLM ``QuantizeMethodBase`` wrapper; the
    compressed-tensors scheme (which owns the selected kernel) hangs off
    ``.scheme``.  Fall back to ``.kernel`` for uncompressed wrappers.
    """
    model = self.model_runner.get_model()
    counts: dict[str, int] = {}
    per_layer = {}
    for name, module in _quant_modules(model).items():
        # vLLM stores the compressed-tensors scheme on the *layer*
        # (``layer.scheme``); the selected NVFP4 linear kernel hangs off it.
        kernel = None
        for holder in (
            getattr(module, "scheme", None),
            getattr(module, "quant_method", None),
            getattr(getattr(module, "quant_method", None), "scheme", None),
        ):
            kernel = getattr(holder, "kernel", None)
            if kernel is not None:
                break
        key = type(kernel).__name__ if kernel is not None else "none"
        counts[key] = counts.get(key, 0) + 1
        per_layer[name] = key
    return {"counts": counts, "per_layer": per_layer}


def census(self, hash_params=(), limit=0, names=None):
    """Pointer/shape/dtype census (+ optional value hashes) of kernel params."""
    model = self.model_runner.get_model()
    modules = _quant_modules(model)
    rows = {}
    for name in _select(modules, names, limit):
        module = modules[name]
        row = {}
        for pname in KERNEL_PARAMS:
            param = getattr(module, pname, None)
            if param is None:
                continue
            tensor = _data(param)
            entry = {
                "ptr": tensor.data_ptr(),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
            }
            if pname in hash_params:
                entry["sha256"] = _sha256(tensor)
            row[pname] = entry
        rows[name] = row
    return {"count": len(rows), "layers": rows}


def graph_census(self):
    """CUDA-graph bookkeeping before/after a reload (recapture detector)."""
    runner = self.model_runner
    out = {}
    for attr, value in vars(runner).items():
        graphs = getattr(value, "graphs", None)
        if isinstance(graphs, dict):
            out[attr] = {
                "num_graphs": len(graphs),
                "keys": [str(key) for key in list(graphs)[:32]],
                "graph_ids": [id(graph) for graph in list(graphs.values())[:32]],
                "graphs_captured": getattr(value, "_graphs_captured", None),
            }
    return out


def dump_kernel_params(self, out_path, names=None):
    """Snapshot the engine's kernel-format params to safetensors."""
    from safetensors.torch import save_file

    model = self.model_runner.get_model()
    modules = _quant_modules(model)
    tensors = {}
    for name in _select(modules, names, 0):
        module = modules[name]
        for pname in KERNEL_PARAMS:
            param = getattr(module, pname, None)
            if param is None:
                continue
            tensors[f"{name}.{pname}"] = (
                _data(param).detach().to("cpu").contiguous().clone()
            )
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
    return {
        "path": str(path),
        "tensors": len(tensors),
        "bytes": int(sum(t.numel() * t.element_size() for t in tensors.values())),
    }


# --------------------------------------------------------------------------- #
# reload paths
# --------------------------------------------------------------------------- #


def reload_checkpoint(self, weights_path):
    """``is_checkpoint_format=True``: load_weights + PWAL, copy into old storage."""
    from vllm.config import set_current_vllm_config

    runner = self.model_runner
    start = _timed_sync()
    with set_current_vllm_config(self.vllm_config):
        runner.reload_weights(weights_path=weights_path, is_checkpoint_format=True)
    seconds = _timed_sync() - start
    return {"seconds": seconds, "weights_path": weights_path}


def kf_prepare(self, path, cache_key, device="cpu"):
    """Load a kernel-format safetensors file into a worker-side cache."""
    from safetensors.torch import load_file

    start = time.perf_counter()
    tensors = load_file(path, device=device)
    seconds = time.perf_counter() - start
    cache = getattr(self, "_kf_cache", None)
    if cache is None:
        cache = {}
        self._kf_cache = cache
    cache[cache_key] = tensors
    return {
        "cache_key": cache_key,
        "tensors": len(tensors),
        "load_seconds": seconds,
        "device": device,
        "bytes": int(sum(t.numel() * t.element_size() for t in tensors.values())),
    }


def _mutate(items, mutate):
    """Apply a small mutation spec to the (name, tensor) list."""
    out = []
    for name, tensor in items:
        rule = None
        for candidate in mutate:
            if name == candidate["name"]:
                rule = candidate
                break
        if rule is None:
            out.append((name, tensor))
            continue
        if rule.get("skip"):
            continue
        if "factor" in rule:
            # Go through float so the mutation also works on fp8 / uint8 params.
            out.append((name, (tensor.float() * float(rule["factor"])).to(tensor.dtype)))
        else:
            out.append((name, tensor))
    return out


def kf_reload(self, cache_key=None, path=None, mutate=None, device="cpu"):
    """``is_checkpoint_format=False``: in-place ``param.copy_`` of kernel tensors."""
    from safetensors.torch import load_file
    from vllm.config import set_current_vllm_config

    runner = self.model_runner
    if cache_key is not None:
        tensors = self._kf_cache[cache_key]
    else:
        tensors = load_file(path, device=device)
    items = list(tensors.items())
    if mutate:
        items = _mutate(items, mutate)
    start = _timed_sync()
    with set_current_vllm_config(self.vllm_config):
        runner.reload_weights(weights_iterator=iter(items), is_checkpoint_format=False)
    seconds = _timed_sync() - start
    return {"seconds": seconds, "num_tensors": len(items)}


def kf_materialize(self, cache_key):
    """Force lazy safetensors tensors into real CPU memory (first-touch cost)."""
    tensors = self._kf_cache[cache_key]
    start = time.perf_counter()
    total = 0
    for name, tensor in list(tensors.items()):
        clone = tensor.clone()
        tensors[name] = clone
        total += clone.numel() * clone.element_size()
    return {
        "seconds": time.perf_counter() - start,
        "bytes": total,
        "tensors": len(tensors),
    }


def kf_reload_repeat(self, cache_key, repeats=5):
    """Steady-state KF reload latency distribution."""
    times = []
    for _ in range(repeats):
        times.append(kf_reload(self, cache_key=cache_key)["seconds"])
    ordered = sorted(times)
    return {
        "times": times,
        "median": ordered[len(ordered) // 2],
        "min": ordered[0],
        "max": ordered[-1],
    }


def kf_copy_floor(self, cache_key):
    """Pure device-side ``copy_`` floor over the same tensor set (no loader)."""
    model = self.model_runner.get_model()
    tensors = self._kf_cache[cache_key]
    params = dict(model.named_parameters())
    missing = [name for name in tensors if name not in params]
    start = _timed_sync()
    for name, tensor in tensors.items():
        if name in params:
            params[name].data.copy_(tensor)
    seconds = _timed_sync() - start
    return {"seconds": seconds, "num_tensors": len(tensors), "missing": missing[:8]}


# --------------------------------------------------------------------------- #
# payload <-> engine bitwise readback
# --------------------------------------------------------------------------- #


def readback_payload(self, payload_path, out_path, names=None, limit=0):
    """Compare engine kernel params with the payload file bit by bit.

    Expected values are rebuilt with vLLM's own helpers
    (``pad_nvfp4_weight_for_cutlass`` / ``swizzle_blockscale``) from the
    compressed-tensors checkpoint tensors, so this tests the *load path*
    end to end rather than a re-implementation.
    """
    import json

    from safetensors import safe_open

    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        pad_nvfp4_weight_for_cutlass,
        swizzle_blockscale,
    )

    model = self.model_runner.get_model()
    modules = _quant_modules(model)
    rows = []
    all_weight_equal = True
    all_scale_equal = True
    all_scalar_equal = True
    with safe_open(payload_path, framework="pt", device="cpu") as handle:
        for name in _select(modules, names, limit):
            module = modules[name]
            packed = handle.get_tensor(f"{name}.weight_packed")
            scale = handle.get_tensor(f"{name}.weight_scale")
            wgs_stored = handle.get_tensor(f"{name}.weight_global_scale").float().reshape(())
            igs_stored = handle.get_tensor(f"{name}.input_global_scale").float().reshape(())

            expected_weight, padding_cols = pad_nvfp4_weight_for_cutlass(packed)
            expected_scale = swizzle_blockscale(scale)

            engine_weight = _data(module.weight)
            engine_scale = _data(module.weight_scale)
            weight_equal = bool(
                engine_weight.shape == expected_weight.shape
                and torch.equal(engine_weight.cpu(), expected_weight)
            )
            scale_equal = bool(
                engine_scale.shape == expected_scale.shape
                and torch.equal(engine_scale.cpu(), expected_scale.cpu())
            )
            expected = {
                "weight_global_scale": float(1.0 / wgs_stored),
                "input_global_scale": float(1.0 / igs_stored),
                "input_global_scale_inv": float(igs_stored),
                "alpha": float((1.0 / wgs_stored) * (1.0 / igs_stored)),
            }
            actual = {
                key: float(_data(getattr(module, key)))
                for key in expected
                if getattr(module, key, None) is not None
            }
            scalar_equal = all(
                actual[key] == expected[key] for key in expected if key in actual
            )
            all_weight_equal &= weight_equal
            all_scale_equal &= scale_equal
            all_scalar_equal &= scalar_equal
            rows.append(
                {
                    "layer": name,
                    "weight_equal": weight_equal,
                    "weight_scale_equal": scale_equal,
                    "scalars_equal": scalar_equal,
                    "padding_cols": int(padding_cols),
                    "engine_weight_shape": list(engine_weight.shape),
                    "expected_weight_shape": list(expected_weight.shape),
                    "engine_weight_sha256": _sha256(engine_weight),
                    "expected_weight_sha256": _sha256(expected_weight),
                    "engine_scale_sha256": _sha256(engine_scale),
                    "expected_scale_sha256": _sha256(expected_scale),
                    "expected_scalars": expected,
                    "engine_scalars": actual,
                    "engine_weight_scale_min": float(engine_scale.float().min()),
                    "engine_weight_scale_max": float(engine_scale.float().max()),
                }
            )
    summary = {
        "payload": payload_path,
        "layers": len(rows),
        "all_weight_bitwise_equal": all_weight_equal,
        "all_weight_scale_bitwise_equal": all_scale_equal,
        "all_scalars_exact_equal": all_scalar_equal,
        "rows": rows,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {key: summary[key] for key in summary if key != "rows"}


# --------------------------------------------------------------------------- #
# per-layer hidden-state capture (hook based, real engine forward)
# --------------------------------------------------------------------------- #


def capture_register(self, layer_regex=r"\.layers\.\d+$", class_name="DecoderLayer"):
    """Register forward hooks on decoder layers; store handles on the worker.

    Falls back to class-name matching (``LlamaDecoderLayer``) when the module
    tree carries compile wrappers that defeat name matching.
    """
    import re

    model = self.model_runner.get_model()
    captured: dict[str, torch.Tensor] = {}
    pattern = re.compile(layer_regex)

    def make_hook(name):
        def hook(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor.detach().to("cpu", torch.float32)

        return hook

    matched = [name for name, _ in model.named_modules() if pattern.search(name)]
    strategy = "regex"
    if not matched:
        matched = [
            name
            for name, module in model.named_modules()
            if class_name in type(module).__name__
        ]
        strategy = "class_name"
    handles = []
    for name, module in model.named_modules():
        if name in matched:
            handles.append(module.register_forward_hook(make_hook(name)))
    # Record the token ids each forward actually saw, so the offline analysis
    # can prove both engines compared the same positions.
    inputs: list[torch.Tensor] = []
    embed = None
    for name, module in model.named_modules():
        if name.endswith("embed_tokens"):
            embed = module
            break
    if embed is not None:
        handles.append(
            embed.register_forward_pre_hook(
                lambda _m, args: inputs.append(args[0].detach().to("cpu"))
            )
        )
    self._capture_state = {
        "captured": captured,
        "handles": handles,
        "inputs": inputs,
    }
    return {
        "registered": len(handles),
        "strategy": strategy,
        "matched": matched[:6],
        "module_count": sum(1 for _ in model.named_modules()),
    }


def capture_collect(self, out_path):
    """Persist the hook-captured activations and drop the hooks."""
    state = getattr(self, "_capture_state", None)
    if state is None:
        raise RuntimeError("capture_register was not called")
    for handle in state["handles"]:
        handle.remove()
    captured = state["captured"]
    inputs = state.get("inputs", [])
    torch.save({"layers": captured, "inputs": inputs}, out_path)
    self._capture_state = None
    return {
        "path": out_path,
        "layers": len(captured),
        "inputs": [list(tensor.shape) for tensor in inputs[-3:]],
        "input_token_ids": [int(value) for value in inputs[-1].reshape(-1)[:24]]
        if inputs
        else None,
        "shapes": {name: list(tensor.shape) for name, tensor in list(captured.items())[:4]},
    }
