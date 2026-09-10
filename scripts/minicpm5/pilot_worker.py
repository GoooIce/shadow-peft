#!/usr/bin/env python3
"""Worker-side RPCs for the ticket-14 pilot loop.

Everything here is handed to ``LLM.collective_rpc`` as a *callable*: vLLM
cloudpickles it into the EngineCore worker and calls ``func(worker, ...)``.
The functions keep the candidate's **kernel-format** tensors resident on the
worker (ticket 11: 31 ms pure-copy refit) and mutate only the nibbles a
candidate actually edited, which is what makes per-candidate refit cheap.

``search_load``      -> load the exported kernel-format file into a worker cache
``search_patch``     -> apply sparse E2M1 code edits to the cached ``weight``
``search_readback``  -> read back codes at the edited coordinates (patch proof)
``search_hash``      -> per-module sha256 of cached ``weight`` (drift detector)
"""

from __future__ import annotations

import hashlib
import time

import torch


def _modules(self):
    model = self.model_runner.get_model()
    out = {}
    for name, module in model.named_modules():
        if hasattr(module, "alpha") and hasattr(module, "weight_scale"):
            out[name] = module            # NVFP4 kernel modules
        elif (hasattr(module, "weight_scale")
              and getattr(getattr(module, "weight", None), "dtype", None) is not None
              and str(module.weight.dtype) == "torch.int8"):
            out[name] = module            # W8A8 int8 kernel modules
    return out


def _payload_view(tensor: "torch.Tensor"):
    """Payload-order ``[out, in]`` zero-copy view of an int8 kernel weight.

    The live weight (and any faithful clone of it) is ``[in, out]`` -- a
    transposed view of payload-layout storage.  ``.t()`` returns a view, but
    any ``reshape(-1)``/``flatten`` on either side materialises a **copy** and
    the write silently never reaches the engine (the trap the 2026-09-10
    probes documented).  Element reads/writes through the 2D view always alias
    the storage.
    """
    return tensor.t()


def _read_code(view, flat_index: int, in_features: int) -> int:
    row, col = divmod(int(flat_index), in_features)
    return int(view[row, col])


def _write_code(view, flat_index: int, in_features: int, code: int) -> bool:
    row, col = divmod(int(flat_index), in_features)
    if int(view[row, col]) == int(code):
        return False
    view[row, col] = int(code)
    return True


def _data(obj):
    return obj.data if hasattr(obj, "data") else obj


def _weight_bytes(tensor: torch.Tensor):
    """Flat uint8 view over the packed kernel weight (dtype is already uint8)."""
    assert tensor.dtype == torch.uint8, f"unexpected kernel weight dtype {tensor.dtype}"
    return tensor.reshape(-1)


def search_load(self, path, cache_key, device="cpu", clone=True):
    """Load a kernel-format safetensors file into a *mutable* worker cache."""
    from safetensors.torch import load_file

    start = time.perf_counter()
    tensors = load_file(path, device=device)
    if clone:
        # safetensors hands out memory-mapped / shared storage; clone so the
        # per-candidate nibble patches never touch the file's pages.
        tensors = {name: tensor.clone() for name, tensor in tensors.items()}
    seconds = time.perf_counter() - start
    cache = getattr(self, "_search_cache", None)
    if cache is None:
        cache = {}
        self._search_cache = cache
    cache[cache_key] = {"tensors": tensors}
    return {
        "cache_key": cache_key,
        "tensors": len(tensors),
        "load_seconds": round(seconds, 3),
        "bytes": int(sum(t.numel() * t.element_size() for t in tensors.values())),
    }


def search_patch(self, cache_key, edits, verify=True):
    """Apply ``{module: [[flat_index, new_code], ...]}`` to cached weights.

    ``flat_index`` is the row-major index inside the fused engine weight in
    **payload order** (``row * in_features + column`` over ``[out_fused, in]``);
    on the NVFP4 kernel layout nibble order follows compressed-tensors (even
    index = low nibble), on the int8 layout codes are single signed bytes
    written through the transposed 2D view (never a reshape -- that copies).
    """
    import torch

    cache = self._search_cache[cache_key]
    tensors = cache["tensors"]
    start = time.perf_counter()
    touched = {}
    changed = 0
    for module, items in edits.items():
        tensor = tensors[f"{module}.weight"]
        if tensor.dtype == torch.int8:
            view = _payload_view(tensor)
            in_features = int(tensor.shape[0])
            touched[module] = (view, in_features)
            for flat_index, new_code in items:
                changed += _write_code(view, flat_index, in_features, new_code)
            continue
        flat = _weight_bytes(tensor)
        touched[module] = (flat, None)
        for flat_index, new_code in items:
            position = int(flat_index) // 2
            shift = 4 * (int(flat_index) % 2)
            byte = int(flat[position])
            if ((byte >> shift) & 0x0F) == int(new_code):
                continue
            flat[position] = (byte & ~(0x0F << shift)) | (int(new_code) << shift)
            changed += 1
    seconds = time.perf_counter() - start
    readback = {}
    if verify:
        for module, items in edits.items():
            source, in_features = touched[module]
            if in_features is None:
                readback[module] = [
                    int((int(source[int(index) // 2]) >> (4 * (int(index) % 2))) & 0x0F)
                    for index, _code in items
                ]
            else:
                readback[module] = [
                    _read_code(source, index, in_features) for index, _code in items
                ]
    return {
        "modules": len(edits),
        "bytes_written": changed,
        "seconds": round(seconds, 4),
        "readback": readback,
    }


def search_readback(self, cache_key, queries, source="cache"):
    """Read codes at ``{module: [flat_index, ...]}``.

    ``source="cache"`` reads the worker-side candidate cache (what the refit
    is about to copy), ``source="engine"`` reads the *live* parameter of the
    running model -- the two must agree, and only the second one is what the
    kernel dump sees.
    """
    import torch

    if source == "engine":
        modules = _modules(self)
        tensors = {f"{module}.weight": modules[module].weight for module in queries}
    else:
        cache = self._search_cache[cache_key]
        tensors = cache["tensors"]
    out = {}
    for module, indices in queries.items():
        tensor = tensors[f"{module}.weight"]
        if tensor.dtype == torch.int8:
            view = _payload_view(tensor)
            in_features = int(tensor.shape[0])
            out[module] = [_read_code(view, index, in_features) for index in indices]
        else:
            flat = _weight_bytes(tensor)
            out[module] = [
                int((int(flat[int(index) // 2]) >> (4 * (int(index) % 2))) & 0x0F)
                for index in indices
            ]
    return out


def search_hash(self, cache_key, modules=None):
    """sha256 of the cached ``weight`` tensor per module (cheap drift detector).

    int8 weights hash over the payload-order view made contiguous, matching
    the kernel-format export byte convention.
    """
    import torch

    cache = self._search_cache[cache_key]
    tensors = cache["tensors"]
    names = modules or sorted({key.rsplit(".", 1)[0] for key in tensors})
    out = {}
    for name in names:
        tensor = tensors.get(f"{name}.weight")
        if tensor is None:
            out[name] = None
            continue
        if tensor.dtype == torch.int8:
            payload = _payload_view(tensor).contiguous().numpy().tobytes()
        else:
            payload = tensor.detach().cpu().contiguous().numpy().tobytes()
        out[name] = hashlib.sha256(payload).hexdigest()[:24]
    return out


def search_reload(self, cache_key):
    """In-place kernel-format reload from the (already patched) worker cache."""
    from vllm.config import set_current_vllm_config

    runner = self.model_runner
    tensors = self._search_cache[cache_key]["tensors"]
    torch.cuda.synchronize()
    start = time.perf_counter()
    with set_current_vllm_config(self.vllm_config):
        runner.reload_weights(
            weights_iterator=iter(list(tensors.items())), is_checkpoint_format=False
        )
    torch.cuda.synchronize()
    return {"seconds": time.perf_counter() - start, "tensors": len(tensors)}


def search_dump(self, out_path, modules=None):
    """Dump kernel params for the hard gate (all params, or a module subset)."""
    from safetensors.torch import save_file

    from refit_worker import KERNEL_PARAMS

    modules_map = _modules(self)
    names = sorted(modules_map) if modules is None else list(modules)
    tensors = {}
    for name in names:
        module = modules_map[name]
        for param in KERNEL_PARAMS:
            value = getattr(module, param, None)
            if value is None:
                continue
            tensors[f"{name}.{param}"] = _data(value).detach().to("cpu").contiguous().clone()
    save_file(tensors, str(out_path))
    return {
        "path": str(out_path),
        "tensors": len(tensors),
        "bytes": int(sum(t.numel() * t.element_size() for t in tensors.values())),
    }
