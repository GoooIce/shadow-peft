"""Export torch-quantized Bonsai-style 1-bit/2-bit models to MLX checkpoints.

The torch quantizers store packed bit patterns in ``torch.int32``. MLX
checkpoints require the same bits under an unsigned ``U32`` safetensors
header, so exporting uses a zero-copy dtype reinterpretation. The supported
layouts are binary 1-bit (32 LSB-first codes per word) and ternary 2-bit
(16 LSB-first codes per word).

This module writes an MLX container that ``mlx_vlm.load`` / ``mlx_lm.load``
recognise via:

* ``config.json`` with a top-level quantization block and, when present, a
  matching ``text_config.quantization`` block. The 1-bit contract contains
  ``{bits, group_size}``; the 2-bit MLX affine contract also contains
  ``mode="affine"``.
* each quantized layer emitted as a ``(weight [U32], scales [F16], biases [F16])``
  triplet; the presence of ``.scales`` is the load-time trigger for
  ``nn.quantize`` to swap the ``nn.Linear`` for a ``QuantizedLinear``.

Weights larger than ``MAX_SHARD_BYTES`` are split into ``model-NNNNN-of-NNNNN``
shards with a ``model.safetensors.index.json`` map (same convention as Bonsai).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import save_file as st_save_file

# Default per-shard size cap (5 GiB), matching the Bonsai unpacked shard budget.
MAX_SHARD_BYTES = 5 * 1024 * 1024 * 1024


def _shard_keys(
    state_dict: dict[str, torch.Tensor], max_bytes: int = MAX_SHARD_BYTES
) -> tuple[dict[str, Any], list[str]]:
    """Greedily pack tensors into shards under ``max_bytes``.

    Returns ``(index, shard_names)`` where ``index`` is the safetensors index
    dict (``{"metadata": ..., "weight_map": {...}}``).
    """
    weight_map: dict[str, str] = {}
    shards: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for name, tensor in state_dict.items():
        t_bytes = tensor.numel() * tensor.element_size()
        if cur and cur_bytes + t_bytes > max_bytes:
            shards.append(cur)
            cur = []
            cur_bytes = 0
        cur.append(name)
        cur_bytes += t_bytes
    if cur:
        shards.append(cur)
    shard_names = []
    n = len(shards)
    for i, names in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors" if n > 1 else "model.safetensors"
        shard_names.append(fname)
        for nm in names:
            weight_map[nm] = fname
    index = {
        "metadata": {
            "total_size": sum(t.numel() * t.element_size() for t in state_dict.values())
        },
        "weight_map": weight_map,
    }
    return index, shard_names


def _remap_mlx_key(key: str) -> str:
    """Remap a HF/transformers weight key to the mlx_vlm gemma4_unified layout.

    transformers' ``Gemma4UnifiedForConditionalGeneration`` re-parents the
    vision tower under ``model.embed_vision.*``: patch/pos tensors land at
    ``model.embed_vision.patch_*`` / ``model.embed_vision.pos_*`` and the
    projection nests under ``model.embed_vision.multimodal_embedder.*``.
    mlx_vlm's ``Model`` expects the patch/pos tensors under
    ``model.vision_embedder.*`` and the projection flattened to
    ``model.embed_vision.embedding_projection.*`` (no ``multimodal_embedder``
    node). mlx_vlm's :py:meth:`sanitize` only strips ``model.`` and re-inserts
    ``language_model.model.``; it does not fix the vision hierarchy, so we do
    it here.
    """
    # vision patch/pos tensors: model.embed_vision.{patch_*,pos_*} -> model.vision_embedder.{...}
    if ".embed_vision.patch_" in key:
        key = key.replace(".embed_vision.patch_", ".vision_embedder.patch_")
    if ".embed_vision.pos_" in key:
        key = key.replace(".embed_vision.pos_", ".vision_embedder.pos_")
    # flatten the projection node
    if ".multimodal_embedder." in key:
        key = key.replace(".multimodal_embedder.", ".")
    return key


def _validate_bonsai_1bit_contract(
    model: nn.Module,
    manifest: dict[str, Any],
    group_size: int,
    exclude_patterns: tuple[str, ...],
) -> int:
    """Reject state that would be mislabeled as a Bonsai binary container."""
    if group_size not in (32, 64, 128):
        raise ValueError(
            f"MLX group_size must be one of 32, 64, or 128, got {group_size}"
        )

    quantization = manifest.get("quantization")
    if not isinstance(quantization, dict):
        raise ValueError("manifest must contain a quantization mapping")
    if quantization.get("bits") != 1:
        raise ValueError("1-bit MLX export requires manifest quantization.bits == 1")
    if quantization.get("group_size") != group_size:
        raise ValueError(
            "manifest quantization.group_size must exactly match the export group_size"
        )
    method = quantization.get("method")
    variant = quantization.get("variant")
    if method not in (None, "bonsai") or variant != "sign":
        raise ValueError(
            "1-bit MLX export requires a Bonsai/sign manifest; call "
            "quantize_model_1bit(..., method='bonsai') first"
        )

    quantized_names = manifest.get("quantized", [])
    if not isinstance(quantized_names, list) or not all(
        isinstance(name, str) for name in quantized_names
    ):
        raise ValueError("manifest['quantized'] must be a list of module names")
    if not quantized_names:
        raise ValueError("manifest['quantized'] must contain at least one module")
    if len(set(quantized_names)) != len(quantized_names):
        raise ValueError("manifest['quantized'] contains duplicate module names")

    state = model.state_dict()
    buffered_bases = {
        name.rsplit(".", 1)[0]
        for name in state
        if name.endswith((".scales", ".biases"))
    }
    observed_quantized = {
        base
        for base in buffered_bases
        if f"{base}.weight" in state
        and (
            state[f"{base}.weight"].dtype == torch.int32
            or (
                f"{base}.scales" in state
                and f"{base}.biases" in state
            )
        )
    }
    observed_quantized.update(
        name.rsplit(".", 1)[0]
        for name, tensor in state.items()
        if name.endswith(".weight") and tensor.dtype == torch.int32
    )
    unlisted = sorted(observed_quantized - set(quantized_names))
    if unlisted:
        raise ValueError(
            f"state_dict contains quantized modules missing from manifest: {unlisted}"
        )

    included_count = 0
    for base in quantized_names:
        weight_key = f"{base}.weight"
        scale_key = f"{base}.scales"
        bias_key = f"{base}.biases"
        triplet_keys = (weight_key, scale_key, bias_key)
        missing = [key for key in triplet_keys if key not in state]
        if missing:
            raise ValueError(f"incomplete 1-bit triplet for {base!r}: missing {missing}")
        excluded_triplet = tuple(
            any(pattern in key for pattern in exclude_patterns) for key in triplet_keys
        )
        if any(excluded_triplet):
            if not all(excluded_triplet):
                raise ValueError(
                    f"exclude_patterns partially remove the 1-bit triplet for "
                    f"{base!r}; exclude the whole module or none of it"
                )
        else:
            included_count += 1

        weight = state[weight_key]
        scales = state[scale_key]
        biases = state[bias_key]
        if weight.dtype != torch.int32 or weight.ndim != 2:
            raise ValueError(
                f"{weight_key} must be a 2-D I32 packed tensor, got "
                f"dtype={weight.dtype}, shape={tuple(weight.shape)}"
            )
        if not scales.is_floating_point() or not biases.is_floating_point():
            raise ValueError(f"{base!r} scales/biases must be floating-point tensors")

        in_features = weight.shape[1] * 32
        if in_features % group_size != 0:
            raise ValueError(
                f"{weight_key} encodes {in_features} inputs, not divisible by "
                f"group_size {group_size}"
            )
        expected_shape = (weight.shape[0], in_features // group_size)
        if tuple(scales.shape) != expected_shape or tuple(biases.shape) != expected_shape:
            raise ValueError(
                f"invalid scale/bias shape for {base!r}: expected {expected_shape}, "
                f"got {tuple(scales.shape)} and {tuple(biases.shape)}"
            )

        stored_scales = scales.to(torch.float16)
        stored_biases = biases.to(torch.float16)
        if not bool(torch.isfinite(stored_scales).all().item()) or not bool(
            torch.isfinite(stored_biases).all().item()
        ):
            raise ValueError(f"{base!r} scale/bias become non-finite in F16 storage")
        if bool((stored_scales < 0).any().item()):
            raise ValueError(f"{base!r} contains negative binary scales")
        expected_biases = -stored_scales / 2
        if not torch.equal(
            stored_biases.view(torch.int16), expected_biases.view(torch.int16)
        ):
            raise ValueError(
                f"{base!r} violates the Bonsai binary relation bias == -scale / 2"
            )
    if included_count == 0:
        raise ValueError("1-bit export would contain no quantized modules")
    return included_count


def _save_lowbit_mlx_checkpoint(
    model: nn.Module,
    config: dict[str, Any] | None,
    manifest: dict[str, Any],
    out_dir: str | Path,
    *,
    bits: int,
    group_size: int = 128,
    max_shard_bytes: int = MAX_SHARD_BYTES,
    exclude_patterns: tuple[str, ...] | list[str] = (),
    cast_all_float_to_fp16: bool = False,
) -> dict[str, Any]:
    """Write an MLX-style packed low-bit checkpoint.

    Parameters
    ----------
    model
        A torch model after 1-bit or 2-bit model quantization. Quantized modules
        carry int32 packed weight plus floating-point scale/bias buffers.
    config
        Original model config dict (e.g. from ``model.config.to_dict()``). Will
        be copied and augmented with top-level and, when present,
        ``text_config.quantization`` blocks. If ``None``, only the top-level
        quantization block is written.
    manifest
        The model quantizer manifest, used to identify packed weight triplets.
    out_dir
        Output directory.
    group_size
        Quantization group size (written into config.json).
    exclude_patterns
        Tensor-name substrings to omit from the MLX container. This supports
        tied architectures whose runtime scores through ``embed_tokens`` and
        therefore has no separate ``lm_head`` parameter.

    Returns
    -------
    dict
        ``{"quantized_count": int, "shards": list[str], "config": dict}``.
    """
    if bits not in (1, 2):
        raise ValueError(f"bits must be 1 or 2, got {bits}")
    manifest_quantization = manifest.get("quantization", {})
    manifest_bits = manifest_quantization.get("bits")
    if manifest_bits is not None and manifest_bits != bits:
        raise ValueError(
            f"manifest bits {manifest_bits} do not match exporter bits {bits}"
        )
    manifest_group_size = manifest_quantization.get("group_size")
    if manifest_group_size is not None and manifest_group_size != group_size:
        raise ValueError(
            f"manifest group_size {manifest_group_size} does not match "
            f"export group_size {group_size}"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    quantized_names = set(manifest.get("quantized", []))
    raw_state = model.state_dict()
    excluded = sorted(
        name for name in raw_state if any(pattern in name for pattern in exclude_patterns)
    )
    state_dict = {
        name: tensor.contiguous()
        for name, tensor in raw_state.items()
        if name not in excluded
    }

    # Reinterpret quantized packed weights int32 -> uint32 (zero-copy) so the
    # safetensors header reads "U32", which MLX's quantized kernel requires.
    # Bonsai's container contract stores group scales and affine biases as
    # FP16 even when the source torch model was loaded in BF16.
    out_tensors: dict[str, torch.Tensor] = {}
    quant_count = 0
    for name, tensor in state_dict.items():
        base = name.rsplit(".", 1)[0] if "." in name else None
        is_quant_wt = (
            name.endswith(".weight")
            and base in quantized_names
            and f"{base}.scales" in state_dict
            and tensor.dtype == torch.int32
        )
        if is_quant_wt:
            out_tensors[name] = tensor.view(torch.uint32)
            quant_count += 1
        elif (
            base in quantized_names and name.endswith((".scales", ".biases"))
        ) or (cast_all_float_to_fp16 and tensor.is_floating_point()):
            out_tensors[name] = tensor.to(torch.float16)
        else:
            out_tensors[name] = tensor

    # Remap keys to match mlx_vlm's gemma4_unified module hierarchy. The HF
    # transformers layout groups vision patch/pos tensors under
    # ``model.vision_embedder.*`` (so on-disk ``embed_vision.patch_*`` after
    # the generic ``model.`` strip done by mlx_vlm sanitize) and nests the
    # projection under ``model.embed_vision.multimodal_embedder.*``. mlx_vlm
    # expects the patch/pos tensors directly under ``vision_embedder.*`` and
    # the projection flattened to ``embed_vision.embedding_projection.*``
    # (no ``multimodal_embedder`` middle node). We pre-apply that remap so the
    # generic sanitize pass produces correct leaf keys.
    out_tensors = {_remap_mlx_key(k): v for k, v in out_tensors.items()}

    # Shard + write safetensors
    index, shard_names = _shard_keys(out_tensors, max_shard_bytes)
    for shard_name in shard_names:
        shard_tensors = {
            k: v for k, v in out_tensors.items() if index["weight_map"][k] == shard_name
        }
        st_save_file(shard_tensors, str(out / shard_name))

    # index.json (only when sharded)
    if len(shard_names) > 1:
        (out / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2), encoding="utf-8"
        )

    # config.json: copy original + inject quantization block
    cfg = dict(config) if config else {}
    quantization_cfg = {"bits": bits, "group_size": group_size}
    if bits == 2:
        quantization_cfg["mode"] = "affine"
    cfg["quantization"] = dict(quantization_cfg)
    if isinstance(cfg.get("text_config"), dict):
        cfg["text_config"] = dict(cfg["text_config"])
        cfg["text_config"]["quantization"] = dict(quantization_cfg)
    trimmed = manifest.get("trimmed", {})
    if not isinstance(trimmed, dict):
        raise ValueError("manifest['trimmed'] must be a mapping")
    effective_trim_sizes: set[int] = set()
    for name, dimensions in trimmed.items():
        if (
            not isinstance(name, str)
            or not isinstance(dimensions, list)
            or len(dimensions) != 2
            or not all(isinstance(value, int) for value in dimensions)
            or dimensions[1] < 1
            or dimensions[1] >= dimensions[0]
        ):
            raise ValueError(f"invalid trim manifest entry: {name!r} -> {dimensions!r}")
        weight_key = f"{name}.weight"
        if weight_key not in raw_state or raw_state[weight_key].shape[0] != dimensions[1]:
            raise ValueError(
                f"trim manifest for {name!r} does not match its stored weight shape"
            )
        effective_trim_sizes.add(dimensions[1])
    if len(effective_trim_sizes) > 1:
        raise ValueError(
            f"trim manifest contains inconsistent vocabulary sizes: "
            f"{sorted(effective_trim_sizes)}"
        )
    if effective_trim_sizes:
        trim_vocab_to = effective_trim_sizes.pop()
        requested_trim = manifest_quantization.get("trim_vocab_to")
        if requested_trim is not None and requested_trim != trim_vocab_to:
            raise ValueError(
                f"manifest trim_vocab_to {requested_trim} does not match actual "
                f"trimmed size {trim_vocab_to}"
            )
        if "vocab_size" in cfg:
            cfg["vocab_size"] = trim_vocab_to
        if isinstance(cfg.get("text_config"), dict) and "vocab_size" in cfg["text_config"]:
            cfg["text_config"]["vocab_size"] = trim_vocab_to
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # quantization.json (manifest, for our own tooling)
    (out / "quantization.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    return {
        "bits": bits,
        "quantized_count": quant_count,
        "shards": shard_names,
        "config": cfg,
        "index": index if len(shard_names) > 1 else None,
        "excluded": excluded,
    }


def _tied_module_role(name: str) -> str | None:
    """Classify common tied input-embedding and output-head module names."""
    lowered = name.lower()
    tail = lowered.rsplit(".", 1)[-1]
    embedding_markers = (
        "embed_tokens",
        "tok_embeddings",
        "word_embeddings",
        ".wte",
    )
    head_markers = ("lm_head", "output_layer", "embed_out")
    if tail == "wte" or any(marker in lowered for marker in embedding_markers):
        return "embedding"
    if tail in ("output", "output_head") or any(
        marker in lowered for marker in head_markers
    ):
        return "head"
    return None


def _deduplicated_tied_patterns(
    model: nn.Module,
    manifest: dict[str, Any],
) -> tuple[str, ...]:
    """Select the duplicate side while preserving the token embedding."""
    quantized_tied = manifest.get("quantized_tied", [])
    tied = manifest.get("tied", [])
    if not isinstance(quantized_tied, list) or not isinstance(tied, list):
        raise ValueError("manifest tied-module fields must be lists")
    tied_names = list(
        dict.fromkeys(
            [
                *quantized_tied,
                *tied,
            ]
        )
    )
    if not tied_names:
        return ()
    if not all(isinstance(name, str) for name in tied_names):
        raise ValueError("manifest tied-module lists must contain module names")
    tied_to = manifest.get("tied_to")
    if not isinstance(tied_to, dict):
        raise ValueError(
            "tied-weight deduplication requires manifest['tied_to']; "
            "regenerate the manifest or set deduplicate_tied_weights=False"
        )

    state = model.state_dict()
    excluded: list[str] = []
    for duplicate in tied_names:
        source = tied_to.get(duplicate)
        if not isinstance(source, str):
            raise ValueError(f"missing tied source for module {duplicate!r}")
        duplicate_role = _tied_module_role(duplicate)
        source_role = _tied_module_role(source)
        if (
            duplicate_role == "embedding" and source_role != "embedding"
        ) or (source_role == "head" and duplicate_role != "head"):
            canonical, discarded = duplicate, source
        elif (source_role == "embedding" and duplicate_role != "embedding") or (
            duplicate_role == "head" and source_role != "head"
        ):
            canonical, discarded = source, duplicate
        else:
            raise ValueError(
                f"cannot choose the embedding side of tied pair "
                f"{source!r} <-> {duplicate!r}; set "
                "deduplicate_tied_weights=False and exclude explicitly"
            )

        canonical_keys = (
            f"{canonical}.weight",
            f"{canonical}.scales",
            f"{canonical}.biases",
        )
        if not all(key in state for key in canonical_keys):
            raise ValueError(
                f"canonical tied embedding {canonical!r} is not quantized; "
                "use quantize_model_1bit(..., method='bonsai')"
            )
        excluded.append(f"{discarded}.")
    return tuple(dict.fromkeys(excluded))


def save_1bit_mlx_checkpoint(
    model: nn.Module,
    config: dict[str, Any] | None,
    manifest: dict[str, Any],
    out_dir: str | Path,
    group_size: int = 128,
    max_shard_bytes: int = MAX_SHARD_BYTES,
    exclude_patterns: tuple[str, ...] | list[str] = (),
    *,
    deduplicate_tied_weights: bool = True,
) -> dict[str, Any]:
    """Write a Bonsai-style binary checkpoint for a 1-bit-capable MLX build.

    The exporter validates the sign manifest, packed triplet shapes and exact
    F16 ``bias == -scale / 2`` relation before writing. Packed weights are U32,
    all floating-point tensors are F16, and config.json records
    bits=1/group_size. For recognized tied embedding/head pairs, the output
    head copy is omitted while the quantized token embedding is retained.
    Set ``deduplicate_tied_weights=False`` for runtimes with a different
    scoring contract. Loading or inference requires MLX 1-bit affine support
    (for example the Prism ML fork).
    """
    effective_excludes = tuple(exclude_patterns)
    if deduplicate_tied_weights:
        effective_excludes += _deduplicated_tied_patterns(model, manifest)
    expected_quantized_count = _validate_bonsai_1bit_contract(
        model,
        manifest,
        group_size,
        effective_excludes,
    )
    result = _save_lowbit_mlx_checkpoint(
        model,
        config,
        manifest,
        out_dir,
        bits=1,
        group_size=group_size,
        max_shard_bytes=max_shard_bytes,
        exclude_patterns=effective_excludes,
        cast_all_float_to_fp16=True,
    )
    if result["quantized_count"] != expected_quantized_count:
        raise RuntimeError(
            "internal error: exported quantized tensor count does not match validation"
        )
    return result


def save_2bit_mlx_checkpoint(
    model: nn.Module,
    config: dict[str, Any] | None,
    manifest: dict[str, Any],
    out_dir: str | Path,
    group_size: int = 128,
    max_shard_bytes: int = MAX_SHARD_BYTES,
    exclude_patterns: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    """Write an MLX-loadable Bonsai-style ternary 2-bit checkpoint."""
    return _save_lowbit_mlx_checkpoint(
        model,
        config,
        manifest,
        out_dir,
        bits=2,
        group_size=group_size,
        max_shard_bytes=max_shard_bytes,
        exclude_patterns=exclude_patterns,
    )
