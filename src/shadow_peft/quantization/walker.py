"""The model walker and the checkpoint save/load pair, shared by every scheme.

One in-place traversal (skip patterns, tied-weight detection, vocab trimming,
manifest) drives any ``QuantScheme``; one save writes ``model.safetensors`` +
``quantization.json``; one load rebuilds the packed structure from a fresh
model of the same architecture and overlays the checkpoint state — a
round-trippable pair at the seam.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file

from .modules import classes_for_scheme
from .scheme import QuantScheme, get_scheme


def quantize_model(
    model: nn.Module,
    scheme: QuantScheme,
    *,
    group_size: int = 128,
    skip_patterns: tuple[str, ...] | list[str] = (),
    refine_iters: int = 0,
    trim_vocab_to: int | None = None,
    thr_ratio: float = 0.5,
) -> dict[str, Any]:
    """
    Replace every ``nn.Linear``/``nn.Embedding`` in ``model`` (in place) with its
    quantized counterpart under ``scheme``. Returns a JSON-serializable manifest.

    - Modules whose full name contains any of ``skip_patterns`` are left as-is.
    - A ``nn.Linear`` whose weight is the same tensor as an already-replaced
      module's weight (e.g. a tied lm_head) is skipped and listed under "tied".
    - Layers whose last dim is not divisible by ``group_size`` are skipped and
      listed under "skipped" with the reason.
    - ``refine_iters > 0`` enables Lloyd-style refinement (scheme-specific).
    - ``trim_vocab_to`` drops dead padding rows (e.g. Qwen3's 151936 -> 151669)
      from ``nn.Embedding`` tables and any module with "lm_head" in its name
      before quantization. A tied lm_head is re-pointed at the trimmed fp
      weight so its output dim stays consistent with the trimmed embedding.
      Trims are recorded under "trimmed" as ``{name: [orig_rows, new_rows]}``.
      Pass the FULL tokenizer length (including special tokens, e.g.
      ``len(AutoTokenizer)``) — a bare ``vocab_size`` attribute may exclude
      special tokens and truncate their rows.
    """
    if trim_vocab_to is not None and trim_vocab_to < 1:
        raise ValueError(f"trim_vocab_to must be >= 1, got {trim_vocab_to}")

    linear_cls, embedding_cls = classes_for_scheme(scheme)
    quantized: list[str] = []
    skipped: dict[str, str] = {}
    tied: list[str] = []
    trimmed: dict[str, list[int]] = {}
    replaced_weights: list[torch.Tensor] = []

    for name, module in list(model.named_modules()):
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            continue
        matched = next((p for p in skip_patterns if p in name), None)
        if matched is not None:
            skipped[name] = f"matched skip pattern '{matched}'"
            continue
        weight = module.weight
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model

        if any(weight is w or weight.data_ptr() == w.data_ptr() for w in replaced_weights):
            tied.append(name)
            # Keep a tied lm_head dimensionally consistent with a trimmed
            # embedding: re-point it at the trimmed slice of the fp weight.
            if (
                trim_vocab_to is not None
                and isinstance(module, nn.Linear)
                and weight.shape[0] > trim_vocab_to
            ):
                with torch.no_grad():
                    new_head = nn.Linear(
                        weight.shape[1],
                        trim_vocab_to,
                        bias=module.bias is not None,
                        device=weight.device,
                        dtype=weight.dtype,
                    )
                    new_head.weight.copy_(weight[:trim_vocab_to])
                    if module.bias is not None:
                        new_head.bias.copy_(module.bias[:trim_vocab_to])
                setattr(parent, child_name, new_head)
                trimmed[name] = [weight.shape[0], trim_vocab_to]
            continue
        if weight.shape[-1] % group_size != 0:
            skipped[name] = (
                f"last dim {weight.shape[-1]} not divisible by group_size {group_size}"
            )
            continue

        # Row-trim vocab-sized modules before packing.
        source = module
        if (
            trim_vocab_to is not None
            and weight.shape[0] > trim_vocab_to
            and (isinstance(module, nn.Embedding) or "lm_head" in name)
        ):
            with torch.no_grad():
                if isinstance(module, nn.Embedding):
                    source = nn.Embedding(
                        trim_vocab_to, weight.shape[1], device=weight.device
                    )
                else:
                    source = nn.Linear(
                        weight.shape[1],
                        trim_vocab_to,
                        bias=module.bias is not None,
                        device=weight.device,
                        dtype=weight.dtype,
                    )
                source.weight = nn.Parameter(
                    weight[:trim_vocab_to].clone(), requires_grad=False
                )
                if getattr(module, "bias", None) is not None:
                    source.bias = nn.Parameter(
                        module.bias[:trim_vocab_to].clone(), requires_grad=False
                    )
            trimmed[name] = [weight.shape[0], trim_vocab_to]

        if isinstance(source, nn.Linear):
            new_module: nn.Module = linear_cls.from_linear(
                source,
                group_size,
                refine_iters=refine_iters,
                thr_ratio=thr_ratio,
                scheme=scheme,
            )
        else:
            new_module = embedding_cls.from_embedding(
                source,
                group_size,
                refine_iters=refine_iters,
                thr_ratio=thr_ratio,
                scheme=scheme,
            )
        setattr(parent, child_name, new_module)
        replaced_weights.append(weight)
        quantized.append(name)

    if (
        trimmed
        and getattr(model, "config", None) is not None
        and getattr(model.config, "vocab_size", 0) > trim_vocab_to
    ):
        # Keep the model config consistent with the trimmed vocab so e.g. HF's
        # internal CE loss (`logits.view(-1, config.vocab_size)`) still works.
        model.config.vocab_size = trim_vocab_to

    quantization: dict[str, Any] = {
        "mode": "affine",
        "bits": scheme.bits,
    }
    if scheme.ternary:
        quantization["ternary"] = True
    quantization["group_size"] = group_size
    quantization["refine_iters"] = refine_iters
    if scheme.ternary:
        quantization["thr_ratio"] = thr_ratio
    quantization["trim_vocab_to"] = trim_vocab_to

    return {
        "quantization": quantization,
        "quantized": quantized,
        "skipped": skipped,
        "tied": tied,
        "trimmed": trimmed,
    }


def save_quantized_checkpoint(
    model: nn.Module, manifest: dict[str, Any], dir_path: str | Path
) -> None:
    """Save ``model.state_dict()`` as ``model.safetensors`` plus ``quantization.json``."""
    save_dir = Path(dir_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    safetensors_save_file(state_dict, str(save_dir / "model.safetensors"))
    (save_dir / "quantization.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_quantized_checkpoint(
    model: nn.Module,
    dir_path: str | Path,
    *,
    skip_patterns: tuple[str, ...] | list[str] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_quantized_checkpoint` into ``model``.

    ``model`` must be a fresh instance of the same architecture (any weights;
    they are only used for shape reconstruction). The packed module structure
    is rebuilt from the manifest (``bits`` selects the scheme, plus
    ``group_size`` / ``trim_vocab_to``), then the checkpoint state dict is
    overlaid — so a save -> load round-trip reproduces the packed model
    bit-for-bit. Returns the stored manifest.

    Notes:
      - ``refine_iters`` is not re-run (shapes do not depend on it).
      - By default (``skip_patterns=None``) the skip patterns are derived from
        the manifest: every module recorded under ``"skipped"`` (whatever the
        reason — custom skip pattern, bad dims) is passed by its full name,
        which the walker matches as a substring. Tied modules re-derive
        naturally from traversal order, so the saved checkpoint is
        self-describing — no need to re-supply the original patterns.
      - Pass an explicit tuple/list to override the derived patterns (e.g. to
        intentionally quantize modules the original run skipped). Skipped fp
        modules keep the checkpoint's weights.
    """
    save_dir = Path(dir_path)
    manifest = json.loads((save_dir / "quantization.json").read_text(encoding="utf-8"))
    q = manifest["quantization"]
    scheme = get_scheme(int(q["bits"]))
    resolved_skips: tuple[str, ...] | list[str]
    if skip_patterns is None:
        # Full module names work as substring patterns for the walker, and
        # every module that was NOT quantized appears in "skipped" or "tied";
        # "tied" re-derives naturally from traversal order, so skipping
        # exactly the recorded "skipped" names reproduces the packed layout.
        resolved_skips = tuple(manifest.get("skipped", {}).keys())
    else:
        resolved_skips = skip_patterns
    quantize_model(
        model,
        scheme,
        group_size=int(q["group_size"]),
        skip_patterns=resolved_skips,
        refine_iters=0,
        trim_vocab_to=q.get("trim_vocab_to"),
        thr_ratio=float(q.get("thr_ratio", 0.5)),
    )
    state_dict = safetensors_load_file(str(save_dir / "model.safetensors"))
    model.load_state_dict(state_dict, strict=strict)
    return manifest
