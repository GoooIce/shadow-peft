from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel


def _unwrap_language_model(backbone: nn.Module) -> nn.Module:
    """
    Descend through multimodal wrappers (Gemma3 4B+ / Gemma4 all sizes) that nest
    the decoder stack under `.language_model` next to their vision tower.

    ``Gemma3ForConditionalGeneration.model`` is a multimodal container with no
    ``.layers`` of its own; the text decoder lives at
    ``model.model.language_model.layers``. The loop handles nested wrappers and
    is a no-op for text-only backbones without a ``language_model`` child.
    """
    seen: set[int] = set()
    while id(backbone) not in seen:
        seen.add(id(backbone))
        inner = getattr(backbone, "language_model", None)
        if inner is None:
            break
        backbone = inner
    return backbone


def _get_backbone(model: PreTrainedModel) -> nn.Module:
    """
    Return the module that contains the transformer decoder stack.

    This mirrors common HF naming conventions (LLaMA/Mistral/Qwen2/Qwen3: `.model`,
    GPT2/Falcon: `.transformer`, multimodal Gemma3/Gemma4: `.model.language_model`).
    """
    for attr in ("model", "transformer", "base_model", "decoder", "language_model"):
        backbone = getattr(model, attr, None)
        if backbone is not None:
            return _unwrap_language_model(backbone)
    # Some HF classes are already the "backbone" (e.g. LlamaModel, Qwen3Model).
    if hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList):
        return model
    if hasattr(model, "h") and isinstance(model.h, nn.ModuleList):
        return model
    if hasattr(model, "decoder") and hasattr(model.decoder, "layers") and isinstance(
        model.decoder.layers, nn.ModuleList
    ):
        return _unwrap_language_model(model.decoder)
    raise AttributeError("Unable to locate transformer backbone inside the supplied model.")


def _get_decoder_layers(model: PreTrainedModel) -> tuple[nn.Module, nn.ModuleList, str]:
    """
    Return (backbone, layers, layers_attr_name).

    Supported:
    - backbone.layers (LLaMA/Mistral/Qwen)
    - backbone.h (GPT2-style)
    """
    backbone = _get_backbone(model)
    if hasattr(backbone, "layers") and isinstance(backbone.layers, nn.ModuleList):
        return backbone, backbone.layers, "layers"
    if hasattr(backbone, "h") and isinstance(backbone.h, nn.ModuleList):
        return backbone, backbone.h, "h"
    if hasattr(backbone, "decoder") and hasattr(backbone.decoder, "layers") and isinstance(
        backbone.decoder.layers, nn.ModuleList
    ):
        # Some model backbones (incl. some Gemma/encoder-decoder style internals) nest layers here.
        return backbone.decoder, backbone.decoder.layers, "layers"
    raise AttributeError(
        "Unsupported model: cannot find a ModuleList of decoder layers on the backbone "
        "(expected `.layers` or `.h`)."
    )


def _get_hidden_size(model: PreTrainedModel) -> int:
    cfg = model.config
    # Multimodal configs (Gemma3/Gemma4 ConditionalGeneration) keep the text
    # decoder's dimensions under `text_config`, not at the top level.
    text_cfg = getattr(cfg, "text_config", None)
    for c in (cfg, text_cfg):
        if c is None:
            continue
        for attr in ("hidden_size", "n_embd", "d_model"):
            if hasattr(c, attr):
                return int(getattr(c, attr))
    raise AttributeError("Unable to infer hidden size from model.config.")


def _set_intermediate_size(cfg: Any, value: int) -> bool:
    """
    Best-effort set of the MLP/GLU intermediate size across common HF configs.

    Returns True if a known field was set.
    """
    for attr in ("intermediate_size", "ffn_dim", "n_inner"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, int(value))
            return True
    return False


def _set_num_attention_heads(cfg: Any, value: int) -> bool:
    for attr in ("num_attention_heads", "n_head", "num_heads", "attention_heads"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, int(value))
            return True
    return False


def _set_num_key_value_heads(cfg: Any, value: int) -> bool:
    for attr in ("num_key_value_heads", "num_kv_heads", "n_kv_head"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, int(value))
            return True
    return False


def _set_head_dim(cfg: Any, value: int) -> bool:
    for attr in ("head_dim", "attention_head_dim"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, int(value))
            return True
    return False


def _set_num_hidden_layers(cfg: Any, n: int) -> None:
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, int(n))
            return
    raise AttributeError("Unable to set number of layers on this config.")


def _get_num_hidden_layers(cfg: Any) -> int:
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    raise AttributeError("Unable to get number of layers from this config.")


def _shrink_per_layer_config(cfg: Any, num_layers: int) -> None:
    """
    Keep per-layer config fields consistent after shrinking ``num_hidden_layers``.

    Gemma3/Gemma4 configs carry per-layer lists/dicts (``layer_types``,
    ``per_layer_config``, ``max_window_layers``) sized to the original layer
    count; leaving them stale makes the smaller shadow config fail validation
    on instantiation.
    """
    lt = getattr(cfg, "layer_types", None)
    if isinstance(lt, (list, tuple)) and len(lt) > num_layers:
        cfg.layer_types = list(lt[:num_layers])
    plc = getattr(cfg, "per_layer_config", None)
    if isinstance(plc, dict):
        stale = [k for k in plc if str(k).isdigit() and int(k) >= num_layers]
        for k in stale:
            del plc[k]
    mw = getattr(cfg, "max_window_layers", None)
    if mw is not None and isinstance(mw, int) and mw > num_layers:
        cfg.max_window_layers = num_layers
    # LFM2-style hybrids derive `full_attn_idxs` from the original layer list;
    # leave it stale and cache construction classifies the shrunken stack wrong.
    lt2 = getattr(cfg, "layer_types", None)
    fa = getattr(cfg, "full_attn_idxs", None)
    if isinstance(fa, (list, tuple)) and isinstance(lt2, (list, tuple)):
        cfg.full_attn_idxs = [i for i, t in enumerate(lt2) if str(t) == "full_attention"]


def build_implicit_shadow_model(
    base_model: PreTrainedModel,
    *,
    num_shadow_layers: int,
    shadow_intermediate_size: int | None = None,
    shadow_num_attention_heads: int | None = None,
    shadow_num_key_value_heads: int | None = None,
    shadow_head_dim: int | None = None,
    shadow_layer_types: list[str] | None = None,
) -> PreTrainedModel:
    """
    Create an implicit shadow model by instantiating the same class as `base_model`
    with a copied config but fewer layers and (optionally) smaller MLP size.

    This shadow model is randomly initialized (like the raw Qwen3 code path).
    """
    # Prefer instantiating the *backbone* class (e.g. Qwen3Model) rather than the task model
    # (e.g. Qwen3ForCausalLM) so the implicit shadow model stays lightweight.
    backbone = None
    for attr in ("model", "transformer", "base_model", "decoder"):
        cand = getattr(base_model, attr, None)
        # In HF, these are typically PreTrainedModel instances too.
        if isinstance(cand, PreTrainedModel):
            backbone = cand
            break
    if backbone is None:
        # Fall back to using the base model itself (covers cases where base_model is already the backbone).
        backbone = base_model
    # Multimodal wrappers (Gemma3/Gemma4) nest the text decoder under
    # `.language_model`; the shadow must be built from the text model class.
    backbone = _unwrap_language_model(backbone)

    cfg = deepcopy(backbone.config)
    # Gemma4 PLE (per-layer embedding): the shadow backbone is driven via base
    # `inputs_embeds` (embedding sharing), so it never sees `input_ids` and the
    # PLE lookup cannot run (it would try to reverse the shared-away
    # `embed_tokens`). Drop the PLE branch from the shadow config; the real
    # Gemma4-12B testbed ships with PLE disabled anyway.
    if getattr(cfg, "hidden_size_per_layer_input", 0):
        cfg.hidden_size_per_layer_input = 0
    if getattr(cfg, "vocab_size_per_layer_input", 0):
        cfg.vocab_size_per_layer_input = 0
    if num_shadow_layers < 1:
        raise ValueError(f"num_shadow_layers must be >= 1, got {num_shadow_layers}")
    _set_num_hidden_layers(cfg, num_shadow_layers)
    _shrink_per_layer_config(cfg, num_shadow_layers)
    if shadow_layer_types is not None:
        # Explicit composition (exp4 knob): the caller owns heterogeneity —
        # validate length and skip the all-conv auto-remap below.
        if len(shadow_layer_types) != num_shadow_layers:
            raise ValueError(
                f"shadow_layer_types has {len(shadow_layer_types)} entries, "
                f"expected num_shadow_layers={num_shadow_layers}"
            )
        cfg.layer_types = list(shadow_layer_types)
        fa = getattr(cfg, "full_attn_idxs", None)
        if isinstance(fa, (list, tuple)):
            cfg.full_attn_idxs = [i for i, t in enumerate(cfg.layer_types) if str(t) == "full_attention"]
    else:
        # Hybrid backbones (e.g. LFM2) can slice to an all-conv prefix. A shadow
        # with zero attention layers cannot use HF caches at all (get_seq_length()
        # requires an attention entry) and a 3-token conv memory is too weak to
        # track sequence state — remap the tiny shadow to the family's attention
        # layer type when shrinking dropped every attention entry.
        lt = getattr(cfg, "layer_types", None)
        if isinstance(lt, (list, tuple)) and not any("attention" in str(t) for t in lt):
            original = getattr(backbone.config, "layer_types", None) or []
            attn_types = [t for t in original if "attention" in str(t)]
            if attn_types:
                cfg.layer_types = [attn_types[0]] * len(lt)
                fa = getattr(cfg, "full_attn_idxs", None)
                if isinstance(fa, (list, tuple)):
                    cfg.full_attn_idxs = list(range(len(cfg.layer_types)))

    # If explicitly provided, override the MLP/GLU intermediate size on the shadow config.
    if shadow_intermediate_size is not None:
        _set_intermediate_size(cfg, shadow_intermediate_size)

    # Optionally shrink attention projections (dominant param cost for large backbones).
    if shadow_num_attention_heads is not None and not _set_num_attention_heads(cfg, shadow_num_attention_heads):
        raise ValueError(
            "shadow_num_attention_heads was set, but this model config does not expose a "
            "recognized attention-heads field (e.g. num_attention_heads / n_head)."
        )
    if shadow_num_key_value_heads is not None and not _set_num_key_value_heads(cfg, shadow_num_key_value_heads):
        raise ValueError(
            "shadow_num_key_value_heads was set, but this model config does not expose a "
            "recognized kv-heads field (e.g. num_key_value_heads / num_kv_heads)."
        )
    if shadow_head_dim is not None and not _set_head_dim(cfg, shadow_head_dim):
        raise ValueError(
            "shadow_head_dim was set, but this model config does not expose a recognized "
            "head-dim field (e.g. head_dim)."
        )

    if hasattr(cfg, "use_cache"):
        cfg.use_cache = False

    shadow = backbone.__class__(cfg)
    return shadow


@torch.no_grad()
def mark_only_shadow_modules_trainable(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False

    for name, module in model.named_modules():
        if name.startswith("shadow_"):
            for p in module.parameters(recurse=True):
                p.requires_grad = True


def count_parameters(module: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


