from __future__ import annotations

from copy import deepcopy

import mlx.nn as nn
from mlx.utils import tree_flatten


def _get_inner_model(model: nn.Module) -> nn.Module:
    """
    Return the inner backbone holding the decoder layer stack.

    mlx-lm top-level models (e.g. `llama.Model`) keep it in `.model`; if the supplied
    module already has `.layers`, it is returned as-is.
    """
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module) and isinstance(getattr(inner, "layers", None), list):
        return inner
    if isinstance(getattr(model, "layers", None), list):
        return model
    raise AttributeError(
        "Unable to locate transformer backbone (expected `.model` or the module itself "
        "to hold a `.layers` list)."
    )


def _get_decoder_layers(model: nn.Module) -> tuple[nn.Module, list]:
    """Return (inner_backbone, layers_list)."""
    inner = _get_inner_model(model)
    return inner, inner.layers


def _get_hidden_size(model: nn.Module) -> int:
    args = getattr(model, "args", None)
    if args is not None and hasattr(args, "hidden_size"):
        return int(args.hidden_size)
    inner = _get_inner_model(model)
    args = getattr(inner, "args", None)
    if args is not None and hasattr(args, "hidden_size"):
        return int(args.hidden_size)
    raise AttributeError("Unable to infer hidden size from model.args.")


def build_implicit_shadow_model(
    base_model: nn.Module,
    *,
    num_shadow_layers: int,
    shadow_intermediate_size: int | None = None,
    shadow_num_attention_heads: int | None = None,
    shadow_num_key_value_heads: int | None = None,
    shadow_head_dim: int | None = None,
) -> nn.Module:
    """
    Create an implicit shadow backbone by instantiating the same inner-model class as
    `base_model` with a copied ModelArgs but fewer layers (randomly initialized, which is
    what plain mlx module construction gives you).
    """
    inner = _get_inner_model(base_model)
    if num_shadow_layers < 1:
        raise ValueError(f"num_shadow_layers must be >= 1, got {num_shadow_layers}")

    args = deepcopy(inner.args)
    args.num_hidden_layers = int(num_shadow_layers)

    if shadow_intermediate_size is not None:
        args.intermediate_size = int(shadow_intermediate_size)
    if shadow_num_attention_heads is not None:
        args.num_attention_heads = int(shadow_num_attention_heads)
    if shadow_num_key_value_heads is not None:
        if not hasattr(args, "num_key_value_heads"):
            raise ValueError(
                "shadow_num_key_value_heads was set, but this model's ModelArgs does not "
                "expose num_key_value_heads."
            )
        args.num_key_value_heads = int(shadow_num_key_value_heads)
    if shadow_head_dim is not None:
        if not hasattr(args, "head_dim"):
            raise ValueError(
                "shadow_head_dim was set, but this model's ModelArgs does not expose head_dim."
            )
        args.head_dim = int(shadow_head_dim)

    return type(inner)(args)


def clone_linear(linear: nn.Linear) -> nn.Linear:
    """Create an independent copy of an mlx Linear layer (no shared arrays)."""
    bias = getattr(linear, "bias", None)  # mlx Linear omits the attr when bias=False
    out = nn.Linear(linear.weight.shape[1], linear.weight.shape[0], bias=bias is not None)
    # `w * 1` forces a new array object so the clone trains independently.
    out.weight = linear.weight * 1
    if bias is not None:
        out.bias = bias * 1
    return out


def clone_embedding(embed: nn.Embedding) -> nn.Embedding:
    """Create an independent copy of an mlx Embedding layer (no shared arrays)."""
    out = nn.Embedding(embed.weight.shape[0], embed.weight.shape[1])
    out.weight = embed.weight * 1
    return out


def count_parameters(module: nn.Module) -> tuple[int, int]:
    trainable = sum(v.size for _, v in tree_flatten(module.trainable_parameters()))
    total = sum(v.size for _, v in tree_flatten(module.parameters()))
    return trainable, total


def print_trainable_parameters(module: nn.Module) -> None:
    trainable, total = count_parameters(module)
    pct = (100.0 * trainable / total) if total else 0.0
    print(
        f"Trainable params: {trainable:,} || Total params: {total:,} || Trainable%: {pct:.2f}%"
    )
