from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache

from .model_utils import _get_decoder_layers, _get_inner_model


def compute_pinv_projection(
    reference_lm_head_weight: mx.array,
    target_lm_head_weight: mx.array,
) -> mx.array:
    """
    Optimally initialize a `shadow_hidden -> base_hidden` projection via pseudo-inverse.

    Solves `W_target @ W_proj = W_ref` for `W_proj`, i.e.
    `W_proj = pinv(W_target) @ W_ref`, where head weights have layout `(vocab, hidden)`
    (mlx `nn.Linear` weight is `(out_features, in_features)`).

    Note: `mx.linalg.pinv` is CPU-only, so the computation is pinned to the CPU stream
    in float32 and cast back to the target dtype.
    """
    with mx.stream(mx.cpu):
        w_pinv = mx.linalg.pinv(target_lm_head_weight.astype(mx.float32), stream=mx.cpu)
        w_proj = w_pinv @ reference_lm_head_weight.astype(mx.float32)
        mx.eval(w_proj)
    return w_proj.astype(target_lm_head_weight.dtype)


class ProjectedCausalLM(nn.Module):
    """
    Standalone causal LM with a hidden-size projection:

      shadow backbone (hidden=D_s) -> shadow_hidden_projection (D_s -> D_b) -> lm_head (D_b -> V)
    """

    def __init__(
        self,
        *,
        shadow_model: nn.Module,
        shadow_hidden_projection: nn.Linear,
        lm_head: nn.Linear,
    ) -> None:
        super().__init__()
        self.shadow_model = shadow_model
        self.shadow_hidden_projection = shadow_hidden_projection
        self.lm_head = lm_head

    @property
    def layers(self) -> list:
        return _get_decoder_layers(self.shadow_model)[1]

    def make_cache(self) -> list:
        fn = getattr(self.shadow_model, "make_cache", None)
        if callable(fn):
            return fn()
        return [KVCache() for _ in self.layers]

    def __call__(
        self,
        inputs: mx.array,
        cache: list | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        inner = _get_inner_model(self.shadow_model)
        kwargs: dict = {}
        if cache is not None:
            kwargs["cache"] = cache
        if input_embeddings is not None:
            kwargs["input_embeddings"] = input_embeddings
        hidden = inner(inputs, **kwargs)
        return self.lm_head(self.shadow_hidden_projection(hidden))

    @classmethod
    def wrap(
        cls,
        *,
        shadow_model: nn.Module,
        shadow_hidden_projection: nn.Linear,
        lm_head: nn.Linear,
        init_optimal_projection: bool = True,
        reference_lm_head_weight: mx.array | None = None,
    ) -> ProjectedCausalLM:
        """
        Wrap an already-instantiated shadow model + projection + head.

        If `init_optimal_projection` is True, initialize the projection via pseudo-inverse
        so that `lm_head @ projection` best approximates `reference_lm_head_weight`
        (shape `(vocab, shadow_hidden)`).
        """
        out = cls(
            shadow_model=shadow_model,
            shadow_hidden_projection=shadow_hidden_projection,
            lm_head=lm_head,
        )
        if init_optimal_projection:
            if reference_lm_head_weight is None:
                raise ValueError(
                    "When init_optimal_projection=True, you must provide "
                    "reference_lm_head_weight (the original model's lm_head to approximate)."
                )
            out.shadow_hidden_projection.weight = compute_pinv_projection(
                reference_lm_head_weight, out.lm_head.weight
            )
            reconstructed = out.lm_head.weight @ out.shadow_hidden_projection.weight
            err = mx.linalg.norm(reconstructed - reference_lm_head_weight) / mx.linalg.norm(
                reference_lm_head_weight
            )
            print(f"Reconstruction error: {float(err):.6f}")
        return out
