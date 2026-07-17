from __future__ import annotations

import weakref
from copy import deepcopy
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.cache import KVCache

from .config import ShadowConfig
from .model_utils import (
    _get_decoder_layers,
    _get_hidden_size,
    _get_inner_model,
    build_implicit_shadow_model,
    clone_embedding,
    clone_linear,
    print_trainable_parameters,
)
from .modules import ShadowInjectionModel, ShadowUpdateModel

_ADAPTER_FILE = "shadow_adapter.safetensors"


class _RuntimeState:
    """
    Plain (non-Module) holder for transient per-forward state.

    mlx registers any `mx.array` stored directly on an `nn.Module` as a parameter, so
    the shadow hidden states and the shadow KV-cache must live behind a plain object
    (verified: `Module.parameters()` does not descend into arbitrary attribute objects).
    """

    def __init__(self) -> None:
        self.shadow_hidden_states: mx.array | None = None
        self.shadow_cache: list | None = None


class _ShadowLayerWrapper(nn.Module):
    """
    Wrap a single decoder layer to apply Shadow injection/update.

    mlx-lm decoder layers have the signature `__call__(x, mask=None, cache=None)` and
    return the new hidden states, which makes wrapping straightforward. Layer 0 is a
    no-op passthrough (mirrors the torch implementation).
    """

    def __init__(self, layer: nn.Module, *, layer_idx: int, adapter: ShadowPeftModel) -> None:
        super().__init__()
        self.layer = layer
        self.layer_idx = int(layer_idx)
        # llama's inner forward and `make_cache()` read `use_sliding` on every layer.
        self.use_sliding = getattr(layer, "use_sliding", False)
        # Weakref so the adapter does not become a submodule of the wrapped layer
        # (which would create a cycle: adapter -> base_model -> wrapped layer -> adapter).
        self._adapter_ref = weakref.ref(adapter)

    def _get_adapter(self) -> ShadowPeftModel:
        adapter = self._adapter_ref()
        if adapter is None:
            raise RuntimeError("Shadow adapter reference is gone.")
        return adapter

    def __getattr__(self, name: str):
        # Only called when normal lookup fails. IMPORTANT: mlx's `nn.Module` is a
        # dict subclass whose own `__getattr__` resolves submodules from the dict
        # store — preserve that first, then delegate arch-specific per-layer
        # attributes (e.g. `attention_type`) to the wrapped layer.
        try:
            return nn.Module.__getattr__(self, name)
        except AttributeError:
            pass
        if name.startswith("__"):
            raise AttributeError(name)
        layer = nn.Module.__getattr__(self, "layer")
        return getattr(layer, name)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        if self.layer_idx == 0:
            return self.layer(x, mask, cache=cache)

        adapter = self._get_adapter()
        shadow = adapter._runtime.shadow_hidden_states
        if shadow is None:
            raise RuntimeError(
                "Shadow state was not initialized. "
                "Call the ShadowPeftModel's __call__, not the base model directly."
            )
        sidx = self.layer_idx - 1
        x = adapter.shadow_injection_model(x, shadow, sidx)
        h = self.layer(x, mask, cache=cache)
        adapter._runtime.shadow_hidden_states = adapter.shadow_update_model(h, shadow, sidx)
        return h


class ShadowPeftModel(nn.Module):
    """
    PEFT-style wrapper that augments a frozen base mlx-lm causal LM with Shadow modules.

    The base model is modified in-place by wrapping its decoder layers, but this wrapper
    owns the adapter modules and provides save/load utilities.
    """

    def __init__(
        self,
        base_model: nn.Module,
        shadow_config: ShadowConfig,
        *,
        shadow_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.shadow_config = shadow_config

        # Freeze the base model first; adapter modules created afterwards stay trainable.
        self.base_model.freeze()

        base_inner, base_layers = _get_decoder_layers(self.base_model)
        num_base_layers = len(base_layers)
        if num_base_layers < 2:
            raise ValueError("Shadow requires at least 2 decoder layers to apply injection.")
        num_adapt_layers = num_base_layers - 1

        hidden_size = _get_hidden_size(self.base_model)

        # Shadow backbone (inner model). Explicit shadow models are normalized to backbone-only.
        self._explicit_shadow_model = shadow_model is not None
        if shadow_model is None:
            self.shadow_model = build_implicit_shadow_model(
                self.base_model,
                num_shadow_layers=shadow_config.num_shadow_layers,
                shadow_intermediate_size=shadow_config.shadow_intermediate_size,
                shadow_num_attention_heads=shadow_config.shadow_num_attention_heads,
                shadow_num_key_value_heads=shadow_config.shadow_num_key_value_heads,
                shadow_head_dim=shadow_config.shadow_head_dim,
            )
        else:
            self.shadow_model = _get_inner_model(shadow_model)

        # Embedding sharing policy (mirrors the torch implementation):
        # - implicit shadow: always share base embeddings, drop the shadow table
        # - explicit shadow without `embed_tokens`: share base embeddings
        self._share_base_embeddings = not self._explicit_shadow_model
        if self._explicit_shadow_model and getattr(self.shadow_model, "embed_tokens", None) is None:
            self._share_base_embeddings = True
        if self._share_base_embeddings and getattr(self.shadow_model, "embed_tokens", None) is not None:
            self.shadow_model.embed_tokens = None

        # If the shadow hidden size differs from the base, project into base hidden size.
        shadow_hidden_size = _get_hidden_size(self.shadow_model)
        self.shadow_hidden_size = int(shadow_hidden_size)
        self.base_hidden_size = int(hidden_size)
        if shadow_hidden_size != hidden_size:
            self.shadow_hidden_projection: nn.Linear | None = nn.Linear(
                shadow_hidden_size, hidden_size, bias=False
            )
        else:
            self.shadow_hidden_projection = None

        # Adapter modules (trainable).
        self.shadow_injection_model = ShadowInjectionModel(
            num_layers=num_adapt_layers,
            hidden_size=hidden_size,
            injection_hidden_size=shadow_config.injection_hidden_size,
            dropout=shadow_config.dropout,
            alpha=shadow_config.alpha,
        )
        self.shadow_update_model = ShadowUpdateModel(
            num_layers=num_adapt_layers,
            hidden_size=hidden_size,
            gate_hidden_size=shadow_config.gate_hidden_size,
            dropout=shadow_config.dropout,
        )

        # Transient per-forward state (kept out of the module tree).
        self._runtime = _RuntimeState()

        # Wrap base layers in-place.
        wrapped = []
        for i, layer in enumerate(base_layers):
            if isinstance(layer, _ShadowLayerWrapper):
                if layer._adapter_ref() is self:
                    wrapped.append(layer)
                else:
                    # The base already carries a stale wrapper from another adapter:
                    # wrap the original layer, not the old wrapper.
                    wrapped.append(
                        _ShadowLayerWrapper(layer.layer, layer_idx=i, adapter=self)
                    )
            else:
                wrapped.append(_ShadowLayerWrapper(layer, layer_idx=i, adapter=self))
        base_inner.layers = wrapped

    # ---- mlx-lm generation compatibility -------------------------------------

    @property
    def layers(self) -> list:
        return _get_decoder_layers(self.base_model)[1]

    def make_cache(self) -> list:
        """Delegate to the base model so `mlx_lm.models.cache.make_prompt_cache` works."""
        fn = getattr(self.base_model, "make_cache", None)
        if callable(fn):
            return fn()
        return [KVCache() for _ in self.layers]

    # ---- shadow forward --------------------------------------------------------

    def _compute_shadow_hidden(
        self,
        input_ids: mx.array,
        *,
        shadow_cache: list | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        """Run the shadow backbone to produce per-token shadow hidden states."""
        if self._share_base_embeddings:
            if input_embeddings is None:
                base_inner = _get_inner_model(self.base_model)
                input_embeddings = base_inner.embed_tokens(input_ids)
            out = self.shadow_model(input_ids, cache=shadow_cache, input_embeddings=input_embeddings)
        else:
            out = self.shadow_model(input_ids, cache=shadow_cache)
        if self.shadow_hidden_projection is not None:
            out = self.shadow_hidden_projection(out)
        return out

    def _resolve_shadow_cache(self, cache: list | None) -> list | None:
        """
        Decide which shadow-backbone cache to use for this forward call.

        Mirrors the torch `_resolve_shadow_cache`:
        - `cache is None` -> uncached full-sequence mode; reset internal shadow cache.
        - empty base cache (offset == 0) -> fresh prefill; create a new shadow cache.
        - non-empty base cache -> incremental decode; reuse the existing shadow cache.
        """
        if cache is None:
            self._runtime.shadow_cache = None
            return None

        first = cache[0] if isinstance(cache, (list, tuple)) else cache
        offset = getattr(first, "offset", 0) if first is not None else 0
        if offset == 0:
            num_shadow_layers = len(_get_decoder_layers(self.shadow_model)[1])
            self._runtime.shadow_cache = [KVCache() for _ in range(num_shadow_layers)]
        return self._runtime.shadow_cache

    def forward_with_shadow(
        self,
        inputs: mx.array,
        cache: list | None = None,
        input_embeddings: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Forward the wrapped base model and also return the **initial** shadow hidden
        states produced by the shadow backbone (used by task wrappers for shadow logits).
        """
        shadow_cache = self._resolve_shadow_cache(cache)
        initial_shadow_hidden = self._compute_shadow_hidden(
            inputs, shadow_cache=shadow_cache, input_embeddings=input_embeddings
        )
        self._runtime.shadow_hidden_states = initial_shadow_hidden

        kwargs: dict = {}
        if cache is not None:
            kwargs["cache"] = cache
        if input_embeddings is not None:
            kwargs["input_embeddings"] = input_embeddings
        try:
            logits = self.base_model(inputs, **kwargs)
            return logits, initial_shadow_hidden
        finally:
            # Avoid holding onto activations across calls.
            self._runtime.shadow_hidden_states = None

    def __call__(
        self,
        inputs: mx.array,
        cache: list | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        logits, _ = self.forward_with_shadow(inputs, cache=cache, input_embeddings=input_embeddings)
        return logits

    def forward_hidden(
        self,
        inputs: mx.array,
        cache: list | None = None,
        input_embeddings: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Like `forward_with_shadow`, but returns the base backbone's final hidden states
        (pre-lm_head) instead of logits. Used by the sequence-classification wrapper.
        """
        shadow_cache = self._resolve_shadow_cache(cache)
        initial_shadow_hidden = self._compute_shadow_hidden(
            inputs, shadow_cache=shadow_cache, input_embeddings=input_embeddings
        )
        self._runtime.shadow_hidden_states = initial_shadow_hidden

        kwargs: dict = {}
        if cache is not None:
            kwargs["cache"] = cache
        if input_embeddings is not None:
            kwargs["input_embeddings"] = input_embeddings
        try:
            base_inner = _get_inner_model(self.base_model)
            hidden = base_inner(inputs, **kwargs)
            return hidden, initial_shadow_hidden
        finally:
            self._runtime.shadow_hidden_states = None

    # ---- trainability ----------------------------------------------------------

    def _adapter_modules(self) -> list[nn.Module]:
        modules = [self.shadow_model, self.shadow_injection_model, self.shadow_update_model]
        if self.shadow_hidden_projection is not None:
            modules.append(self.shadow_hidden_projection)
        return modules

    def set_trainable(self, trainable: bool = True) -> None:
        """(Un)freeze the Shadow adapter modules. The base model always stays frozen."""
        for module in self._adapter_modules():
            if trainable:
                module.unfreeze()
            else:
                module.freeze()

    def print_trainable_parameters(self) -> None:
        print_trainable_parameters(self)

    # ---- save / load -------------------------------------------------------------

    def adapter_parameters(self) -> dict[str, mx.array]:
        """Flat `{name: array}` dict of adapter-owned parameters (never base weights)."""
        params: dict[str, mx.array] = {}
        for name in (
            "shadow_model",
            "shadow_hidden_projection",
            "shadow_injection_model",
            "shadow_update_model",
        ):
            module = getattr(self, name, None)
            if module is None:
                continue
            for k, v in tree_flatten(module.parameters()):
                params[f"{name}.{k}"] = v
        return params

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.shadow_config.save_pretrained(save_dir)
        mx.save_safetensors(str(save_dir / _ADAPTER_FILE), self.adapter_parameters())

    @classmethod
    def from_pretrained(
        cls,
        model: nn.Module,
        pretrained_shadow_path: str | Path,
        *,
        is_trainable: bool = False,
        shadow_model: nn.Module | None = None,
    ) -> ShadowPeftModel:
        ckpt_dir = Path(pretrained_shadow_path)
        cfg = ShadowConfig.from_pretrained(ckpt_dir)
        peft_model = cls(model, cfg, shadow_model=shadow_model)
        st_path = ckpt_dir / _ADAPTER_FILE
        if not st_path.exists():
            raise FileNotFoundError(f"Missing adapter checkpoint: {st_path}")
        peft_model.load_weights(str(st_path), strict=False)
        peft_model.set_trainable(is_trainable)
        return peft_model

    # ---- export ------------------------------------------------------------------

    def export_shadow(self) -> nn.Module:
        """
        Export a standalone mlx-lm causal LM from the shadow backbone.

        The exported model reuses the base model's input embeddings and lm_head (they may
        have been removed from the shadow backbone for embedding sharing). If the shadow
        hidden size differs from the base, a `ProjectedCausalLM` bundling the trained
        `shadow_hidden_projection` is returned instead.
        """
        from .projected_causal_lm import ProjectedCausalLM

        base_inner = _get_inner_model(self.base_model)
        shadow_args = deepcopy(self.shadow_model.args)

        exported = type(self.base_model)(shadow_args)
        exported_inner = _get_inner_model(exported)
        exported_inner.load_weights(
            list(tree_flatten(self.shadow_model.parameters())), strict=False
        )

        # Embeddings: prefer the shadow table if it still exists; otherwise take the base's.
        embed = getattr(self.shadow_model, "embed_tokens", None)
        if embed is None:
            embed = base_inner.embed_tokens
        exported_inner.embed_tokens = clone_embedding(embed)

        hidden_match = self.shadow_hidden_size == self.base_hidden_size
        if hidden_match:
            base_head = getattr(self.base_model, "lm_head", None)
            if base_head is not None and getattr(exported, "lm_head", None) is not None:
                exported.lm_head = clone_linear(base_head)
            # Tied architectures score via `embed_tokens.as_linear`, already covered above.
            return exported

        # Hidden sizes differ: bundle backbone + trained projection + base lm_head.
        base_head = getattr(self.base_model, "lm_head", None)
        if base_head is None:
            vocab_size = int(base_inner.args.vocab_size)
            base_head = nn.Linear(self.base_hidden_size, vocab_size, bias=False)
            base_head.weight = base_inner.embed_tokens.weight * 1
        return ProjectedCausalLM(
            shadow_model=exported,
            shadow_hidden_projection=clone_linear(self.shadow_hidden_projection),
            lm_head=clone_linear(base_head),
        )


def get_shadow_model(
    model: nn.Module,
    shadow_config: ShadowConfig,
    *,
    shadow_model: nn.Module | None = None,
) -> ShadowPeftModel:
    return ShadowPeftModel(model, shadow_config, shadow_model=shadow_model)


def prepare_shadow_model(
    shadow_model: nn.Module,
    *,
    remove_embed_tokens: bool = False,
) -> nn.Module:
    """
    Prepare an **explicit** shadow model for ShadowPEFT (MLX version).

    Keeps backbone-only (the inner model); optionally removes `embed_tokens` so the
    shadow backbone is driven by base `input_embeddings` (embedding-sharing mode).
    """
    inner = _get_inner_model(shadow_model)
    if remove_embed_tokens and getattr(inner, "embed_tokens", None) is not None:
        inner.embed_tokens = None
    return inner
