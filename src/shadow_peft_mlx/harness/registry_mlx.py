"""Adapter registry (MLX): holds all specialist adapter weights and swaps them.

MLX counterpart of ``shadow_peft/harness/registry.py``. One live
``ShadowForCausalLM`` serves all specialists; swapping is an in-place weight
overwrite via ``ShadowPeftModel.load_weights(..., strict=False)``. Because
shadow is **stateful** (each adapter carries its own evolving shadow hidden
state + shadow KV-cache), a swap also resets ``_runtime.shadow_cache`` so the
new adapter starts clean.

Key MLX differences from torch:
- No ``load_adapter_state_dict`` -- use ``load_weights(state, strict=False)``.
- Shadow state lives on ``peft._runtime`` (a plain object, not instance attrs),
  because MLX would otherwise register mx.arrays as parameters.
- Adapter snapshot: ``peft.adapter_parameters()`` (== torch ``adapter_state_dict``).
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..peft_model import ShadowPeftModel
from .config import ShadowHarnessConfig, SpecialistConfig

logger = logging.getLogger(__name__)

_ADAPTER_FILE = "shadow_adapter.safetensors"
_MODULES_FILE = "shadow_modules.safetensors"
_HEAD_PREFIXES = ("lm_head", "shadow_lm_head")


def _load_adapter_state(path: str | Path) -> dict:
    """Load ``shadow_adapter.safetensors`` as a flat ``{name: mx.array}`` dict."""
    import mlx.core as mx

    st = Path(path) / _ADAPTER_FILE
    if not st.exists():
        raise FileNotFoundError(f"Missing adapter checkpoint: {st}")
    return mx.load(str(st))


def _load_head_state(path: str | Path) -> dict:
    """Load optional ``shadow_modules.safetensors`` (trained task heads)."""
    import mlx.core as mx

    st = Path(path) / _MODULES_FILE
    if not st.exists():
        return {}
    return mx.load(str(st))


def _capture_live_adapter(peft: ShadowPeftModel) -> dict:
    """Snapshot the live adapter parameters (for tests / single-specialist use)."""
    import mlx.core as mx

    # Force-evaluate the lazy arrays so the snapshot is a stable copy.
    return {k: mx.array(v) for k, v in peft.adapter_parameters().items()}


def _capture_live_heads(model) -> dict:
    """Snapshot lm_head / shadow_lm_head weights, if present."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    flat: dict[str, object] = {}
    for prefix in _HEAD_PREFIXES:
        mod = getattr(model, prefix, None)
        if mod is None:
            continue
        for k, v in tree_flatten(mod.parameters()):
            flat[f"{prefix}.{k}"] = mx.array(v)
    return flat


class AdapterRegistryMlx:
    """Holds all specialist adapter weights and swaps them onto one live MLX model."""

    def __init__(self, model, config: ShadowHarnessConfig) -> None:
        self.model = model
        self._config = config
        # specialist name -> adapter param dict {name: mx.array}
        self._adapter_states: dict[str, dict] = {}
        # specialist name -> task-head state dict, possibly empty
        self._head_states: dict[str, dict] = {}
        self._current: str | None = None

        for spec in config.specialists:
            self._adapter_states[spec.name], self._head_states[spec.name] = self._load(spec)

    def _load(self, spec: SpecialistConfig) -> tuple[dict, dict]:
        if spec.adapter_path is None:
            adapter = _capture_live_adapter(self.model.peft_model)
            heads = _capture_live_heads(self.model)
            return adapter, heads
        return _load_adapter_state(spec.adapter_path), _load_head_state(spec.adapter_path)

    def ensure(self, specialist: str) -> None:
        """Make sure ``specialist``'s adapter weights are loaded into the live model.

        If ``specialist`` is already active this is a no-op. Otherwise it
        overwrites the four adapter modules in place and resets the shadow
        KV-cache (shadow is stateful -- a stale cache would mix adapters).
        """
        if specialist == self._current:
            return
        if specialist not in self._adapter_states:
            raise KeyError(f"Unknown specialist: {specialist!r}")

        peft: ShadowPeftModel = self.model.peft_model
        # swap_adapter loads the adapter parameters in place AND resets the
        # shadow runtime state (hidden states + KV-cache) in one atomic step.
        peft.swap_adapter(self._adapter_states[specialist])

        self._swap_task_heads(specialist)
        self._current = specialist
        logger.debug("Swapped shadow adapter (mlx) -> %s", specialist)

    def _swap_task_heads(self, specialist: str) -> None:
        """Swap trained task heads (lm_head / shadow_lm_head) if present."""
        flat = self._head_states.get(specialist, {})
        if not flat:
            return
        # Heads live on the ShadowForCausalLM wrapper; load_weights overwrites
        # matching lm_head.* / shadow_lm_head.* params, ignoring the rest.
        self.model.load_weights(list(flat.items()), strict=False)

    @property
    def current(self) -> str | None:
        """Name of the currently-active specialist, or ``None`` if none yet."""
        return self._current
