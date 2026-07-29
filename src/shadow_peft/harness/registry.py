"""Adapter registry: holds all specialist adapter weights and swaps them on demand.

Torch-specific. One live ``ShadowForCausalLM`` serves all specialists; swapping
is an in-place weight overwrite of the four adapter modules via
``ShadowPeftModel.load_adapter_state_dict``. Because shadow is **stateful**
(each adapter carries its own evolving shadow hidden state + shadow KV-cache),
a swap must also reset ``_shadow_cache`` so the new adapter starts clean.

Base model weights are never touched by a swap -- they stay resident on device.
Adapter state dicts live on CPU and are only moved onto the live model on swap.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..peft_model import ShadowPeftModel, resolve_shadow_checkpoint_dir
from .config import ShadowHarnessConfig, SpecialistConfig

logger = logging.getLogger(__name__)

_MODULES_PREFIXES = ("lm_head", "shadow_lm_head")


def _load_adapter_state(path: str | Path) -> dict:
    """Load ``shadow_adapter.safetensors`` (or legacy ``.pt``) as a flat dict on CPU."""
    from safetensors.torch import load_file as safetensors_load_file

    ckpt_dir = resolve_shadow_checkpoint_dir(path)
    st = ckpt_dir / "shadow_adapter.safetensors"
    if st.exists():
        return safetensors_load_file(str(st))
    legacy = ckpt_dir / "shadow_adapter.pt"
    if legacy.exists():
        import torch

        return torch.load(str(legacy), map_location="cpu")
    raise FileNotFoundError(f"No shadow_adapter.safetensors or shadow_adapter.pt under {ckpt_dir}")


def _load_task_head_state(path: str | Path) -> dict:
    """Load optional ``shadow_modules.safetensors`` (trained task heads) as a flat dict."""
    from safetensors.torch import load_file as safetensors_load_file

    ckpt_dir = resolve_shadow_checkpoint_dir(path)
    st = ckpt_dir / "shadow_modules.safetensors"
    if not st.exists():
        return {}
    return safetensors_load_file(str(st))


def _sub_state(flat: dict, prefix: str) -> dict:
    """Strip a ``prefix.`` from keys, returning the sub-dict (possibly empty)."""
    p = prefix + "."
    return {k[len(p):]: v for k, v in flat.items() if k.startswith(p)}


class AdapterRegistry:
    """Holds all specialist adapter weights and swaps them onto one live model.

    Parameters
    ----------
    model:
        The live ``ShadowForCausalLM`` (or any object exposing
        ``.peft_model: ShadowPeftModel`` and ``.lm_head`` / ``.shadow_lm_head``).
        A single base model serves all specialists.
    config:
        Harness config describing all specialists.
    """

    def __init__(self, model, config: ShadowHarnessConfig) -> None:
        self.model = model
        self._config = config
        # specialist name -> adapter state dict (CPU tensors)
        self._adapter_states: dict[str, dict] = {}
        # specialist name -> task-head state dict (CPU tensors), possibly empty
        self._head_states: dict[str, dict] = {}
        self._current: str | None = None

        for spec in config.specialists:
            self._adapter_states[spec.name], self._head_states[spec.name] = self._load(spec)

    def _load(self, spec: SpecialistConfig) -> tuple[dict, dict]:
        if spec.adapter_path is None:
            # Capture the live adapter weights as-is (for tests / single-specialist use).
            adapter = _clone_to_cpu(self.model.peft_model.adapter_state_dict())
            heads = _capture_live_heads(self.model)
            return adapter, heads
        adapter = _load_adapter_state(spec.adapter_path)
        heads = _load_task_head_state(spec.adapter_path)
        return adapter, heads

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
        peft.load_adapter_state_dict(self._adapter_states[specialist])
        # Shadow is stateful: the shadow backbone weights just changed, so its
        # KV-cache is now invalid. Force a clean rebuild on the next forward.
        peft._shadow_cache = None

        self._swap_task_heads(specialist)
        self._current = specialist
        logger.debug("Swapped shadow adapter -> %s", specialist)

    def _swap_task_heads(self, specialist: str) -> None:
        """Swap trained task heads (lm_head / shadow_lm_head) if present."""
        flat = self._head_states.get(specialist, {})
        if not flat:
            return
        for prefix in _MODULES_PREFIXES:
            mod = getattr(self.model, prefix, None)
            if mod is None:
                continue
            sub = _sub_state(flat, prefix)
            if sub:
                mod.load_state_dict(sub, strict=True)

    @property
    def current(self) -> str | None:
        """Name of the currently-active specialist, or ``None`` if none yet."""
        return self._current


def _capture_live_heads(model) -> dict:
    """Snapshot lm_head / shadow_lm_head weights (CPU clone), if present."""
    flat: dict[str, object] = {}
    for prefix in _MODULES_PREFIXES:
        mod = getattr(model, prefix, None)
        if mod is None:
            continue
        for k, v in mod.state_dict().items():
            flat[f"{prefix}.{k}"] = v
    return _clone_to_cpu(flat)


def _clone_to_cpu(state: dict) -> dict:
    """Clone every tensor in ``state`` to CPU (detach + clone)."""
    import torch

    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        out[k] = v.detach().clone().cpu()
    return out
