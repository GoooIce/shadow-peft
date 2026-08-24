"""Spectral diagnostics suite (ticket 08, P1).

Independent-report machinery — never a Pareto axis. Three pieces:

- :class:`ShadowStateCollector` — forward hooks on the wrapped decoder layers
  capture the evolving shadow state entering each layer (the tensor the
  adapter holds at that layer's entry), stacked per batch and savable to disk.
- spectral metrics — effective rank (Roy–Vetterli erank over the singular
  distribution) and **LAR** defined here as the top-direction energy share
  ``λ_max / Σλ`` (1 = fully anisotropic/rank-1-leaning, → 1/r for isotropic).
  Both are scale-invariant ratios.
- spectral drift between two captures (pre/post quantization) — cosine
  distance between normalized singular spectra, principal angle between the
  top-k singular subspaces, and the scale-sensitive ``energy_ratio`` (Σλ
  ratio; ≈1 means the states carry the same magnitude).

Gemma4 Peri-LN caveat: layer outputs carry a per-layer ``layer_scalar``
buffer multiplier, so states captured from different models must be divided
by the producing layer's scalar before magnitude-sensitive comparisons —
the collector does this by default (``normalize_layer_scalar=True``), which
leaves the scale-invariant metrics untouched but makes ``energy_ratio``
meaningful across models.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import torch

from shadow_peft.model_utils import _get_decoder_layers
from shadow_peft.peft_model import ShadowPeftModel

TOP_K_DEFAULT = 16
MAX_SVD_ROWS = 4096  # row cap for the flat SVD (deterministic stride)


def _divide_layer_scalar(state: torch.Tensor, source_module: torch.nn.Module) -> torch.Tensor:
    """Undo the Gemma4 Peri-LN `layer_scalar` multiplier if present."""
    s = getattr(source_module, "layer_scalar", None)
    if isinstance(s, torch.Tensor) and s.numel() == 1 and s.item() != 0.0:
        return state / s.item()
    return state


class ShadowStateCollector:
    """Capture the per-layer shadow state trajectory via forward hooks.

    ``captures[layer_idx]`` is a list of ``[batch, seq, hidden]`` float32 CPU
    tensors, one per forwarded batch. The hook is a *pre*-hook on each wrapped
    decoder layer, reading the adapter's current shadow state at that layer's
    entry (layer 0 records the shadow backbone's initial output).
    """

    def __init__(
        self,
        peft_model: ShadowPeftModel,
        layer_indices: Sequence[int] | None = None,
        *,
        normalize_layer_scalar: bool = True,
    ) -> None:
        self.peft_model = peft_model
        self.normalize = normalize_layer_scalar
        backbone, layers, _ = _get_decoder_layers(peft_model.base_model)
        self.layer_indices = list(range(len(layers))) if layer_indices is None else list(layer_indices)
        self.captures: dict[int, list[torch.Tensor]] = {}
        self._handles: list[Any] = []

    def _make_hook(self, layer_idx: int):
        def pre_hook(module, args):
            state = self.peft_model._shadow_hidden_states
            if state is None:
                return
            st = state.detach().to(torch.float32).cpu()
            if self.normalize:
                st = _divide_layer_scalar(st, module)
            self.captures.setdefault(layer_idx, []).append(st)

        return pre_hook

    def attach(self) -> None:
        backbone, layers, _ = _get_decoder_layers(self.peft_model.base_model)
        self.detach()
        for idx in self.layer_indices:
            self._handles.append(layers[idx].register_forward_pre_hook(self._make_hook(idx)))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    @torch.no_grad()
    def run(self, model: torch.nn.Module, batches: list[dict[str, torch.Tensor]]) -> None:
        """Forward ``batches`` through ``model`` (peft or task wrapper)."""
        self.attach()
        was_training = model.training
        model.eval()
        try:
            for batch in batches:
                model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                )
        finally:
            self.detach()
            if was_training:
                model.train()

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        stacked = {idx: torch.stack(ts) for idx, ts in self.captures.items()}
        torch.save({"layer_indices": self.layer_indices, "states": stacked}, out)

    @staticmethod
    def load(path: str | Path) -> dict[int, torch.Tensor]:
        obj = torch.load(path, weights_only=True)
        return {int(k): v for k, v in obj["states"].items()}


# ---------------------------------------------------------------------------
# Spectral metrics
# ---------------------------------------------------------------------------


def _flat_states(states: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    """``[n, b, t, h]`` (or list of ``[b, t, h]``) -> row-capped ``[N, h]``."""
    if isinstance(states, list):
        states = torch.stack(states)
    flat = states.reshape(-1, states.shape[-1])
    if flat.shape[0] > MAX_SVD_ROWS:
        stride = math.ceil(flat.shape[0] / MAX_SVD_ROWS)
        flat = flat[::stride]
    return flat


def singular_spectrum(states: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    """Singular values (descending) of the flattened state matrix."""
    flat = _flat_states(states).to(torch.float32)
    return torch.linalg.svdvals(flat)


def effective_rank(states: torch.Tensor | list[torch.Tensor]) -> float:
    """Roy–Vetterli erank: exp(entropy of the normalized singular energy)."""
    s = singular_spectrum(states)
    e = s**2
    p = e / e.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def lar(states: torch.Tensor | list[torch.Tensor]) -> float:
    """Top-direction energy share λ_max / Σλ (see module docstring)."""
    s = singular_spectrum(states)
    e = s**2
    return float(e[0] / e.sum())


def layer_metrics(states: torch.Tensor | list[torch.Tensor], top_k: int = TOP_K_DEFAULT) -> dict[str, Any]:
    s = singular_spectrum(states)
    e = s**2
    return {
        "erank": effective_rank(states),
        "lar": float(e[0] / e.sum()),
        "energy": float(e.sum()),
        "top_singular_values": [float(v) for v in s[:top_k]],
        "rows": int(_flat_states(states).shape[0]),
    }


def spectral_drift(
    pre_states: torch.Tensor | list[torch.Tensor],
    post_states: torch.Tensor | list[torch.Tensor],
    top_k: int = TOP_K_DEFAULT,
) -> dict[str, Any]:
    """Drift between two captures of the same layer (pre vs post quantization).

    ``spectrum_cosine`` ∈ [0, 1] (1 = identical shapes), ``subspace_angle_deg``
    is the largest principal angle between top-k left singular subspaces,
    ``energy_ratio`` = Σλ(post) / Σλ(pre) (scale-sensitive; ≈1 after layer-
    scalar normalization when only scaling changed).
    """
    a = _flat_states(pre_states).to(torch.float32)
    b = _flat_states(post_states).to(torch.float32)
    sa, sb = torch.linalg.svdvals(a), torch.linalg.svdvals(b)
    ea, eb = sa**2, sb**2

    def _cos(x: torch.Tensor, y: torch.Tensor) -> float:
        n = min(x.numel(), y.numel())
        xa, xb = x[:n], y[:n]
        denom = xa.norm() * xb.norm()
        return float((xa @ xb) / denom) if denom > 0 else 0.0

    k = min(top_k, a.shape[1], b.shape[1])
    ua = torch.linalg.svd(a)[0][:, :k]
    ub = torch.linalg.svd(b)[0][:, :k]
    sv = torch.linalg.svdvals(ua.T @ ub)
    sv = sv.clamp(-1.0, 1.0)
    angle = float(torch.rad2deg(torch.acos(sv.min())))

    return {
        "spectrum_cosine": _cos(ea, eb),
        "subspace_angle_deg": angle,
        "energy_ratio": float(eb.sum() / ea.sum()) if ea.sum() > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def metrics_from_capture(capture: dict[int, torch.Tensor | list[torch.Tensor]]) -> dict[str, Any]:
    return {str(idx): layer_metrics(states) for idx, states in sorted(capture.items())}


def drift_report(
    pre: dict[int, torch.Tensor | list[torch.Tensor]],
    post: dict[int, torch.Tensor | list[torch.Tensor]],
) -> dict[str, Any]:
    common = sorted(set(pre) & set(post))
    layers = {str(i): spectral_drift(pre[i], post[i]) for i in common}
    cos = [v["spectrum_cosine"] for v in layers.values()]
    ang = [v["subspace_angle_deg"] for v in layers.values()]
    enr = [v["energy_ratio"] for v in layers.values()]
    return {
        "layers": layers,
        "summary": {
            "mean_spectrum_cosine": sum(cos) / len(cos) if cos else None,
            "max_subspace_angle_deg": max(ang) if ang else None,
            "mean_energy_ratio": sum(enr) / len(enr) if enr else None,
        },
    }


def render_report(
    pre_metrics: dict[str, Any],
    post_metrics: dict[str, Any] | None = None,
    drift: dict[str, Any] | None = None,
    *,
    title: str = "Shadow-state spectral diagnostics",
) -> str:
    """Markdown report (independent — not a Pareto axis)."""
    lines = [
        f"# {title}",
        "",
        "> Independent spectral report; not part of the Pareto axes.",
        "",
        "| layer | erank | LAR | energy |",
        "|---|---|---|---|",
    ]
    for idx, m in pre_metrics.items():
        lines.append(
            f"| pre/{idx} | {m['erank']:.3f} | {m['lar']:.3f} | {m['energy']:.3e} |"
        )
    if post_metrics:
        for idx, m in post_metrics.items():
            lines.append(
                f"| post/{idx} | {m['erank']:.3f} | {m['lar']:.3f} | {m['energy']:.3e} |"
            )
    if drift:
        lines += [
            "",
            "## spectral drift (pre → post)",
            "",
            "| layer | spectrum cos | subspace angle (deg) | energy ratio |",
            "|---|---|---|---|",
        ]
        for idx, d in drift["layers"].items():
            lines.append(
                f"| {idx} | {d['spectrum_cosine']:.4f} | "
                f"{d['subspace_angle_deg']:.2f} | {d['energy_ratio']:.4f} |"
            )
        s = drift["summary"]
        lines += [
            "",
            f"- mean spectrum cosine: {s['mean_spectrum_cosine']:.4f}",
            f"- max subspace angle: {s['max_subspace_angle_deg']:.2f}°",
            f"- mean energy ratio: {s['mean_energy_ratio']:.4f}",
        ]
    return "\n".join(lines) + "\n"
