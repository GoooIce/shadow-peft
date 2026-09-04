"""Greedy layer-group schedule machinery (exp6: docs/exp6-layer-group-recovery.md).

Capture-and-replay drives the mounted student layer by layer without any
family-specific mask/rotary code: one no-grad forward through ``peft`` records
every wrapped layer's call frame (args/kwargs minus ``hidden_states``) plus the
layer-boundary states — base hidden ``h_k`` and the evolved shadow state
``s_k``. Training a group then re-executes only the group's layers with grad,
fed the detached boundary state; the recorded frames replay the model's own
attention-mask / position math verbatim.

The shadow-chain recompute exploits the update algebra —
``s' = s + g(h)·(T(h) − s)`` reads only the base hidden trajectory — so the
prefix chain can be rebuilt cheaply under grad, letting gradients reach the
prefix ShadowUpdate modules that a purely local group loss would starve (the
local blind spot: ``update[k]`` runs after layer k and only feeds layer k+1's
injection).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from shadow_peft.model_utils import _get_backbone, _get_decoder_layers


@dataclass
class LayerFrame:
    """One wrapped base layer's captured call frame + boundary outputs."""

    call_args: tuple = ()
    call_kwargs: dict = field(default_factory=dict)
    hidden_out: torch.Tensor | None = None  # output of layer k (detached)
    shadow_out: torch.Tensor | None = None  # shadow state after layer k (detached)


@dataclass
class CaptureBatch:
    frames: list[LayerFrame]
    shadow_initial: torch.Tensor  # trunk output (projected) entering layer 1
    model_hidden_states: tuple | None = None  # the forward's own tuple, for preflight parity


def _wrapped_layers(peft) -> list:
    return _get_decoder_layers(peft.base_model)[1]


@torch.no_grad()
def capture_frames(peft, *, input_ids, attention_mask) -> CaptureBatch:
    """No-grad student forward that records per-layer call frames and boundaries."""
    layers = _wrapped_layers(peft)
    frames = [LayerFrame() for _ in layers]
    initial: dict[str, torch.Tensor] = {}

    def pre(k: int):
        def hook(module, args, kwargs):
            f = frames[k]
            f.call_args = tuple(args[1:]) if args else ()
            f.call_kwargs = {key: v for key, v in kwargs.items() if key != "hidden_states"}
            if k == 0:
                initial["s"] = peft._shadow_hidden_states.detach().clone()

        return hook

    def post(k: int):
        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            frames[k].hidden_out = hs.detach()
            frames[k].shadow_out = peft._shadow_hidden_states.detach()

        return hook

    handles = []
    for k, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(pre(k), with_kwargs=True))
        handles.append(layer.register_forward_hook(post(k)))
    try:
        out = peft(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
    finally:
        for h in handles:
            h.remove()
    return CaptureBatch(
        frames=frames,
        shadow_initial=initial["s"],
        model_hidden_states=getattr(out, "hidden_states", None),
    )


def recompute_shadow_chain(peft, cap: CaptureBatch, *, upto: int) -> torch.Tensor:
    """Rebuild the shadow state entering layer ``upto`` under grad.

    Runs the update modules over the captured (detached) prefix hidden
    trajectory, so gradients flow into the prefix ShadowUpdate parameters —
    the mitigation for the local blind spot. Cheap by construction: LayerNorm
    plus two small MLPs per prefix layer.
    """
    s = cap.shadow_initial
    for j in range(1, upto):
        s = peft.shadow_update_model(cap.frames[j].hidden_out, s, j - 1)
    return s


def group_hidden_mse(
    student_hiddens, teacher_hiddens, labels: torch.Tensor
) -> torch.Tensor:
    """OneBit-style normalized hidden MSE over a group's layers.

    Same math as the end-to-end hidden-MSE term: L2-normalize each token
    vector, per-layer MSE, averaged over the group's layers and non-ignored
    positions (position-wise mask, no shift — matching the existing term).
    """
    if len(student_hiddens) != len(teacher_hiddens):
        raise ValueError(
            f"group target mismatch: {len(student_hiddens)} student layers vs "
            f"{len(teacher_hiddens)} teacher layers"
        )
    valid = (labels != -100).unsqueeze(-1).to(torch.float32)
    denom = valid.sum().clamp_min(1.0)
    total = None
    for student, teacher in zip(student_hiddens, teacher_hiddens, strict=True):
        s_n = F.normalize(student.float(), dim=-1)
        t_n = F.normalize(teacher.float(), dim=-1)
        mse = ((s_n - t_n) ** 2).sum(dim=-1, keepdim=True)
        layer_loss = (mse * valid).sum() / denom
        total = layer_loss if total is None else total + layer_loss
    return total / len(student_hiddens)


def layer_mse_table(student_hidden_states, teacher_hidden_states, labels) -> list[float]:
    """Per-layer normalized MSE (attribution axis), skipping the embeddings row."""
    valid = (labels != -100).unsqueeze(-1).to(torch.float32)
    denom = valid.sum().clamp_min(1.0).item()
    table = []
    for student, teacher in zip(student_hidden_states[1:], teacher_hidden_states[1:], strict=True):
        s_n = F.normalize(student.float(), dim=-1)
        t_n = F.normalize(teacher.float(), dim=-1)
        mse = ((s_n - t_n) ** 2).sum(dim=-1, keepdim=True)
        table.append(round(((mse * valid).sum() / denom).item(), 6))
    return table


def _final_norm(peft, hidden: torch.Tensor) -> torch.Tensor:
    """Apply the student backbone's final norm (HF's hidden_states tuple reports
    the LAST layer's output post-norm; replayed layer outputs are pre-norm, so
    the final group's last target must be normalized on the student side too).
    Family naming varies: Llama/Qwen use ``norm``, LFM2 uses ``embedding_norm``.
    """
    backbone = _get_backbone(peft.base_model)
    for name in ("norm", "embedding_norm", "final_norm", "final_layernorm", "norm_f"):
        norm = getattr(backbone, name, None)
        if norm is not None:
            return norm(hidden)
    raise AttributeError(
        "student backbone has no recognizable final norm (tried norm/embedding_norm/"
        "final_norm/final_layernorm/norm_f); cannot align the last-layer replay "
        "target with the teacher's post-norm hidden state"
    )


def group_forward_loss(
    peft,
    cap: CaptureBatch,
    *,
    group: tuple[int, int],
    teacher_hidden_states,
    labels: torch.Tensor,
    recompute_chain: bool = True,
) -> torch.Tensor:
    """Replay one group's layers with grad; normalized-MSE loss vs teacher.

    Boundary tensors are detached constants (prefix ran under no_grad); the
    trainable surface is the group's injections plus — via the wrapper's own
    per-layer update and the optional prefix chain recompute — the update
    modules at or before the group's last loss-relevant layer.
    """
    a, b = group
    layers = _wrapped_layers(peft)
    if not (1 <= a <= b < len(layers)):
        raise ValueError(f"group [{a},{b}] outside adapted layers 1..{len(layers) - 1}")
    s_in = (
        recompute_shadow_chain(peft, cap, upto=a)
        if recompute_chain
        else cap.frames[a - 1].shadow_out
    )
    h = cap.frames[a - 1].hidden_out
    peft._shadow_hidden_states = s_in
    outs = []
    try:
        for k in range(a, b + 1):
            out = layers[k](h, *cap.frames[k].call_args, **cap.frames[k].call_kwargs)
            h = out[0] if isinstance(out, tuple) else out
            outs.append(h)
    finally:
        peft._shadow_hidden_states = None
    if b == len(layers) - 1:
        outs[-1] = _final_norm(peft, outs[-1])
    return group_hidden_mse(outs, teacher_hidden_states[a + 1 : b + 2], labels)


def layer_groups(num_layers: int, schedule: str, group_size: int) -> list[tuple[int, int]]:
    """Chunk the adapted base layers 1..num_layers-1 into inclusive (a, b) groups."""
    if schedule not in ("per_layer", "groups"):
        raise ValueError(f"unsupported layer_schedule={schedule!r}")
    size = 1 if schedule == "per_layer" else group_size
    if size < 1:
        raise ValueError(f"group_size must be >= 1, got {size}")
    adapted = list(range(1, num_layers))
    return [
        (chunk[0], chunk[-1])
        for chunk in (adapted[i : i + size] for i in range(0, len(adapted), size))
    ]


@torch.no_grad()
def replay_parity(peft, *, input_ids, attention_mask) -> dict:
    """Preflight gate: replay ALL layers from the captured boundary and diff
    against the model's own forward. Zero → the capture/replay contract holds
    on this architecture (mask/rotary frames replayed verbatim). Any nonzero
    drift means the family does something the frames do not carry — refuse to
    train on it.
    """
    cap = capture_frames(peft, input_ids=input_ids, attention_mask=attention_mask)
    layers = _wrapped_layers(peft)
    if cap.model_hidden_states is None or len(cap.model_hidden_states) != len(layers) + 1:
        raise RuntimeError("capture forward did not return a full hidden_states tuple")
    h = cap.model_hidden_states[0].detach()
    peft._shadow_hidden_states = cap.shadow_initial
    diffs = {}
    try:
        for k, layer in enumerate(layers):
            out = layer(h, *cap.frames[k].call_args, **cap.frames[k].call_kwargs)
            h = out[0] if isinstance(out, tuple) else out
            ref = cap.model_hidden_states[k + 1]
            if k == len(layers) - 1:
                h = _final_norm(peft, h)
            diffs[f"layer_{k}"] = (h.float() - ref.float()).abs().max().item()
    finally:
        peft._shadow_hidden_states = None
    return {"max_abs_diff": max(diffs.values()), "per_layer": diffs}
