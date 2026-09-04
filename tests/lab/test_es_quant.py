"""Tests for the flat ES genome to per-Linear clip bridge."""

from __future__ import annotations

import torch
from torch import nn

from shadow_lab.es_quant import ClipGenomeLayout, FrozenLinearWeights, quasar_initial_clips


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(128, 16), nn.GELU(), nn.Linear(256, 32, bias=False))


def test_layout_round_trips_flat_clips_in_named_layer_order() -> None:
    model = _model()
    layout = ClipGenomeLayout.from_model(model)
    clips = torch.linspace(0.3, 1.0, layout.dimension)

    split = layout.split(clips)

    assert layout.dimension == 16 + 64
    assert list(split) == ["0", "2"]
    assert split["0"].shape == (16, 1)
    assert split["2"].shape == (32, 2)
    assert torch.equal(torch.cat([value.reshape(-1) for value in split.values()]), clips)


def test_cpu_master_restores_weights_and_freezes_every_parameter() -> None:
    model = _model()
    original = {name: module.weight.detach().clone() for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    layout = ClipGenomeLayout.from_model(model)
    master = FrozenLinearWeights.capture(model, layout)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.weight.data.zero_()

    master.restore(model, "cpu")

    assert master.parameter_count == sum(value.numel() for value in original.values())
    assert all(torch.equal(model.get_submodule(name).weight, value) for name, value in original.items())
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_quasar_initialization_has_one_group_clip_per_layout_dimension() -> None:
    model = _model()
    layout = ClipGenomeLayout.from_model(model)

    clips, report = quasar_initial_clips(model, layout)

    assert clips.shape == (layout.dimension,)
    assert torch.all(clips >= 0.3)
    assert torch.all(clips <= 1.0)
    assert sum(report["clip_factor_counts"].values()) == layout.dimension
