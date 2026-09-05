"""Bridge a flat replayable ES genome to Mage NVFP4 Linear reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .nvfp4_runtime import LinearCensusRow, census_linear_layers, materialize_native_nvfp4
from .quasar import QuasarNVFP4Config, reconstruct_nvfp4


@dataclass(frozen=True)
class LayerClipSlice:
    name: str
    shape: tuple[int, int]
    start: int
    stop: int

    @property
    def dimensions(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class ClipGenomeLayout:
    layers: tuple[LayerClipSlice, ...]
    dimension: int

    @classmethod
    def from_census(cls, rows: list[LinearCensusRow]) -> ClipGenomeLayout:
        layers = []
        cursor = 0
        for row in rows:
            if not row.eligible:
                continue
            shape = (row.out_features, (row.in_features + 127) // 128)
            stop = cursor + row.reconstruction_groups
            layers.append(LayerClipSlice(row.name, shape, cursor, stop))
            cursor = stop
        if not layers:
            raise ValueError("clip genome layout has no eligible layers")
        return cls(tuple(layers), cursor)

    @classmethod
    def from_model(
        cls,
        root: nn.Module,
        *,
        target_modules: tuple[str, ...] | None = None,
        exclude_modules: tuple[str, ...] = (),
    ) -> ClipGenomeLayout:
        return cls.from_census(
            census_linear_layers(
                root,
                target_modules=target_modules,
                exclude_modules=exclude_modules,
            )
        )

    def split(self, clips: torch.Tensor) -> dict[str, torch.Tensor]:
        values = clips.reshape(-1)
        if values.numel() != self.dimension or not torch.isfinite(values).all():
            raise ValueError("clip genome shape/value mismatch")
        return {
            layer.name: values[layer.start : layer.stop].reshape(layer.shape)
            for layer in self.layers
        }


@dataclass(frozen=True)
class FrozenLinearWeight:
    name: str
    value: torch.Tensor


class FrozenLinearWeights:
    """CPU master weights used to restore one materialized candidate at a time."""

    def __init__(self, weights: tuple[FrozenLinearWeight, ...]):
        if not weights or len({item.name for item in weights}) != len(weights):
            raise ValueError("frozen Linear master must be non-empty and uniquely named")
        if any(item.value.device.type != "cpu" for item in weights):
            raise ValueError("frozen Linear master weights must live on CPU")
        self._weights = weights

    @classmethod
    def capture(cls, root: nn.Module, layout: ClipGenomeLayout) -> FrozenLinearWeights:
        weights = []
        for layer in layout.layers:
            module = root.get_submodule(layer.name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"{layer.name} is not a Linear")
            weights.append(
                FrozenLinearWeight(
                    layer.name,
                    module.weight.detach().to(device="cpu").clone().contiguous(),
                )
            )
        return cls(tuple(weights))

    @property
    def saliency_by_layer(self) -> dict[str, torch.Tensor] | None:
        return None

    def validate_layout(self, layout: ClipGenomeLayout) -> None:
        if tuple(item.name for item in self._weights) != tuple(layer.name for layer in layout.layers):
            raise ValueError("master/layout layer mismatch")

    @property
    def parameter_count(self) -> int:
        return sum(item.value.numel() for item in self._weights)

    @torch.no_grad()
    def restore(self, root: nn.Module, device: str | torch.device) -> None:
        for item in self._weights:
            module = root.get_submodule(item.name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"{item.name} is not a Linear")
            module.weight = nn.Parameter(
                item.value.to(device=device, copy=True),
                requires_grad=False,
            )
        root.requires_grad_(False)


@torch.no_grad()
def quasar_initial_clips(
    root: nn.Module,
    layout: ClipGenomeLayout,
    *,
    config: QuasarNVFP4Config | None = None,
) -> tuple[torch.Tensor, dict]:
    """Run the 15-point QUASAR grid once and flatten its winning group clips."""

    config = config or QuasarNVFP4Config(reconstruction_group_size=128)
    if config.reconstruction_group_size != 128:
        raise ValueError("Phase-1 ES clip layout requires reconstruction group 128")
    clips = torch.empty(layout.dimension, dtype=torch.float32, device="cpu")
    weighted_error = 0.0
    counts = {str(factor): 0 for factor in config.clip_factors}
    for layer in layout.layers:
        module = root.get_submodule(layer.name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"{layer.name} is not a Linear")
        reconstruction = reconstruct_nvfp4(module.weight, saliency=None, config=config)
        values = reconstruction.clip_factor_value.detach().float().cpu().reshape(-1)
        clips[layer.start : layer.stop].copy_(values)
        weighted_error += float(reconstruction.weighted_error)
        for factor, count in zip(
            config.clip_factors,
            torch.bincount(
                reconstruction.clip_factor_index.reshape(-1).long(),
                minlength=len(config.clip_factors),
            ).cpu(),
            strict=True,
        ):
            counts[str(factor)] += int(count)
    return clips, {
        "scheme": "quasar-grid",
        "dimension": layout.dimension,
        "weighted_reconstruction_error": weighted_error,
        "clip_factor_counts": counts,
    }


@torch.no_grad()
def materialize_es_candidate(
    root: nn.Module,
    master: FrozenLinearWeights,
    layout: ClipGenomeLayout,
    clips: torch.Tensor,
    *,
    device: str | torch.device,
    quasar_config: QuasarNVFP4Config | None = None,
    use_triton_kernel: bool = False,
) -> dict:
    """Restore BF16 masters, then materialize exactly one native NVFP4 candidate."""

    master.validate_layout(layout)
    master.restore(root, device)
    saliency = master.saliency_by_layer
    options = {"saliency_by_layer": saliency} if saliency is not None else {}
    return materialize_native_nvfp4(
        root,
        scheme="external-clip",
        quasar_config=quasar_config,
        clip_by_layer=layout.split(clips),
        target_modules=tuple(layer.name for layer in layout.layers),
        use_triton_kernel=use_triton_kernel,
        **options,
    )
