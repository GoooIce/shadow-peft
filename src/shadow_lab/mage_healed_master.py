"""Immutable QAD latent + AdamW saliency handoff for clip-only ES."""

from __future__ import annotations

from dataclasses import asdict

import torch

from .es_quant import ClipGenomeLayout, FrozenLinearWeight, FrozenLinearWeights, LayerClipSlice
from .mage_qad_checkpoint import content_hash, cpu_copy


class HealedMaster(FrozenLinearWeights):
    def __init__(self, checkpoint: dict, *, selected_sha256: str):
        payload = {k: v for k, v in checkpoint.items() if k != "sha256"}
        if checkpoint["sha256"] != selected_sha256 or content_hash(payload) != selected_sha256:
            raise ValueError("healed master must be the selected QAD checkpoint")
        super().__init__(
            tuple(
                FrozenLinearWeight(name, value.detach().cpu().clone())
                for name, value in checkpoint["latent"].items()
            )
        )
        self.selected_sha256 = selected_sha256
        self._saliency = cpu_copy(checkpoint["saliency"])
        self._initial_clips = checkpoint["clips"].detach().cpu().clone()
        layout = checkpoint["layout"]
        self.layout = ClipGenomeLayout(
            tuple(
                LayerClipSlice(
                    row["name"], (row["shape"][0], row["shape"][1]), row["start"], row["stop"]
                )
                for row in layout["layers"]
            ),
            layout["dimension"],
        )
        if set(self._saliency) != {item.name for item in self._weights}:
            raise ValueError("healed master saliency is incomplete")
        self._fingerprint = self.fingerprint()

    def fingerprint(self) -> str:
        return content_hash(
            {
                "weights": {item.name: item.value for item in self._weights},
                "saliency": self._saliency,
                "clips": self._initial_clips,
                "layout": asdict(self.layout),
            }
        )

    def verify(self) -> None:
        if self.fingerprint() != self._fingerprint:
            raise ValueError("healed master was mutated after QAD")

    @property
    def initial_clips(self) -> torch.Tensor:
        self.verify()
        return self._initial_clips.clone()

    @property
    def saliency_by_layer(self) -> dict[str, torch.Tensor]:
        self.verify()
        return cpu_copy(self._saliency)

    def validate_layout(self, layout: ClipGenomeLayout) -> None:
        self.verify()
        if layout != self.layout:
            raise ValueError("healed master/layout identity mismatch")

    def restore(self, root, device) -> None:
        self.verify()
        if any(p.requires_grad for p in root.parameters()):
            raise ValueError("healed handoff rejects trainable latent weights")
        super().restore(root, device)
