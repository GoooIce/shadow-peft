"""Atomic, content-addressed Mage QAD training and deployment artifacts."""

from __future__ import annotations

import hashlib
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import torch

from .es_quant import ClipGenomeLayout, LayerClipSlice
from .image_eval import canonical_json
from .quasar import QuasarNVFP4Controller


def content_hash(value) -> str:
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(canonical_json([str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(str(key))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"[")
            for entry in item:
                visit(entry)
            digest.update(b"]")
        else:
            digest.update(canonical_json(item).encode())
        digest.update(b"\x00")

    visit(value)
    return digest.hexdigest()


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return value


def atomic_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".qad-")
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def link_checkpoint(source: Path, target: Path) -> None:
    """Atomically make ``target`` a second name for the already committed ``source``.

    A best/latest pair with identical content needs one serialization, not two: the
    save window is the campaign's OOM-prone phase. ``target`` keeps pointing at a
    complete checkpoint at every instant; a later ``atomic_save`` to ``target``
    replaces only that name and leaves ``source`` untouched.
    """
    if not source.is_file():
        raise FileNotFoundError(f"cannot link missing checkpoint {source}")
    if source.resolve() == target.resolve():
        raise ValueError("checkpoint link source and target are the same path")
    fd, name = tempfile.mkstemp(dir=target.parent, prefix=".qad-link-")
    os.close(fd)
    os.unlink(name)
    try:
        os.link(source, name)
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_checkpoint(
    path: Path,
    controller: QuasarNVFP4Controller,
    optimizer: torch.optim.Optimizer,
    identity: dict,
    progress: Mapping,
) -> dict:
    packed, latent, saliency, slices, clips = {}, {}, {}, [], []
    cursor = 0
    for name, layer in controller.layers.items():
        moment = optimizer.state.get(layer.weight, {}).get("exp_avg_sq")
        if moment is None:
            raise ValueError(f"missing AdamW second moment for {name}")
        reconstruction = layer.reconstruction()
        clip = reconstruction.clip_factor_value
        slices.append(
            LayerClipSlice(name, (clip.shape[0], clip.shape[1]), cursor, cursor + clip.numel())
        )
        cursor += clip.numel()
        clips.append(clip.reshape(-1))
        latent[name] = layer.weight
        saliency[name] = moment
        packed[name] = {
            "weight_packed": reconstruction.weight_packed,
            "weight_scale": reconstruction.weight_scale,
            "tensor_scale": reconstruction.tensor_scale,
            "input_tensor_scale": layer.input_tensor_scale,
        }
    value = cpu_copy(
        {
            "schema_version": 1,
            "identity": identity,
            "progress": progress,
            "model": controller.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "latent": latent,
            "saliency": saliency,
            "packed": packed,
            "clips": torch.cat(clips),
            "layout": asdict(ClipGenomeLayout(tuple(slices), cursor)),
            "quasar_config": asdict(controller.config),
            "rng": {
                "torch": torch.get_rng_state(),
                "python": random.getstate(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        }
    )
    value["sha256"] = content_hash(value)
    atomic_save(value, path)
    return value


def read_checkpoint(path: Path) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=True)
    expected = value.pop("sha256")
    if content_hash(value) != expected:
        raise ValueError("QAD checkpoint content hash mismatch")
    value["sha256"] = expected
    return value


def load_checkpoint(
    path: Path, controller: QuasarNVFP4Controller, optimizer: torch.optim.Optimizer, identity: dict
) -> dict:
    value = read_checkpoint(path)
    if value["identity"] != identity or value["quasar_config"] != asdict(controller.config):
        raise ValueError("QAD checkpoint identity mismatch")
    controller.model.load_state_dict(value["model"], strict=True)
    optimizer.load_state_dict(value["optimizer"])
    controller.bind_optimizer(optimizer)
    torch.set_rng_state(value["rng"]["torch"])
    random.setstate(value["rng"]["python"])
    if value["rng"]["cuda"]:
        torch.cuda.set_rng_state_all(value["rng"]["cuda"])
    return value
