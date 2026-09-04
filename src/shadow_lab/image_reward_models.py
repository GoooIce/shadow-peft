"""Lazy GPU scorers for the Mage image-reward protocol.

All checkpoints are local paths so scoring never triggers a Hugging Face fetch.
The intended assets are the ModelScope HPS v2.1 and LAION OpenCLIP mirrors.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import types
from pathlib import Path

from .image_eval import RewardInput


def _pil_images(inputs):
    from PIL import Image

    return [Image.open(io.BytesIO(item.image_bytes)).convert("RGB") for item in inputs]


def _hps_open_clip_symbols():
    # HPSv2 factory.py imports ``turtle.forward`` but never references that
    # symbol.  Headless GB10 Python intentionally has no Tk, so provide only
    # the unused symbol instead of installing a desktop GUI stack.
    if importlib.util.find_spec("tkinter") is None and "turtle" not in sys.modules:
        turtle_stub = types.ModuleType("turtle")
        turtle_stub.forward = lambda *args, **kwargs: None
        sys.modules["turtle"] = turtle_stub
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

    return create_model_and_transforms, get_tokenizer


class HPSv21Scorer:
    name = "hps_v2_1"

    def __init__(self, checkpoint: str | Path, *, device: str = "cuda", batch_size: int = 8):
        import torch

        create_model_and_transforms, get_tokenizer = _hps_open_clip_symbols()

        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        model, _, preprocess = create_model_and_transforms(
            "ViT-H-14",
            pretrained=None,
            precision="amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state_dict = payload.get("state_dict") if isinstance(payload, dict) else None
        if not isinstance(state_dict, dict):
            raise ValueError("HPS v2.1 checkpoint has no state_dict")
        model.load_state_dict(state_dict, strict=True)
        self._torch = torch
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._model = model.to(self._device).eval()
        self._preprocess = preprocess
        self._tokenizer = get_tokenizer("ViT-H-14")

    def score_batch(self, inputs: list[RewardInput]) -> list[float]:
        torch = self._torch
        images = _pil_images(inputs)
        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(inputs), self._batch_size):
                stop = start + self._batch_size
                image_tensor = torch.stack(
                    [self._preprocess(image) for image in images[start:stop]]
                ).to(self._device)
                text_tensor = self._tokenizer(
                    [item.sample.prompt for item in inputs[start:stop]]
                ).to(self._device)
                with torch.autocast(
                    device_type=self._device.type,
                    dtype=torch.bfloat16,
                    enabled=self._device.type == "cuda",
                ):
                    outputs = self._model(image_tensor, text_tensor)
                    values = (outputs["image_features"] * outputs["text_features"]).sum(-1)
                scores.extend(values.float().cpu().tolist())
        return scores


class OpenCLIPH14Scorer:
    name = "clip_h14"

    def __init__(self, checkpoint: str | Path, *, device: str = "cuda", batch_size: int = 8):
        import open_clip
        import torch

        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14",
            pretrained=str(checkpoint),
            precision="amp",
            device=device,
        )
        self._torch = torch
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._model = model.eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer("ViT-H-14")

    def score_batch(self, inputs: list[RewardInput]) -> list[float]:
        torch = self._torch
        images = _pil_images(inputs)
        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(inputs), self._batch_size):
                stop = start + self._batch_size
                image_tensor = torch.stack(
                    [self._preprocess(image) for image in images[start:stop]]
                ).to(self._device)
                text_tensor = self._tokenizer(
                    [item.sample.prompt for item in inputs[start:stop]]
                ).to(self._device)
                with torch.autocast(
                    device_type=self._device.type,
                    dtype=torch.bfloat16,
                    enabled=self._device.type == "cuda",
                ):
                    image_features = self._model.encode_image(image_tensor, normalize=True)
                    text_features = self._model.encode_text(text_tensor, normalize=True)
                    values = (image_features * text_features).sum(-1)
                scores.extend(values.float().cpu().tolist())
        return scores
