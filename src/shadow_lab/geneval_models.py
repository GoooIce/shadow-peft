"""GB10-compatible detector and color backend for held-out GenEval diagnostics."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, replace
from pathlib import Path

from .geneval import Detection, evaluate_metadata, filter_detections
from .image_eval import RewardInput

COLORS = (
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
)

# The ModelScope Transformers mirror uses older COCO/VOC spellings while the
# official GenEval metadata follows MMDetection's current COCO class names.
COCO_LABEL_ALIASES = {
    "motorbike": "motorcycle",
    "aeroplane": "airplane",
    "sofa": "couch",
    "pottedplant": "potted plant",
    "diningtable": "dining table",
    "tvmonitor": "tv",
}


@dataclass(frozen=True)
class _Segment:
    detection: Detection
    mask: object


class _OpenCLIPColorClassifier:
    def __init__(self, checkpoint: Path, device: str):
        import open_clip
        import torch

        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14",
            pretrained=str(checkpoint),
            precision="amp",
            device=device,
        )
        self._torch = torch
        self._device = torch.device(device)
        self._model = model.eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer("ViT-H-14")
        self._text_cache = {}

    def _classifier(self, class_name: str):
        torch = self._torch
        cached = self._text_cache.get(class_name)
        if cached is not None:
            return cached
        prompts = [
            template.format(color=color, class_name=class_name)
            for color in COLORS
            for template in (
                "a photo of a {color} {class_name}",
                "a photo of a {color}-colored {class_name}",
                "a photo of a {color} object",
            )
        ]
        tokens = self._tokenizer(prompts).to(self._device)
        with torch.inference_mode(), torch.autocast(
            device_type=self._device.type,
            dtype=torch.bfloat16,
            enabled=self._device.type == "cuda",
        ):
            features = self._model.encode_text(tokens, normalize=True)
            features = features.reshape(len(COLORS), 3, -1).mean(dim=1)
            features = features / features.norm(dim=-1, keepdim=True)
        self._text_cache[class_name] = features
        return features

    def classify(self, image, class_name: str, segments: list[_Segment]) -> list[str]:
        import numpy as np
        from PIL import Image

        torch = self._torch
        crops = []
        for segment in segments:
            mask = Image.fromarray(np.asarray(segment.mask, dtype=np.uint8) * 255)
            blank = Image.new("RGB", image.size, color="#999")
            composite = Image.composite(image, blank, mask)
            x1, y1, x2, y2 = segment.detection.bbox
            crops.append(composite.crop((x1, y1, x2, y2)))
        batch = torch.stack([self._preprocess(crop) for crop in crops]).to(self._device)
        classifier = self._classifier(class_name)
        with torch.inference_mode(), torch.autocast(
            device_type=self._device.type,
            dtype=torch.bfloat16,
            enabled=self._device.type == "cuda",
        ):
            features = self._model.encode_image(batch, normalize=True)
            indices = (features @ classifier.T).argmax(dim=-1)
        return [COLORS[index] for index in indices.cpu().tolist()]


class TransformersGenEvalEvaluator:
    """Port GenEval decisions to a local Transformers Mask2Former backend.

    This is intentionally named a compatible evaluator: the current SM121
    runtime and ModelScope checkpoint differ from the legacy MMDetection 2.x
    environment used by the paper, so its scores must not be compared to the
    published leaderboard.
    """

    def __init__(
        self,
        detector_path: str | Path,
        clip_checkpoint: str | Path,
        *,
        device: str = "cuda",
        batch_size: int = 4,
    ):
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        detector_path = Path(detector_path).expanduser().resolve()
        if not (detector_path / "model.safetensors").is_file():
            raise FileNotFoundError(detector_path / "model.safetensors")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._torch = torch
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._processor = AutoImageProcessor.from_pretrained(
            detector_path,
            local_files_only=True,
        )
        model, loading_info = Mask2FormerForUniversalSegmentation.from_pretrained(
            detector_path,
            local_files_only=True,
            output_loading_info=True,
        )
        missing = set(loading_info["missing_keys"])
        unexpected = set(loading_info["unexpected_keys"])
        allowed_missing = {
            "model.pixel_level_module.encoder.swin.layernorm.weight",
            "model.pixel_level_module.encoder.swin.layernorm.bias",
        }
        if missing - allowed_missing or any(
            not key.endswith("attention.self.relative_position_index")
            for key in unexpected
        ):
            raise RuntimeError(
                "unexpected Mask2Former checkpoint conversion gaps: "
                f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
            )
        self.loading_info = {
            "missing_keys": sorted(missing),
            "unexpected_keys": sorted(unexpected),
            "mismatched_keys": sorted(loading_info["mismatched_keys"]),
        }
        self._model = model.to(self._device).eval()
        self._id2label = {
            int(index): COCO_LABEL_ALIASES.get(label, label)
            for index, label in model.config.id2label.items()
        }
        self._color = _OpenCLIPColorClassifier(
            Path(clip_checkpoint),
            device,
        )

    @staticmethod
    def _images(inputs: list[RewardInput]):
        from PIL import Image

        return [Image.open(io.BytesIO(item.image_bytes)).convert("RGB") for item in inputs]

    def _segments(self, result) -> list[_Segment]:
        import numpy as np

        segmentation = result["segmentation"].detach().cpu().numpy()
        segments = []
        for info in result["segments_info"]:
            mask = segmentation == int(info["id"])
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            label = self._id2label[int(info["label_id"])]
            detection = Detection(
                class_name=label,
                score=float(info["score"]),
                bbox=(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
            )
            segments.append(_Segment(detection, mask))
        return segments

    def _evaluate_one(self, item: RewardInput, image, result) -> dict:
        metadata = item.sample.metadata
        tag = metadata.get("tag")
        if item.sample.source != "geneval" or tag not in {
            "single_object",
            "two_object",
            "counting",
            "colors",
            "position",
            "color_attr",
        }:
            raise ValueError(f"sample {item.sample.sample_id} is not a GenEval sample")
        raw_segments = self._segments(result)
        mask_by_identity = {id(segment.detection): segment.mask for segment in raw_segments}
        objects = filter_detections(
            [segment.detection for segment in raw_segments],
            tag=tag,
        )
        needed_colors: dict[str, int] = {}
        for requirement in metadata.get("include", []):
            if "color" in requirement:
                class_name = requirement["class"]
                needed_colors[class_name] = max(
                    needed_colors.get(class_name, 0),
                    int(requirement["count"]),
                )
        for class_name, count in needed_colors.items():
            selected = objects.get(class_name, [])[:count]
            if not selected:
                continue
            segments = [
                _Segment(detection, mask_by_identity[id(detection)])
                for detection in selected
            ]
            colors = self._color.classify(image, class_name, segments)
            replacements = {
                id(detection): replace(detection, color=color)
                for detection, color in zip(selected, colors, strict=True)
            }
            objects[class_name] = [
                replacements.get(id(detection), detection)
                for detection in objects[class_name]
            ]
        decision = evaluate_metadata(metadata, objects)
        protocol = metadata.get("_geneval_protocol", {})
        return {
            "sample_id": item.sample.sample_id,
            "cluster_id": protocol.get("parent_sample_id", item.sample.sample_id),
            "source_id": item.sample.source_id,
            "seed": item.sample.seed,
            "tag": tag,
            "correct": decision.correct,
            "reasons": list(decision.reasons),
            "detections": {
                class_name: [
                    {
                        "score": detection.score,
                        "bbox": list(detection.bbox),
                        "color": detection.color,
                    }
                    for detection in detections
                ]
                for class_name, detections in sorted(objects.items())
            },
        }

    def evaluate_batch(self, inputs: list[RewardInput]) -> list[dict]:
        torch = self._torch
        records = []
        for start in range(0, len(inputs), self._batch_size):
            batch_inputs = inputs[start : start + self._batch_size]
            images = self._images(batch_inputs)
            encoded = self._processor(images=images, return_tensors="pt")
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with torch.inference_mode():
                outputs = self._model(**encoded)
            results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=0.0,
                target_sizes=[(image.height, image.width) for image in images],
            )
            records.extend(
                self._evaluate_one(item, image, result)
                for item, image, result in zip(batch_inputs, images, results, strict=True)
            )
        if not all(math.isfinite(item["score"]) for record in records for values in record["detections"].values() for item in values):
            raise RuntimeError("non-finite GenEval detector score")
        return records
