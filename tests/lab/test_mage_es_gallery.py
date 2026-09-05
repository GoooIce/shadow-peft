from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest
from PIL import Image

from shadow_lab.mage_es_gallery import CapturingPipeline


class _Pipeline:
    model = object()

    def generate(self, prompts, **kwargs):
        del kwargs
        return [
            Image.new("RGB", (4, 4), (index, 20, 30)) for index, _ in enumerate(prompts)
        ]


def _png_sha256(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def test_capture_saves_the_exact_generated_pngs(tmp_path: Path) -> None:
    pipeline = CapturingPipeline(_Pipeline(), tmp_path)

    with pipeline.capture("small", ["sample/a", "sample:b"]) as records:
        images = pipeline.generate(["one", "two"], seeds=[1, 2])

    assert pipeline.model is _Pipeline.model
    assert [record["sample_id"] for record in records] == ["sample/a", "sample:b"]
    assert [record["image_sha256"] for record in records] == [
        _png_sha256(image) for image in images
    ]
    assert all((tmp_path / record["relative_path"]).is_file() for record in records)


def test_capture_rejects_unsafe_or_incomplete_sessions(tmp_path: Path) -> None:
    pipeline = CapturingPipeline(_Pipeline(), tmp_path)

    with pytest.raises(ValueError, match="label"):
        with pipeline.capture("../escape", ["sample"]):
            pass

    with pytest.raises(RuntimeError, match="expected 2"):
        with pipeline.capture("initial", ["one", "two"]):
            pipeline.generate(["one"])
