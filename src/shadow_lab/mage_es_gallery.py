"""Lossless, opt-in image capture for fixed-center Mage ES evaluations."""

from __future__ import annotations

import hashlib
import io
import os
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


_SAFE_LABEL = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*")


def _png_bytes(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _sample_slug(sample_id: str) -> str:
    readable = re.sub(r"[^a-zA-Z0-9_.-]+", "-", sample_id).strip("-.")[:48]
    digest = hashlib.sha256(sample_id.encode()).hexdigest()[:10]
    return f"{readable or 'sample'}-{digest}"


class CapturingPipeline:
    """Delegate generation and atomically persist the exact PNG scoring inputs."""

    def __init__(self, pipeline: Any, image_root: Path) -> None:
        self._pipeline = pipeline
        self.image_root = image_root.expanduser().resolve()
        self._session: dict[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pipeline, name)

    @contextmanager
    def capture(
        self, label: str, sample_ids: Sequence[str]
    ) -> Iterator[list[dict[str, Any]]]:
        if not _SAFE_LABEL.fullmatch(label):
            raise ValueError(f"invalid capture label: {label}")
        if self._session is not None:
            raise RuntimeError("nested image capture sessions are not supported")
        records: list[dict[str, Any]] = []
        self._session = {
            "label": label,
            "sample_ids": [str(value) for value in sample_ids],
            "records": records,
        }
        try:
            yield records
            captured = len(records)
            expected = len(sample_ids)
            if captured != expected:
                raise RuntimeError(
                    f"capture session expected {expected} images but saved {captured}"
                )
        finally:
            self._session = None

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        images = self._pipeline.generate(*args, **kwargs)
        if self._session is None:
            return images
        records = self._session["records"]
        sample_ids = self._session["sample_ids"]
        label = self._session["label"]
        if len(records) + len(images) > len(sample_ids):
            raise RuntimeError("capture generated more images than its sample manifest")
        destination = self.image_root / label
        destination.mkdir(parents=True, exist_ok=True)
        for image in images:
            index = len(records)
            sample_id = sample_ids[index]
            relative = Path(label) / f"{index:03d}-{_sample_slug(sample_id)}.png"
            path = self.image_root / relative
            payload = _png_bytes(image)
            temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            try:
                temporary.write_bytes(payload)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            records.append(
                {
                    "sample_id": sample_id,
                    "relative_path": relative.as_posix(),
                    "image_sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
            )
        return images
