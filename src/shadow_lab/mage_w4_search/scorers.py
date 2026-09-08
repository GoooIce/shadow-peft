"""Frozen external model adapters; OCR receives pixels and generic config only."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

from .contracts import digest_json
from .ocr import EngineIdentity, GeometryPolicy, OCRConfig, RawSegment

RESPONSE_PREFIX = "MAGE_W4_OCR_RESULT_V1 "
REQUEST_SCHEMA = "mage-w4-ocr-request-v1"
RESPONSE_SCHEMA = "mage-w4-ocr-response-v1"


class OCRProcessError(ValueError):
    """An isolated OCR process failed; it supplied no valid recognition."""


def adapter_source_sha256() -> str:
    """Bind adapter, matching and geometry implementation with stable file labels."""
    directory = Path(__file__).parent
    return digest_json(
        {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("scorers.py", "ocr.py")
        }
    )


def decode_ocr_config(raw: dict) -> OCRConfig:
    """Decode the exact public config schema without accepting answer fields."""
    try:
        if not isinstance(raw, dict) or set(raw) != {field.name for field in fields(OCRConfig)}:
            raise ValueError("invalid OCR config schema")
        identity = dict(raw["identity"])
        if set(identity) != {field.name for field in fields(EngineIdentity)}:
            raise ValueError("invalid OCR identity schema")
        identity["model_sha256"] = tuple(tuple(row) for row in identity["model_sha256"])
        geometry = raw["geometry"]
        if not isinstance(geometry, dict) or set(geometry) != {
            field.name for field in fields(GeometryPolicy)
        }:
            raise ValueError("invalid OCR geometry schema")
        return OCRConfig(
            **{
                **raw,
                "identity": EngineIdentity(**identity),
                "geometry": GeometryPolicy(**geometry),
            }
        )
    except (TypeError, KeyError, AttributeError) as error:
        raise ValueError("invalid OCR config schema") from error


def _settings(config):
    settings = json.loads(config.settings_json)
    required = {
        "runtime_versions",
        "device",
        "model_dirs",
        "init_options",
        "predict_options",
        "adapter_source_sha256",
        "worker_timeout_seconds",
    }
    if set(settings) != required or settings["device"] != "cpu":
        raise ValueError("unsupported frozen OCR adapter settings (pilot CPU only)")
    if settings["adapter_source_sha256"] != adapter_source_sha256():
        raise ValueError("OCR adapter implementation differs from frozen source identity")
    timeout = settings["worker_timeout_seconds"]
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("OCR worker timeout must be finite and positive")
    if any(
        not isinstance(settings[key], dict)
        for key in ("runtime_versions", "model_dirs", "init_options", "predict_options")
    ):
        raise ValueError("invalid frozen OCR adapter settings")
    return settings


class SubprocessOCRAdapter:
    """One image per isolated Python; inherit the supervisor's process group.

    Per-call latency includes model initialization. No prompt or expected text
    crosses this boundary, and the parent never imports Paddle/Torch OCR models.
    """

    def __init__(self, config, model_root, *, python_executable):
        self.config = config
        self.model_root = Path(model_root).resolve(strict=True)
        self.python_executable = Path(python_executable).absolute()
        if not self.python_executable.is_file() or not os.access(self.python_executable, os.X_OK):
            raise ValueError("OCR Python executable must exist and be executable")
        self.settings = _settings(config)
        self.last_receipt = None

    def initialize(self):
        self._invoke("initialize", None)
        return self.last_receipt

    def __call__(self, image_bytes, config):
        if config.sha256 != self.config.sha256:
            raise ValueError("per-image OCR configuration changed")
        if not isinstance(image_bytes, bytes) or not image_bytes:
            raise ValueError("OCR requires nonempty image bytes")
        return self._invoke("recognize", image_bytes)

    def _invoke(self, action, image_bytes):
        _settings(self.config)  # A source edit after construction invalidates the lock.
        request = {
            "schema": REQUEST_SCHEMA,
            "action": action,
            "config": asdict(self.config),
            "config_sha256": self.config.sha256,
            "image_sha256": hashlib.sha256(image_bytes).hexdigest()
            if image_bytes is not None
            else None,
            "image_base64": base64.b64encode(image_bytes).decode("ascii")
            if image_bytes is not None
            else None,
        }
        fingerprint = digest_json(request)
        started = time.monotonic()
        stdout, stderr, returncode = "", "", None
        try:
            child = subprocess.run(
                [
                    str(self.python_executable),
                    "-m",
                    "shadow_lab.mage_w4_search.scorers",
                    "--worker",
                    "--model-root",
                    str(self.model_root),
                ],
                input=json.dumps(request, ensure_ascii=False, allow_nan=False),
                text=True,
                capture_output=True,
                timeout=self.settings["worker_timeout_seconds"],
                check=False,
                start_new_session=False,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            )
            stdout, stderr, returncode = child.stdout, child.stderr, child.returncode
            if returncode != 0:
                raise OCRProcessError(f"OCR child exited with return code {returncode}")
            lines = [
                line[len(RESPONSE_PREFIX) :]
                for line in stdout.splitlines()
                if line.startswith(RESPONSE_PREFIX)
            ]
            if len(lines) != 1:
                raise OCRProcessError("OCR child must return exactly one prefixed response")
            response = json.loads(lines[0])
            required = {
                "schema",
                "action",
                "request_sha256",
                "config_sha256",
                "image_sha256",
                "status",
                "segments",
                "worker",
                "timing",
                "error",
            }
            if not isinstance(response, dict) or set(response) != required:
                raise OCRProcessError("invalid OCR response schema")
            if (
                response["schema"] != RESPONSE_SCHEMA
                or response["action"] != action
                or response["request_sha256"] != fingerprint
                or response["config_sha256"] != self.config.sha256
                or response["image_sha256"] != request["image_sha256"]
            ):
                raise OCRProcessError("OCR response identity mismatch")
            worker, timing = response["worker"], response["timing"]
            if (
                not isinstance(worker, dict)
                or set(worker) != {"pid", "process_group", "adapter_source_sha256"}
                or worker["adapter_source_sha256"] != self.settings["adapter_source_sha256"]
                or worker["process_group"] != os.getpgrp()
                or type(worker["pid"]) is not int
                or worker["pid"] <= 0
                or worker["pid"] == os.getpid()
            ):
                raise OCRProcessError("OCR child source or process-group identity mismatch")
            if (
                not isinstance(timing, dict)
                or set(timing) != {"wall_seconds", "host_cpu_seconds"}
                or any(
                    type(value) not in {int, float} or not math.isfinite(value) or value < 0
                    for value in timing.values()
                )
            ):
                raise OCRProcessError("invalid OCR child timing receipt")
            if response["status"] != "ok" or response["error"] is not None:
                raise OCRProcessError("OCR child did not complete successfully")
            if not isinstance(response["segments"], list) or (
                action == "initialize" and response["segments"]
            ):
                raise OCRProcessError("invalid OCR response segments")
            segments = tuple(
                RawSegment(**{**row, "box": tuple(row["box"])}) for row in response["segments"]
            )
            self.last_receipt = {
                **response,
                "parent_wall_seconds": time.monotonic() - started,
                "returncode": returncode,
                "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
            }
            return segments
        except (subprocess.TimeoutExpired, OSError, ValueError, TypeError, KeyError) as error:
            if isinstance(error, subprocess.TimeoutExpired):
                stdout = _log_text(error.stdout)
                stderr = _log_text(error.stderr)
            self.last_receipt = {
                "status": "failed",
                "action": action,
                "request_sha256": fingerprint,
                "config_sha256": self.config.sha256,
                "image_sha256": request["image_sha256"],
                "returncode": returncode,
                "parent_wall_seconds": time.monotonic() - started,
                "timed_out": isinstance(error, subprocess.TimeoutExpired),
                "error": f"{type(error).__name__}: {error}",
                "stdout_tail": stdout[-2000:],
                "stderr_tail": stderr[-2000:],
            }
            raise OCRProcessError(self.last_receipt["error"]) from error


def _log_text(value):
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""


def ocr_segments(engine: str, result) -> tuple[RawSegment, ...]:
    if engine == "PaddleOCR":
        boxes, texts, scores = (result[key] for key in ("rec_polys", "rec_texts", "rec_scores"))
        if not len(boxes) == len(texts) == len(scores):
            raise ValueError("incomplete PaddleOCR output")
        entries = zip(boxes, texts, scores, strict=True)
    elif engine == "EasyOCR":
        entries = result
    else:
        raise ValueError("unsupported OCR engine")
    segments = []
    for quad, text, confidence in entries:
        if len(quad) != 4 or any(len(point) != 2 for point in quad):
            raise ValueError("OCR output requires four-corner boxes")
        points = [(float(x), float(y)) for x, y in quad]
        xs, ys = zip(*points, strict=True)
        angle = math.degrees(math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0]))
        segments.append(
            RawSegment(text, (min(xs), min(ys), max(xs), max(ys)), float(confidence), angle)
        )
    return tuple(segments)


def verify_ocr_assets(config, model_root) -> dict[str, Path]:
    root = Path(model_root).resolve(strict=True)
    paths = {}
    for name, expected in config.identity.model_sha256:
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("OCR asset escapes frozen root")
        actual = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                actual.update(block)
        if actual.hexdigest() != expected:
            raise ValueError(f"OCR model hash mismatch: {name}")
        paths[name] = path
    return paths


class FrozenOCRAdapter:
    """Load one frozen OCR tool once; per-image calls cannot read target text."""

    def __init__(self, config, model_root):
        self.config = config
        root = Path(model_root).resolve(strict=True)
        paths = verify_ocr_assets(config, root)
        package = {"PaddleOCR": "paddleocr", "EasyOCR": "easyocr"}.get(config.identity.engine)
        if (
            package is None
            or importlib.metadata.version(package) != config.identity.package_version
        ):
            raise ValueError("OCR package version differs from registered identity")
        settings = _settings(config)
        for name, version in settings["runtime_versions"].items():
            if importlib.metadata.version(name) != version:
                raise ValueError(f"OCR runtime version mismatch: {name}")
        init, predict = settings["init_options"], settings["predict_options"]
        if any(
            key in predict or key in init
            for key in ("allowlist", "blocklist", "character_dict_path")
        ):
            raise ValueError("answer-constraining OCR options are prohibited")
        self.predict_options = predict

        def directory(key):
            path = (root / settings["model_dirs"][key]).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_dir():
                raise ValueError("OCR model directory must be inside frozen root")
            all_files = {p.resolve() for p in path.rglob("*") if p.is_file()}
            if not all_files or not all_files <= set(paths.values()):
                raise ValueError("OCR model directory contains unregistered files")
            return str(path)

        if package == "easyocr":
            import easyocr

            if set(init) != {"languages"} or set(predict) != {
                "decoder",
                "batch_size",
                "workers",
                "detail",
                "paragraph",
            }:
                raise ValueError("unsupported EasyOCR frozen configuration")
            if predict["detail"] != 1 or predict["paragraph"] is not False:
                raise ValueError("OCR must return individual boxes and text")
            self.model = easyocr.Reader(
                init["languages"],
                gpu=False,
                model_storage_directory=directory("reader"),
                download_enabled=False,
                verbose=False,
            )
        else:
            # Paddle must initialize before PaddleOCR (and before any Torch import)
            # in the dedicated OCR process to avoid the observed C++ import crash.
            importlib.import_module("paddle")
            from paddleocr import PaddleOCR

            if set(init) != {"cpu_threads", "enable_mkldnn"} or set(predict):
                raise ValueError("unsupported PaddleOCR frozen configuration")
            self.model = PaddleOCR(
                text_detection_model_name="PP-OCRv5_server_det",
                text_recognition_model_name="PP-OCRv5_server_rec",
                text_detection_model_dir=directory("detector"),
                text_recognition_model_dir=directory("recognizer"),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="cpu",
                **init,
            )

    def __call__(self, image_bytes, config):
        if config.sha256 != self.config.sha256:
            raise ValueError("per-image OCR configuration changed")
        if config.identity.engine == "EasyOCR":
            return ocr_segments("EasyOCR", self.model.readtext(image_bytes, **self.predict_options))
        import numpy as np
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as image:
            bgr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
        results = list(self.model.predict(bgr))
        if len(results) != 1:
            raise ValueError("PaddleOCR did not return exactly one image")
        return ocr_segments("PaddleOCR", results[0])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Isolated target-blind OCR worker")
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    wall, cpu = time.monotonic(), time.process_time()
    request, fingerprint, config = {}, None, None
    response = {"schema": RESPONSE_SCHEMA, "status": "failed", "segments": [], "error": None}
    code = 1
    try:
        request = json.loads(sys.stdin.read())
        required = {"schema", "action", "config", "config_sha256", "image_sha256", "image_base64"}
        if (
            not isinstance(request, dict)
            or set(request) != required
            or request["schema"] != REQUEST_SCHEMA
            or request["action"] not in {"initialize", "recognize"}
        ):
            raise ValueError("invalid OCR request schema")
        fingerprint = digest_json(request)
        config = decode_ocr_config(request["config"])
        if config.sha256 != request["config_sha256"]:
            raise ValueError("OCR request config identity mismatch")
        image_bytes = None
        if request["action"] == "initialize":
            if request["image_sha256"] is not None or request["image_base64"] is not None:
                raise ValueError("initialization must not receive an image")
        else:
            if not isinstance(request["image_base64"], str):
                raise ValueError("missing base64 image bytes")
            image_bytes = base64.b64decode(request["image_base64"], validate=True)
            if (
                not image_bytes
                or hashlib.sha256(image_bytes).hexdigest() != request["image_sha256"]
            ):
                raise ValueError("OCR request image identity mismatch")
        model = FrozenOCRAdapter(config, args.model_root)
        result = model(image_bytes, config) if request["action"] == "recognize" else ()
        response.update(status="ok", segments=[asdict(segment) for segment in result])
        code = 0
    except Exception as error:
        response["error"] = f"{type(error).__name__}: {error}"
    response.update(
        action=request.get("action") if isinstance(request, dict) else None,
        request_sha256=fingerprint,
        config_sha256=config.sha256 if config is not None else None,
        image_sha256=request.get("image_sha256") if isinstance(request, dict) else None,
        worker={
            "pid": os.getpid(),
            "process_group": os.getpgrp(),
            "adapter_source_sha256": adapter_source_sha256(),
        },
        timing={
            "wall_seconds": time.monotonic() - wall,
            "host_cpu_seconds": time.process_time() - cpu,
        },
    )
    print(
        "\n" + RESPONSE_PREFIX + json.dumps(response, ensure_ascii=False, allow_nan=False),
        flush=True,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
