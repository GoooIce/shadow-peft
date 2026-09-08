#!/usr/bin/env python3
"""Download and fingerprint OCR models on a GB10 CPU; perform no image scoring."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import tarfile
import time
import urllib.request
from pathlib import Path


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.machine() != "aarch64" or platform.system() != "Linux":
        parser.error("OCR model preparation belongs on a GB10 host")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    start, cpu = time.monotonic(), time.process_time()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    paddle = root / "paddle"
    paddle.mkdir(exist_ok=True)
    for name in ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"):
        archive = paddle / (name + ".tar")
        if not archive.exists():
            url = f"https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/{name}_infer.tar"
            print(f"download {name}", flush=True)
            urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as stream:
            stream.extractall(paddle, filter="data")
    import easyocr

    easy = root / "easy"
    easyocr.Reader(
        ["ch_sim", "en"],
        gpu=False,
        model_storage_directory=str(easy),
        download_enabled=True,
        verbose=False,
    )
    files = {
        str(p.relative_to(root)): sha(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix not in {".tar", ".zip"} and p.name != "inventory.json"
    }
    payload = {
        "schema": "mage-ocr-model-inventory/v1",
        "root": str(root),
        "files": files,
        "runtime": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "host": platform.node(),
        "host_cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.monotonic() - start,
        "device_occupied": False,
        "images_recognized": 0,
        "human_transcriptions": 0,
        "sources": [
            "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/v3.2.0/docs/version3.x/pipeline_usage/OCR.en.md",
            "https://jaided.ai/easyocr/documentation/",
        ],
    }
    (root / "inventory.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in payload.items() if k not in {"files", "runtime"}}), flush=True
    )


if __name__ == "__main__":
    main()
