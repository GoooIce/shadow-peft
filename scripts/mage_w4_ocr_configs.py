#!/usr/bin/env python3
"""Freeze both generic OCR configs before isolated loading; never read an image."""

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

from shadow_lab.mage_w4_search.ocr import EngineIdentity, OCRConfig
from shadow_lab.mage_w4_search.scorers import SubprocessOCRAdapter, adapter_source_sha256


def _write(path, payload):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def prepare_configs(model_root, output, *, python_executable, timeout_seconds=300):
    """A C++ crash in either engine leaves both configs and a failure receipt."""
    if (
        type(timeout_seconds) not in {int, float}
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout must be finite and positive")
    model_root, output = Path(model_root), Path(output)
    inventory = json.loads((model_root / "inventory.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    configs, records = [], []
    versions = {
        name.lower().replace("_", "-"): version for name, version in inventory["runtime"].items()
    }
    for engine, package, prefix, dirs, init, predict in (
        (
            "PaddleOCR",
            "paddleocr",
            "paddle/",
            {
                "detector": "paddle/PP-OCRv5_server_det_infer",
                "recognizer": "paddle/PP-OCRv5_server_rec_infer",
            },
            {"cpu_threads": 8, "enable_mkldnn": False},
            {},
        ),
        (
            "EasyOCR",
            "easyocr",
            "easy/",
            {"reader": "easy"},
            {"languages": ["ch_sim", "en"]},
            {"decoder": "greedy", "batch_size": 1, "workers": 0, "detail": 1, "paragraph": False},
        ),
    ):
        config = OCRConfig(
            EngineIdentity(
                engine,
                versions[package],
                tuple(
                    sorted(
                        (name, value)
                        for name, value in inventory["files"].items()
                        if name.startswith(prefix)
                    )
                ),
            ),
            settings_json=json.dumps(
                {
                    "runtime_versions": inventory["runtime"],
                    "device": "cpu",
                    "model_dirs": dirs,
                    "init_options": init,
                    "predict_options": predict,
                    "adapter_source_sha256": adapter_source_sha256(),
                    "worker_timeout_seconds": timeout_seconds,
                },
                sort_keys=True,
            ),
        )
        record = {
            "schema": "mage-w4-ocr-config-preparation-v2",
            "config": asdict(config),
            "config_sha256": config.sha256,
            "status": "prepared",
            "device_occupied": False,
            "images_scored": 0,
            "initialization_kind": "isolated-process-inherited-group",
            "ocr_python_executable": str(Path(python_executable).absolute()),
        }
        records.append(json.loads(json.dumps(record)))
        configs.append(config)
    # Both alternatives are frozen before either potentially crashes in C++.
    for record in records:
        _write(output / f"{record['config']['identity']['engine']}.json", record)
    _write(output / "configs.json", records)
    for config, record in zip(configs, records, strict=True):
        start, cpu, adapter = time.monotonic(), time.process_time(), None
        try:
            adapter = SubprocessOCRAdapter(config, model_root, python_executable=python_executable)
            record["ocr_process"] = adapter.initialize()
            record["status"] = "loaded"
        except Exception as error:
            record.update(
                status="load_failed",
                error=f"{type(error).__name__}: {error}",
                ocr_process=adapter.last_receipt if adapter is not None else None,
            )
        record.update(
            wall_seconds=time.monotonic() - start, coordinator_cpu_seconds=time.process_time() - cpu
        )
        _write(output / f"{config.identity.engine}.json", record)
        _write(output / "configs.json", records)
        print(
            json.dumps(
                {key: value for key, value in record.items() if key != "config"}, ensure_ascii=False
            ),
            flush=True,
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ocr-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout-seconds", type=float, default=300)
    args = parser.parse_args()
    prepare_configs(
        args.model_root,
        args.output,
        python_executable=args.ocr_python,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    main()
