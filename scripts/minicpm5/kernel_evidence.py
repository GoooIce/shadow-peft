#!/usr/bin/env python3
"""Three-fold native-kernel evidence for an NVFP4 checkpoint served by vLLM.

Follows the evidence protocol fixed by ticket 01 (E7):

1. startup log line ``Using <Kernel> for NVFP4 GEMM``;
2. explicit ``--linear-backend`` contrast (including a W4A16 counter-example);
3. a profiler trace that must contain FP4-input GEMM kernels and must not be
   dominated by dequantisation / BF16 wmma compute.

The script starts ``vllm serve`` itself, waits for readiness, drives requests
and profiles, then writes ``kernel_evidence_<label>.json`` next to the log.

Usage:
    python kernel_evidence.py --model <payload> --out <dir> --label nvfp4-auto \
        [--linear-backend marlin] [--quantization compressed-tensors]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

KERNEL_LINE = re.compile(r"Using (\S+) for NVFP4 GEMM")
NATIVE_MARKERS = ("fp4", "nvfp4", "e2m1", "block_scaled")
W4A16_MARKERS = ("marlin",)
DEQUANT_MARKERS = ("dequant",)
BF16_WMMA_MARKERS = ("wmma", "bf16")


def _post(url: str, payload: dict | None, timeout: float = 120.0):
    data = json.dumps(payload).encode() if payload is not None else b""
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body else None


def _wait_health(port: int, deadline: float) -> bool:
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5):
                return True
        except Exception:
            time.sleep(3)
    return False


def _classify(events: list[dict]) -> dict:
    """Classify CUDA kernels and report duration share per class."""
    total = sum(event["dur"] for event in events) or 1.0
    classes: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        name = event["name"]
        lowered = name.lower()
        if any(marker in lowered for marker in W4A16_MARKERS):
            bucket = "w4a16_marlin"
        elif any(marker in lowered for marker in DEQUANT_MARKERS):
            bucket = "dequantization"
        elif any(marker in lowered for marker in NATIVE_MARKERS):
            bucket = "native_fp4"
        elif any(marker in lowered for marker in BF16_WMMA_MARKERS):
            bucket = "bf16_or_wmma"
        else:
            bucket = "other"
        classes[bucket].append(event)

    summary = {}
    for bucket, items in classes.items():
        per_name: dict[str, dict] = {}
        for item in items:
            entry = per_name.setdefault(item["name"], {"calls": 0, "dur_us": 0.0})
            entry["calls"] += 1
            entry["dur_us"] += item["dur"]
        summary[bucket] = {
            "calls": sum(v["calls"] for v in per_name.values()),
            "dur_us": sum(v["dur_us"] for v in per_name.values()),
            "dur_share": round(
                sum(v["dur_us"] for v in per_name.values()) / total, 4
            ),
            "kernels": dict(
                sorted(per_name.items(), key=lambda kv: -kv[1]["dur_us"])
            ),
        }
    return summary


def _load_trace(path: Path) -> list[dict]:
    import gzip

    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            trace = json.load(fh)
    else:
        trace = json.loads(path.read_text())
    events = trace.get("traceEvents", trace)
    return [
        {"name": event["name"], "dur": float(event.get("dur", 0.0))}
        for event in events
        if event.get("cat") == "kernel" and "dur" in event
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--port", type=int, default=8130)
    parser.add_argument("--linear-backend", default=None)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--wait-seconds", type=int, default=420)
    parser.add_argument("--profile", action="store_true", default=True)
    parser.add_argument("--no-profile", dest="profile", action="store_false")
    parser.add_argument("--keep-server", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"serve_{args.label}.log"
    profiler_dir = out_dir / f"profiler_{args.label}"
    profiler_dir.mkdir(exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--served-model-name",
        "minicpm5",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--port",
        str(args.port),
        "--disable-log-stats",
        # vLLM 0.29.0 attaches the /start_profile and /stop_profile endpoints
        # only when a profiler is configured; the old VLLM_TORCH_PROFILER_DIR
        # env var alone is no longer enough.
        "--profiler-config.profiler",
        "torch",
        "--profiler-config.torch_profiler_dir",
        str(profiler_dir),
    ]
    if args.linear_backend:
        command += ["--linear-backend", args.linear_backend]
    if args.quantization:
        command += ["--quantization", args.quantization]

    environment = dict(os.environ)
    environment["VLLM_TORCH_PROFILER_DIR"] = str(profiler_dir)
    environment["VLLM_LOGGING_LEVEL"] = "INFO"
    # GB10 fixes from ticket 06: ninja lives in the venv bin, and Python.h for
    # the flashinfer/triton JIT lives in the user-space python3.12-dev unpack.
    venv_bin = str(Path(sys.executable).parent)
    environment["PATH"] = venv_bin + os.pathsep + environment.get("PATH", "")
    headers = Path.home() / ".cache/minicpm5/python-dev/usr/include"
    if (headers / "python3.12").is_dir():
        environment["CPATH"] = (
            f"{headers}/python3.12:{headers}"
            + (os.pathsep + environment["CPATH"] if environment.get("CPATH") else "")
        )

    print(f"[evidence] launching: {' '.join(command)}", flush=True)
    log_file = log_path.open("w")
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )

    report: dict = {
        "label": args.label,
        "model": args.model,
        "command": command,
        "linear_backend": args.linear_backend,
        "quantization": args.quantization,
        "gpu": None,
        "capability": None,
        "startup_kernel_line": None,
        "ready": False,
        "profiler_trace": None,
        "kernel_classes": None,
    }
    try:
        import torch

        report["gpu"] = torch.cuda.get_device_name(0)
        report["capability"] = list(torch.cuda.get_device_capability())

        deadline = time.time() + args.wait_seconds
        report["ready"] = _wait_health(args.port, deadline)
        if not report["ready"]:
            report["tail"] = log_path.read_text()[-4000:]
            raise SystemExit(f"server not ready; see {log_path}")

        text = log_path.read_text(errors="replace")
        matches = KERNEL_LINE.findall(text)
        report["startup_kernel_line"] = matches[-1] if matches else None
        report["startup_kernel_all"] = matches
        report["kernel_evidence_fold_1"] = {
            "pattern": KERNEL_LINE.pattern,
            "kernel": report["startup_kernel_line"],
            "pass": bool(matches),
        }
        print(f"[evidence] startup kernel line: {matches}", flush=True)

        # Drive a prefill-heavy request so the GEMM kernels actually run.
        prompt = "The quick brown fox " * 400
        _post(
            f"http://127.0.0.1:{args.port}/v1/completions",
            {
                "model": "minicpm5",
                "prompt": prompt,
                "max_tokens": 16,
                "temperature": 0.0,
            },
        )
        print("[evidence] warm-up request done", flush=True)

        if args.profile:
            _post(f"http://127.0.0.1:{args.port}/start_profile", None, timeout=60)
            _post(
                f"http://127.0.0.1:{args.port}/v1/completions",
                {
                    "model": "minicpm5",
                    "prompt": prompt,
                    "max_tokens": 16,
                    "temperature": 0.0,
                },
            )
            _post(f"http://127.0.0.1:{args.port}/stop_profile", None, timeout=180)
            time.sleep(5)
            traces = sorted(
                profiler_dir.rglob("*.json*"), key=lambda p: p.stat().st_mtime
            )
            if traces:
                trace_path = traces[-1]
                report["profiler_trace"] = str(trace_path)
                events = _load_trace(trace_path)
                report["kernel_event_count"] = len(events)
                report["kernel_classes"] = _classify(events)
                native = report["kernel_classes"].get("native_fp4", {})
                report["kernel_evidence_fold_3"] = {
                    "native_fp4_calls": native.get("calls", 0),
                    "native_fp4_dur_share": native.get("dur_share", 0.0),
                    "dequant_calls": report["kernel_classes"]
                    .get("dequantization", {})
                    .get("calls", 0),
                    "w4a16_marlin_calls": report["kernel_classes"]
                    .get("w4a16_marlin", {})
                    .get("calls", 0),
                    "pass": native.get("calls", 0) > 0
                    and report["kernel_classes"]
                    .get("dequantization", {})
                    .get("calls", 0)
                    == 0,
                }
                print(
                    "[evidence] kernel classes: "
                    + json.dumps(
                        {k: (v["calls"], v["dur_share"])
                         for k, v in report["kernel_classes"].items()}
                    ),
                    flush=True,
                )
    finally:
        if not args.keep_server:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=60)
            except Exception:
                process.kill()
        log_file.close()

    report_path = out_dir / f"kernel_evidence_{args.label}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[evidence] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
