#!/usr/bin/env python3
"""Native-kernel evidence for an **INT8 W8A8** checkpoint served by vLLM.

Ticket 10's triple criterion (startup log / explicit ``--linear-backend``
contrast / profiler) was designed for NVFP4, where the startup log names a
kernel.  For INT8 that criterion is *not sufficient*: vLLM logs
``Selected <Kernel> for <Scheme>`` before anything runs, and on SM121 the
selected CUTLASS kernel then fails at the first forward.  A log-only check is
therefore a false positive, and this script replaces the first fold with a
*witness* fold: the engine either serves a request or it does not.

What it produces, per requested backend:

  fold 0  static: which INT8 kernel classes survive ``--linear-backend=<b>``
          (computed in-process from vLLM's own registry, no GPU work);
  fold 1  witness: did the server become healthy and answer a real completion?
  fold 2  startup-log lines (``Selected ... for ...``) plus any hard error the
          engine raised, with the exact text;
  fold 3  profiler: CUDA kernel names + duration shares for the served model;
  fold 4  attribution: the Triton cache entry for the kernel this run actually
          compiled, checked for an INT8 tensor-core MMA instruction and for the
          target architecture.  This is what separates "ran a real int8 GEMM"
          from "ran something that happened to be named scaled_mm_kernel".

Usage:
    python int8_kernel_evidence.py --model <payload> --out <dir> \
        [--backends auto cutlass triton marlin] [--profile-backends triton]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

SELECTED_LINE = re.compile(r"Selected (\S+) for (\S+)")
KERNEL_LINE = re.compile(r"Using (\S+) for NVFP4 GEMM")
ERROR_LINES = (
    "Int8 not supported",
    "Failed to find a kernel that can implement",
    "was requested but no",
    "RuntimeError",
    "ValueError",
    "AssertionError",
)
INT8_MMA = re.compile(r"mma\.sync\.aligned\.[a-z0-9_.]*s8\.s8\.s32")
TARGET = re.compile(r"\.target\s+(\S+)")
TT_DOT = re.compile(r"tt\.dot[^\"]{0,160}")


def _post(url: str, payload: dict | None, timeout: float = 180.0):
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


FATAL_MARKERS = (
    "Engine core initialization failed",
    "EngineCore failed to start",
    "Int8 not supported",
    "Failed to find a kernel",
    "was requested, but no",
)


def _wait_ready_or_dead(port: int, deadline: float, process, log_path: Path) -> dict:
    """Wait for /health, but stop as soon as the engine is provably dead.

    A refused backend never reaches /health, so a plain health poll would burn
    the whole timeout; watching the log for the refusal lets the sweep stay
    short.  `process` gives the same answer when the engine core dies with a
    non-zero exit instead of a logged traceback.
    """
    state = {"ready": False, "died": False, "fatal": None}
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5):
                state["ready"] = True
                return state
        except Exception:
            pass
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
        for marker in FATAL_MARKERS:
            if marker in text:
                # give the failing engine a moment to flush its traceback
                time.sleep(6)
                state["fatal"] = marker
                state["died"] = True
                return state
        if process.poll() is not None:
            state["died"] = True
            state["fatal"] = f"process exited with code {process.returncode}"
            return state
        time.sleep(3)
    return state


def _load_trace(path: Path) -> list[dict]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            trace = json.load(handle)
    else:
        trace = json.loads(path.read_text())
    events = trace.get("traceEvents", trace)
    return [
        {"name": event["name"], "dur": float(event.get("dur", 0.0))}
        for event in events
        if event.get("cat") == "kernel" and "dur" in event
    ]


def _classify(events: list[dict]) -> dict:
    """INT8-oriented classification. No name-only int8 claim is made here."""
    total = sum(event["dur"] for event in events) or 1.0
    classes: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        lowered = event["name"].lower()
        if "dequant" in lowered:
            bucket = "dequantization"
        elif "scaled_mm_kernel" in lowered or "int8" in lowered or "igemm" in lowered:
            bucket = "int8_candidate"
        elif "wmma" in lowered or "gemm" in lowered or "cutlass" in lowered:
            bucket = "other_gemm"
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
            "dur_us": round(sum(v["dur_us"] for v in per_name.values()), 1),
            "dur_share": round(
                sum(v["dur_us"] for v in per_name.values()) / total, 4
            ),
            "kernels": dict(sorted(per_name.items(), key=lambda kv: -kv[1]["dur_us"])),
        }
    return summary


def _static_matrix() -> dict:
    """Which INT8 kernel class does each legal backend select? (no GPU work)"""
    import vllm
    from vllm.config.kernel import LinearBackend

    from vllm.model_executor.kernels.linear import (
        _filter_kernels_by_backend,
        _POSSIBLE_INT8_KERNELS,
    )
    from vllm.platforms import current_platform

    possible = _POSSIBLE_INT8_KERNELS.get(current_platform._enum, [])
    legal = list(getattr(LinearBackend, "__args__", []))
    table = {}
    for backend in legal:
        try:
            survivors = _filter_kernels_by_backend(backend, possible)
        except Exception as error:  # pragma: no cover
            table[backend] = {"error": f"{type(error).__name__}: {error}"}
            continue
        table[backend] = {
            "int8_kernels": [k.__name__ for k in survivors],
            "support": {
                k.__name__: (
                    (lambda r: r if isinstance(r, tuple) else (r, None))(
                        k.is_supported()
                    )
                )
                for k in survivors
            },
        }
    return {
        "vllm": vllm.__version__,
        "possible_int8_kernels_cuda": [k.__name__ for k in possible],
        "legal_linear_backends": legal,
        "per_backend": table,
    }


def _ptx_attribution(cache_dir: Path) -> dict:
    """Scan a Triton cache for the scaled_mm kernel compiled during this run."""
    found = []
    for ptx in cache_dir.rglob("*.ptx"):
        try:
            text = ptx.read_text(errors="replace")
        except Exception:
            continue
        if "scaled_mm" not in text:
            continue
        mma = sorted(set(INT8_MMA.findall(text)))
        target = TARGET.search(text)
        found.append(
            {
                "ptx": str(ptx),
                "mma_instructions": mma,
                "target": target.group(1) if target else None,
                "is_int8_tensor_core": bool(mma),
            }
        )
    return {
        "cache_dir": str(cache_dir),
        "scaled_mm_ptx_files": len(found),
        "int8_tensor_core_files": sum(1 for f in found if f["is_int8_tensor_core"]),
        "entries": found[:12],
    }


def run_backend(args, backend: str, out_dir: Path) -> dict:
    label = f"int8-{backend}"
    log_path = out_dir / f"serve_{label}.log"
    profiler_dir = out_dir / f"profiler_{label}"
    profiler_dir.mkdir(parents=True, exist_ok=True)
    triton_cache = out_dir / f"triton_cache_{backend}"
    triton_cache.mkdir(parents=True, exist_ok=True)
    port = args.port + abs(hash(backend)) % 200

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
        str(port),
        "--disable-log-stats",
        "--quantization",
        "compressed-tensors",
        "--linear-backend",
        backend,
        "--profiler-config.profiler",
        "torch",
        "--profiler-config.torch_profiler_dir",
        str(profiler_dir),
    ]

    environment = dict(os.environ)
    environment["VLLM_LOGGING_LEVEL"] = "INFO"
    environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    venv_bin = str(Path(sys.executable).parent)
    environment["PATH"] = venv_bin + os.pathsep + environment.get("PATH", "")
    headers = Path.home() / ".cache/minicpm5/python-dev/usr/include"
    if (headers / "python3.12").is_dir():
        environment["CPATH"] = f"{headers}/python3.12:{headers}" + (
            os.pathsep + environment["CPATH"] if environment.get("CPATH") else ""
        )
    environment["TRITON_CACHE_DIR"] = str(triton_cache)

    report: dict = {
        "label": label,
        "backend": backend,
        "model": args.model,
        "command": command,
        "env_note": {"TRITON_CACHE_DIR": str(triton_cache)},
        "ready": False,
        "witness": None,
        "startup_selected_lines": [],
        "nvfp4_gemm_lines": [],
        "error_lines": [],
        "log_tail": None,
        "profiler_trace": None,
        "kernel_classes": None,
        "ptx_attribution": None,
    }

    print(f"[evidence] launching {label}: {' '.join(command)}", flush=True)
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command, stdout=log_file, stderr=subprocess.STDOUT, env=environment,
            start_new_session=True,
        )
        try:
            deadline = time.time() + args.wait_seconds
            started = time.time()
            state = _wait_ready_or_dead(port, deadline, process, log_path)
            report["ready"] = state["ready"]
            report["startup_seconds"] = round(time.time() - started, 1)
            report["engine_died_early"] = state["died"]
            report["engine_fatal_marker"] = state["fatal"]
            if report["ready"]:
                print(f"[evidence] {label} healthy", flush=True)
                prompt = "The quick brown fox " * 400
                response = _post(
                    f"http://127.0.0.1:{port}/v1/completions",
                    {"model": "minicpm5", "prompt": prompt, "max_tokens": 16,
                     "temperature": 0.0},
                )
                report["witness"] = {
                    "request": "POST /v1/completions (1600-token prompt, 16 new tokens)",
                    "text_head": (response["choices"][0]["text"] or "")[:120],
                    "usage": response.get("usage"),
                }
                print(f"[evidence] {label} witness ok: {report['witness']['text_head']!r}",
                      flush=True)
                if backend in args.profile_backends:
                    _post(f"http://127.0.0.1:{port}/start_profile", None, timeout=60)
                    _post(
                        f"http://127.0.0.1:{port}/v1/completions",
                        {"model": "minicpm5", "prompt": prompt, "max_tokens": 16,
                         "temperature": 0.0},
                    )
                    _post(f"http://127.0.0.1:{port}/stop_profile", None, timeout=300)
                    time.sleep(5)
                    traces = sorted(profiler_dir.rglob("*.json*"),
                                    key=lambda p: p.stat().st_mtime)
                    if traces:
                        trace_path = traces[-1]
                        report["profiler_trace"] = str(trace_path)
                        events = _load_trace(trace_path)
                        report["kernel_event_count"] = len(events)
                        report["kernel_classes"] = _classify(events)
            else:
                print(f"[evidence] {label} never became healthy", flush=True)
        finally:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=90)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            log_file.flush()

    text = log_path.read_text(errors="replace")
    report["startup_selected_lines"] = [
        f"Selected {match[0]} for {match[1]}"
        for match in SELECTED_LINE.findall(text)
    ]
    report["nvfp4_gemm_lines"] = [
        f"Using {match} for NVFP4 GEMM" for match in KERNEL_LINE.findall(text)
    ]
    report["error_lines"] = sorted(
        {
            line.strip()
            for line in text.splitlines()
            if any(marker in line for marker in ERROR_LINES)
        }
    )[:20]
    report["log_tail"] = text[-2500:]
    if report["ready"]:
        report["ptx_attribution"] = _ptx_attribution(triton_cache)
        # the kernels vLLM actually ran, with the int8 claim made explicit
        classes = report["kernel_classes"] or {}
        cand = classes.get("int8_candidate", {})
        report["folds"] = {
            "fold1_witness_served_a_request": bool(report["witness"]),
            "fold2_selected_line": report["startup_selected_lines"][:1],
            "fold2_hard_error": report["error_lines"][:1],
            "fold3_int8_candidate_calls": cand.get("calls", 0),
            "fold3_int8_candidate_dur_share": cand.get("dur_share", 0.0),
            "fold3_dequant_calls": classes.get("dequantization", {}).get("calls", 0),
            "fold4_int8_tensor_core_ptx_files":
                report["ptx_attribution"]["int8_tensor_core_files"],
        }
    else:
        report["folds"] = {
            "fold1_witness_served_a_request": False,
            "fold2_hard_error": report["error_lines"][:1],
        }

    report_path = out_dir / f"kernel_evidence_{label}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[evidence] wrote {report_path}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backends", nargs="*", default=["auto", "cutlass", "triton"])
    parser.add_argument("--profile-backends", nargs="*", default=["triton"])
    parser.add_argument("--port", type=int, default=8140)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--wait-seconds", type=int, default=300)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"model": args.model, "backends": {}}
    print("[evidence] building static INT8 backend matrix", flush=True)
    try:
        summary["static_matrix"] = _static_matrix()
    except Exception as error:
        summary["static_matrix_error"] = f"{type(error).__name__}: {error}"
    if args.static_only:
        (out_dir / "int8_kernel_evidence_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(summary.get("static_matrix", {}), indent=2)[:3000])
        return 0

    for backend in args.backends:
        report = run_backend(args, backend, out_dir)
        summary["backends"][backend] = {
            "ready": report["ready"],
            "witness": report["witness"],
            "startup_selected_lines": report["startup_selected_lines"][:3],
            "error_lines": report["error_lines"][:3],
            "folds": report.get("folds"),
        }

    (out_dir / "int8_kernel_evidence_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(
        {b: {"ready": r["ready"],
             "selected": r["startup_selected_lines"][:1],
             "error": [e[:140] for e in r["error_lines"][:1]],
             "folds": r["folds"]}
         for b, r in summary["backends"].items()},
        indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
