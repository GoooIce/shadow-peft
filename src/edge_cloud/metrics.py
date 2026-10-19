"""Thread-safe counters and latency quantiles for both ends."""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque


class Metrics:
    def __init__(self, max_samples: int = 4096) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._latencies: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=max_samples)
        )

    def incr(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    def observe(self, name: str, value_ms: float) -> None:
        with self._lock:
            self._latencies[name].append(float(value_ms))

    def snapshot(self) -> dict:
        with self._lock:
            counters = {k: round(v, 4) for k, v in sorted(self._counters.items())}
            latencies = {
                name: _summarize(list(samples))
                for name, samples in sorted(self._latencies.items())
                if samples
            }
        return {"counters": counters, "latency_ms": latencies, "timestamp": time.time()}

    def prometheus(self) -> str:
        with self._lock:
            counters = dict(self._counters)
            latencies = {name: list(samples) for name, samples in self._latencies.items() if samples}
        lines = ["# HELP edge_cloud_counter Monotonic counters", "# TYPE edge_cloud_counter counter"]
        for name, value in sorted(counters.items()):
            lines.append(f'edge_cloud_counter{{name="{name}"}} {value:.6f}')
        lines.extend(
            ["# HELP edge_cloud_latency_ms Observed latency in milliseconds", "# TYPE edge_cloud_latency_ms gauge"]
        )
        for name, samples in sorted(latencies.items()):
            summary = _summarize(samples)
            for quantile, field in (("0.5", "p50"), ("0.95", "p95"), ("1.0", "max")):
                lines.append(
                    f'edge_cloud_latency_ms{{name="{name}",quantile="{quantile}"}} {summary[field]:.3f}'
                )
            lines.append(
                f'edge_cloud_latency_ms{{name="{name}",quantile="count"}} {summary["count"]:.0f}'
            )
        return "\n".join(lines) + "\n"


def _summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    count = len(ordered)
    return {
        "count": float(count),
        "avg": round(sum(ordered) / count, 3),
        "p50": round(_quantile(ordered, 0.50), 3),
        "p95": round(_quantile(ordered, 0.95), 3),
        "max": round(ordered[-1], 3),
    }


def _quantile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]
