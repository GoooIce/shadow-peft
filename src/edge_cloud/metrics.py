"""Thread-safe counters and latency quantiles for both ends."""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from collections.abc import Sequence


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

    def observe_hist(self, name: str, value: float, boundaries: Sequence[float]) -> None:
        """Bucket *value* into half-open ``[b_i, b_i+1)`` counters.

        Boundaries are the upper edges of the buckets except the last, which is
        inclusive on its lower side: ``(4, 8)`` produces ``name_bucket_lt_4``,
        ``name_bucket_lt_8`` and ``name_bucket_ge_8``. Only counters are
        written; raw histogram samples never enter the latency quantiles.
        """

        ordered = _hist_boundaries(name, boundaries)
        number = float(value)
        for boundary in ordered:
            if number < boundary:
                self.incr(f"{name}_bucket_lt_{_numeric_label(boundary)}")
                return
        self.incr(f"{name}_bucket_ge_{_numeric_label(ordered[-1])}")

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
            latencies = {
                name: list(samples)
                for name, samples in self._latencies.items()
                if samples
            }
        lines = [
            "# HELP edge_cloud_counter Monotonic counters",
            "# TYPE edge_cloud_counter counter",
        ]
        for name, value in sorted(counters.items()):
            lines.append(f'edge_cloud_counter{{name="{name}"}} {value:.6f}')
        lines.extend(
            [
                "# HELP edge_cloud_latency_ms Observed latency in milliseconds",
                "# TYPE edge_cloud_latency_ms gauge",
            ]
        )
        for name, samples in sorted(latencies.items()):
            summary = _summarize(samples)
            for quantile, field in (("0.5", "p50"), ("0.95", "p95"), ("1.0", "max")):
                lines.append(
                    f'edge_cloud_latency_ms{{name="{name}",quantile="{quantile}"}} '
                    f"{summary[field]:.3f}"
                )
            lines.append(
                f'edge_cloud_latency_ms{{name="{name}",quantile="count"}} {summary["count"]:.0f}'
            )
        return "\n".join(lines) + "\n"


def _hist_boundaries(name: str, boundaries: Sequence[float]) -> list[float]:
    ordered = sorted({float(boundary) for boundary in boundaries})
    if not ordered:
        raise ValueError(f"{name} 直方图至少需要一个边界")
    if any(not math.isfinite(boundary) or boundary <= 0.0 for boundary in ordered):
        raise ValueError(f"{name} 直方图边界必须是正数")
    return ordered


def _numeric_label(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


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
