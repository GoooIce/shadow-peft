from __future__ import annotations

import pytest

from edge_cloud.metrics import Metrics


def test_observe_hist_increments_half_open_bucket_counters_only() -> None:
    metrics = Metrics()

    for value in (0, 3, 4, 7, 8, 16, 127, 128, 500):
        metrics.observe_hist("handoff_index", value, (4, 8, 16, 32, 64, 128))

    counters = metrics.snapshot()["counters"]
    assert counters["handoff_index_bucket_lt_4"] == 2
    assert counters["handoff_index_bucket_lt_8"] == 2
    assert counters["handoff_index_bucket_lt_16"] == 1
    assert counters["handoff_index_bucket_lt_32"] == 1
    assert counters.get("handoff_index_bucket_lt_64", 0) == 0
    assert counters["handoff_index_bucket_lt_128"] == 1
    assert counters["handoff_index_bucket_ge_128"] == 2
    assert "handoff_index" not in metrics.snapshot()["latency_ms"]


def test_observe_hist_keeps_observing_separate_from_raw_latency_samples() -> None:
    metrics = Metrics()
    metrics.observe("handoff_index", 9.0)
    metrics.observe_hist("handoff_index", 9, (4, 16))

    snapshot = metrics.snapshot()
    assert snapshot["latency_ms"]["handoff_index"]["count"] == 1
    assert snapshot["counters"]["handoff_index_bucket_lt_16"] == 1


def test_observe_hist_rejects_empty_or_non_positive_boundaries() -> None:
    metrics = Metrics()

    with pytest.raises(ValueError):
        metrics.observe_hist("handoff_index", 3, ())
    with pytest.raises(ValueError):
        metrics.observe_hist("handoff_index", 3, (4, 0))
