"""Confidence estimation for edge outputs (端侧置信度).

The estimator deliberately combines model-internal signals (token log
probabilities) with output-level heuristics. On a production line a single
signal is too brittle: a small model can be confidently wrong, and a correct
structured answer can contain one low-probability formatting token.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from .protocol import ConfidenceReport

_DEFAULT_WEIGHTS = {
    # Token probabilities carry the calibration; heuristics only catch
    # degeneration (repetition, empty/truncated output, bad format) and must
    # not rescue a low-probability answer.
    "logprob_geomean": 0.72,
    "logprob_min": 0.16,
    "low_prob_ok": 0.07,
    "repetition_ok": 0.02,
    "length_ok": 0.02,
    "format_ok": 0.01,
}

_HEURISTIC_WEIGHTS = {
    "repetition_ok": 0.50,
    "length_ok": 0.30,
    "format_ok": 0.20,
}


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _repetition_ok(text: str) -> float:
    # Chinese has no whitespace boundaries; split CJK characters individually
    # so 4-gram repetition is still detectable in Chinese answers.
    tokens = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_.+-]+", text)
    if len(tokens) < 6:
        return 1.0
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(tokens) - 3):
        gram = tuple(tokens[i : i + 4])
        counts[gram] = counts.get(gram, 0) + 1
    worst = max(counts.values(), default=1)
    repeated_ratio = max(0, worst - 1) / max(1, len(tokens) - 3)
    return _clip(1.0 - repeated_ratio * 3.0)


def _length_ok(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    length = len(stripped)
    if length < 4:
        return 0.35
    if length < 12:
        return 0.75
    return 1.0


def format_ok(text: str, expected_format: str | None) -> float:
    """Return 1.0 when *text* satisfies *expected_format*, else 0.0.

    Public because the router applies the same final-answer rule after a
    token-level handoff; unknown formats are not checked and score 1.0.
    """

    if not expected_format:
        return 1.0
    stripped = text.strip()
    if expected_format == "json":
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            return 0.0
        return 1.0
    if expected_format == "number":
        return 1.0 if re.search(r"-?\d+(?:\.\d+)?", stripped) else 0.0
    if expected_format == "label":
        return 1.0 if re.fullmatch(r"[\w\u4e00-\u9fff-]{1,24}", stripped) else 0.0
    if expected_format == "verdict":
        return 1.0 if re.search(r"(判定|结论|放行|拦截|合格|不合格|返修)", stripped) else 0.0
    return 1.0


# Backwards-compatible private alias for callers written before the rename.
_format_ok = format_ok


@dataclass(slots=True)
class ConfidenceEstimator:
    """Score in [0, 1]; values below the router threshold trigger escalation."""

    low_logprob: float = -2.0
    weights: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))

    def estimate(
        self,
        text: str,
        token_logprobs: list[float] | None = None,
        *,
        finish_reason: str = "stop",
        expected_format: str | None = None,
    ) -> ConfidenceReport:
        repetition_ok = _repetition_ok(text)
        length_ok = _length_ok(text)
        format_ok_value = format_ok(text, expected_format)
        signals: dict[str, float] = {
            "repetition_ok": round(repetition_ok, 4),
            "length_ok": round(length_ok, 4),
            "format_ok": round(format_ok_value, 4),
        }
        weights = self.weights

        if token_logprobs:
            finite = [lp for lp in token_logprobs if math.isfinite(lp)]
            if finite:
                mean_lp = sum(finite) / len(finite)
                geomean = math.exp(mean_lp)
                min_prob = math.exp(min(finite))
                low_frac = sum(1 for lp in finite if lp < self.low_logprob) / len(finite)
                signals.update(
                    {
                        "logprob_geomean": round(_clip(geomean), 4),
                        "logprob_min": round(_clip(min_prob), 4),
                        "low_prob_frac": round(low_frac, 4),
                        "low_prob_ok": round(_clip(1.0 - low_frac), 4),
                        "tokens_scored": float(len(finite)),
                    }
                )
                score = (
                    weights["logprob_geomean"] * _clip(geomean)
                    + weights["logprob_min"] * _clip(min_prob)
                    + weights["low_prob_ok"] * _clip(1.0 - low_frac)
                    + weights["repetition_ok"] * repetition_ok
                    + weights["length_ok"] * length_ok
                    + weights["format_ok"] * format_ok_value
                )
                method = "logprob+heuristic"
            else:
                score = self._heuristic_score(repetition_ok, length_ok, format_ok_value)
                method = "heuristic"
        else:
            score = self._heuristic_score(repetition_ok, length_ok, format_ok_value)
            method = "heuristic"

        # A truncated answer is usable but should not be treated as complete.
        if finish_reason == "length":
            score *= 0.85
            signals["truncated"] = 1.0

        return ConfidenceReport(score=round(_clip(score), 4), method=method, signals=signals)

    @staticmethod
    def _heuristic_score(repetition_ok: float, length_ok: float, format_ok: float) -> float:
        return (
            _HEURISTIC_WEIGHTS["repetition_ok"] * repetition_ok
            + _HEURISTIC_WEIGHTS["length_ok"] * length_ok
            + _HEURISTIC_WEIGHTS["format_ok"] * format_ok
        )
