"""Cost model for token-level handoff (交接成本模型).

The production budget used to count cloud calls and escalation ratio only. A
token-level handoff needs to price the work that each side actually performs:
edge decode, cloud prefill, cloud decode, the cloud call itself, and the extra
round-trip. Prices are supplied by configuration (the spec does not fetch real
bills), so the same model runs offline in the evaluation harness.

All components are additive: with token prices left at zero the estimate
degrades to per-call pricing, which is exactly the legacy budget unit. RTT is
reported for cost-accuracy comparisons but is not priced here because the spec
defines no per-millisecond unit price.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

_PRECISION = 9


def _round(value: float) -> float:
    return round(value, _PRECISION)


def _check_price(value: float, name: str) -> float:
    try:
        price = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数字") from exc
    if price < 0:
        raise ValueError(f"{name} 不能为负数")
    return price


def _check_count(value: int, name: str) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if count < 0:
        raise ValueError(f"{name} 不能为负数")
    return count


@dataclass(slots=True)
class CostEstimate:
    """Estimated cost of one request, broken down by component.

    ``estimated_cost`` is the sum of the four cost components. ``rtt_ms`` is
    informational (network latency is not priced); ``None`` is never used here
    because an estimate always has explicit counts, including zeros.
    """

    edge_tokens: int = 0
    cloud_prefill_tokens: int = 0
    cloud_decode_tokens_expected: int = 0
    cloud_calls: int = 0
    rtt_ms: float = 0.0
    estimated_cost: float = 0.0
    edge_cost: float = 0.0
    cloud_prefill_cost: float = 0.0
    cloud_decode_cost: float = 0.0
    cloud_call_cost: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(slots=True)
class CostModel:
    """Configuration-driven unit prices used to compare edge vs cloud work."""

    cost_per_edge_token: float = 0.0
    cost_per_cloud_input_token: float = 0.0
    cost_per_cloud_output_token: float = 0.0
    cost_per_cloud_call: float = 0.0
    cloud_rtt_ms: float = 0.0

    def __post_init__(self) -> None:
        self.cost_per_edge_token = _check_price(self.cost_per_edge_token, "cost_per_edge_token")
        self.cost_per_cloud_input_token = _check_price(
            self.cost_per_cloud_input_token, "cost_per_cloud_input_token"
        )
        self.cost_per_cloud_output_token = _check_price(
            self.cost_per_cloud_output_token, "cost_per_cloud_output_token"
        )
        self.cost_per_cloud_call = _check_price(self.cost_per_cloud_call, "cost_per_cloud_call")
        self.cloud_rtt_ms = _check_price(self.cloud_rtt_ms, "cloud_rtt_ms")

    def estimate(
        self,
        *,
        edge_tokens: int = 0,
        cloud_prefill_tokens: int = 0,
        cloud_decode_tokens_expected: int = 0,
        cloud_calls: int = 0,
        rtt_ms: float | None = None,
    ) -> CostEstimate:
        """Price one request; ``rtt_ms=None`` uses the configured ``cloud_rtt_ms``.

        With every token price at zero the token components are zero and the
        total equals ``cloud_calls * cost_per_cloud_call``. A request without a
        cloud call reports ``rtt_ms=0``: no round trip happens. Negative counts
        or a negative RTT raise ``ValueError``.
        """
        edge_tokens = _check_count(edge_tokens, "edge_tokens")
        cloud_prefill_tokens = _check_count(cloud_prefill_tokens, "cloud_prefill_tokens")
        cloud_decode_tokens_expected = _check_count(
            cloud_decode_tokens_expected, "cloud_decode_tokens_expected"
        )
        cloud_calls = _check_count(cloud_calls, "cloud_calls")
        override_rtt = None if rtt_ms is None else _check_price(rtt_ms, "rtt_ms")
        if cloud_calls == 0:
            rtt_ms = 0.0
        else:
            rtt_ms = self.cloud_rtt_ms if override_rtt is None else override_rtt

        edge_cost = _round(edge_tokens * self.cost_per_edge_token)
        prefill_cost = _round(cloud_prefill_tokens * self.cost_per_cloud_input_token)
        decode_cost = _round(cloud_decode_tokens_expected * self.cost_per_cloud_output_token)
        call_cost = _round(cloud_calls * self.cost_per_cloud_call)

        return CostEstimate(
            edge_tokens=edge_tokens,
            cloud_prefill_tokens=cloud_prefill_tokens,
            cloud_decode_tokens_expected=cloud_decode_tokens_expected,
            cloud_calls=cloud_calls,
            rtt_ms=rtt_ms,
            estimated_cost=_round(edge_cost + prefill_cost + decode_cost + call_cost),
            edge_cost=edge_cost,
            cloud_prefill_cost=prefill_cost,
            cloud_decode_cost=decode_cost,
            cloud_call_cost=call_cost,
        )
