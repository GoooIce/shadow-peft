"""Replaceable handoff policies (可注入的交接策略).

Policies only decide *whether* to hand off; the router owns cloud calls,
budgets, deadlines and the single-handoff rule. Two policies are provided:

``ProbePolicy``
    Scores the current prefix with an offline-calibrated probe and requests a
    handoff when ``prefix_trust`` falls below the configured threshold. It
    exposes ``available``/``unavailable_reason`` so the router can decide to
    stay on the legacy path *before* streaming starts. A probe file that is
    missing, version-mismatched, malformed, or unable to supply its required
    features never fails the request: the policy reports itself unavailable and
    ``observe`` continues on the edge. Missing features are never scored as 0.

``OraclePolicy``
    Evaluation-only upper bound. It reads the gold handoff position from
    ``reset(context={"gold_handoff_index": n})`` and never touches production
    configuration or a probe artifact.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .handoff import HandoffDecision, HandoffStep
from .probe import ProbeFeatureError, ProbeFormatError, ProbeModel, ProbeVersionError

# Used only when a policy is unavailable and no explicit threshold was given;
# an unavailable policy never scores, so the value is informational.
_DEFAULT_THRESHOLD = 0.5


def _unit_threshold(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数字") from exc
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} 必须在 [0, 1] 内")
    return number


class ProbePolicy:
    """Threshold policy backed by an offline-calibrated probe artifact.

    Args:
        probe_path: Path to the probe JSON written by ``probe.write_probe``.
        threshold: Decision threshold; a handoff is requested when
            ``prefix_trust < threshold``. ``None`` uses the artifact's stored
            threshold.
        threshold_by_task: Per-task overrides; tasks not present use the global
            threshold.
        feature_names: Features the serving backend can provide. When given, a
            probe whose ``required_features`` are not all available makes the
            policy unavailable before streaming starts.

    The policy never raises because of a bad artifact: load failures and
    missing features mark it unavailable instead. Constructor arguments that
    are programmer errors (thresholds outside [0, 1]) still raise ``ValueError``.
    """

    policy_name = "probe"

    def __init__(
        self,
        probe_path: str,
        threshold: float | None = None,
        threshold_by_task: Mapping[str, float] | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        self.probe_path = str(probe_path)
        self._threshold = (
            _unit_threshold(threshold, name="threshold")
            if threshold is not None
            else _DEFAULT_THRESHOLD
        )
        if threshold_by_task is not None and not isinstance(threshold_by_task, Mapping):
            raise ValueError("threshold_by_task 必须是对象")
        self._threshold_by_task = {
            str(task): _unit_threshold(value, name=f"threshold_by_task.{task}")
            for task, value in (threshold_by_task or {}).items()
        }
        if isinstance(feature_names, str):
            raise ValueError("feature_names 必须是非字符串列表")

        self._model: ProbeModel | None = None
        self._available = False
        self._unavailable_reason = ""
        try:
            model = ProbeModel.load(probe_path)
        except FileNotFoundError:
            self._unavailable_reason = f"probe_file_missing: {self.probe_path}"
        except ProbeVersionError as exc:
            self._unavailable_reason = f"probe_version_mismatch: {exc}"
        except ProbeFormatError as exc:
            self._unavailable_reason = f"probe_invalid: {exc}"
        except OSError as exc:
            self._unavailable_reason = f"probe_unreadable: {exc}"
        else:
            missing = sorted(
                set(model.required_features) - {str(name) for name in feature_names or ()}
            )
            if feature_names is not None and missing:
                self._unavailable_reason = (
                    "probe_feature_unavailable: runtime 无法提供 "
                    + ", ".join(missing)
                )
            else:
                self._model = model
                self._available = True
                if threshold is None:
                    self._threshold = model.threshold

    @property
    def available(self) -> bool:
        """Whether the probe loaded and its required features can be supplied."""
        return self._available

    @property
    def unavailable_reason(self) -> str:
        """A stable reason code plus detail; empty when available."""
        return self._unavailable_reason

    @property
    def model(self) -> ProbeModel | None:
        """The loaded probe, or ``None`` when unavailable."""
        return self._model

    def reset(self, context: dict[str, Any] | None = None) -> None:
        """Called once per request; a probe keeps no per-request state."""
        del context

    def _mark_unavailable(self, reason: str) -> None:
        self._available = False
        self._unavailable_reason = reason

    def observe(self, step: HandoffStep) -> HandoffDecision:
        features = step.features
        if not self._available or self._model is None:
            return HandoffDecision(action="continue", boundary=features.boundary_kind)
        try:
            trust = self._model.score(features)
        except ProbeFeatureError as exc:
            self._mark_unavailable(f"probe_feature_missing: {exc}")
            return HandoffDecision(action="continue", boundary=features.boundary_kind)

        threshold = self._threshold_by_task.get(features.task, self._threshold)
        if trust < threshold:
            return HandoffDecision(
                action="handoff",
                reason="low_prefix_trust",
                boundary=features.boundary_kind,
                handoff_index=features.prefix_tokens,
                prefix_trust=trust,
            )
        return HandoffDecision(
            action="continue",
            boundary=features.boundary_kind,
            prefix_trust=trust,
        )


def _json_braces_balanced(text: str) -> bool:
    """Return whether braces/brackets and string literals are balanced."""

    stack: list[str] = []
    opening = {"}": "{", "]": "["}
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]" and (not stack or stack.pop() != opening[char]):
            return False
    return not stack and not in_string


def _json_prefix_invalid(text: str) -> bool:
    """A balanced prefix that no longer parses is structurally broken JSON."""

    if not text.strip() or not _json_braces_balanced(text):
        return False
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return True
    return False


class FormatGuardPolicy:
    """Conservative mid-stream format gate around Probe/Oracle policies.

    Only JSON is checked mid-generation, and only at a ``json_field`` boundary
    (the prefix already ends with ``,``, ``}`` or ``]``). The guard triggers
    ``format_invalid`` when the backend's step context declares
    ``format_ok == 0.0`` or when the prefix is balanced but ``json.loads``
    fails; an unbalanced prefix is left alone because the object is still being
    written. Every other format is validated once on the final answer by the
    router, never mid-stream, and the guard never fabricates a format failure
    for non-JSON tasks. When the guard does not trigger, the wrapped policy
    decides as usual.
    """

    policy_name = "format_guard"

    def __init__(self, delegate: Any) -> None:
        if not callable(getattr(delegate, "observe", None)):
            raise ValueError("delegate 必须实现 observe")
        self.delegate = delegate
        self._expected_format: str | None = None

    @property
    def available(self) -> bool:
        return bool(getattr(self.delegate, "available", True))

    @property
    def unavailable_reason(self) -> str:
        return str(getattr(self.delegate, "unavailable_reason", ""))

    def reset(self, context: dict[str, Any] | None = None) -> None:
        expected = context.get("expected_format") if isinstance(context, Mapping) else None
        self._expected_format = expected if isinstance(expected, str) and expected else None
        self.delegate.reset(context)

    def observe(self, step: HandoffStep) -> HandoffDecision:
        features = step.features
        if (
            self._expected_format == "json"
            and features.boundary_kind == "json_field"
            and (features.format_ok == 0.0 or _json_prefix_invalid(step.prefix_text))
        ):
            return HandoffDecision(
                action="handoff",
                reason="format_invalid",
                boundary="json_field",
                handoff_index=features.prefix_tokens,
                prefix_trust=None,
            )
        return self.delegate.observe(step)


class OraclePolicy:
    """Evaluation-only policy that knows the best handoff position.

    This is the upper bound used by the offline harness, not a serving policy.
    It intentionally has no probe path, threshold or config dependency; the
    label arrives through ``reset``. Once ``prefix_tokens >= gold_handoff_index``
    it keeps requesting a handoff (the router still enforces a single handoff)
    with reason ``low_prefix_trust``; ``prefix_trust`` stays ``None`` because an
    oracle has no calibrated trust estimate.

    Raises:
        ValueError: the injected ``gold_handoff_index`` is not a non-negative
            integer.
    """

    policy_name = "oracle"

    def __init__(self) -> None:
        self._gold_handoff_index: int | None = None

    @property
    def gold_handoff_index(self) -> int | None:
        return self._gold_handoff_index

    def reset(self, context: dict[str, Any] | None = None) -> None:
        gold = None if context is None else context.get("gold_handoff_index")
        if gold is None:
            self._gold_handoff_index = None
            return
        if isinstance(gold, bool) or not isinstance(gold, int) or gold < 0:
            raise ValueError("gold_handoff_index 必须是非负整数")
        self._gold_handoff_index = gold

    def observe(self, step: HandoffStep) -> HandoffDecision:
        features = step.features
        gold = self._gold_handoff_index
        if gold is not None and features.prefix_tokens >= gold:
            return HandoffDecision(
                action="handoff",
                reason="low_prefix_trust",
                boundary=features.boundary_kind,
                handoff_index=features.prefix_tokens,
                prefix_trust=None,
            )
        return HandoffDecision(action="continue", boundary=features.boundary_kind)
