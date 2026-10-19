from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from edge_cloud.handoff import HandoffFeatures, HandoffPolicy, HandoffStep
from edge_cloud.policies import FormatGuardPolicy, OraclePolicy, ProbePolicy
from edge_cloud.probe import PROBE_VERSION, fit_probe, write_probe

_FIXED_TRAINED_AT = "2026-01-01T00:00:00+00:00"


def _write_one_feature_probe(
    path: Path,
    *,
    feature_name: str = "base_margin",
    coef: float = 2.0,
    intercept: float = 0.0,
    threshold: float = 0.5,
) -> Path:
    payload: dict[str, Any] = {
        "version": PROBE_VERSION,
        "feature_names": [feature_name],
        "means": [0.0],
        "stds": [1.0],
        "coefs": [coef],
        "intercept": intercept,
        "threshold": threshold,
        "calibration": [],
        "fingerprint": "",
        "required_features": [feature_name],
        "trained_at": _FIXED_TRAINED_AT,
    }
    write_probe(payload, path)
    return path


def _step(
    *,
    prefix_tokens: int = 0,
    boundary_kind: str = "token",
    task: str = "general",
    base_margin: float = 0.0,
    kl_base_shadow: float | None = None,
    token: str = "",
    prefix_text: str = "prefix",
    format_ok: float = 1.0,
) -> HandoffStep:
    return HandoffStep(
        features=HandoffFeatures(
            base_margin=base_margin,
            kl_base_shadow=kl_base_shadow,
            prefix_tokens=prefix_tokens,
            boundary_kind=boundary_kind,  # type: ignore[arg-type]
            task=task,
            format_ok=format_ok,
        ),
        prefix_text=prefix_text,
        token=token,
        token_logprob=None,
    )


def test_probe_policy_hands_off_below_the_threshold_and_continues_above_it(
    tmp_path: Path,
) -> None:
    path = _write_one_feature_probe(tmp_path / "probe.json")
    policy = ProbePolicy(path, threshold=0.5)
    assert policy.available is True
    assert policy.unavailable_reason == ""

    low = policy.observe(
        _step(prefix_tokens=5, boundary_kind="sentence", base_margin=-1.0, token="。")
    )

    # coef=2, mean=0, std=1: sigmoid(-2) = 0.1192029...
    assert low.action == "handoff"
    assert low.reason == "low_prefix_trust"
    assert low.boundary == "sentence"
    assert low.handoff_index == 5
    assert low.prefix_trust == pytest.approx(0.1192029220, abs=1e-9)

    high = policy.observe(_step(prefix_tokens=6, base_margin=1.0))

    assert high.action == "continue"
    assert high.reason is None
    assert high.handoff_index == 0
    assert high.prefix_trust == pytest.approx(0.8807970780, abs=1e-9)


def test_probe_policy_uses_the_global_threshold_when_the_task_has_no_override(
    tmp_path: Path,
) -> None:
    path = _write_one_feature_probe(tmp_path / "probe.json")
    policy = ProbePolicy(path, threshold=0.4, threshold_by_task={"quality_verdict": 0.6})

    # trust = sigmoid(0) = 0.5 for every step.
    general = policy.observe(_step(task="general"))
    verdict = policy.observe(_step(task="quality_verdict"))
    unknown = policy.observe(_step(task="diagnosis"))

    assert general.action == "continue"
    assert verdict.action == "handoff"
    assert verdict.reason == "low_prefix_trust"
    assert unknown.action == "continue"


def test_probe_policy_falls_back_to_the_artifact_threshold(tmp_path: Path) -> None:
    path = _write_one_feature_probe(tmp_path / "probe.json", threshold=0.4)

    # trust 0.5 clears the artifact threshold 0.4 ...
    assert ProbePolicy(path).observe(_step()).action == "continue"
    # ... but not an explicit 0.6 override.
    assert ProbePolicy(path, threshold=0.6).observe(_step()).action == "handoff"


def test_probe_policy_with_a_missing_file_is_unavailable_and_never_hands_off(
    tmp_path: Path,
) -> None:
    policy = ProbePolicy(tmp_path / "missing.json", threshold=0.5)

    assert policy.available is False
    assert policy.unavailable_reason.startswith("probe_file_missing")

    decision = policy.observe(_step(prefix_tokens=4))

    assert decision.action == "continue"
    assert decision.reason is None
    assert decision.prefix_trust is None


def test_probe_policy_with_an_unsupported_version_is_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "probe.json"
    payload = {
        "version": "handoff-probe/v99",
        "feature_names": ["base_margin"],
        "means": [0.0],
        "stds": [1.0],
        "coefs": [2.0],
        "intercept": 0.0,
        "threshold": 0.5,
        "calibration": [],
        "fingerprint": "0" * 64,
        "required_features": ["base_margin"],
        "trained_at": _FIXED_TRAINED_AT,
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    policy = ProbePolicy(path, threshold=0.5)

    assert policy.available is False
    assert policy.unavailable_reason.startswith("probe_version_mismatch")
    assert policy.observe(_step()).action == "continue"


def test_probe_policy_degrades_when_a_required_feature_is_none(tmp_path: Path) -> None:
    path = _write_one_feature_probe(tmp_path / "probe.json", feature_name="kl_base_shadow")
    policy = ProbePolicy(path, threshold=0.5)
    assert policy.available is True

    # kl_base_shadow=None must not score as 0 (which would trigger a handoff).
    decision = policy.observe(
        _step(prefix_tokens=3, base_margin=0.0, kl_base_shadow=None)
    )

    assert decision.action == "continue"
    assert decision.reason is None
    assert decision.prefix_trust is None
    assert policy.available is False
    assert policy.unavailable_reason.startswith("probe_feature_missing")


def test_probe_policy_is_unavailable_when_runtime_cannot_supply_a_required_feature(
    tmp_path: Path,
) -> None:
    path = _write_one_feature_probe(tmp_path / "probe.json", feature_name="kl_base_shadow")

    unavailable = ProbePolicy(path, threshold=0.5, feature_names=["base_margin"])
    available = ProbePolicy(
        path,
        threshold=0.5,
        feature_names=["base_margin", "kl_base_shadow"],
    )

    assert unavailable.available is False
    assert unavailable.unavailable_reason.startswith("probe_feature_unavailable")
    assert "kl_base_shadow" in unavailable.unavailable_reason
    assert available.available is True


def test_probe_policy_rejects_invalid_thresholds(tmp_path: Path) -> None:
    path = _write_one_feature_probe(tmp_path / "probe.json")

    with pytest.raises(ValueError):
        ProbePolicy(path, threshold=1.5)
    with pytest.raises(ValueError):
        ProbePolicy(path, threshold=0.5, threshold_by_task={"general": -0.1})


def test_probe_policy_from_fitted_handoff_rows_triggers_only_on_handoff_features(
    tmp_path: Path,
) -> None:
    rows = [
        {"kl_base_shadow": 0.9},
        {"kl_base_shadow": 0.8},
        {"kl_base_shadow": 0.95},
        {"kl_base_shadow": 0.1},
        {"kl_base_shadow": 0.15},
        {"kl_base_shadow": 0.05},
    ]
    labels = [1, 1, 1, 0, 0, 0]
    payload = fit_probe(rows, labels, threshold=0.5, trained_at=_FIXED_TRAINED_AT)
    path = tmp_path / "fitted.json"
    write_probe(payload, path)
    policy = ProbePolicy(path, threshold=0.5)

    assert policy.available is True
    handoff = policy.observe(_step(prefix_tokens=4, kl_base_shadow=0.9))
    stay = policy.observe(_step(prefix_tokens=2, kl_base_shadow=0.1))

    assert handoff.action == "handoff"
    assert handoff.reason == "low_prefix_trust"
    assert handoff.handoff_index == 4
    assert handoff.prefix_trust is not None and handoff.prefix_trust < 0.5
    assert stay.action == "continue"
    assert stay.reason is None
    assert stay.prefix_trust is not None and stay.prefix_trust > 0.5


def test_oracle_policy_hands_off_once_the_gold_index_is_reached() -> None:
    policy = OraclePolicy()
    policy.reset({"gold_handoff_index": 3})

    before = policy.observe(_step(prefix_tokens=2))
    exact = policy.observe(_step(prefix_tokens=3, boundary_kind="newline"))
    after = policy.observe(_step(prefix_tokens=5))

    assert before.action == "continue"
    assert before.reason is None
    assert exact.action == "handoff"
    assert exact.reason == "low_prefix_trust"
    assert exact.boundary == "newline"
    assert exact.handoff_index == 3
    assert exact.prefix_trust is None
    assert after.action == "handoff"
    assert after.handoff_index == 5
    assert policy.policy_name == "oracle"


def test_oracle_policy_without_a_gold_index_never_hands_off() -> None:
    policy = OraclePolicy()

    assert policy.observe(_step(prefix_tokens=10)).action == "continue"

    policy.reset({})
    assert policy.observe(_step(prefix_tokens=10)).action == "continue"

    policy.reset(None)
    assert policy.observe(_step(prefix_tokens=10)).action == "continue"


@pytest.mark.parametrize("gold", [-1, 1.5, "3", True])
def test_oracle_policy_rejects_an_invalid_gold_index(gold: Any) -> None:
    policy = OraclePolicy()

    with pytest.raises(ValueError):
        policy.reset({"gold_handoff_index": gold})


def test_both_policies_satisfy_the_handoff_policy_protocol(tmp_path: Path) -> None:
    path = _write_one_feature_probe(tmp_path / "probe.json")

    assert isinstance(ProbePolicy(path, threshold=0.5), HandoffPolicy)
    assert isinstance(OraclePolicy(), HandoffPolicy)


def test_format_guard_hands_off_on_balanced_invalid_json_at_a_field_boundary() -> None:
    delegate = OraclePolicy()
    guard = FormatGuardPolicy(delegate)
    guard.reset({"expected_format": "json"})

    decision = guard.observe(
        _step(
            prefix_tokens=7,
            boundary_kind="json_field",
            prefix_text='{"a": 1, "b": }',
        )
    )

    assert decision.action == "handoff"
    assert decision.reason == "format_invalid"
    assert decision.boundary == "json_field"
    assert decision.handoff_index == 7
    assert decision.prefix_trust is None


def test_format_guard_hands_off_when_the_step_declares_format_failure() -> None:
    guard = FormatGuardPolicy(OraclePolicy())
    guard.reset({"expected_format": "json"})

    decision = guard.observe(
        _step(
            prefix_tokens=2,
            boundary_kind="json_field",
            prefix_text='{"a": 1,',
            format_ok=0.0,
        )
    )

    assert decision.action == "handoff"
    assert decision.reason == "format_invalid"


def test_format_guard_delegates_for_valid_balanced_json() -> None:
    guard = FormatGuardPolicy(OraclePolicy())
    guard.reset({"expected_format": "json", "gold_handoff_index": 3})

    decision = guard.observe(
        _step(
            prefix_tokens=3,
            boundary_kind="json_field",
            prefix_text='{"a": 1}',
        )
    )

    # The valid JSON does not trigger the guard; the oracle delegate does.
    assert decision.action == "handoff"
    assert decision.reason == "low_prefix_trust"


def test_format_guard_delegates_for_unbalanced_json_prefixes() -> None:
    guard = FormatGuardPolicy(OraclePolicy())
    guard.reset({"expected_format": "json"})

    decision = guard.observe(
        _step(
            prefix_tokens=2,
            boundary_kind="json_field",
            prefix_text='{"a": 1,',
        )
    )

    assert decision.action == "continue"
    assert decision.reason is None


def test_format_guard_never_fabricates_format_failures_for_other_formats() -> None:
    guard = FormatGuardPolicy(OraclePolicy())
    guard.reset({"expected_format": "number"})

    decision = guard.observe(
        _step(
            prefix_tokens=2,
            boundary_kind="sentence",
            prefix_text="标准答案 42。",
            format_ok=0.0,
        )
    )

    assert decision.action == "continue"


def test_format_guard_delegates_reset_and_availability_to_the_wrapped_policy(
    tmp_path: Path,
) -> None:
    probe = ProbePolicy(tmp_path / "missing.json", threshold=0.5)
    guard = FormatGuardPolicy(probe)

    assert guard.available is False
    assert guard.unavailable_reason.startswith("probe_file_missing")
    guard.reset({"expected_format": None})
    assert guard.observe(_step()).action == "continue"
