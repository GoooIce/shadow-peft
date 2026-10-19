from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from edge_cloud.probe import (
    PROBE_VERSION,
    ProbeFeatureError,
    ProbeFormatError,
    ProbeModel,
    ProbeVersionError,
    fit_probe,
    write_probe,
)

_FIXED_TRAINED_AT = "2026-01-01T00:00:00+00:00"


def _one_feature_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": PROBE_VERSION,
        "feature_names": ["base_margin"],
        "means": [0.0],
        "stds": [1.0],
        "coefs": [2.0],
        "intercept": 0.0,
        "threshold": 0.5,
        "calibration": [],
        "fingerprint": "",
        "required_features": ["base_margin"],
        "trained_at": _FIXED_TRAINED_AT,
    }
    payload.update(overrides)
    return payload


def test_score_on_a_hand_built_probe_matches_sigmoid_of_standardized_input(
    tmp_path: Path,
) -> None:
    path = tmp_path / "probe.json"
    write_probe(_one_feature_payload(), path)
    model = ProbeModel.load(path)

    # mean=0, std=1, coef=2, intercept=0 -> sigmoid(2 * 0.5) = 0.731058...
    assert model.score({"base_margin": 0.5}) == pytest.approx(0.7310585786, abs=1e-9)
    assert model.score({"base_margin": 0.0}) == pytest.approx(0.5, abs=1e-12)
    assert 0.0 < model.score({"base_margin": -100.0}) < 1.0


def test_boolean_features_are_encoded_as_zero_and_one(tmp_path: Path) -> None:
    payload = _one_feature_payload(
        feature_names=["base_top1_in_shadow_topk"],
        required_features=["base_top1_in_shadow_topk"],
    )
    path = tmp_path / "probe.json"
    write_probe(payload, path)
    model = ProbeModel.load(path)

    # sigmoid(2 * 1) vs sigmoid(2 * 0)
    assert model.score({"base_top1_in_shadow_topk": True}) == pytest.approx(
        0.8807970780, abs=1e-9
    )
    assert model.score({"base_top1_in_shadow_topk": False}) == pytest.approx(0.5, abs=1e-12)


def test_missing_required_feature_raises_instead_of_scoring_as_zero(tmp_path: Path) -> None:
    path = tmp_path / "probe.json"
    write_probe(_one_feature_payload(), path)
    model = ProbeModel.load(path)

    with pytest.raises(ProbeFeatureError, match="base_margin"):
        model.score({"base_margin": None})
    with pytest.raises(ProbeFeatureError, match="base_margin"):
        model.score({})


def test_optional_missing_feature_is_imputed_at_the_training_mean(tmp_path: Path) -> None:
    payload = _one_feature_payload(
        feature_names=["base_margin", "kl_base_shadow"],
        means=[0.0, 1.0],
        stds=[1.0, 1.0],
        coefs=[0.0, 1.0],
        required_features=["base_margin"],
    )
    path = tmp_path / "probe.json"
    write_probe(payload, path)
    model = ProbeModel.load(path)

    # Missing optional feature -> z = 0, not raw 0: score stays 0.5 and differs
    # from feeding an actual 0.0, which would be standardized to -1.
    assert model.score({"base_margin": 0.0, "kl_base_shadow": None}) == pytest.approx(
        0.5, abs=1e-12
    )
    assert model.score({"base_margin": 0.0, "kl_base_shadow": 0.0}) == pytest.approx(
        1.0 / (1.0 + math.e), abs=1e-9
    )


def _handoff_direction_rows() -> tuple[list[dict[str, Any]], list[int]]:
    """Rows whose feature is high exactly where the prefix should hand off."""
    handoff = [{"kl_base_shadow": 0.9}, {"kl_base_shadow": 0.8}, {"kl_base_shadow": 0.95}]
    stay = [{"kl_base_shadow": 0.1}, {"kl_base_shadow": 0.15}, {"kl_base_shadow": 0.05}]
    return handoff + stay, [1, 1, 1, 0, 0, 0]


def test_label_one_means_handoff_and_scores_low_prefix_trust() -> None:
    rows, labels = _handoff_direction_rows()

    payload = fit_probe(
        rows, labels, threshold=0.5, trained_at=_FIXED_TRAINED_AT
    )
    model = ProbeModel.from_dict(payload)

    handoff_trust = model.score(rows[0])
    stay_trust = model.score(rows[-1])
    assert handoff_trust < stay_trust
    assert handoff_trust < 0.5 < stay_trust


def test_fit_load_preserves_the_trust_direction(tmp_path: Path) -> None:
    rows, labels = _handoff_direction_rows()
    payload = fit_probe(
        rows, labels, threshold=0.5, trained_at=_FIXED_TRAINED_AT
    )
    path = tmp_path / "probe.json"
    write_probe(payload, path)

    loaded = ProbeModel.load(path)

    assert loaded.score(rows[0]) < 0.5
    assert loaded.score(rows[-1]) > 0.5
    assert loaded.score(rows[0]) < loaded.score(rows[-1])


def test_calibration_positive_rate_is_the_handoff_rate() -> None:
    rows, labels = _handoff_direction_rows()

    payload = fit_probe(
        rows,
        labels,
        threshold=0.5,
        calibration_bins=2,
        trained_at=_FIXED_TRAINED_AT,
    )
    model = ProbeModel.from_dict(payload)
    lowest, highest = payload["calibration"]

    # Both bins are populated: the handoff rows score low, the stay rows high.
    assert lowest["count"] == 3
    assert highest["count"] == 3
    assert lowest["positive_rate"] == pytest.approx(1.0)
    assert highest["positive_rate"] == pytest.approx(0.0)
    assert lowest["mean_score"] == pytest.approx(
        sum(model.score(row) for row in rows[:3]) / 3, abs=1e-6
    )


def test_fit_probe_returns_the_documented_artifact_schema() -> None:
    rows = [
        {"base_margin": 0.9, "low_prob_frac": 0.0},
        {"base_margin": 0.8, "low_prob_frac": 0.1},
        {"base_margin": 0.2, "low_prob_frac": 0.8},
        {"base_margin": 0.1, "low_prob_frac": 0.9},
    ]
    labels = [1, 1, 0, 0]

    payload = fit_probe(
        rows,
        labels,
        version=PROBE_VERSION,
        threshold=0.5,
        trained_at=_FIXED_TRAINED_AT,
    )

    assert set(payload) == {
        "version",
        "feature_names",
        "means",
        "stds",
        "coefs",
        "intercept",
        "threshold",
        "calibration",
        "fingerprint",
        "required_features",
        "trained_at",
    }
    assert payload["version"] == PROBE_VERSION
    assert payload["feature_names"] == ["base_margin", "low_prob_frac"]
    assert payload["required_features"] == ["base_margin", "low_prob_frac"]
    assert len(payload["means"]) == len(payload["stds"]) == len(payload["coefs"]) == 2
    assert all(std > 0.0 for std in payload["stds"])
    assert payload["trained_at"] == _FIXED_TRAINED_AT
    assert isinstance(payload["fingerprint"], str) and len(payload["fingerprint"]) == 64


def test_fit_load_score_round_trip_preserves_scores(tmp_path: Path) -> None:
    rows = [
        {"base_margin": 0.9, "kl_base_shadow": 0.05},
        {"base_margin": 0.7, "kl_base_shadow": 0.2},
        {"base_margin": 0.2, "kl_base_shadow": 0.7},
        {"base_margin": 0.1, "kl_base_shadow": 0.9},
    ]
    labels = [1, 1, 0, 0]
    payload = fit_probe(
        rows,
        labels,
        threshold=0.5,
        trained_at=_FIXED_TRAINED_AT,
    )
    path = tmp_path / "probe.json"
    write_probe(payload, path)

    written = ProbeModel.from_dict(payload)
    loaded = ProbeModel.load(path)

    for row in rows:
        assert loaded.score(row) == pytest.approx(written.score(row), abs=1e-12)
    # Labels 1 mean "should hand off", so the first rows must score lower trust.
    assert loaded.score(rows[0]) < loaded.score(rows[2])
    assert loaded.score(rows[0]) < 0.5 < loaded.score(rows[2])


def test_fit_is_deterministic_and_fingerprint_ignores_trained_at() -> None:
    rows = [
        {"base_margin": 0.9, "low_prob_frac": 0.0},
        {"base_margin": 0.2, "low_prob_frac": 0.8},
        {"base_margin": 0.6, "low_prob_frac": 0.3},
        {"base_margin": 0.3, "low_prob_frac": 0.7},
    ]
    labels = [1, 0, 1, 0]

    first = fit_probe(rows, labels, threshold=0.5, trained_at=_FIXED_TRAINED_AT)
    second = fit_probe(rows, labels, threshold=0.5, trained_at=_FIXED_TRAINED_AT)
    later = fit_probe(rows, labels, threshold=0.5, trained_at="2026-06-01T00:00:00+00:00")

    assert first == second
    assert first["fingerprint"] == second["fingerprint"]
    assert first["fingerprint"] == later["fingerprint"]
    assert first["coefs"] == later["coefs"]
    assert first["intercept"] == later["intercept"]


def test_write_probe_is_byte_stable_and_sorts_keys(tmp_path: Path) -> None:
    payload = fit_probe(
        [{"base_margin": 0.9}, {"base_margin": 0.2}, {"base_margin": 0.6}],
        [1, 0, 1],
        trained_at=_FIXED_TRAINED_AT,
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    write_probe(payload, first_path)
    write_probe(payload, second_path)

    first_text = first_path.read_text(encoding="utf-8")
    assert first_text == second_path.read_text(encoding="utf-8")
    assert first_text.endswith("\n")

    pairs = json.loads(first_text, object_pairs_hook=lambda items: items)
    assert [key for key, _ in pairs] == sorted(key for key, _ in pairs)


def test_fit_probe_calibration_bins_cover_every_training_row() -> None:
    rows = [{"base_margin": value / 10.0} for value in range(10)]
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    payload = fit_probe(rows, labels, calibration_bins=5, trained_at=_FIXED_TRAINED_AT)
    bins = payload["calibration"]

    assert len(bins) == 5
    assert sum(entry["count"] for entry in bins) == len(rows)
    for entry in bins:
        assert {"lower", "upper", "count", "positives", "positive_rate", "mean_score"} <= set(entry)
        if entry["count"] == 0:
            assert entry["positive_rate"] is None
            assert entry["mean_score"] is None
        else:
            assert entry["positive_rate"] == pytest.approx(
                entry["positives"] / entry["count"], abs=1e-9
            )
            assert 0.0 <= entry["mean_score"] <= 1.0


def test_fit_probe_one_class_continue_rows_score_high_trust() -> None:
    payload = fit_probe(
        [{"base_margin": 0.9}, {"base_margin": 0.8}],
        [0, 0],
        trained_at=_FIXED_TRAINED_AT,
    )
    model = ProbeModel.from_dict(payload)

    assert all(math.isfinite(coef) for coef in payload["coefs"])
    assert math.isfinite(payload["intercept"])
    assert model.score({"base_margin": 0.9}) > 0.5


def test_fit_probe_one_class_handoff_rows_score_low_trust() -> None:
    payload = fit_probe(
        [{"base_margin": 0.9}, {"base_margin": 0.8}],
        [1, 1],
        trained_at=_FIXED_TRAINED_AT,
    )
    model = ProbeModel.from_dict(payload)

    assert all(math.isfinite(coef) for coef in payload["coefs"])
    assert math.isfinite(payload["intercept"])
    assert model.score({"base_margin": 0.9}) < 0.5


@pytest.mark.parametrize(
    ("rows", "labels"),
    [
        ([], []),
        ([{"base_margin": 0.5}], []),
        ([{"base_margin": 0.5}], [2]),
        ([{"base_margin": None}], [1]),
    ],
)
def test_fit_probe_rejects_unusable_training_input(
    rows: list[dict[str, Any]], labels: list[int]
) -> None:
    with pytest.raises(ValueError):
        fit_probe(rows, labels, trained_at=_FIXED_TRAINED_AT)


def test_fit_probe_requires_requested_features_to_exist() -> None:
    with pytest.raises(ValueError):
        fit_probe(
            [{"base_margin": 0.5}, {"base_margin": 0.1}],
            [1, 0],
            feature_names=["base_margin", "kl_base_shadow"],
            trained_at=_FIXED_TRAINED_AT,
        )


def test_fit_payload_written_with_plain_json_is_still_loadable(tmp_path: Path) -> None:
    payload = fit_probe(
        [{"base_margin": 0.9}, {"base_margin": 0.2}],
        [1, 0],
        trained_at=_FIXED_TRAINED_AT,
    )
    path = tmp_path / "plain.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    model = ProbeModel.load(path)

    # base_margin 0.9 is labelled should_handoff -> low trust; 0.2 stays on edge.
    assert model.score({"base_margin": 0.9}) < 0.5 < model.score({"base_margin": 0.2})


def test_load_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ProbeModel.load(tmp_path / "missing.json")


def test_load_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ProbeFormatError):
        ProbeModel.load(path)


def test_load_rejects_an_unsupported_version(tmp_path: Path) -> None:
    payload = _one_feature_payload(version="handoff-probe/v99")
    path = tmp_path / "future.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ProbeVersionError, match="v99"):
        ProbeModel.load(path)


def test_load_rejects_a_missing_schema_key(tmp_path: Path) -> None:
    payload = _one_feature_payload()
    del payload["calibration"]
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ProbeFormatError, match="calibration"):
        ProbeModel.load(path)


def test_load_rejects_a_tampered_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "probe.json"
    write_probe(_one_feature_payload(), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["intercept"] = 1.0
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ProbeFormatError, match="fingerprint"):
        ProbeModel.load(path)
