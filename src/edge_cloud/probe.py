"""Offline-calibrated prefix-trust probe (离线校准的 probe 产物).

A probe is a plain logistic-regression model over handoff features. It is
trained offline by the evaluation harness (``scripts/exp9_fit_probe.py``) and
loaded read-only on the serving path; the serving path never trains. The JSON
artifact is intentionally dependency-free (no numpy/sklearn) and stable:

    version          artifact schema version
    feature_names    ordered input features (``HandoffFeatures`` field names)
    means, stds      standardization statistics per feature
    coefs, intercept logistic weights on the standardized features
    threshold        calibrated operating point stored with the model
    calibration      trust bins; ``positive_rate`` is the observed handoff
                     rate in that bin (mean of the original labels), kept under
                     its historical name for schema compatibility
    fingerprint      SHA-256 of the canonical payload (excluding fingerprint
                     and trained_at), so identical fits share a fingerprint
    required_features features that must be present at scoring time
    trained_at       ISO-8601 UTC timestamp (informational; not fingerprinted)

Score direction: ``ProbeModel.score`` returns **prefix trust**, so a high score
means "keep generating on the edge" and a low score means "this prefix should
hand off". Candidate labels keep the spec convention ``1 =
prefix_should_handoff``; ``fit_probe`` therefore fits against ``1 - label`` and
never against the raw label. ``ProbePolicy`` requests a handoff when trust
falls below its threshold.

Missing-value semantics: a required feature that is ``None``/absent makes
``ProbeModel.score`` raise ``ProbeFeatureError``; it is never read as 0. An
optional feature that is missing is imputed at its training mean, i.e. its
standardized value is 0, which is also not the same as feeding a raw 0.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROBE_VERSION = "handoff-probe/v1"
SUPPORTED_PROBE_VERSIONS = frozenset({PROBE_VERSION})

_ARTIFACT_KEYS = frozenset(
    {
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
)
_CALIBRATION_KEYS = frozenset(
    {"lower", "upper", "count", "positives", "positive_rate", "mean_score"}
)

_MISSING = object()
_RIDGE = 1e-6
_MAX_IRLS_ITER = 100
_IRLS_TOL = 1e-9
_SINGULAR_EPS = 1e-12


class ProbeFormatError(ValueError):
    """The probe artifact is missing fields, malformed, or tampered with."""


class ProbeVersionError(ProbeFormatError):
    """The probe artifact was written by an unsupported schema version."""


class ProbeFeatureError(ValueError):
    """A required feature is missing or not a finite number."""


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def probe_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 fingerprint of a probe payload.

    ``fingerprint`` and ``trained_at`` are excluded so refitting the same data
    at a different time produces the same fingerprint.
    """
    body = {
        key: value
        for key, value in payload.items()
        if key not in {"fingerprint", "trained_at"}
    }
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def _require_mapping(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ProbeFormatError("probe payload 必须是 JSON 对象")
    return payload


def _require_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ProbeFormatError(f"probe 字段 {key} 必须是非空字符串")
    return value


def _require_str_list(payload: Mapping[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ProbeFormatError(f"probe 字段 {key} 必须是非空字符串列表")
    items = list(value)
    if len(set(items)) != len(items):
        raise ProbeFormatError(f"probe 字段 {key} 不能包含重复项")
    return items


def _require_float(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProbeFormatError(f"probe 字段 {where} 必须是数值")
    number = float(value)
    if not math.isfinite(number):
        raise ProbeFormatError(f"probe 字段 {where} 必须是有限数值")
    return number


def _require_float_list(
    payload: Mapping[str, Any], key: str, *, expected_length: int
) -> list[float]:
    value = payload.get(key)
    if not isinstance(value, (list, tuple)):
        raise ProbeFormatError(f"probe 字段 {key} 必须是数值列表")
    if len(value) != expected_length:
        raise ProbeFormatError(
            f"probe 字段 {key} 长度必须为 {expected_length}, 实际为 {len(value)}"
        )
    return [_require_float(item, where=f"{key}[{index}]") for index, item in enumerate(value)]


def _validate_calibration(value: Any) -> list[dict[str, Any]]:
    """Validate calibration bins.

    ``positive_rate`` (and ``positives``) count rows labelled
    ``prefix_should_handoff=1`` inside the trust bin, so the value is the
    observed handoff rate, not a probability that the prefix is trustworthy.
    """
    if not isinstance(value, (list, tuple)):
        raise ProbeFormatError("probe 字段 calibration 必须是列表")
    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(value):
        where = f"calibration[{index}]"
        if not isinstance(entry, Mapping):
            raise ProbeFormatError(f"probe 字段 {where} 必须是对象")
        missing = sorted(_CALIBRATION_KEYS - set(entry))
        if missing:
            raise ProbeFormatError(f"probe 字段 {where} 缺少: {', '.join(missing)}")
        lower = _require_float(entry["lower"], where=f"{where}.lower")
        upper = _require_float(entry["upper"], where=f"{where}.upper")
        if upper < lower:
            raise ProbeFormatError(f"probe 字段 {where} 的 upper 小于 lower")
        count = entry["count"]
        positives = entry["positives"]
        for name, number in (("count", count), ("positives", positives)):
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise ProbeFormatError(f"probe 字段 {where}.{name} 必须是非负整数")
        if positives > count:
            raise ProbeFormatError(f"probe 字段 {where}.positives 不能大于 count")
        positive_rate = entry["positive_rate"]
        mean_score = entry["mean_score"]
        if positive_rate is not None:
            positive_rate = _require_float(positive_rate, where=f"{where}.positive_rate")
            if not 0.0 <= positive_rate <= 1.0:
                raise ProbeFormatError(f"probe 字段 {where}.positive_rate 必须在 [0, 1] 内")
        if mean_score is not None:
            mean_score = _require_float(mean_score, where=f"{where}.mean_score")
            if not 0.0 <= mean_score <= 1.0:
                raise ProbeFormatError(f"probe 字段 {where}.mean_score 必须在 [0, 1] 内")
        entries.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "positives": positives,
                "positive_rate": positive_rate,
                "mean_score": mean_score,
            }
        )
    return entries


@dataclass(frozen=True, slots=True)
class ProbeModel:
    """A loaded probe artifact; ``score`` returns prefix trust in [0, 1].

    The score is **trust**, not handoff probability: values near 1 mean the
    prefix is likely safe to keep on the edge, values near 0 mean it should be
    handed off. ``score`` accepts a mapping of feature name to value or any
    object exposing the feature names as attributes (``HandoffFeatures``).
    Booleans are encoded as 1/0. A required feature that is missing, ``None`` or
    non-finite raises ``ProbeFeatureError`` — missing information is never
    treated as 0. An optional missing feature is mean-imputed (standardized
    value 0).
    """

    version: str
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    coefs: tuple[float, ...]
    intercept: float
    threshold: float
    calibration: tuple[dict[str, Any], ...]
    fingerprint: str
    required_features: tuple[str, ...]
    trained_at: str

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, verify_fingerprint: bool = True
    ) -> ProbeModel:
        """Validate a probe payload and build a model.

        Raises:
            ProbeVersionError: the payload declares an unsupported version.
            ProbeFormatError: required fields are missing, malformed, or the
                stored fingerprint does not match the payload.
        """
        data = _require_mapping(payload)
        missing = sorted(_ARTIFACT_KEYS - set(data))
        if missing:
            raise ProbeFormatError(f"probe 缺少字段: {', '.join(missing)}")

        version = _require_str(data, "version")
        if version not in SUPPORTED_PROBE_VERSIONS:
            raise ProbeVersionError(
                f"probe version {version!r} 不受支持, 期望 "
                f"{', '.join(sorted(SUPPORTED_PROBE_VERSIONS))!r}"
            )

        feature_names = _require_str_list(data, "feature_names")
        if not feature_names:
            raise ProbeFormatError("probe 字段 feature_names 不能为空")
        means = _require_float_list(data, "means", expected_length=len(feature_names))
        stds = _require_float_list(data, "stds", expected_length=len(feature_names))
        if any(std <= 0.0 for std in stds):
            raise ProbeFormatError("probe 字段 stds 必须全部大于 0")
        coefs = _require_float_list(data, "coefs", expected_length=len(feature_names))
        intercept = _require_float(data.get("intercept"), where="intercept")
        threshold = _require_float(data.get("threshold"), where="threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ProbeFormatError("probe 字段 threshold 必须在 [0, 1] 内")

        required_features = _require_str_list(data, "required_features")
        unknown = sorted(set(required_features) - set(feature_names))
        if unknown:
            raise ProbeFormatError(
                f"probe 字段 required_features 包含未知特征: {', '.join(unknown)}"
            )

        calibration = _validate_calibration(data.get("calibration"))
        trained_at = _require_str(data, "trained_at")

        fingerprint = _require_str(data, "fingerprint")
        if verify_fingerprint:
            expected = probe_fingerprint(data)
            if fingerprint != expected:
                raise ProbeFormatError(
                    "probe fingerprint 不匹配: 产物可能被修改或写入时未使用稳定 JSON"
                )

        return cls(
            version=version,
            feature_names=tuple(feature_names),
            means=tuple(means),
            stds=tuple(stds),
            coefs=tuple(coefs),
            intercept=intercept,
            threshold=threshold,
            calibration=tuple(calibration),
            fingerprint=fingerprint,
            required_features=tuple(required_features),
            trained_at=trained_at,
        )

    @classmethod
    def load(cls, path: str | Path) -> ProbeModel:
        """Read and validate a probe JSON artifact.

        Raises:
            FileNotFoundError/OSError: the file cannot be read.
            ProbeFormatError: the file is not valid JSON or fails validation.
            ProbeVersionError: the artifact version is unsupported.
        """
        text = Path(path).read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProbeFormatError(f"probe JSON 无法解析: {exc}") from exc
        return cls.from_dict(payload)

    def _feature_value(self, features: Any, name: str) -> Any:
        if isinstance(features, Mapping):
            return features.get(name, _MISSING)
        return getattr(features, name, _MISSING)

    def score(self, features: Any) -> float:
        """Return ``sigmoid(w·z + b)``: the prefix trust for one observation.

        High trust (near 1) means the prefix looks safe to continue on the edge;
        low trust (near 0) means it should hand off. The model is fitted against
        ``1 - prefix_should_handoff``, so this is never a handoff probability.

        Args:
            features: Mapping or object with attributes named by
                ``feature_names``.

        Returns:
            Prefix trust in (0, 1).

        Raises:
            ProbeFeatureError: a required feature is missing, ``None``,
                non-numeric, or non-finite.
        """
        total = self.intercept
        missing: list[str] = []
        for name, mean, std, coef in zip(
            self.feature_names, self.means, self.stds, self.coefs, strict=True
        ):
            value = self._feature_value(features, name)
            if value is _MISSING or value is None:
                if name in self.required_features:
                    missing.append(name)
                continue
            if isinstance(value, bool):
                number = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                number = float(value)
            else:
                raise ProbeFeatureError(f"probe 特征 {name} 必须是数值或 None")
            if not math.isfinite(number):
                if name in self.required_features:
                    missing.append(name)
                continue
            total += coef * ((number - mean) / std)
        if missing:
            raise ProbeFeatureError(f"probe 必需特征缺失: {', '.join(missing)}")
        return _sigmoid(total)


def _solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve ``matrix @ x = rhs`` with partial-pivoting Gaussian elimination."""
    size = len(rhs)
    rows = [row[:] + [rhs[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) < _SINGULAR_EPS:
            raise ValueError("probe IRLS 法方程奇异, 无法求解")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        inverse = 1.0 / rows[column][column]
        for row in range(column + 1, size):
            factor = rows[row][column] * inverse
            if factor == 0.0:
                continue
            for index in range(column, size + 1):
                rows[row][index] -= factor * rows[column][index]
    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        remainder = sum(rows[row][index] * solution[index] for index in range(row + 1, size))
        solution[row] = (rows[row][size] - remainder) / rows[row][row]
    return solution


def _fit_irls(
    standardized: list[list[float]], labels: list[float], *, ridge: float
) -> tuple[list[float], float]:
    """Fit logistic regression by IRLS from a deterministic all-zero start."""
    width = len(standardized[0])
    params = [0.0] * (width + 1)  # params[0] is the intercept.
    for _ in range(_MAX_IRLS_ITER):
        gradient = [0.0] * (width + 1)
        hessian = [[0.0] * (width + 1) for _ in range(width + 1)]
        for row, label in zip(standardized, labels, strict=True):
            eta = params[0] + sum(
                params[index + 1] * value for index, value in enumerate(row)
            )
            probability = _sigmoid(eta)
            residual = probability - label
            weight = max(probability * (1.0 - probability), 1e-12)
            gradient[0] += residual
            hessian[0][0] += weight
            for left in range(width):
                gradient[left + 1] += residual * row[left]
                term = weight * row[left]
                hessian[0][left + 1] += term
                hessian[left + 1][0] += term
                for right in range(width):
                    hessian[left + 1][right + 1] += term * row[right]
        for index in range(width + 1):
            hessian[index][index] += ridge
            gradient[index] += ridge * params[index]
        step = _solve_linear(hessian, [-value for value in gradient])
        params = [value + delta for value, delta in zip(params, step, strict=True)]
        if max(abs(delta) for delta in step) < _IRLS_TOL:
            break
    return params[1:], params[0]


def _calibration_table(
    probabilities: Sequence[float], labels: Sequence[float], bins: int
) -> list[dict[str, Any]]:
    """Bin trust scores and report the observed handoff rate per bin.

    ``labels`` are the original ``prefix_should_handoff`` labels, so
    ``positive_rate`` is the fraction of rows in that trust bin that should
    hand off. The field name is kept for schema compatibility.
    """
    table: list[dict[str, Any]] = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            (probability, label)
            for probability, label in zip(probabilities, labels, strict=True)
            if lower <= probability < upper or (index == bins - 1 and probability == upper)
        ]
        count = len(members)
        positives = int(sum(label for _, label in members))
        positive_rate = round(positives / count, 6) if count else None
        mean_score = (
            round(sum(probability for probability, _ in members) / count, 6)
            if count
            else None
        )
        table.append(
            {
                "lower": round(lower, 12),
                "upper": round(upper, 12),
                "count": count,
                "positives": positives,
                "positive_rate": positive_rate,
                "mean_score": mean_score,
            }
        )
    return table


def _label_value(value: Any, *, index: int) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and float(value) in (0.0, 1.0):
        return float(value)
    raise ValueError(f"labels[{index}] 必须是 0 或 1, 实际为 {value!r}")


def fit_probe(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Any],
    *,
    version: str = PROBE_VERSION,
    threshold: float = 0.5,
    feature_names: Sequence[str] | None = None,
    required_features: Sequence[str] | None = None,
    calibration_bins: int = 10,
    trained_at: str | None = None,
) -> dict[str, Any]:
    """Fit a standardized logistic-regression probe on offline rows.

    Pure Python (IRLS with a tiny ridge), deterministic: the fit starts from all
    zeros, feature order defaults to sorted names, and repeating the same call
    yields identical parameters and fingerprint.

    Labels use the candidate-export convention ``1 = prefix_should_handoff``,
    but the model is fitted against ``1 - label`` because it predicts **prefix
    trust**: high score = safe to continue on the edge, low score = should hand
    off. ``ProbePolicy`` then hands off when trust is below its threshold.

    Args:
        rows: Feature mappings. ``None`` values are mean-imputed; a feature that
            is ``None`` in every row is unusable and raises ``ValueError``.
        labels: Binary ``prefix_should_handoff`` labels, one per row
            (``True``/``False`` or 0/1); the fit target is ``1 - label``.
        version: Artifact schema version to record. Only values in
            ``SUPPORTED_PROBE_VERSIONS`` can be loaded back; use the default
            unless a newer schema exists.
        threshold: Operating point stored in the artifact.
        feature_names: Feature order; defaults to the sorted keys of row 0.
        required_features: Features required at scoring time; defaults to every
            fitted feature.
        calibration_bins: Number of equal-width trust bins (at least 1); each
            reports ``positive_rate`` = observed handoff rate in that bin.
        trained_at: ISO timestamp override for deterministic artifacts; defaults
            to the current UTC time.

    Returns:
        The artifact dict documented at module level. ``score`` on this model
        returns trust, not the probability of the input label. Write the
        artifact with ``write_probe`` to keep the fingerprint and JSON
        formatting consistent.

    Raises:
        ValueError: empty/ragged/malformed training data, no usable feature, or
            invalid labels.
    """
    row_list = [dict(row) for row in rows]
    if not row_list:
        raise ValueError("probe 训练至少需要一行数据")
    if len(labels) != len(row_list):
        raise ValueError("rows 与 labels 长度必须一致")
    if not isinstance(version, str) or not version:
        raise ValueError("version 必须是非空字符串")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold 必须在 [0, 1] 内")
    if calibration_bins < 1:
        raise ValueError("calibration_bins 必须至少为 1")

    names = (
        list(feature_names)
        if feature_names is not None
        else sorted(row_list[0].keys())
    )
    if not names:
        raise ValueError("probe 训练需要至少一个特征")
    if len(set(names)) != len(names):
        raise ValueError("feature_names 不能包含重复项")
    if required_features is None:
        required = tuple(names)
    else:
        required = tuple(required_features)
        unknown = sorted(set(required) - set(names))
        if unknown:
            raise ValueError(f"required_features 包含未知特征: {', '.join(unknown)}")

    label_values = [_label_value(value, index=index) for index, value in enumerate(labels)]
    # Candidate labels are 1 = should_handoff; trust is the opposite.
    handoff_count = sum(label_values)
    continue_count = len(row_list) - handoff_count
    trust_labels = [1.0 - value for value in label_values]

    means: list[float] = []
    stds: list[float] = []
    for name in names:
        values: list[float] = []
        for row in row_list:
            value = row.get(name)
            if value is None:
                continue
            if isinstance(value, bool):
                values.append(1.0 if value else 0.0)
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"特征 {name} 必须全部是有限数值或 None")
            values.append(float(value))
        if not values:
            raise ValueError(f"特征 {name} 全部缺失, 无法训练")
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        means.append(mean)
        stds.append(math.sqrt(variance) if variance > 0.0 else 1.0)

    raw_rows: list[list[float]] = []
    standardized: list[list[float]] = []
    for row in row_list:
        raw: list[float] = []
        z_row: list[float] = []
        for name, mean, std in zip(names, means, stds, strict=True):
            value = row.get(name)
            if value is None:
                number = mean
            elif isinstance(value, bool):
                number = 1.0 if value else 0.0
            else:
                number = float(value)
            raw.append(number)
            z_row.append((number - mean) / std)
        raw_rows.append(raw)
        standardized.append(z_row)

    if continue_count == 0 or handoff_count == 0:
        # One-class data has no identifiable slope; record a smoothed trust base
        # rate instead of letting the intercept diverge. All-handoff rows must
        # land near 0 trust and all-continue rows near 1.
        intercept = math.log((continue_count + 0.5) / (handoff_count + 0.5))
        coefs = [0.0] * len(names)
    else:
        coefs, intercept = _fit_irls(standardized, trust_labels, ridge=_RIDGE)

    rounded_coefs = [round(coef, 12) for coef in coefs]
    rounded_intercept = round(intercept, 12)
    rounded_means = [round(mean, 12) for mean in means]
    rounded_stds = [round(std, 12) for std in stds]

    probabilities = [
        _sigmoid(
            rounded_intercept
            + sum(
                coef * ((number - mean) / std)
                for number, mean, std, coef in zip(
                    row, rounded_means, rounded_stds, rounded_coefs, strict=True
                )
            )
        )
        for row in raw_rows
    ]

    payload: dict[str, Any] = {
        "version": version,
        "feature_names": names,
        "means": rounded_means,
        "stds": rounded_stds,
        "coefs": rounded_coefs,
        "intercept": rounded_intercept,
        "threshold": float(threshold),
        "calibration": _calibration_table(probabilities, label_values, calibration_bins),
        "fingerprint": "",
        "required_features": [name for name in names if name in required],
        "trained_at": trained_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    payload["fingerprint"] = probe_fingerprint(payload)
    return payload


def write_probe(payload: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    """Write a probe payload as stable, sorted JSON and return it.

    The fingerprint is recomputed from the payload (ignoring ``fingerprint`` and
    ``trained_at``) and the result is validated with ``ProbeModel.from_dict``
    before anything is written.
    """
    data = dict(payload)
    data["fingerprint"] = probe_fingerprint(data)
    ProbeModel.from_dict(data)
    text = (
        json.dumps(
            data,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return data
