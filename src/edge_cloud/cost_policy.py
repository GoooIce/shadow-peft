"""纯成本策略拟合器（cost-aware task thresholds）.

The fitter is deliberately independent from ``Router`` and every backend: it
only consumes replay samples with counterfactual labels and returns a policy
that can be rendered as ``EdgeConfig`` thresholds. Costs come from the caller
(``CostModel`` in the experiment); this module performs no I/O and uses no
randomness. Only the evaluator's bootstrap uses an explicit seed.

Decision convention: a sample stays on the edge when ``score >= threshold``
and escalates otherwise. Higher thresholds therefore escalate more, matching
the router's legacy ``score < escalation_threshold`` rule.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

_PRECISION = 9
_SPLITS = ("train", "val")
_THRESHOLD_STEP = 0.05
DEFAULT_THRESHOLD_GRID: tuple[float, ...] = tuple(
    round(index * _THRESHOLD_STEP, _PRECISION) for index in range(21)
)
TIE_BREAK_HIGHER_THRESHOLD = "higher_threshold"
FALLBACK_INSUFFICIENT_SAMPLES = "insufficient_samples"


def _round(value: float) -> float:
    return round(float(value), _PRECISION)


def _finite(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数字") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} 必须是有限数值")
    return parsed


def _unit(value: Any, name: str) -> float:
    parsed = _finite(value, name)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} 必须在 [0, 1] 内")
    return parsed


def _non_negative(value: Any, name: str) -> float:
    parsed = _finite(value, name)
    if parsed < 0.0:
        raise ValueError(f"{name} 不能为负数")
    return parsed


def _non_negative_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if parsed < 0:
        raise ValueError(f"{name} 不能为负数")
    return parsed


def _integer(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是整数") from exc


def _flag(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} 必须是布尔值")


def _task_name(value: Any, name: str) -> str:
    task = str(value).strip()
    if not task:
        raise ValueError(f"{name} 不能为空")
    return task


def _task_tuple(values: Iterable[str] | str) -> tuple[str, ...]:
    if isinstance(values, str):
        items: Iterable[str] = (values,)
    else:
        try:
            items = list(values)
        except TypeError as exc:
            raise ValueError("require_cloud_tasks 必须是任务名序列") from exc
    seen: list[str] = []
    for item in items:
        task = _task_name(item, "require_cloud_tasks")
        if task not in seen:
            seen.append(task)
    return tuple(seen)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile_sorted(ordered: Sequence[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = max(0, min(len(ordered) - 1, int(math.floor(position))))
    upper = max(0, min(len(ordered) - 1, int(math.ceil(position))))
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    return _round(ordered[index])


@dataclass(frozen=True, slots=True)
class CostSample:
    """One replay row with counterfactual edge/cloud outcomes and costs.

    ``split`` is explicit: the fitter never splits data itself. Rows without a
    cloud label cannot be represented here and must stay out of training.
    """

    task: str
    score: float
    edge_correct: bool
    cloud_correct: bool
    edge_cost: float
    escalated_cost: float
    cloud_only_cost: float
    split: str = "train"
    sample_id: str | None = None
    edge_latency_ms: float | None = None
    escalated_latency_ms: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", _task_name(self.task, "task"))
        object.__setattr__(self, "score", _unit(self.score, "score"))
        object.__setattr__(self, "edge_correct", _flag(self.edge_correct, "edge_correct"))
        object.__setattr__(
            self, "cloud_correct", _flag(self.cloud_correct, "cloud_correct")
        )
        object.__setattr__(
            self, "edge_cost", _non_negative(self.edge_cost, "edge_cost")
        )
        object.__setattr__(
            self,
            "escalated_cost",
            _non_negative(self.escalated_cost, "escalated_cost"),
        )
        object.__setattr__(
            self,
            "cloud_only_cost",
            _non_negative(self.cloud_only_cost, "cloud_only_cost"),
        )
        split = str(self.split).strip().lower()
        if split not in _SPLITS:
            raise ValueError(f"split 必须是 {'|'.join(_SPLITS)} 之一")
        object.__setattr__(self, "split", split)
        if self.sample_id is not None:
            object.__setattr__(self, "sample_id", str(self.sample_id))
        if self.edge_latency_ms is not None:
            object.__setattr__(
                self,
                "edge_latency_ms",
                _non_negative(self.edge_latency_ms, "edge_latency_ms"),
            )
        if self.escalated_latency_ms is not None:
            object.__setattr__(
                self,
                "escalated_latency_ms",
                _non_negative(self.escalated_latency_ms, "escalated_latency_ms"),
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CostSample:
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in payload.items() if key in allowed})


@dataclass(frozen=True, slots=True)
class CostPolicy:
    """Fitted thresholds plus the cost weight that produced them."""

    global_threshold: float
    thresholds_by_task: Mapping[str, float] | None = None
    cost_weight: float = 0.0
    always_cloud_tasks: Iterable[str] | str = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "global_threshold", _unit(self.global_threshold, "global_threshold")
        )
        raw = {} if self.thresholds_by_task is None else self.thresholds_by_task
        if not isinstance(raw, Mapping):
            raise ValueError("thresholds_by_task 必须是对象")
        thresholds: dict[str, float] = {}
        for task, value in raw.items():
            name = _task_name(task, "thresholds_by_task.task")
            thresholds[name] = _unit(value, f"thresholds_by_task.{name}")
        object.__setattr__(self, "thresholds_by_task", thresholds)
        object.__setattr__(
            self, "cost_weight", _non_negative(self.cost_weight, "cost_weight")
        )
        object.__setattr__(
            self, "always_cloud_tasks", _task_tuple(self.always_cloud_tasks)
        )

    def is_always_cloud(self, task: str) -> bool:
        return str(task) in self.always_cloud_tasks

    def threshold_for(self, task: str) -> float | None:
        """Effective threshold, or ``None`` for tasks that always escalate."""

        task = str(task)
        if task in self.always_cloud_tasks:
            return None
        return self.thresholds_by_task.get(task, self.global_threshold)

    def should_escalate(self, task: str, score: float) -> bool:
        """Legacy router decision: escalate when the score is below the threshold."""

        if self.is_always_cloud(task):
            return True
        threshold = self.threshold_for(task)
        return _unit(score, "score") < threshold

    def to_config(self) -> dict[str, Any]:
        return {
            "escalation_threshold": self.global_threshold,
            "escalation_threshold_by_task": dict(self.thresholds_by_task),
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, Any],
        *,
        cost_weight: float = 0.0,
        always_cloud_tasks: Iterable[str] | str = (),
    ) -> CostPolicy:
        if not isinstance(payload, Mapping):
            raise ValueError("cost policy 配置必须是对象")
        if "escalation_threshold" not in payload:
            raise ValueError("cost policy 配置缺少 escalation_threshold")
        thresholds = payload.get("escalation_threshold_by_task") or {}
        return cls(
            global_threshold=payload["escalation_threshold"],
            thresholds_by_task=thresholds,
            cost_weight=payload.get("cost_weight", cost_weight),
            always_cloud_tasks=payload.get("always_cloud_tasks", always_cloud_tasks),
        )


@dataclass(frozen=True, slots=True)
class TaskFitReport:
    """Fit diagnostics for one task (or the global aggregate)."""

    task: str
    n: int
    threshold: float | None
    accuracy: float
    cost: float
    cost_norm: float
    objective: float
    escalation_rate: float
    cloud_calls: int = 0
    fallback_reason: str | None = None
    always_cloud: bool = False
    tie_break: str | None = None

    @property
    def cost_per_request(self) -> float:
        return self.cost

    @property
    def total_cost(self) -> float:
        return _round(self.cost * self.n)

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass(frozen=True, slots=True)
class CostPolicyFit:
    """Result of ``fit_cost_policy``; ``policy`` renders the deployable subset."""

    global_threshold: float
    thresholds_by_task: dict[str, float]
    cost_weight: float
    min_samples: int
    always_cloud_tasks: tuple[str, ...]
    per_task: dict[str, TaskFitReport]
    global_report: TaskFitReport
    threshold_grid: tuple[float, ...]
    tie_break: str | None = None

    @property
    def policy(self) -> CostPolicy:
        return CostPolicy(
            global_threshold=self.global_threshold,
            thresholds_by_task=self.thresholds_by_task,
            cost_weight=self.cost_weight,
            always_cloud_tasks=self.always_cloud_tasks,
        )

    @property
    def task_reports(self) -> dict[str, TaskFitReport]:
        return self.per_task

    @property
    def reports(self) -> dict[str, TaskFitReport]:
        return self.per_task

    @property
    def global_fit(self) -> TaskFitReport:
        return self.global_report

    def to_policy(self) -> CostPolicy:
        return self.policy

    def to_config(self) -> dict[str, Any]:
        return self.policy.to_config()

    def threshold_for(self, task: str) -> float | None:
        return self.policy.threshold_for(task)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TaskEvalReport:
    """Evaluation metrics for one task on one split."""

    task: str
    n: int
    accuracy: float
    cost: float
    cost_norm: float
    objective: float
    escalation_rate: float
    cloud_calls: int = 0
    threshold: float | None = None
    always_cloud: bool = False
    latency_p95_ms: float | None = None

    @property
    def cost_per_request(self) -> float:
        return self.cost

    @property
    def total_cost(self) -> float:
        return _round(self.cost * self.n)

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass(frozen=True, slots=True)
class CostPolicyReport:
    """Validation-split metrics for one policy, including bootstrap intervals."""

    split: str
    n: int
    accuracy: float
    cost: float
    cost_norm: float
    objective: float
    escalation_rate: float
    cloud_calls: int
    per_task: dict[str, TaskEvalReport]
    accuracy_ci: dict[str, Any]
    cost_ci: dict[str, Any]
    cost_weight: float
    global_threshold: float
    thresholds_by_task: dict[str, float]
    latency_p95_ms: float | None = None

    @property
    def cost_per_request(self) -> float:
        return self.cost

    @property
    def total_cost(self) -> float:
        return _round(self.cost * self.n)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_grid(values: Iterable[float] | None) -> tuple[float, ...]:
    if values is None:
        return DEFAULT_THRESHOLD_GRID
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError("threshold_grid 必须是阈值序列") from exc
    if not items:
        raise ValueError("threshold_grid 不能为空")
    grid = sorted({_round(_unit(value, "threshold_grid")) for value in items})
    return tuple(grid)


def _normalize_samples(samples: Iterable[CostSample]) -> list[CostSample]:
    try:
        items = list(samples)
    except TypeError as exc:
        raise ValueError("samples 必须是 CostSample 序列") from exc
    for item in items:
        if not isinstance(item, CostSample):
            raise TypeError("samples 必须包含 CostSample")
    return items


def _is_escalated(item: CostSample, threshold: float | None, always_cloud: bool) -> bool:
    if always_cloud or threshold is None:
        return True
    return item.score < threshold


def _metrics(
    samples: Sequence[CostSample],
    *,
    threshold: float | None,
    cost_weight: float,
    always_cloud: bool = False,
) -> dict[str, Any]:
    n = len(samples)
    if n == 0:
        return {
            "n": 0,
            "accuracy": 0.0,
            "cost": 0.0,
            "cost_norm": 0.0,
            "objective": 0.0,
            "escalation_rate": 0.0,
            "cloud_calls": 0,
        }
    correct: list[float] = []
    chosen_costs: list[float] = []
    cloud_costs: list[float] = []
    cloud_calls = 0
    for item in samples:
        escalated = always_cloud or threshold is None or item.score < threshold
        if escalated:
            cloud_calls += 1
            correct.append(1.0 if item.cloud_correct else 0.0)
            chosen_costs.append(item.escalated_cost)
        else:
            correct.append(1.0 if item.edge_correct else 0.0)
            chosen_costs.append(item.edge_cost)
        cloud_costs.append(item.cloud_only_cost)
    accuracy = _mean(correct)
    cost = _mean(chosen_costs)
    cloud_only_mean = _mean(cloud_costs)
    cost_norm = cost / cloud_only_mean if cloud_only_mean > 0.0 else 0.0
    objective = accuracy - cost_weight * cost_norm
    return {
        "n": n,
        "accuracy": _round(accuracy),
        "cost": _round(cost),
        "cost_norm": _round(cost_norm),
        "objective": _round(objective),
        "escalation_rate": _round(cloud_calls / n),
        "cloud_calls": cloud_calls,
    }


def _fit_threshold(
    samples: Sequence[CostSample], grid: Sequence[float], cost_weight: float
) -> tuple[float, dict[str, Any], str | None]:
    best_threshold = grid[0]
    best_metrics = _metrics(
        samples, threshold=best_threshold, cost_weight=cost_weight
    )
    tie_break: str | None = None
    for threshold in grid[1:]:
        metrics = _metrics(samples, threshold=threshold, cost_weight=cost_weight)
        if metrics["objective"] > best_metrics["objective"]:
            best_threshold = threshold
            best_metrics = metrics
            tie_break = None
        elif metrics["objective"] == best_metrics["objective"]:
            if threshold > best_threshold:
                best_threshold = threshold
                best_metrics = metrics
            tie_break = TIE_BREAK_HIGHER_THRESHOLD
    return best_threshold, best_metrics, tie_break


def _report(
    task: str,
    metrics: dict[str, Any],
    *,
    threshold: float | None,
    fallback_reason: str | None = None,
    always_cloud: bool = False,
    tie_break: str | None = None,
) -> TaskFitReport:
    return TaskFitReport(
        task=task,
        n=metrics["n"],
        threshold=threshold,
        accuracy=metrics["accuracy"],
        cost=metrics["cost"],
        cost_norm=metrics["cost_norm"],
        objective=metrics["objective"],
        escalation_rate=metrics["escalation_rate"],
        cloud_calls=metrics["cloud_calls"],
        fallback_reason=fallback_reason,
        always_cloud=always_cloud,
        tie_break=tie_break,
    )


def fit_cost_policy(
    samples: Sequence[CostSample],
    *,
    cost_weight: float,
    threshold_grid: Iterable[float] | None = None,
    min_samples: int = 30,
    require_cloud_tasks: Iterable[str] | str = (),
) -> CostPolicyFit:
    """Fit a global threshold, then one threshold per task with enough rows."""

    weight = _non_negative(cost_weight, "cost_weight")
    grid = _normalize_grid(threshold_grid)
    minimum = _non_negative_int(min_samples, "min_samples")
    always_cloud = _task_tuple(require_cloud_tasks)
    always_set = set(always_cloud)
    items = _normalize_samples(samples)
    train = [item for item in items if item.split == "train"]
    if not train:
        raise ValueError("训练数据为空")
    eligible = [item for item in train if item.task not in always_set]
    if not eligible:
        raise ValueError("没有可用于拟合阈值的训练样本（全部任务要求升级云端）")

    global_threshold, global_metrics, global_tie = _fit_threshold(
        eligible, grid, weight
    )
    global_report = _report(
        "__global__",
        global_metrics,
        threshold=global_threshold,
        tie_break=global_tie,
    )

    groups: dict[str, list[CostSample]] = {}
    for item in train:
        groups.setdefault(item.task, []).append(item)

    thresholds: dict[str, float] = {}
    reports: dict[str, TaskFitReport] = {}
    for task in sorted(groups):
        group = groups[task]
        if task in always_set:
            metrics = _metrics(
                group, threshold=None, cost_weight=weight, always_cloud=True
            )
            reports[task] = _report(
                task,
                metrics,
                threshold=None,
                always_cloud=True,
            )
            continue
        if len(group) < minimum:
            threshold = global_threshold
            metrics = _metrics(group, threshold=threshold, cost_weight=weight)
            reports[task] = _report(
                task,
                metrics,
                threshold=threshold,
                fallback_reason=FALLBACK_INSUFFICIENT_SAMPLES,
            )
        else:
            threshold, metrics, tie_break = _fit_threshold(group, grid, weight)
            reports[task] = _report(
                task,
                metrics,
                threshold=threshold,
                tie_break=tie_break,
            )
            thresholds[task] = threshold

    return CostPolicyFit(
        global_threshold=global_threshold,
        thresholds_by_task=thresholds,
        cost_weight=weight,
        min_samples=minimum,
        always_cloud_tasks=always_cloud,
        per_task=reports,
        global_report=global_report,
        threshold_grid=grid,
        tie_break=global_tie,
    )


def _bootstrap_ci(
    values: Sequence[float], *, n_boot: int, seed: int, level: float = 0.95
) -> dict[str, Any]:
    data = [float(value) for value in values]
    if not data:
        return {"point": None, "low": None, "high": None, "samples": 0, "level": level}
    point = _mean(data)
    if n_boot < 1:
        return {
            "point": _round(point),
            "low": None,
            "high": None,
            "samples": 0,
            "level": level,
        }
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        total = sum(data[rng.randrange(len(data))] for _ in range(len(data)))
        means.append(total / len(data))
    means.sort()
    alpha = (1.0 - level) / 2.0
    return {
        "point": _round(point),
        "low": _round(_percentile_sorted(means, alpha)),
        "high": _round(_percentile_sorted(means, 1.0 - alpha)),
        "samples": n_boot,
        "level": level,
    }


def _task_eval_report(
    task: str,
    samples: Sequence[CostSample],
    policy: CostPolicy,
    *,
    weight: float,
) -> TaskEvalReport:
    threshold = policy.threshold_for(task)
    always_cloud = policy.is_always_cloud(task)
    metrics = _metrics(
        samples,
        threshold=threshold,
        cost_weight=weight,
        always_cloud=always_cloud,
    )
    latencies: list[float] = []
    for item in samples:
        escalated = _is_escalated(item, threshold, always_cloud)
        latency = item.escalated_latency_ms if escalated else item.edge_latency_ms
        if latency is not None:
            latencies.append(latency)
    return TaskEvalReport(
        task=task,
        n=metrics["n"],
        accuracy=metrics["accuracy"],
        cost=metrics["cost"],
        cost_norm=metrics["cost_norm"],
        objective=metrics["objective"],
        escalation_rate=metrics["escalation_rate"],
        cloud_calls=metrics["cloud_calls"],
        threshold=threshold,
        always_cloud=always_cloud,
        latency_p95_ms=_p95(latencies),
    )


def evaluate_cost_policy(
    policy: CostPolicy | CostPolicyFit,
    samples: Sequence[CostSample],
    *,
    split: str = "val",
    n_boot: int = 1000,
    seed: int = 0,
) -> CostPolicyReport:
    """Evaluate a policy on one explicit split; bootstrap CIs are seeded."""

    if isinstance(policy, CostPolicyFit):
        policy = policy.policy
    if not isinstance(policy, CostPolicy):
        raise TypeError("policy 必须是 CostPolicy 或 CostPolicyFit")
    split_name = str(split).strip().lower()
    if split_name not in _SPLITS:
        raise ValueError(f"split 必须是 {'|'.join(_SPLITS)} 之一")
    boot = _integer(n_boot, "n_boot")
    seed_value = _integer(seed, "seed")
    items = [item for item in _normalize_samples(samples) if item.split == split_name]
    if not items:
        raise ValueError(f"split={split_name} 没有样本")

    weight = policy.cost_weight
    chosen_correct: list[float] = []
    chosen_costs: list[float] = []
    cloud_costs: list[float] = []
    cloud_calls = 0
    latencies: list[float] = []
    groups: dict[str, list[CostSample]] = {}
    for item in items:
        threshold = policy.threshold_for(item.task)
        escalated = _is_escalated(
            item,
            threshold,
            always_cloud=policy.is_always_cloud(item.task),
        )
        groups.setdefault(item.task, []).append(item)
        if escalated:
            cloud_calls += 1
            chosen_correct.append(1.0 if item.cloud_correct else 0.0)
            chosen_costs.append(item.escalated_cost)
            if item.escalated_latency_ms is not None:
                latencies.append(item.escalated_latency_ms)
        else:
            chosen_correct.append(1.0 if item.edge_correct else 0.0)
            chosen_costs.append(item.edge_cost)
            if item.edge_latency_ms is not None:
                latencies.append(item.edge_latency_ms)
        cloud_costs.append(item.cloud_only_cost)

    accuracy = _mean(chosen_correct)
    cost = _mean(chosen_costs)
    cloud_only_mean = _mean(cloud_costs)
    cost_norm = cost / cloud_only_mean if cloud_only_mean > 0.0 else 0.0
    objective = accuracy - weight * cost_norm
    per_task = {
        task: _task_eval_report(task, groups[task], policy, weight=weight)
        for task in sorted(groups)
    }
    return CostPolicyReport(
        split=split_name,
        n=len(items),
        accuracy=_round(accuracy),
        cost=_round(cost),
        cost_norm=_round(cost_norm),
        objective=_round(objective),
        escalation_rate=_round(cloud_calls / len(items)),
        cloud_calls=cloud_calls,
        per_task=per_task,
        accuracy_ci=_bootstrap_ci(chosen_correct, n_boot=boot, seed=seed_value),
        cost_ci=_bootstrap_ci(chosen_costs, n_boot=boot, seed=seed_value),
        cost_weight=weight,
        global_threshold=policy.global_threshold,
        thresholds_by_task=dict(policy.thresholds_by_task),
        latency_p95_ms=_p95(latencies),
    )
