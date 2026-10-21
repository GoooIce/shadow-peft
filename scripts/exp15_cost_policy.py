#!/usr/bin/env python3
"""exp15: cost-aware per-task escalation policy experiment (issue #52).

Spec: issue #48 §6–§7. The experiment fits ``edge_cloud.cost_policy`` policies
over a lambda grid, compares every per-task policy against a global-threshold
baseline fitted on the same train split and the same lambda, and writes a
``policy.json`` that can be pasted into ``EdgeConfig``.

Two data sources:

``--synthetic``
    Deterministic ``CostSample`` rows derived from ``edge_cloud.knowledge``
    entries and the configured ``CostModel``. Synthetic rows only validate the
    pipeline; the report always sets ``claims_gain=false`` and never emits a
    go/no-go verdict.

``--dataset <jsonl>``
    Real replay rows with an explicit ``split``. Only replay data can claim a
    gain, and only when the evidence gate (>= 500 rows and >= ``min_samples``
    train/val rows for every fitted task) passes.

Examples::

    PYTHONPATH=src python scripts/exp15_cost_policy.py --synthetic --smoke \
        --out runs-local/exp15/smoke

    PYTHONPATH=src python scripts/exp15_cost_policy.py \
        --dataset runs/exp15/replay.jsonl --out runs-local/exp15/replay

The script never loads a model, never touches the router and never writes
production configuration. ``policy.json`` is a review artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from edge_cloud import knowledge
from edge_cloud.config import EdgeConfig
from edge_cloud.cost import CostModel
from edge_cloud.cost_policy import (
    DEFAULT_THRESHOLD_GRID,
    CostPolicy,
    CostPolicyFit,
    CostSample,
    evaluate_cost_policy,
    fit_cost_policy,
)
from edge_cloud.handoff_eval import DEFAULT_COST_MODEL, LatencyModel, count_tokens

EXPERIMENT = "exp15-cost-policy"
SCHEMA_VERSION = "exp15-cost-policy/v1"
DEFAULT_SEED = 15
DEFAULT_VARIANTS_PER_ENTRY = 60
DEFAULT_LAMBDA_GRID: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)
DEFAULT_SELECT_LAMBDA = 0.1
DEFAULT_MIN_SAMPLES = 30
DEFAULT_BOOTSTRAP = 200
DEFAULT_MAX_TASK_ESCALATION_RATE = 0.6
CURRENT_DEFAULT_THRESHOLD = 0.62
REPLAY_TOTAL_MIN = 500

_PROMPT_TEMPLATES: tuple[str, ...] = (
    "现场提问 {tag}：请就 {keyword} 给出处理建议。",
    "产线工单 {tag}：值班工程师询问 {keyword}，请按步骤回答。",
    "异常记录 {tag}：涉及 {keyword}，需要结论与依据。",
    "交接复盘 {tag}：{keyword} 的要点是什么？",
)

_REPLAY_REQUIRED_FIELDS: tuple[str, ...] = (
    "task",
    "score",
    "edge_correct",
    "cloud_correct",
    "edge_cost",
    "escalated_cost",
    "cloud_only_cost",
    "split",
)


def _finite_non_negative(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数字") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} 必须是有限非负数")
    return parsed


def _unit(value: Any, name: str) -> float:
    parsed = _finite_non_negative(value, name)
    if parsed > 1.0:
        raise ValueError(f"{name} 必须在 [0, 1] 内")
    return parsed


def build_synthetic_samples(
    *,
    variants_per_entry: int = DEFAULT_VARIANTS_PER_ENTRY,
    seed: int = DEFAULT_SEED,
    cost_model: CostModel | None = None,
    latency_model: LatencyModel | None = None,
) -> list[CostSample]:
    """Build deterministic cost samples from every knowledge-base entry.

    ``score`` is the entry's ``edge_quality`` and labels use the 0.5 cutoff as
    required by the spec. Costs come from ``count_tokens`` and the supplied
    ``CostModel``; latency uses the closed-form ``LatencyModel``. Even variant
    indexes are train rows, odd indexes are val rows, so every task has both
    splits for any ``variants_per_entry >= 2``.
    """

    if int(variants_per_entry) < 2:
        raise ValueError("variants_per_entry 必须 >= 2（train 与 val 各至少一条）")
    model = cost_model or DEFAULT_COST_MODEL
    latency = latency_model or LatencyModel()
    samples: list[CostSample] = []
    for entry in knowledge.ENTRIES:
        for variant in range(int(variants_per_entry)):
            template = _PROMPT_TEMPLATES[(variant + int(seed)) % len(_PROMPT_TEMPLATES)]
            keyword = entry.keywords[(variant + int(seed)) % len(entry.keywords)]
            tag = f"E15-{entry.key}-{variant:03d}"
            prompt = template.format(tag=tag, keyword=keyword)
            prompt_tokens = count_tokens(prompt)
            edge_tokens = count_tokens(entry.edge_answer)
            cloud_tokens = count_tokens(entry.cloud_answer)
            samples.append(
                CostSample(
                    sample_id=f"syn-{entry.key}-{variant:03d}",
                    task=entry.task,
                    score=entry.edge_quality,
                    edge_correct=entry.edge_quality >= 0.5,
                    cloud_correct=entry.cloud_quality >= 0.5,
                    edge_cost=model.estimate(edge_tokens=edge_tokens).estimated_cost,
                    escalated_cost=model.estimate(
                        edge_tokens=edge_tokens,
                        cloud_prefill_tokens=prompt_tokens + edge_tokens,
                        cloud_decode_tokens_expected=cloud_tokens,
                        cloud_calls=1,
                    ).estimated_cost,
                    cloud_only_cost=model.estimate(
                        cloud_prefill_tokens=prompt_tokens,
                        cloud_decode_tokens_expected=cloud_tokens,
                        cloud_calls=1,
                    ).estimated_cost,
                    split="train" if variant % 2 == 0 else "val",
                    edge_latency_ms=latency.estimate(
                        edge_tokens=edge_tokens,
                        cloud_prefill_tokens=0,
                        cloud_decode_tokens=0,
                        cloud_calls=0,
                    ),
                    escalated_latency_ms=latency.estimate(
                        edge_tokens=edge_tokens,
                        cloud_prefill_tokens=prompt_tokens + edge_tokens,
                        cloud_decode_tokens=cloud_tokens,
                        cloud_calls=1,
                    ),
                )
            )
    return samples


def load_cost_samples(path: str | Path) -> list[CostSample]:
    """Read a replay JSONL file; every row must carry an explicit split."""

    samples: list[CostSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"replay 第 {line_number} 行不是合法 JSON: {exc}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"replay 第 {line_number} 行必须是 JSON 对象")
            missing = [name for name in _REPLAY_REQUIRED_FIELDS if name not in row]
            if missing:
                raise ValueError(f"replay 第 {line_number} 行缺少字段: {', '.join(missing)}")
            split = row["split"]
            if not isinstance(split, str) or split.strip().lower() not in {
                "train",
                "val",
            }:
                raise ValueError(
                    f"replay 第 {line_number} 行 split 必须是 train|val，收到 {split!r}"
                )
            try:
                samples.append(CostSample.from_dict(row))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"replay 第 {line_number} 行无效: {exc}") from exc
    if not samples:
        raise ValueError("replay 数据集为空")
    return samples


def _task_sequence(value: Iterable[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items: Iterable[str] = [value]
    else:
        items = value
    seen: list[str] = []
    for item in items:
        task = str(item).strip()
        if task and task not in seen:
            seen.append(task)
    return tuple(seen)


def _counts(samples: Sequence[CostSample]) -> dict[str, Any]:
    tasks: dict[str, dict[str, int]] = {}
    train = 0
    val = 0
    for sample in samples:
        bucket = tasks.setdefault(sample.task, {"train": 0, "val": 0})
        bucket[sample.split] += 1
        if sample.split == "train":
            train += 1
        else:
            val += 1
    return {
        "total": len(samples),
        "train": train,
        "val": val,
        "tasks": {task: tasks[task] for task in sorted(tasks)},
    }


def evaluate_evidence(
    samples: Sequence[CostSample],
    *,
    dataset_source: str,
    fit: CostPolicyFit,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    total_min: int = REPLAY_TOTAL_MIN,
) -> dict[str, Any]:
    """Decide whether the dataset may claim a gain over the baseline.

    Synthetic rows never qualify. Replay rows need at least ``total_min``
    samples and at least ``min_samples`` train/val rows for every task that
    actually received a fitted threshold.
    """

    if dataset_source not in {"synthetic", "replay"}:
        raise ValueError("dataset_source 必须是 synthetic|replay")
    items = list(samples)
    counts = _counts(items)
    reasons: list[str] = []
    requirements: dict[str, Any] = {
        "dataset_source": {"value": dataset_source, "required": "replay"},
        "total_samples": {
            "value": counts["total"],
            "minimum": int(total_min),
            "passed": counts["total"] >= int(total_min),
        },
        "per_task_min_samples": {"value": int(min_samples), "tasks": {}},
    }
    if dataset_source != "replay":
        reasons.append("dataset_source=synthetic：合成知识库只用于管线验证，不能作为收益证据")
    else:
        if counts["total"] < int(total_min):
            reasons.append(f"总样本 {counts['total']} < {int(total_min)}，不足以支撑 go/no-go")
        fitted = sorted(fit.thresholds_by_task)
        if not fitted:
            reasons.append("没有任务达到 min_samples，未产生任何 per-task 阈值")
        for task in fitted:
            train_n = int(fit.per_task[task].n)
            val_n = int(counts["tasks"].get(task, {}).get("val", 0))
            passed = train_n >= int(min_samples) and val_n >= int(min_samples)
            requirements["per_task_min_samples"]["tasks"][task] = {
                "train": train_n,
                "val": val_n,
                "passed": passed,
            }
            if not passed:
                reasons.append(
                    f"task={task} 的 train/val 样本 {train_n}/{val_n} 未同时达到 {int(min_samples)}"
                )
    claims_gain = dataset_source == "replay" and not reasons
    return {
        "claims_gain": claims_gain,
        "evidence": "replay" if claims_gain else "insufficient",
        "reasons": reasons,
        "requirements": requirements,
    }


def _objective(report: Mapping[str, Any]) -> float:
    return float(report.get("objective", 0.0))


def evaluate_go_no_go(
    per_task_val: Mapping[str, Any],
    baseline_val: Mapping[str, Any],
    per_task_train: Mapping[str, Any],
    baseline_train: Mapping[str, Any],
    *,
    fitted_tasks: Sequence[str] = (),
    min_samples: int = DEFAULT_MIN_SAMPLES,
    max_task_escalation_rate: float = DEFAULT_MAX_TASK_ESCALATION_RATE,
    always_cloud_tasks: Sequence[str] | str = (),
) -> dict[str, Any]:
    """Apply spec §7's go/no-go gates to one per-task-vs-global comparison."""

    cap = _unit(max_task_escalation_rate, "max_task_escalation_rate")
    minimum = int(min_samples)
    if minimum < 0:
        raise ValueError("min_samples 不能为负数")
    always_cloud = set(_task_sequence(always_cloud_tasks))

    per_accuracy = float(per_task_val["accuracy"])
    base_accuracy = float(baseline_val["accuracy"])
    per_cost = float(per_task_val["cost"])
    base_cost = float(baseline_val["cost"])
    per_norm = float(per_task_val.get("cost_norm", 0.0))
    base_norm = float(baseline_val.get("cost_norm", 0.0))
    epsilon = 1e-9

    accuracy_within_half_pp = per_accuracy >= base_accuracy - 0.005 - epsilon
    cost_at_most_80pct = per_cost <= base_cost * 0.8 + epsilon
    cost_at_most_105pct = per_cost <= base_cost * 1.05 + epsilon
    accuracy_plus_1pp = per_accuracy >= base_accuracy + 0.01 - epsilon
    primary = (accuracy_within_half_pp and cost_at_most_80pct) or (
        cost_at_most_105pct and accuracy_plus_1pp
    )

    violations: list[dict[str, Any]] = []
    for task, entry in sorted(per_task_val.get("per_task", {}).items()):
        if task in always_cloud:
            continue
        rate = float(entry.get("escalation_rate", 0.0))
        if rate > cap + epsilon:
            violations.append({"task": task, "rate": round(rate, 6)})

    size_failures: list[dict[str, Any]] = []
    val_tasks = per_task_val.get("per_task", {})
    for task in sorted(_task_sequence(fitted_tasks)):
        entry = val_tasks.get(task)
        n = int(entry.get("n", 0)) if entry else 0
        if n < minimum:
            size_failures.append({"task": task, "val": n})

    train_delta = _objective(per_task_train) - _objective(baseline_train)
    val_delta = _objective(per_task_val) - _objective(baseline_val)
    overall_inverted = train_delta > epsilon and val_delta < -epsilon

    inverted_tasks: list[str] = []
    train_tasks = per_task_train.get("per_task", {})
    baseline_train_tasks = baseline_train.get("per_task", {})
    baseline_val_tasks = baseline_val.get("per_task", {})
    for task in sorted(set(val_tasks) & set(train_tasks)):
        if task in always_cloud:
            continue
        if task not in baseline_train_tasks or task not in baseline_val_tasks:
            continue
        task_train_delta = _objective(train_tasks[task]) - _objective(baseline_train_tasks[task])
        task_val_delta = _objective(val_tasks[task]) - _objective(baseline_val_tasks[task])
        if task_train_delta > epsilon and task_val_delta < -epsilon:
            inverted_tasks.append(task)

    checks = {
        "accuracy_within_half_pp": accuracy_within_half_pp,
        "cost_at_most_80pct": cost_at_most_80pct,
        "cost_at_most_105pct": cost_at_most_105pct,
        "accuracy_plus_1pp": accuracy_plus_1pp,
        "primary_condition": primary,
        "no_task_over_escalation_cap": not violations,
        "val_sample_sizes": not size_failures,
        "no_direction_inversion": not overall_inverted and not inverted_tasks,
    }
    failed = [
        name
        for name in (
            "primary_condition",
            "no_task_over_escalation_cap",
            "val_sample_sizes",
            "no_direction_inversion",
        )
        if not checks[name]
    ]
    reasons: list[str] = []
    if not primary:
        reasons.append("未满足同准确率成本 −20% 或同成本准确率 +1pp")
    for violation in violations:
        reasons.append(f"task={violation['task']} val 升级率 {violation['rate']:.3f} > {cap:.3f}")
    for failure in size_failures:
        reasons.append(f"task={failure['task']} val 样本 {failure['val']} < {minimum}")
    if overall_inverted:
        reasons.append(
            f"整体 objective 在 train 提升 {train_delta:+.4f} 但 val 下降 {val_delta:+.4f}"
        )
    if inverted_tasks:
        reasons.append(f"task 方向反转: {', '.join(inverted_tasks)}")

    return {
        "status": "go" if not failed else "no_go",
        "checks": checks,
        "failed_checks": failed,
        "reasons": reasons,
        "deltas": {
            "accuracy_pp": round((per_accuracy - base_accuracy) * 100.0, 6),
            "cost_ratio": (round(per_cost / base_cost, 6) if base_cost > 0.0 else None),
            "cost_norm_delta": round(per_norm - base_norm, 9),
            "train_objective_delta": round(train_delta, 9),
            "val_objective_delta": round(val_delta, 9),
        },
        "escalation_violations": violations,
        "escalation_size_failures": size_failures,
        "inverted_tasks": inverted_tasks,
        "overall_inverted": overall_inverted,
        "always_cloud_exempt_tasks": sorted(always_cloud),
    }


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Everything ``write_artifacts`` needs, plus the report for tests."""

    report: dict[str, Any]
    policy: dict[str, Any]
    samples: tuple[CostSample, ...]
    summary_rows: tuple[dict[str, Any], ...] = ()
    curve_rows: tuple[dict[str, Any], ...] = ()


def _normalize_lambdas(values: Iterable[float]) -> list[float]:
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError("λ 网格必须是数值序列") from exc
    if not items:
        raise ValueError("λ 网格不能为空")
    grid = sorted({_finite_non_negative(value, "lambda") for value in items})
    return [round(value, 9) for value in grid]


def _normalize_threshold_grid(values: Iterable[float]) -> tuple[float, ...]:
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError("阈值网格必须是数值序列") from exc
    if not items:
        raise ValueError("阈值网格不能为空")
    return tuple(sorted({round(_unit(value, "threshold"), 9) for value in items}))


def _references(
    samples: Sequence[CostSample],
    *,
    select_lambda: float,
    require_cloud_tasks: Sequence[str],
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    val_tasks = sorted({sample.task for sample in samples if sample.split == "val"})
    policies = {
        "edge_only": CostPolicy(
            global_threshold=0.0,
            thresholds_by_task={},
            cost_weight=select_lambda,
            always_cloud_tasks=(),
        ),
        "cloud_only": CostPolicy(
            global_threshold=0.0,
            thresholds_by_task={},
            cost_weight=select_lambda,
            always_cloud_tasks=val_tasks,
        ),
        "current_0.62": CostPolicy(
            global_threshold=CURRENT_DEFAULT_THRESHOLD,
            thresholds_by_task={},
            cost_weight=select_lambda,
            always_cloud_tasks=require_cloud_tasks,
        ),
    }
    return {
        name: evaluate_cost_policy(
            policy, samples, split="val", n_boot=bootstrap, seed=seed
        ).to_dict()
        for name, policy in policies.items()
    }


def _caveats(dataset_source: str) -> list[str]:
    caveats = [
        "policy.json 只是审核产物；本脚本不修改生产配置、不写 Router 状态、不联网。",
        "CostModel 的价格是配置值，不是云厂商账单；判断时同时看 cost_norm 与原始 token/call 数。",
        "require_cloud_tasks 不受 per-task 阈值影响（Router 中最优先），也不计入升级率上限；"
        "它们按设计永远升级，go/no-go 会在 always_cloud_exempt_tasks 中单列。",
    ]
    if dataset_source == "synthetic":
        caveats.insert(
            0,
            "dataset_source=synthetic：合成 KB 的 edge_quality/cloud_quality 不是真实准确率证据，"
            "claims_gain 永远为 false。",
        )
    else:
        caveats.insert(
            0,
            "真实日志通常只有被升级请求的云端答案；本报告的 counterfactual 标签由导出方提供，"
            "不是无偏因果收益（策略选择偏差方向取决于导出样本）。",
        )
        caveats.insert(
            1,
            "replay 成本由 edge_cost/escalated_cost/cloud_only_cost 直接给出；token 数来源由数据导出方记录。",
        )
    return caveats


def run_experiment(
    samples: Sequence[CostSample],
    *,
    dataset_source: str,
    seed: int = DEFAULT_SEED,
    lambdas: Sequence[float] = DEFAULT_LAMBDA_GRID,
    selected_lambda: float = DEFAULT_SELECT_LAMBDA,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    threshold_grid: Sequence[float] = DEFAULT_THRESHOLD_GRID,
    require_cloud_tasks: Sequence[str] | str = (),
    max_task_escalation_rate: float = DEFAULT_MAX_TASK_ESCALATION_RATE,
    cost_model: CostModel | None = None,
    latency_model: LatencyModel | None = None,
    smoke: bool = False,
) -> ExperimentResult:
    """Fit and evaluate the λ sweep, then assemble the report and policy."""

    if dataset_source not in {"synthetic", "replay"}:
        raise ValueError("dataset_source 必须是 synthetic|replay")
    items = list(samples)
    if not items:
        raise ValueError("样本为空")
    if not any(sample.split == "train" for sample in items):
        raise ValueError("train split 为空")
    if not any(sample.split == "val" for sample in items):
        raise ValueError("val split 为空")

    weight_grid = _normalize_lambdas(lambdas)
    select_weight = round(_finite_non_negative(selected_lambda, "selected_lambda"), 9)
    if select_weight not in weight_grid:
        weight_grid = sorted({*weight_grid, select_weight})
    grid = _normalize_threshold_grid(threshold_grid)
    minimum = int(min_samples)
    if minimum < 0:
        raise ValueError("min_samples 不能为负数")
    boot = int(bootstrap)
    if boot < 0:
        raise ValueError("bootstrap 不能为负数")
    require_cloud = _task_sequence(require_cloud_tasks)
    seed_value = int(seed)

    entries: list[dict[str, Any]] = []
    selected_entry: dict[str, Any] | None = None
    for weight in weight_grid:
        fit = fit_cost_policy(
            items,
            cost_weight=weight,
            threshold_grid=grid,
            min_samples=minimum,
            require_cloud_tasks=require_cloud,
        )
        per_task_val = evaluate_cost_policy(
            fit, items, split="val", n_boot=boot, seed=seed_value
        ).to_dict()
        per_task_train = evaluate_cost_policy(
            fit, items, split="train", n_boot=boot, seed=seed_value
        ).to_dict()
        baseline_policy = CostPolicy(
            global_threshold=fit.global_threshold,
            thresholds_by_task={},
            cost_weight=weight,
            always_cloud_tasks=require_cloud,
        )
        baseline_val = evaluate_cost_policy(
            baseline_policy, items, split="val", n_boot=boot, seed=seed_value
        ).to_dict()
        baseline_train = evaluate_cost_policy(
            baseline_policy, items, split="train", n_boot=boot, seed=seed_value
        ).to_dict()
        gate = evaluate_go_no_go(
            per_task_val,
            baseline_val,
            per_task_train,
            baseline_train,
            fitted_tasks=sorted(fit.thresholds_by_task),
            min_samples=minimum,
            max_task_escalation_rate=max_task_escalation_rate,
            always_cloud_tasks=require_cloud,
        )
        entry = {
            "lambda": weight,
            "fit": fit,
            "per_task_val": per_task_val,
            "per_task_train": per_task_train,
            "baseline_val": baseline_val,
            "baseline_train": baseline_train,
            "gate": gate,
        }
        entries.append(entry)
        if weight == select_weight:
            selected_entry = entry

    assert selected_entry is not None
    selected_fit = selected_entry["fit"]

    evidence = evaluate_evidence(
        items,
        dataset_source=dataset_source,
        fit=selected_fit,
        min_samples=minimum,
    )

    def status_for(gate: Mapping[str, Any]) -> dict[str, Any]:
        if dataset_source == "synthetic":
            return {
                "status": "pipeline_only",
                "reason": "synthetic 数据只验证管线，不输出 go/no-go",
            }
        if not evidence["claims_gain"]:
            return {
                **gate,
                "status": "evidence_insufficient",
                "evidence_reasons": evidence["reasons"],
            }
        return dict(gate)

    sweep: list[dict[str, Any]] = []
    for entry in entries:
        sweep.append(
            {
                "lambda": entry["lambda"],
                "fit": asdict(entry["fit"]),
                "train": entry["per_task_train"],
                "val": entry["per_task_val"],
                "baseline": {
                    "global_threshold": entry["baseline_val"]["global_threshold"],
                    "train": entry["baseline_train"],
                    "val": entry["baseline_val"],
                },
                "go_no_go": status_for(entry["gate"]),
            }
        )

    selected_gate = status_for(selected_entry["gate"])
    policy_payload = selected_fit.to_config()
    validate_policy_payload(policy_payload)
    references = _references(
        items,
        select_lambda=select_weight,
        require_cloud_tasks=require_cloud,
        bootstrap=boot,
        seed=seed_value,
    )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "spec": "issue #48 §6–§7",
        "dataset_source": dataset_source,
        "smoke": bool(smoke),
        "seed": seed_value,
        "config": {
            "lambda_grid": weight_grid,
            "selected_lambda": select_weight,
            "min_samples": minimum,
            "bootstrap": boot,
            "threshold_grid": list(grid),
            "require_cloud_tasks": list(require_cloud),
            "max_task_escalation_rate": float(max_task_escalation_rate),
            "current_default_threshold": CURRENT_DEFAULT_THRESHOLD,
            "replay_total_min": REPLAY_TOTAL_MIN,
            "cost_model": (
                asdict(cost_model) if dataset_source == "synthetic" and cost_model else None
            ),
            "latency_model": (
                asdict(latency_model) if dataset_source == "synthetic" and latency_model else None
            ),
        },
        "counts": _counts(items),
        "evidence": evidence,
        "selected": {
            "lambda": select_weight,
            "policy": policy_payload,
            "fit": asdict(selected_fit),
            "train": selected_entry["per_task_train"],
            "val": selected_entry["per_task_val"],
            "baseline": {
                "global_threshold": selected_entry["baseline_val"]["global_threshold"],
                "train": selected_entry["baseline_train"],
                "val": selected_entry["baseline_val"],
            },
            "references": references,
            "go_no_go": selected_gate,
        },
        "lambda_sweep": sweep,
        "caveats": _caveats(dataset_source),
    }
    # Normalize tuples/dataclass leftovers so the in-memory report is exactly
    # what report.json round-trips to.
    report = json.loads(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return ExperimentResult(
        report=report,
        policy=policy_payload,
        samples=tuple(items),
        summary_rows=tuple(_summary_rows(report)),
        curve_rows=tuple(
            _curve_rows(
                items,
                fit=selected_fit,
                select_weight=select_weight,
                grid=grid,
                require_cloud_tasks=require_cloud,
                seed=seed_value,
            )
        ),
    )


SUMMARY_COLUMNS: tuple[str, ...] = (
    "lambda",
    "arm",
    "task",
    "n",
    "accuracy",
    "cost_per_request",
    "cost_norm",
    "objective",
    "escalation_rate",
    "cloud_calls",
    "cloud_calls_per_example",
    "latency_p95_ms",
    "threshold",
    "always_cloud",
    "accuracy_ci_low",
    "accuracy_ci_high",
    "cost_ci_low",
    "cost_ci_high",
    "status",
    "failed_checks",
)

CURVE_COLUMNS: tuple[str, ...] = (
    "lambda",
    "scope",
    "task",
    "threshold",
    "n",
    "accuracy",
    "cost_per_request",
    "cost_norm",
    "objective",
    "escalation_rate",
    "is_selected_threshold",
    "is_baseline_threshold",
)


def _cloud_calls_per_example(metrics: Mapping[str, Any]) -> float:
    n = int(metrics.get("n", 0))
    return round(float(metrics.get("cloud_calls", 0)) / n, 9) if n else 0.0


def _aggregate_summary_row(
    weight: float,
    arm: str,
    metrics: Mapping[str, Any],
    *,
    status: str,
    failed_checks: Sequence[str],
) -> dict[str, Any]:
    accuracy_ci = metrics.get("accuracy_ci") or {}
    cost_ci = metrics.get("cost_ci") or {}
    return {
        "lambda": weight,
        "arm": arm,
        "task": "",
        "n": metrics.get("n"),
        "accuracy": metrics.get("accuracy"),
        "cost_per_request": metrics.get("cost"),
        "cost_norm": metrics.get("cost_norm"),
        "objective": metrics.get("objective"),
        "escalation_rate": metrics.get("escalation_rate"),
        "cloud_calls": metrics.get("cloud_calls"),
        "cloud_calls_per_example": _cloud_calls_per_example(metrics),
        "latency_p95_ms": metrics.get("latency_p95_ms"),
        "threshold": metrics.get("global_threshold"),
        "always_cloud": "",
        "accuracy_ci_low": accuracy_ci.get("low"),
        "accuracy_ci_high": accuracy_ci.get("high"),
        "cost_ci_low": cost_ci.get("low"),
        "cost_ci_high": cost_ci.get("high"),
        "status": status,
        "failed_checks": ";".join(failed_checks),
    }


def _per_task_summary_rows(
    weight: float,
    arm: str,
    metrics: Mapping[str, Any],
    *,
    status: str,
    failed_checks: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, entry in sorted(metrics.get("per_task", {}).items()):
        rows.append(
            {
                "lambda": weight,
                "arm": arm,
                "task": task,
                "n": entry.get("n"),
                "accuracy": entry.get("accuracy"),
                "cost_per_request": entry.get("cost"),
                "cost_norm": entry.get("cost_norm"),
                "objective": entry.get("objective"),
                "escalation_rate": entry.get("escalation_rate"),
                "cloud_calls": entry.get("cloud_calls"),
                "cloud_calls_per_example": _cloud_calls_per_example(entry),
                "latency_p95_ms": entry.get("latency_p95_ms"),
                "threshold": entry.get("threshold"),
                "always_cloud": entry.get("always_cloud"),
                "accuracy_ci_low": None,
                "accuracy_ci_high": None,
                "cost_ci_low": None,
                "cost_ci_high": None,
                "status": status,
                "failed_checks": ";".join(failed_checks),
            }
        )
    return rows


def _summary_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in report["lambda_sweep"]:
        gate = entry["go_no_go"]
        status = gate["status"]
        failed = gate.get("failed_checks", ())
        weight = entry["lambda"]
        rows.append(
            _aggregate_summary_row(
                weight, "per_task", entry["val"], status=status, failed_checks=failed
            )
        )
        rows.append(
            _aggregate_summary_row(
                weight,
                "global_baseline",
                entry["baseline"]["val"],
                status=status,
                failed_checks=failed,
            )
        )
        rows.extend(
            _per_task_summary_rows(
                weight, "per_task", entry["val"], status=status, failed_checks=failed
            )
        )
        rows.extend(
            _per_task_summary_rows(
                weight,
                "global_baseline",
                entry["baseline"]["val"],
                status=status,
                failed_checks=failed,
            )
        )
    selected = report["selected"]
    for name, metrics in selected["references"].items():
        rows.append(
            _aggregate_summary_row(
                selected["lambda"], name, metrics, status="reference", failed_checks=()
            )
        )
    return rows


def _curve_row(
    weight: float,
    scope: str,
    task: str,
    threshold: float,
    metrics: Mapping[str, Any],
    *,
    selected_threshold: float,
    baseline_threshold: float,
) -> dict[str, Any]:
    return {
        "lambda": weight,
        "scope": scope,
        "task": task,
        "threshold": threshold,
        "n": metrics.get("n"),
        "accuracy": metrics.get("accuracy"),
        "cost_per_request": metrics.get("cost"),
        "cost_norm": metrics.get("cost_norm"),
        "objective": metrics.get("objective"),
        "escalation_rate": metrics.get("escalation_rate"),
        "is_selected_threshold": threshold == selected_threshold,
        "is_baseline_threshold": threshold == baseline_threshold,
    }


def _curve_rows(
    samples: Sequence[CostSample],
    *,
    fit: CostPolicyFit,
    select_weight: float,
    grid: Sequence[float],
    require_cloud_tasks: Sequence[str],
    seed: int,
) -> list[dict[str, Any]]:
    """Threshold curves for the global policy and every non-always-cloud task."""

    rows: list[dict[str, Any]] = []
    for threshold in grid:
        policy = CostPolicy(
            global_threshold=threshold,
            thresholds_by_task={},
            cost_weight=select_weight,
            always_cloud_tasks=require_cloud_tasks,
        )
        metrics = evaluate_cost_policy(policy, samples, split="val", n_boot=0, seed=seed).to_dict()
        rows.append(
            _curve_row(
                select_weight,
                "global",
                "__all__",
                threshold,
                metrics,
                selected_threshold=fit.global_threshold,
                baseline_threshold=fit.global_threshold,
            )
        )

    val_tasks = sorted({sample.task for sample in samples if sample.split == "val"})
    for task in val_tasks:
        if task in require_cloud_tasks:
            continue
        subset = [sample for sample in samples if sample.split == "val" and sample.task == task]
        selected_threshold = fit.threshold_for(task)
        if selected_threshold is None:
            continue
        for threshold in grid:
            policy = CostPolicy(
                global_threshold=threshold,
                thresholds_by_task={},
                cost_weight=select_weight,
                always_cloud_tasks=(),
            )
            metrics = evaluate_cost_policy(
                policy, subset, split="val", n_boot=0, seed=seed
            ).to_dict()
            rows.append(
                _curve_row(
                    select_weight,
                    "task",
                    task,
                    threshold,
                    metrics,
                    selected_threshold=selected_threshold,
                    baseline_threshold=fit.global_threshold,
                )
            )
    return rows


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_csv(
    path: str | Path,
    columns: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_cell(row.get(column)) for column in columns})
    return target


def write_artifacts(result: ExperimentResult, out_dir: str | Path) -> dict[str, Path]:
    """Write policy.json, report.json, summary.csv, curves.csv and dataset.jsonl."""

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "policy": write_json(out / "policy.json", result.policy),
        "report": write_json(out / "report.json", result.report),
        "summary": write_csv(out / "summary.csv", SUMMARY_COLUMNS, result.summary_rows),
        "curves": write_csv(out / "curves.csv", CURVE_COLUMNS, result.curve_rows),
        "dataset": write_jsonl(
            out / "dataset.jsonl", (sample.to_dict() for sample in result.samples)
        ),
    }
    return paths


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return target


def validate_policy_payload(payload: Mapping[str, Any]) -> EdgeConfig:
    """Raise ``ValueError`` unless ``EdgeConfig`` accepts the policy snippet."""

    return EdgeConfig.from_dict(dict(payload))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exp15_cost_policy.py",
        description=(
            "exp15 成本策略实验：按 λ 拟合 per-task 升级阈值，与同 train split 的"
            "全局阈值 baseline 对比，输出可粘贴到 EdgeConfig 的 policy.json。"
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="从 knowledge.ENTRIES 生成确定性 CostSample（只验证管线）",
    )
    source.add_argument("--dataset", help="真实 replay JSONL；每行必须带 train|val split")
    parser.add_argument("--out", default="runs-local/exp15", help="输出目录")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="bootstrap 随机种子")
    parser.add_argument(
        "--variants-per-entry",
        type=int,
        default=None,
        help=f"synthetic 每个知识条目的变体数（默认 {DEFAULT_VARIANTS_PER_ENTRY}）",
    )
    parser.add_argument(
        "--lambdas",
        default=",".join(str(value) for value in DEFAULT_LAMBDA_GRID),
        help="λ 网格，逗号分隔",
    )
    parser.add_argument(
        "--select-lambda",
        type=float,
        default=DEFAULT_SELECT_LAMBDA,
        help="写入 policy.json 的 λ（默认 0.1，不在网格中时自动加入）",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help=f"拟合 per-task 阈值所需 train 样本数（默认 {DEFAULT_MIN_SAMPLES}）",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=None,
        help=f"val bootstrap 次数（默认 {DEFAULT_BOOTSTRAP}，0 关闭）",
    )
    parser.add_argument(
        "--threshold-grid",
        default=",".join(str(value) for value in DEFAULT_THRESHOLD_GRID),
        help="候选阈值网格，逗号分隔（默认 0.00–1.00 步长 0.05）",
    )
    parser.add_argument(
        "--require-cloud-tasks",
        default="quality_verdict",
        help="永远升级的任务，逗号分隔；空字符串表示无",
    )
    parser.add_argument(
        "--max-task-escalation-rate",
        type=float,
        default=DEFAULT_MAX_TASK_ESCALATION_RATE,
        help="go/no-go 的 per-task val 升级率上限（默认 0.6）",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="极小规模端到端运行（2 个变体/条目、min-samples=1、bootstrap=0）",
    )
    return parser


def _parse_float_list(value: str, name: str) -> list[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise SystemExit(f"{name} 不能为空")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise SystemExit(f"{name} 必须是逗号分隔的数字: {exc}") from exc


def _parse_tasks(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    variants = (
        args.variants_per_entry
        if args.variants_per_entry is not None
        else (2 if args.smoke else DEFAULT_VARIANTS_PER_ENTRY)
    )
    min_samples = (
        args.min_samples
        if args.min_samples is not None
        else (1 if args.smoke else DEFAULT_MIN_SAMPLES)
    )
    bootstrap = (
        args.bootstrap if args.bootstrap is not None else (0 if args.smoke else DEFAULT_BOOTSTRAP)
    )
    require_cloud = _parse_tasks(args.require_cloud_tasks)
    lambdas = _parse_float_list(args.lambdas, "--lambdas")
    grid = _parse_float_list(args.threshold_grid, "--threshold-grid")
    cost_model = DEFAULT_COST_MODEL
    latency_model = LatencyModel()

    try:
        if args.synthetic:
            samples = build_synthetic_samples(
                variants_per_entry=variants,
                seed=args.seed,
                cost_model=cost_model,
                latency_model=latency_model,
            )
            dataset_source = "synthetic"
        else:
            samples = load_cost_samples(args.dataset)
            dataset_source = "replay"
        result = run_experiment(
            samples,
            dataset_source=dataset_source,
            seed=args.seed,
            lambdas=lambdas,
            selected_lambda=args.select_lambda,
            min_samples=min_samples,
            bootstrap=bootstrap,
            threshold_grid=grid,
            require_cloud_tasks=require_cloud,
            max_task_escalation_rate=args.max_task_escalation_rate,
            cost_model=cost_model if dataset_source == "synthetic" else None,
            latency_model=latency_model if dataset_source == "synthetic" else None,
            smoke=args.smoke,
        )
        paths = write_artifacts(result, args.out)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"exp15: {exc}") from exc

    report = result.report
    print(f"artifacts: {args.out}")
    print(
        json.dumps(
            {
                "dataset_source": report["dataset_source"],
                "smoke": report["smoke"],
                "counts": report["counts"],
                "selected_lambda": report["selected"]["lambda"],
                "claims_gain": report["evidence"]["claims_gain"],
                "evidence": report["evidence"]["evidence"],
                "go_no_go": report["selected"]["go_no_go"]["status"],
                "artifacts": {name: str(path) for name, path in sorted(paths.items())},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
