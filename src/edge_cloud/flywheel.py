"""Data flywheel（数据飞轮）: feedback capture and SFT export.

Every production-line outcome is appended to a JSONL log. Hard cases (escalated,
rejected or manually corrected) are exported as instruction-tuning data for the
next edge adapter. The actual training job is intentionally outside this module:
it can be a ShadowPEFT run, LoRA, or a vendor distillation job.
"""

from __future__ import annotations

import inspect
import json
import os
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from shadow_lab.restrain import build_self_train_row, cluster_answers, consistency

from .protocol import FeedbackRequest


def _accepts_token_budget(callback: Callable[..., Any]) -> bool:
    """True when *callback* accepts a ``max_new_tokens`` keyword."""

    try:
        parameters = tuple(inspect.signature(callback).parameters.values())
    except (TypeError, ValueError):
        return False
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters):
        return True
    return any(
        param.name == "max_new_tokens"
        and param.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        for param in parameters
    )


@dataclass(slots=True)
class FeedbackRecord:
    request_id: str
    accepted: bool
    prompt: str
    task: str
    edge_text: str
    cloud_text: str
    final_text: str
    corrected_text: str
    source: str
    confidence: float
    escalation_reason: str | None
    adapter_version: str
    note: str
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    handoff_index: int | None = None
    handoff_reason: str | None = None
    prefix_trust: float | None = None
    continued_from_prefix: bool = False
    corrected_prefix: bool = False
    cost_estimate: dict[str, Any] | None = None

    @classmethod
    def from_request(cls, request: FeedbackRequest) -> FeedbackRecord:
        return cls(
            request_id=request.request_id,
            accepted=request.accepted,
            prompt=request.prompt,
            task=request.task,
            edge_text=request.edge_text,
            cloud_text=request.cloud_text,
            final_text=request.final_text,
            corrected_text=request.corrected_text,
            source=request.source,
            confidence=request.confidence,
            escalation_reason=request.escalation_reason,
            adapter_version=request.adapter_version,
            note=request.note,
            metadata=request.metadata,
            handoff_index=request.handoff_index,
            handoff_reason=request.handoff_reason,
            prefix_trust=request.prefix_trust,
            continued_from_prefix=request.continued_from_prefix,
            corrected_prefix=request.corrected_prefix,
            cost_estimate=request.cost_estimate,
        )

    @property
    def is_hard_case(self) -> bool:
        return (
            not self.accepted
            or bool(self.corrected_text.strip())
            or bool(self.escalation_reason)
        )

    @property
    def best_answer(self) -> str:
        for candidate in (self.corrected_text, self.final_text, self.cloud_text, self.edge_text):
            if candidate.strip():
                return candidate.strip()
        return ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["is_hard_case"] = self.is_hard_case
        payload["best_answer"] = self.best_answer
        return payload


class FeedbackStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()

    def record(self, request: FeedbackRequest) -> FeedbackRecord:
        record = FeedbackRecord.from_request(request)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return record

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._read_locked()

    def _read_locked(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def stats(self) -> dict[str, Any]:
        records = self.records()
        accepted = sum(1 for record in records if record.get("accepted"))
        corrected = sum(1 for record in records if str(record.get("corrected_text", "")).strip())
        hard_cases = sum(1 for record in records if record.get("is_hard_case"))
        by_task = Counter(str(record.get("task", "general")) for record in records)
        by_source = Counter(str(record.get("source", "unknown")) for record in records)
        unique_prompts = len({str(record.get("prompt", "")).strip() for record in records})
        return {
            "total": len(records),
            "accepted": accepted,
            "corrected": corrected,
            "hard_cases": hard_cases,
            "acceptance_rate": round(accepted / len(records), 4) if records else 0.0,
            "unique_prompts": unique_prompts,
            "by_task": dict(by_task),
            "by_source": dict(by_source),
        }

    def export_sft(
        self,
        out_path: str | Path,
        *,
        hard_only: bool = True,
        system_prompt: str = "你是生产线工艺与质量助手。回答必须给出可执行步骤，涉及数值时说明计算口径。",
    ) -> dict[str, Any]:
        records = self.records()
        selected = [record for record in records if record.get("is_hard_case") or not hard_only]

        deduped: dict[str, dict[str, Any]] = {}
        skipped = 0
        for record in selected:
            answer = str(record.get("best_answer", "")).strip()
            prompt = str(record.get("prompt", "")).strip()
            if not prompt or not answer:
                skipped += 1
                continue
            deduped[prompt] = record

        output = Path(out_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(output.suffix + ".tmp")
        task_counts: Counter[str] = Counter()
        with temp.open("w", encoding="utf-8") as handle:
            for prompt, record in deduped.items():
                answer = str(record.get("best_answer", "")).strip()
                task = str(record.get("task", "general"))
                task_counts[task] += 1
                item = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer},
                    ],
                    "meta": {
                        "request_id": record.get("request_id"),
                        "task": task,
                        "source": record.get("source"),
                        "escalation_reason": record.get("escalation_reason"),
                        "confidence": record.get("confidence"),
                        "accepted": record.get("accepted"),
                        "corrected": bool(str(record.get("corrected_text", "")).strip()),
                        "adapter_version": record.get("adapter_version"),
                    },
                }
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        os.replace(temp, output)
        return {
            "path": str(output),
            "records": len(records),
            "selected": len(selected),
            "exported": len(deduped),
            "skipped_empty": skipped,
            "by_task": dict(task_counts),
            "hard_only": hard_only,
        }

    def export_prompts(
        self,
        out_path: str | Path,
        *,
        hard_only: bool = True,
        dedupe: bool = True,
    ) -> dict[str, Any]:
        """Export hard-case prompts (without answers) for on-policy rollouts.

        Selection matches :meth:`export_sft`: with ``hard_only`` only records
        whose stored ``is_hard_case`` flag is true are considered.

        Deduplication is per prompt after ``strip()``. When ``dedupe`` is true
        the **last** record for a prompt wins (same rule as ``export_sft``), and
        each replaced record is counted in ``skipped_duplicate``. Blank prompts
        are always skipped and counted in ``skipped_empty``. The output is a
        JSONL file with a fixed field set and sorted keys, written to a
        temporary file and atomically moved into place with ``os.replace``.
        """
        records = self.records()
        selected = [record for record in records if record.get("is_hard_case") or not hard_only]

        exported: list[dict[str, Any]] = []
        positions: dict[str, int] = {}
        skipped_empty = 0
        skipped_duplicate = 0
        for record in selected:
            prompt = str(record.get("prompt", "")).strip()
            if not prompt:
                skipped_empty += 1
                continue
            item = {
                "prompt": prompt,
                "task": str(record.get("task", "general")),
                "source": record.get("source"),
                "confidence": record.get("confidence"),
                "request_id": record.get("request_id"),
                "escalation_reason": record.get("escalation_reason"),
                "adapter_version": record.get("adapter_version"),
                "is_hard_case": bool(record.get("is_hard_case", False)),
                "accepted": bool(record.get("accepted", False)),
                "corrected": bool(str(record.get("corrected_text", "")).strip()),
            }
            if not dedupe:
                exported.append(item)
                continue
            if prompt in positions:
                exported[positions[prompt]] = item
                skipped_duplicate += 1
            else:
                positions[prompt] = len(exported)
                exported.append(item)

        output = Path(out_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(output.suffix + ".tmp")
        task_counts: Counter[str] = Counter()
        with temp.open("w", encoding="utf-8") as handle:
            for item in exported:
                task_counts[item["task"]] += 1
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temp, output)
        return {
            "path": str(output),
            "records": len(records),
            "selected": len(selected),
            "exported": len(exported),
            "skipped_empty": skipped_empty,
            "skipped_duplicate": skipped_duplicate,
            "by_task": dict(task_counts),
            "hard_only": hard_only,
        }

    @staticmethod
    def _rollouts(
        generate: Callable[[str, str], str],
        prompt: str,
        task: str,
        n: int,
        *,
        max_new_tokens: int,
    ) -> tuple[list[str], int]:
        """Run *n* rollouts, dropping failed or blank answers.

        Returns the usable answers and the number of failed calls. A single
        failed rollout never aborts the export (User story 19). The token
        budget is forwarded only to callbacks that declare it.
        """

        forward_budget = _accepts_token_budget(generate)
        answers: list[str] = []
        failures = 0
        for _ in range(n):
            try:
                if forward_budget:
                    answer = generate(prompt, task, max_new_tokens=max_new_tokens)
                else:
                    answer = generate(prompt, task)
            except Exception:
                failures += 1
                continue
            if not isinstance(answer, str) or not answer.strip():
                failures += 1
                continue
            answers.append(answer)
        return answers, failures

    def export_self_train(
        self,
        out_path: str | Path,
        *,
        generate: Callable[[str, str], str],
        n: int = 8,
        tasks: Sequence[str] = ("quality_verdict", "process_calc"),
        max_new_tokens: int = 512,
        reference_generate: Callable[[str, str], str] | None = None,
        kappa: int = 3,
        sigma: float = 0.5,
    ) -> dict[str, Any]:
        """Export RESTRAIN-lite rows from hard-case prompts (issue #32).

        ``generate(prompt, task)`` is called ``n`` times per prompt; the caller
        owns sampling determinism. Failed or blank rollouts are counted in
        ``rollout_failures`` and dropped without aborting the export. Answers
        are clustered per task with ``shadow_lab.restrain``; prompts whose
        answers do not cluster stably are counted in ``skipped_unclusterable``
        and never written.

        Only hard cases (``is_hard_case``) whose task is in ``tasks`` are
        considered; other prompts are counted as ``skipped_unsupported`` (task)
        or ``skipped_empty`` (blank prompt). Prompts are de-duplicated with the
        last record per prompt winning, matching ``export_sft``.

        When ``reference_generate`` is provided it is sampled ``n`` times with
        the same contract; ``reference_counts`` is aligned to the exported
        clusters and ``reference_consistency`` holds the reference answer
        consistency. Without it both are empty/``None`` (no reference signal).
        Failed reference rollouts are counted in ``rollout_failures`` too.

        ``max_new_tokens`` is the rollout token budget: it is validated here and
        forwarded to callbacks that declare a ``max_new_tokens`` keyword;
        callbacks with the plain ``(prompt, task)`` contract own their decoding.
        ``kappa`` and ``sigma`` follow ``build_self_train_row``. The JSONL is
        written atomically with sorted keys and a stable field set.

        The report carries two low-consistency rates, both using the
        ``prompt_weight`` gate (largest cluster < ``kappa``):

        - ``low_consistency_rate``: fraction of exported rows below the gate
          (the historical, exported-only field).
        - ``low_consistency_rate_all``: every rolled-out prompt as the
          denominator (exported rows plus ``skipped_unclusterable``), with
          each unclusterable prompt counted as low consistency. Blank and
          unsupported prompts were never rolled out and are excluded because
          they carry no consistency signal.
        """

        if n < 1:
            raise ValueError("n must be >= 1")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        allowed_tasks = (tasks,) if isinstance(tasks, str) else tuple(tasks)

        records = self.records()
        selected = [record for record in records if record.get("is_hard_case")]

        deduped: dict[str, dict[str, Any]] = {}
        skipped_empty = 0
        for record in selected:
            prompt = str(record.get("prompt", "")).strip()
            if not prompt:
                skipped_empty += 1
                continue
            deduped[prompt] = record

        rows: list[dict[str, Any]] = []
        task_counts: Counter[str] = Counter()
        method_counts: Counter[str] = Counter()
        skipped_unsupported = 0
        skipped_unclusterable = 0
        rollout_failures = 0

        for prompt, record in deduped.items():
            task = str(record.get("task", "general"))
            if task not in allowed_tasks:
                skipped_unsupported += 1
                continue

            answers, failures = self._rollouts(
                generate, prompt, task, n, max_new_tokens=max_new_tokens
            )
            rollout_failures += failures
            if cluster_answers(answers, task).unclusterable:
                skipped_unclusterable += 1
                continue

            reference_counts: dict[str, int] | None = None
            reference_consistency: float | None = None
            if reference_generate is not None:
                reference_answers, failures = self._rollouts(
                    reference_generate,
                    prompt,
                    task,
                    n,
                    max_new_tokens=max_new_tokens,
                )
                rollout_failures += failures
                reference = cluster_answers(reference_answers, task)
                if reference.counts:
                    reference_counts = dict(reference.counts)
                    reference_consistency = consistency(reference.counts, sigma=sigma)

            row = build_self_train_row(
                prompt,
                task,
                answers,
                reference_counts=reference_counts,
                adapter_version=str(record.get("adapter_version") or ""),
                sigma=sigma,
                kappa=kappa,
            )
            if row is None:
                skipped_unclusterable += 1
                continue
            if reference_counts is None:
                row["reference_counts"] = []
            row["reference_consistency"] = reference_consistency
            rows.append(row)
            task_counts[task] += 1
            method_counts[str(row["cluster_method"])] += 1

        output = Path(out_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(output.suffix + ".tmp")
        try:
            with temp.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            os.replace(temp, output)
        finally:
            temp.unlink(missing_ok=True)

        low_consistency = sum(1 for row in rows if row["prompt_weight"] == 0)
        rolled_out = len(rows) + skipped_unclusterable
        return {
            "path": str(output),
            "records": len(records),
            "selected": len(selected),
            "exported": len(rows),
            "skipped_empty": skipped_empty,
            "skipped_unsupported": skipped_unsupported,
            "skipped_unclusterable": skipped_unclusterable,
            "rollout_failures": rollout_failures,
            "by_task": dict(task_counts),
            "cluster_methods": dict(method_counts),
            "low_consistency_rate": low_consistency / len(rows) if rows else 0.0,
            "low_consistency_rate_all": (
                (low_consistency + skipped_unclusterable) / rolled_out
                if rolled_out
                else 0.0
            ),
        }
