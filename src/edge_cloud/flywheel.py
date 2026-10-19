"""Data flywheel（数据飞轮）: feedback capture and SFT export.

Every production-line outcome is appended to a JSONL log. Hard cases (escalated,
rejected or manually corrected) are exported as instruction-tuning data for the
next edge adapter. The actual training job is intentionally outside this module:
it can be a ShadowPEFT run, LoRA, or a vendor distillation job.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .protocol import FeedbackRequest


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
