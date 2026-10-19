"""Durable offline outbox (断网续传).

When the cloud cannot be reached, the edge still answers locally and records
the request here. A background flusher replays pending items once connectivity
returns, so the data flywheel does not lose production-line hard cases.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .protocol import InferRequest, InferResponse


@dataclass(slots=True)
class OutboxItem:
    item_id: str
    request: dict[str, Any]
    edge_response: dict[str, Any]
    reason: str
    created_at: float = field(default_factory=time.time)
    attempts: int = 0
    status: str = "pending"
    cloud_text: str = ""
    cloud_confidence: float = 0.0
    last_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Outbox:
    def __init__(self, path: str | Path, *, max_items: int = 5000) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.max_items = max_items
        self._lock = threading.Lock()

    def _read_locked(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return items

    def _write_locked(self, items: list[dict[str, Any]]) -> None:
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            for item in items[-self.max_items :]:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        os.replace(temp, self.path)

    def enqueue(
        self,
        request: InferRequest,
        edge_response: InferResponse,
        reason: str,
    ) -> OutboxItem:
        with self._lock:
            items = self._read_locked()
            for existing in items:
                if (
                    existing.get("request", {}).get("request_id") == request.request_id
                    and existing.get("status") == "pending"
                ):
                    return OutboxItem(**existing)
            item = OutboxItem(
                item_id=uuid.uuid4().hex,
                request=request.to_dict(),
                edge_response=edge_response.to_dict(),
                reason=reason,
            )
            items.append(item.to_dict())
            self._write_locked(items)
            return item

    def items(self, status: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items = self._read_locked()
        if status is None:
            return items
        return [item for item in items if item.get("status") == status]

    def pending(self) -> list[dict[str, Any]]:
        return self.items("pending")

    def update(self, item_id: str, **fields: Any) -> None:
        with self._lock:
            items = self._read_locked()
            for item in items:
                if item.get("item_id") == item_id:
                    item.update(fields)
                    break
            self._write_locked(items)

    def stats(self) -> dict[str, int]:
        with self._lock:
            items = self._read_locked()
        counts: dict[str, int] = {}
        for item in items:
            status = str(item.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
        counts["total"] = len(items)
        return counts
