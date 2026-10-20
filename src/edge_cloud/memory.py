"""ExperienceMemory（经验记忆）: JitRL-lite runtime memory for candidate reranking.

The memory stores compact experience entries derived from ``FeedbackRecord``
(prompt, task, candidate answers, return, adapter version, tenant and time).
Retrieval is a dependency-free character n-gram + word TF-IDF cosine search.
It never calls a model and never mutates model weights.

Privacy: only the fields listed in the spec are kept.  Work-order ids,
personnel, notes and raw metadata never reach a ``MemoryEntry``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .flywheel import FeedbackRecord

DEFAULT_TTL_S = 30 * 24 * 60 * 60
DEFAULT_TENANT = "default"
CORRECTED_RETURN = 1.0
ACCEPTED_RETURN = 0.5
REJECTED_RETURN = -1.0


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _record_field(record: Any, name: str, default: Any) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _tenant_from_metadata(metadata: Any) -> str:
    if not isinstance(metadata, Mapping):
        return DEFAULT_TENANT
    line = _as_text(metadata.get("line"))
    if line:
        return line
    station = _as_text(metadata.get("station"))
    if station:
        return station
    return DEFAULT_TENANT


def _clean_candidates(values: Iterable[Any]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _as_text(value)
        if text and text not in seen:
            seen.add(text)
            candidates.append(text)
    return candidates


def _make_entry_id(
    *,
    prompt: str,
    task: str,
    candidates: Sequence[str],
    return_value: float,
    accepted: bool,
    adapter_version: str,
    tenant: str,
    created_at: float,
) -> str:
    payload = json.dumps(
        [
            prompt,
            task,
            list(candidates),
            return_value,
            accepted,
            adapter_version,
            tenant,
            created_at,
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"mem-{digest[:16]}"


@dataclass(slots=True)
class MemoryEntry:
    """One reusable experience derived from a feedback record."""

    entry_id: str
    prompt: str
    task: str
    candidates: list[str]
    return_value: float
    accepted: bool
    corrected: bool
    adapter_version: str
    tenant: str
    created_at: float
    ttl_s: int = DEFAULT_TTL_S
    embedding: list[float] | None = None

    @classmethod
    def from_feedback(
        cls,
        record: FeedbackRecord | Mapping[str, Any],
        *,
        ttl_s: int = DEFAULT_TTL_S,
        tenant: str | None = None,
        now: float | None = None,
    ) -> MemoryEntry:
        """Build an entry from a feedback record.

        Only the spec fields are read: prompt, task, edge/cloud/final/corrected
        text, accepted and adapter_version; the tenant comes from
        ``metadata.line`` or ``metadata.station`` (default ``default``).
        Candidates are the non-empty answers in record order, de-duplicated.
        """

        if ttl_s < 0:
            raise ValueError("ttl_s must be >= 0")
        prompt = _as_text(_record_field(record, "prompt", ""))
        task = _as_text(_record_field(record, "task", "general")) or "general"
        adapter_version = _as_text(_record_field(record, "adapter_version", ""))
        accepted = bool(_record_field(record, "accepted", False))

        corrected_text = _as_text(_record_field(record, "corrected_text", ""))
        final_text = _as_text(_record_field(record, "final_text", ""))
        cloud_text = _as_text(_record_field(record, "cloud_text", ""))
        edge_text = _as_text(_record_field(record, "edge_text", ""))
        corrected = bool(corrected_text)
        candidates = _clean_candidates((edge_text, cloud_text, final_text, corrected_text))

        if corrected:
            return_value = CORRECTED_RETURN
        elif accepted:
            return_value = ACCEPTED_RETURN
        else:
            return_value = REJECTED_RETURN

        if tenant is None:
            tenant = _tenant_from_metadata(_record_field(record, "metadata", None))
        tenant = _as_text(tenant) or DEFAULT_TENANT

        created = _record_field(record, "created_at", None)
        if created is None:
            created_at = time.time() if now is None else float(now)
        else:
            created_at = float(created)

        entry_id = _make_entry_id(
            prompt=prompt,
            task=task,
            candidates=candidates,
            return_value=return_value,
            accepted=accepted,
            adapter_version=adapter_version,
            tenant=tenant,
            created_at=created_at,
        )
        return cls(
            entry_id=entry_id,
            prompt=prompt,
            task=task,
            candidates=candidates,
            return_value=return_value,
            accepted=accepted,
            corrected=corrected,
            adapter_version=adapter_version,
            tenant=tenant,
            created_at=created_at,
            ttl_s=int(ttl_s),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MemoryEntry:
        """Rebuild an entry from persisted data, ignoring unknown fields."""

        prompt = _as_text(payload.get("prompt"))
        task = _as_text(payload.get("task", "general")) or "general"
        adapter_version = _as_text(payload.get("adapter_version", ""))
        tenant = _as_text(payload.get("tenant", DEFAULT_TENANT)) or DEFAULT_TENANT
        accepted = bool(payload.get("accepted", False))
        corrected = bool(payload.get("corrected", False))

        raw_candidates = payload.get("candidates", ())
        if isinstance(raw_candidates, str):
            raw_candidates = (raw_candidates,)
        elif not isinstance(raw_candidates, Sequence):
            raw_candidates = ()
        candidates = _clean_candidates(raw_candidates)

        if "return_value" in payload:
            return_value = float(payload["return_value"])
        elif corrected:
            return_value = CORRECTED_RETURN
        elif accepted:
            return_value = ACCEPTED_RETURN
        else:
            return_value = REJECTED_RETURN

        created_at = float(payload.get("created_at", time.time()))
        ttl_s = int(payload.get("ttl_s", DEFAULT_TTL_S))
        if ttl_s < 0:
            raise ValueError("ttl_s must be >= 0")

        entry_id = _as_text(payload.get("entry_id"))
        if not entry_id:
            entry_id = _make_entry_id(
                prompt=prompt,
                task=task,
                candidates=candidates,
                return_value=return_value,
                accepted=accepted,
                adapter_version=adapter_version,
                tenant=tenant,
                created_at=created_at,
            )

        embedding: list[float] | None = None
        raw_embedding = payload.get("embedding")
        if isinstance(raw_embedding, (list, tuple)):
            embedding = [float(value) for value in raw_embedding]

        return cls(
            entry_id=entry_id,
            prompt=prompt,
            task=task,
            candidates=candidates,
            return_value=return_value,
            accepted=accepted,
            corrected=corrected,
            adapter_version=adapter_version,
            tenant=tenant,
            created_at=created_at,
            ttl_s=ttl_s,
            embedding=embedding,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize exactly the spec fields (no metadata, notes or ids)."""

        return {
            "entry_id": self.entry_id,
            "prompt": self.prompt,
            "task": self.task,
            "candidates": list(self.candidates),
            "return_value": float(self.return_value),
            "accepted": bool(self.accepted),
            "corrected": bool(self.corrected),
            "adapter_version": self.adapter_version,
            "tenant": self.tenant,
            "created_at": float(self.created_at),
            "ttl_s": int(self.ttl_s),
            "embedding": None if self.embedding is None else list(self.embedding),
        }


@dataclass(frozen=True, slots=True)
class MemoryHit:
    """A retrieved entry and its cosine similarity to the query prompt."""

    entry: MemoryEntry
    score: float

    @property
    def memory_score(self) -> float:
        return self.score


class ExperienceMemory:
    """No-model experience memory with TTL, tenant and adapter isolation.

    Retrieval uses a dependency-free TF-IDF cosine score over character
    n-grams (CJK) plus English/number words.  ``compatible_versions`` may be a
    mapping from the queried adapter version to extra accepted memory
    versions, or a flat iterable of versions accepted for every query.
    """

    def __init__(
        self,
        entries: Iterable[MemoryEntry] | None = None,
        *,
        compatible_versions: Mapping[str, Iterable[str]] | Iterable[str] | None = None,
        default_ttl_s: int = DEFAULT_TTL_S,
    ) -> None:
        if default_ttl_s < 0:
            raise ValueError("default_ttl_s must be >= 0")
        self._entries: list[MemoryEntry] = []
        self._compatible_versions = compatible_versions
        self.default_ttl_s = int(default_ttl_s)
        self.last_retrieve_ms = 0.0
        for entry in entries or ():
            self.add(entry)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    @property
    def entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    def add(self, entry: MemoryEntry | Mapping[str, Any]) -> MemoryEntry:
        """Append an entry (mappings are parsed with the spec field set)."""

        if isinstance(entry, Mapping):
            entry = MemoryEntry.from_dict(entry)
        self._entries.append(entry)
        return entry

    def add_feedback(
        self,
        record: FeedbackRecord | Mapping[str, Any],
        *,
        ttl_s: int | None = None,
        tenant: str | None = None,
    ) -> MemoryEntry:
        """Build an entry from feedback and add it to the memory."""

        entry = MemoryEntry.from_feedback(
            record,
            ttl_s=self.default_ttl_s if ttl_s is None else ttl_s,
            tenant=tenant,
        )
        return self.add(entry)

    def retrieve(
        self,
        prompt: str,
        *,
        task: str,
        tenant: str,
        adapter_version: str,
        k: int = 4,
        threshold: float = 0.75,
        now: float | None = None,
    ) -> list[MemoryHit]:
        """Return up to ``k`` similar unexpired entries above ``threshold``.

        Only entries with the same ``task`` and ``tenant``, a compatible
        ``adapter_version`` and ``created_at + ttl_s > now`` are considered.
        An empty memory, an empty token overlap or any failure in the filters
        yields ``[]`` (``no_memory``).  ``last_retrieve_ms`` records the wall
        time of the last call for latency accounting.
        """

        started = time.perf_counter()
        try:
            if k <= 0:
                return []
            now_value = time.time() if now is None else float(now)
            accepted_versions = self._accepted_versions(_as_text(adapter_version))
            threshold_value = float(threshold)

            eligible = [
                entry
                for entry in self._entries
                if entry.task == task
                and entry.tenant == tenant
                and entry.adapter_version in accepted_versions
                and entry.created_at + entry.ttl_s > now_value
            ]
            if not eligible:
                return []

            query_tokens = _tokenize(prompt)
            if not query_tokens:
                return []
            document_tokens = [_tokenize(entry.prompt) for entry in eligible]
            scores = _tfidf_scores(query_tokens, document_tokens)

            ranked = sorted(
                zip(eligible, scores, strict=True),
                key=lambda item: (-item[1], item[0].entry_id),
            )
            return [
                MemoryHit(entry=entry, score=score)
                for entry, score in ranked
                if score >= threshold_value
            ][:k]
        finally:
            self.last_retrieve_ms = (time.perf_counter() - started) * 1000.0

    def _accepted_versions(self, adapter_version: str) -> set[str]:
        versions = {adapter_version}
        configured = self._compatible_versions
        if not configured:
            return versions
        if isinstance(configured, Mapping):
            extra: Iterable[str] = configured.get(adapter_version, ())
        else:
            extra = configured
        if isinstance(extra, str):
            extra = (extra,)
        versions.update(_as_text(value) for value in extra)
        versions.discard("")
        return versions

    def save(self, path: str | Path) -> int:
        """Write entries as stable JSONL (sorted keys) and return the count."""

        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_name(output.name + ".tmp")
        try:
            with temp.open("w", encoding="utf-8") as handle:
                for entry in self._entries:
                    handle.write(
                        json.dumps(entry.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
                    )
            os.replace(temp, output)
        finally:
            temp.unlink(missing_ok=True)
        return len(self._entries)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        compatible_versions: Mapping[str, Iterable[str]] | Iterable[str] | None = None,
        default_ttl_s: int = DEFAULT_TTL_S,
    ) -> ExperienceMemory:
        """Load entries from a JSONL file or a JSON array/object.

        Malformed lines are skipped (matching ``FeedbackStore``); unknown
        fields are ignored by ``MemoryEntry.from_dict``.
        """

        source = Path(path).expanduser()
        if not source.exists():
            return cls(compatible_versions=compatible_versions, default_ttl_s=default_ttl_s)
        text = source.read_text(encoding="utf-8")
        if not text.strip():
            return cls(compatible_versions=compatible_versions, default_ttl_s=default_ttl_s)

        payloads: list[Any]
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            payloads = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payloads.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        else:
            if isinstance(decoded, list):
                payloads = decoded
            elif isinstance(decoded, Mapping):
                nested = decoded.get("entries")
                payloads = list(nested) if isinstance(nested, list) else [decoded]
            else:
                payloads = []

        entries: list[MemoryEntry] = []
        for payload in payloads:
            if not isinstance(payload, Mapping):
                continue
            try:
                entries.append(MemoryEntry.from_dict(payload))
            except (TypeError, ValueError):
                continue
        return cls(
            entries,
            compatible_versions=compatible_versions,
            default_ttl_s=default_ttl_s,
        )


_ASCII_WORD = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_ASCII_PART = re.compile(r"[a-z0-9]+")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


def _tokenize(text: str) -> Counter[str]:
    """Character n-grams for CJK, word tokens for English/numbers."""

    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    tokens: Counter[str] = Counter()
    for match in _ASCII_WORD.finditer(normalized):
        parts = _ASCII_PART.findall(match.group())
        for part in parts:
            tokens[f"w:{part}"] += 1
        if len(parts) > 1:
            tokens[f"w:{''.join(parts)}"] += 1
    for run in _CJK_RUN.findall(normalized):
        for size in (1, 2, 3):
            for index in range(len(run) - size + 1):
                tokens[f"c{size}:{run[index:index + size]}"] += 1
    return tokens


def _tfidf_scores(
    query_tokens: Counter[str],
    document_tokens: Sequence[Counter[str]],
) -> list[float]:
    """Cosine similarity between the query and each document, in order."""

    documents = len(document_tokens)
    if documents == 0:
        return []
    document_frequency: Counter[str] = Counter()
    for tokens in document_tokens:
        document_frequency.update(tokens.keys())
    inverse_frequency = {
        token: math.log((1.0 + documents) / (1.0 + frequency)) + 1.0
        for token, frequency in document_frequency.items()
    }

    def vector(tokens: Counter[str]) -> tuple[dict[str, float], float]:
        weighted = {
            token: (1.0 + math.log(count)) * inverse_frequency[token]
            for token, count in tokens.items()
            if token in inverse_frequency and count > 0
        }
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        return weighted, norm

    query_vector, query_norm = vector(query_tokens)
    if query_norm == 0.0:
        return [0.0] * documents

    scores: list[float] = []
    for tokens in document_tokens:
        document_vector, document_norm = vector(tokens)
        if document_norm == 0.0:
            scores.append(0.0)
            continue
        dot = sum(
            weight * document_vector.get(token, 0.0)
            for token, weight in query_vector.items()
        )
        scores.append(dot / (query_norm * document_norm))
    return scores
