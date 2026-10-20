"""ExperienceMemory — external-behavior tests (issue #37).

Spec: ``docs/specs/jitrl-memory-hotfix.md`` §Implementation Decisions 1/2/5,
Testing Decisions「记忆缝」「检索缝」.

All tests drive the public seams (``MemoryEntry.from_feedback``,
``ExperienceMemory.add/add_feedback/retrieve/load/save``) and assert on
returned entries, hits, and persisted JSON.  No private index structure is
inspected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from edge_cloud.flywheel import FeedbackRecord
from edge_cloud.memory import ExperienceMemory, MemoryEntry

DEFAULT_TENANT = "default"
TASK = "quality_verdict"
ADAPTER = "edge-v1"
ENTRY_FIELDS = {
    "entry_id",
    "prompt",
    "task",
    "candidates",
    "return_value",
    "accepted",
    "corrected",
    "adapter_version",
    "tenant",
    "created_at",
    "ttl_s",
    "embedding",
}


def _record(
    *,
    prompt: str = "E104 真空报警怎么处理",
    task: str = TASK,
    accepted: bool = False,
    edge_text: str = "检查真空泵",
    cloud_text: str = "",
    final_text: str = "",
    corrected_text: str = "",
    adapter_version: str = ADAPTER,
    metadata: dict[str, Any] | None = None,
    created_at: float = 1_000.0,
    note: str = "",
    request_id: str = "req-1",
) -> FeedbackRecord:
    return FeedbackRecord(
        request_id=request_id,
        accepted=accepted,
        prompt=prompt,
        task=task,
        edge_text=edge_text,
        cloud_text=cloud_text,
        final_text=final_text,
        corrected_text=corrected_text,
        source="edge",
        confidence=0.4,
        escalation_reason=None,
        adapter_version=adapter_version,
        note=note,
        created_at=created_at,
        metadata=dict(metadata or {}),
    )


def _entry(**kwargs: Any) -> MemoryEntry:
    return MemoryEntry.from_feedback(_record(**kwargs))


def test_corrected_feedback_maps_to_positive_return() -> None:
    entry = _entry(accepted=True, corrected_text="先泄压再清理吸嘴")

    assert entry.return_value == 1.0
    assert entry.corrected is True
    assert entry.accepted is True


def test_accepted_feedback_maps_to_half_return() -> None:
    entry = _entry(accepted=True)

    assert entry.return_value == 0.5
    assert entry.corrected is False


def test_rejected_feedback_maps_to_negative_return() -> None:
    entry = _entry(accepted=False)

    assert entry.return_value == -1.0
    assert entry.corrected is False


def test_correction_wins_over_acceptance() -> None:
    accepted = _entry(accepted=True, corrected_text="人工纠正")
    rejected = _entry(accepted=False, corrected_text="人工纠正")

    assert accepted.return_value == rejected.return_value == 1.0
    assert accepted.corrected is rejected.corrected is True


def test_candidates_are_deduped_non_empty_answers_in_record_order() -> None:
    entry = _entry(
        edge_text="端侧答",
        cloud_text="云端答",
        final_text="端侧答",
        corrected_text="人工纠正",
    )

    assert entry.candidates == ["端侧答", "云端答", "人工纠正"]


def test_candidates_skip_blank_answers() -> None:
    entry = _entry(edge_text="  ", cloud_text="", final_text="最终答", corrected_text=" ")

    assert entry.candidates == ["最终答"]


def test_tenant_prefers_line_then_station_then_default() -> None:
    assert _entry(metadata={"line": "L1", "station": "S1"}).tenant == "L1"
    assert _entry(metadata={"line": " ", "station": "S2"}).tenant == "S2"
    assert _entry(metadata={}).tenant == DEFAULT_TENANT


def test_entry_serialization_contains_only_spec_fields_and_no_private_data() -> None:
    record = _record(
        prompt="E104 真空报警怎么处理",
        metadata={"line": "L1", "ticket": "WO-7788", "operator": "张三"},
        note="工单 WO-7788 由张三处理",
        request_id="REQ-SECRET-9",
    )

    payload = MemoryEntry.from_feedback(record).to_dict()
    dumped = json.dumps(payload, ensure_ascii=False)

    assert set(payload) == ENTRY_FIELDS
    assert payload["tenant"] == "L1"
    for secret in ("WO-7788", "REQ-SECRET-9", "张三", "operator", "ticket", "metadata", "note"):
        assert secret not in dumped


def test_entry_id_is_deterministic_for_the_same_feedback() -> None:
    first = _entry()
    second = _entry()

    assert first.entry_id
    assert first.entry_id == second.entry_id
    assert first.entry_id != _entry(prompt="另一条故障").entry_id


def test_default_ttl_is_thirty_days() -> None:
    assert _entry().ttl_s == 30 * 24 * 60 * 60


def _memory(*entries: MemoryEntry, **kwargs: Any) -> ExperienceMemory:
    memory = ExperienceMemory(**kwargs)
    for entry in entries:
        memory.add(entry)
    return memory


def _retrieve(
    memory: ExperienceMemory,
    prompt: str,
    *,
    task: str = TASK,
    tenant: str = DEFAULT_TENANT,
    adapter_version: str = ADAPTER,
    **kwargs: Any,
) -> list[Any]:
    kwargs.setdefault("now", 1_050.0)
    return memory.retrieve(
        prompt,
        task=task,
        tenant=tenant,
        adapter_version=adapter_version,
        **kwargs,
    )


def test_synonymous_rewrite_hits_and_unrelated_prompt_misses() -> None:
    entry = _entry(
        prompt="E-104 真空压力不足，检查吸嘴",
        edge_text="检查真空泵与吸嘴",
    )
    memory = _memory(entry)

    hits = _retrieve(memory, "E104 吸嘴真空报警怎么处理", threshold=0.4)
    misses = _retrieve(memory, "轴承温度过高如何处理", threshold=0.4)

    assert [hit.entry.entry_id for hit in hits] == [entry.entry_id]
    assert hits[0].score > 0.4
    assert misses == []


def test_close_rewrite_clears_default_threshold_but_weaker_match_does_not() -> None:
    entry = _entry(prompt="E104 真空压力不足")
    memory = _memory(entry)

    assert len(_retrieve(memory, "E104 真空压力不足")) == 1
    assert len(_retrieve(memory, "真空压力不足", threshold=0.9)) == 1
    assert _retrieve(memory, "真空压力不足", threshold=0.99) == []
    assert _retrieve(memory, "E104 吸嘴真空报警", threshold=0.9) == []


def test_expired_entries_are_filtered_but_fresh_entries_remain() -> None:
    expired = MemoryEntry.from_feedback(
        _record(prompt="E104 真空报警", created_at=1_000.0),
        ttl_s=60,
    )
    fresh = MemoryEntry.from_feedback(
        _record(prompt="E104 真空报警", created_at=1_090.0),
        ttl_s=60,
    )
    memory = _memory(expired, fresh)

    both = {hit.entry.entry_id for hit in _retrieve(memory, "E104 真空报警", now=1_050.0)}
    assert both == {expired.entry_id, fresh.entry_id}
    assert [hit.entry.entry_id for hit in _retrieve(memory, "E104 真空报警", now=1_100.0)] == [
        fresh.entry_id
    ]
    assert _retrieve(memory, "E104 真空报警", now=1_150.0) == []


def test_adapter_version_filter_accepts_only_exact_or_compatible_versions() -> None:
    entry = _entry(adapter_version="edge-v1")

    assert _retrieve(_memory(entry), "E104 真空报警", adapter_version="edge-v2", threshold=0.5) == []
    compatible = _memory(entry, compatible_versions={"edge-v2": ("edge-v1",)})
    assert [hit.entry.entry_id for hit in _retrieve(
        compatible, "E104 真空报警", adapter_version="edge-v2", threshold=0.5
    )] == [entry.entry_id]
    assert [hit.entry.entry_id for hit in _retrieve(
        compatible, "E104 真空报警", adapter_version="edge-v1", threshold=0.5
    )] == [entry.entry_id]


def test_tenant_and_task_isolation() -> None:
    entry = _entry(metadata={"line": "L1"})
    prompt = "E104 真空报警怎么处理"

    assert _retrieve(_memory(entry), prompt, tenant="L2") == []
    assert _retrieve(_memory(entry), prompt, task="process_calc") == []
    assert len(_retrieve(_memory(entry), prompt, tenant="L1")) == 1


def test_top_k_is_bounded_and_ordering_is_stable() -> None:
    entries = [
        _entry(prompt="E104 真空报警 处理步骤一", metadata={"line": "L1"}),
        _entry(prompt="E104 真空报警 处理步骤二", metadata={"line": "L1"}),
        _entry(prompt="E104 真空报警 处理步骤三", metadata={"line": "L1"}),
    ]
    memory = _memory(*entries, compatible_versions=None)

    first = _retrieve(memory, "E104 真空报警", tenant="L1", threshold=0.0, k=2)
    second = _retrieve(memory, "E104 真空报警", tenant="L1", threshold=0.0, k=2)

    assert len(first) == 2
    assert [hit.entry.entry_id for hit in first] == [hit.entry.entry_id for hit in second]
    assert [hit.score for hit in first] == sorted(
        (hit.score for hit in first), reverse=True
    )
    assert _retrieve(memory, "E104 真空报警", tenant="L1", threshold=0.0, k=0) == []


def test_empty_memory_is_safe_and_returns_no_hits() -> None:
    memory = ExperienceMemory()

    assert _retrieve(memory, "E104 真空报警") == []
    assert len(memory) == 0


def test_add_feedback_builds_entry_and_keeps_tenant() -> None:
    memory = ExperienceMemory()

    entry = memory.add_feedback(_record(accepted=True, metadata={"station": "S9"}))

    assert len(memory) == 1
    assert entry.tenant == "S9"
    assert entry.return_value == 0.5
    assert len(_retrieve(memory, "E104 真空报警怎么处理", tenant="S9")) == 1


def test_jsonl_round_trip_preserves_entries_and_hits(tmp_path: Path) -> None:
    memory = _memory(
        _entry(prompt="E104 真空报警怎么处理", edge_text="检查真空泵"),
        _entry(
            prompt="E104 真空报警处理步骤",
            corrected_text="先泄压再清理吸嘴",
            metadata={"line": "L1"},
        ),
    )
    path = tmp_path / "memory.jsonl"

    assert memory.save(path) == 2

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        payload = json.loads(line)
        assert set(payload) == ENTRY_FIELDS
        assert line == json.dumps(payload, ensure_ascii=False, sort_keys=True)

    loaded = ExperienceMemory.load(path)
    assert loaded.entries == memory.entries
    before = _retrieve(memory, "E104 真空报警怎么处理", threshold=0.5, k=2)
    after = _retrieve(loaded, "E104 真空报警怎么处理", threshold=0.5, k=2)
    assert [hit.entry.entry_id for hit in after] == [hit.entry.entry_id for hit in before]
    assert [hit.score for hit in after] == pytest.approx([hit.score for hit in before])


def test_load_accepts_json_arrays_skips_corrupt_rows_and_ignores_extra_fields(
    tmp_path: Path,
) -> None:
    entry = _entry()
    payload = entry.to_dict()
    payload["ticket"] = "WO-PRIVATE"
    payload["embedding"] = [0.25, 0.75]

    json_path = tmp_path / "memory.json"
    json_path.write_text(json.dumps([payload], ensure_ascii=False), encoding="utf-8")
    loaded_json = ExperienceMemory.load(json_path)
    assert len(loaded_json) == 1
    assert loaded_json.entries[0].embedding == [0.25, 0.75]
    assert "ticket" not in loaded_json.entries[0].to_dict()

    jsonl_path = tmp_path / "memory.jsonl"
    jsonl_path.write_text(
        json.dumps(entry.to_dict(), ensure_ascii=False) + "\nnot-json\n", encoding="utf-8"
    )
    assert len(ExperienceMemory.load(jsonl_path)) == 1
    assert len(ExperienceMemory.load(tmp_path / "missing.jsonl")) == 0


def test_retrieval_latency_is_recorded_and_within_budget() -> None:
    memory = ExperienceMemory()
    for index in range(200):
        memory.add_feedback(
            _record(
                prompt=f"E{100 + index} 真空压力报警 处理步骤 {index}",
                edge_text=f"步骤 {index}",
            )
        )

    durations: list[float] = []
    for _ in range(20):
        hits = _retrieve(memory, "E104 真空压力报警 处理步骤 4", threshold=0.0, k=4)
        assert hits
        durations.append(memory.last_retrieve_ms)

    ordered = sorted(durations)
    p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
    assert all(duration >= 0.0 for duration in durations)
    assert p95 < 50.0
