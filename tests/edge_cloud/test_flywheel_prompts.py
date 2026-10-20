from __future__ import annotations

import json
from pathlib import Path

from edge_cloud.flywheel import FeedbackStore
from edge_cloud.protocol import FeedbackRequest

SYSTEM_PROMPT = "你是生产线工艺与质量助手。回答必须给出可执行步骤，涉及数值时说明计算口径。"
PROMPT_FIELDS = {
    "prompt",
    "task",
    "source",
    "confidence",
    "request_id",
    "escalation_reason",
    "adapter_version",
    "is_hard_case",
    "accepted",
    "corrected",
}


def _add(
    store: FeedbackStore,
    *,
    request_id: str,
    prompt: str,
    task: str = "general",
    accepted: bool = True,
    confidence: float = 0.5,
    source: str = "edge",
    escalation_reason: str | None = None,
    corrected_text: str = "",
    adapter_version: str = "edge-v1",
    edge_text: str = "端侧回答",
    final_text: str | None = None,
) -> None:
    store.record(
        FeedbackRequest(
            request_id=request_id,
            accepted=accepted,
            prompt=prompt,
            task=task,
            edge_text=edge_text,
            final_text=edge_text if final_text is None else final_text,
            corrected_text=corrected_text,
            source=source,
            confidence=confidence,
            escalation_reason=escalation_reason,
            adapter_version=adapter_version,
        )
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]


def test_export_prompts_defaults_to_hard_cases_with_metadata(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    _add(
        store,
        request_id="r-escalated",
        prompt="问题-低置信",
        task="standard_qa",
        confidence=0.42,
        source="cloud",
        escalation_reason="low_confidence",
        adapter_version="edge-v7",
    )
    _add(store, request_id="r-easy", prompt="问题-简单", task="process_qa")
    _add(
        store,
        request_id="r-corrected",
        prompt="问题-纠错",
        task="process_calc",
        corrected_text="正确步骤",
    )

    out = tmp_path / "prompts.jsonl"
    report = store.export_prompts(out)

    assert report == {
        "path": str(out),
        "records": 3,
        "selected": 2,
        "exported": 2,
        "skipped_empty": 0,
        "skipped_duplicate": 0,
        "by_task": {"standard_qa": 1, "process_calc": 1},
        "hard_only": True,
    }

    lines = _read_jsonl(out)
    assert [line["prompt"] for line in lines] == ["问题-低置信", "问题-纠错"]
    assert all(set(line) == PROMPT_FIELDS for line in lines)

    first = lines[0]
    assert first["request_id"] == "r-escalated"
    assert first["task"] == "standard_qa"
    assert first["source"] == "cloud"
    assert first["confidence"] == 0.42
    assert first["escalation_reason"] == "low_confidence"
    assert first["adapter_version"] == "edge-v7"
    assert first["is_hard_case"] is True
    assert first["accepted"] is True
    assert first["corrected"] is False

    second = lines[1]
    assert second["request_id"] == "r-corrected"
    assert second["escalation_reason"] is None
    assert second["corrected"] is True
    assert second["is_hard_case"] is True


def test_export_prompts_can_include_every_prompt(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    _add(store, request_id="r-hard", prompt="难例", accepted=False)
    _add(store, request_id="r-easy", prompt="易例")

    out = tmp_path / "prompts.jsonl"
    report = store.export_prompts(out, hard_only=False)

    assert report["hard_only"] is False
    assert report["records"] == 2
    assert report["selected"] == 2
    assert report["exported"] == 2
    assert [line["prompt"] for line in _read_jsonl(out)] == ["难例", "易例"]
    assert [line["accepted"] for line in _read_jsonl(out)] == [False, True]


def test_export_prompts_dedupes_keeping_last_record(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    _add(
        store,
        request_id="r-dup-old",
        prompt="同一问题",
        confidence=0.1,
        escalation_reason="low_confidence",
    )
    _add(
        store,
        request_id="r-other",
        prompt="别的问题",
        confidence=0.2,
        escalation_reason="low_confidence",
    )
    _add(
        store,
        request_id="r-dup-new",
        prompt="同一问题",
        confidence=0.9,
        escalation_reason="low_confidence",
    )

    deduped_path = tmp_path / "deduped.jsonl"
    deduped_report = store.export_prompts(deduped_path)
    assert deduped_report["exported"] == 2
    assert deduped_report["skipped_duplicate"] == 1
    assert deduped_report["by_task"] == {"general": 2}
    assert [line["request_id"] for line in _read_jsonl(deduped_path)] == [
        "r-dup-new",
        "r-other",
    ]
    assert [line["confidence"] for line in _read_jsonl(deduped_path)] == [0.9, 0.2]

    full_path = tmp_path / "full.jsonl"
    full_report = store.export_prompts(full_path, dedupe=False)
    assert full_report["exported"] == 3
    assert full_report["skipped_duplicate"] == 0
    assert [line["request_id"] for line in _read_jsonl(full_path)] == [
        "r-dup-old",
        "r-other",
        "r-dup-new",
    ]


def test_export_prompts_skips_blank_prompts(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    _add(store, request_id="r-empty", prompt="", accepted=False)
    _add(store, request_id="r-blank", prompt="   ", accepted=False)
    _add(store, request_id="r-valid", prompt="有效问题", accepted=False)

    out = tmp_path / "prompts.jsonl"
    report = store.export_prompts(out)

    assert report["records"] == 3
    assert report["selected"] == 3
    assert report["exported"] == 1
    assert report["skipped_empty"] == 2
    assert report["skipped_duplicate"] == 0
    assert [line["prompt"] for line in _read_jsonl(out)] == ["有效问题"]


def test_export_prompts_on_empty_store_writes_empty_file(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")

    out = tmp_path / "prompts.jsonl"
    report = store.export_prompts(out)

    assert report["records"] == 0
    assert report["selected"] == 0
    assert report["exported"] == 0
    assert report["skipped_empty"] == 0
    assert report["skipped_duplicate"] == 0
    assert report["by_task"] == {}
    assert out.read_text(encoding="utf-8") == ""


def test_export_prompts_is_stable_and_atomic(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    _add(store, request_id="r1", prompt="问题一", task="b_task", accepted=False)
    _add(
        store,
        request_id="r2",
        prompt="问题二",
        task="a_task",
        escalation_reason="low_confidence",
    )

    out = tmp_path / "prompts.jsonl"
    store.export_prompts(out)
    first = out.read_bytes()
    store.export_prompts(out)
    second = out.read_bytes()

    assert first == second
    assert not list(tmp_path.glob("*.tmp"))


def test_export_sft_output_unchanged_after_prompt_export(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    _add(
        store,
        request_id="r1",
        prompt="问题A",
        task="standard_qa",
        edge_text="端侧A",
        final_text="云端A",
        source="cloud",
        escalation_reason="low_confidence",
    )
    _add(store, request_id="r2", prompt="问题B", task="process_qa", edge_text="端侧B")
    _add(
        store,
        request_id="r3",
        prompt="问题B",
        task="process_qa",
        edge_text="端侧B",
        final_text="端侧B",
        source="edge",
    )
    _add(
        store,
        request_id="r4",
        prompt="问题C",
        task="process_calc",
        accepted=False,
        edge_text="端侧C",
        corrected_text="正确C",
        source="edge",
    )

    out = tmp_path / "sft.jsonl"
    report = store.export_sft(out)

    assert report == {
        "path": str(out),
        "records": 4,
        "selected": 2,
        "exported": 2,
        "skipped_empty": 0,
        "by_task": {"standard_qa": 1, "process_calc": 1},
        "hard_only": True,
    }
    assert _read_jsonl(out) == [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "问题A"},
                {"role": "assistant", "content": "云端A"},
            ],
            "meta": {
                "request_id": "r1",
                "task": "standard_qa",
                "source": "cloud",
                "escalation_reason": "low_confidence",
                "confidence": 0.5,
                "accepted": True,
                "corrected": False,
                "adapter_version": "edge-v1",
            },
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "问题C"},
                {"role": "assistant", "content": "正确C"},
            ],
            "meta": {
                "request_id": "r4",
                "task": "process_calc",
                "source": "edge",
                "escalation_reason": None,
                "confidence": 0.5,
                "accepted": False,
                "corrected": True,
                "adapter_version": "edge-v1",
            },
        },
    ]
