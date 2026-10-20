"""FeedbackStore.export_self_train — external-behavior tests (issue #32).

Spec: ``docs/specs/restrain-lite-structured-flywheel.md`` §Implementation
Decisions 1, User stories 1/2/6/19.

Every test drives the exporter through its public seam only: a fake
``generate(prompt, task)`` callback with a preset answer queue, then asserts on
the report and the JSONL file. No model, no torch, no internals.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from edge_cloud.flywheel import FeedbackStore
from edge_cloud.protocol import FeedbackRequest


def _add(
    store: FeedbackStore,
    *,
    request_id: str,
    prompt: str,
    task: str = "quality_verdict",
    accepted: bool = False,
    escalation_reason: str | None = "low_confidence",
    corrected_text: str = "",
    adapter_version: str = "edge-v1",
    edge_text: str = "端侧回答",
) -> None:
    store.record(
        FeedbackRequest(
            request_id=request_id,
            accepted=accepted,
            prompt=prompt,
            task=task,
            edge_text=edge_text,
            final_text=edge_text,
            corrected_text=corrected_text,
            source="edge",
            confidence=0.4,
            escalation_reason=escalation_reason,
            adapter_version=adapter_version,
        )
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]


class FakeGenerate:
    """Deterministic rollout callback driven by a per-prompt answer queue.

    An answer entry may be an ``Exception`` instance to simulate a failed
    rollout.
    """

    def __init__(self, answers: dict[str, list[Any]]) -> None:
        self._answers = {prompt: list(items) for prompt, items in answers.items()}
        self.calls: list[tuple[str, str]] = []

    def __call__(self, prompt: str, task: str) -> str:
        self.calls.append((prompt, task))
        queue = self._answers[prompt]
        assert queue, f"unexpected extra rollout for {prompt!r}"
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def test_export_self_train_clusters_verdict_rollouts(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    prompt = "这批零件能放行吗？"
    _add(
        store,
        request_id="r1",
        prompt=prompt,
        task="quality_verdict",
        adapter_version="edge-v7",
    )
    generate = FakeGenerate(
        {
            prompt: [
                "结论：放行",
                "合格",
                "放行",
                "判定：返修",
                "放行",
                "放行",
                "合格",
                "放行",
            ]
        }
    )
    out = tmp_path / "self_train.jsonl"

    report = store.export_self_train(out, generate=generate, n=8)

    assert report == {
        "path": str(out),
        "records": 1,
        "selected": 1,
        "exported": 1,
        "skipped_empty": 0,
        "skipped_unsupported": 0,
        "skipped_unclusterable": 0,
        "rollout_failures": 0,
        "by_task": {"quality_verdict": 1},
        "cluster_methods": {"verdict_label": 1},
        "low_consistency_rate": 0.0,
        "low_consistency_rate_all": 0.0,
    }
    assert generate.calls == [(prompt, "quality_verdict")] * 8

    [row] = _read_jsonl(out)
    assert row["prompt"] == prompt
    assert row["task"] == "quality_verdict"
    assert row["adapter_version"] == "edge-v7"
    assert row["cluster_method"] == "verdict_label"
    assert row["answers"] == ["pass", "rework"]
    assert row["counts"] == [7, 1]
    assert row["prompt_weight"] == 1
    assert row["reference_counts"] == []
    assert row["reference_consistency"] is None

    # RESTRAIN Eq. 2-3 (spec §Implementation Decisions 3): f_j = c_j / n,
    # g(f) = exp(-(1 - f)^2 / (2 * sigma^2)), w_j = g(f_j) / sum(g).
    g_major = math.exp(-((1 - 7 / 8) ** 2) / (2 * 0.5**2))
    g_minor = math.exp(-((1 - 1 / 8) ** 2) / (2 * 0.5**2))
    assert row["weights"][0] == pytest.approx(g_major / (g_major + g_minor))
    assert row["weights"][1] == pytest.approx(g_minor / (g_major + g_minor))
    assert sum(row["weights"]) == pytest.approx(1.0)
    assert row["weights"][0] > row["weights"][1]
    assert row["consistency"] == pytest.approx(g_major)


def test_export_self_train_clusters_number_answers_by_value_and_unit(
    tmp_path: Path,
) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    prompt = "出炉温度是多少？"
    _add(store, request_id="r1", prompt=prompt, task="process_calc")
    generate = FakeGenerate(
        {
            prompt: [
                "245℃",
                "245 摄氏度",
                "245°C",
                "245.1 ℃",
                "245 ℃",
                "结论 245℃",
                "245度c",
                "245℃",
            ]
        }
    )
    out = tmp_path / "self_train.jsonl"

    report = store.export_self_train(out, generate=generate)

    assert report["by_task"] == {"process_calc": 1}
    assert report["cluster_methods"] == {"number_tolerance": 1}
    [row] = _read_jsonl(out)
    assert row["task"] == "process_calc"
    assert row["cluster_method"] == "number_tolerance"
    assert row["answers"] == ["245℃", "245.1℃"]
    assert row["counts"] == [7, 1]


def test_rollout_failures_are_counted_and_do_not_abort_the_export(
    tmp_path: Path,
) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    _add(store, request_id="r1", prompt="能放行吗？", task="quality_verdict")
    _add(store, request_id="r2", prompt="全失败的问题", task="quality_verdict")
    generate = FakeGenerate(
        {
            "能放行吗？": [
                RuntimeError("backend timeout"),
                "   ",
                "放行",
                "合格",
                "放行",
                "合格",
                "放行",
                "合格",
            ],
            "全失败的问题": [RuntimeError("down")] * 8,
        }
    )
    out = tmp_path / "self_train.jsonl"

    report = store.export_self_train(out, generate=generate)

    assert report["exported"] == 1
    assert report["skipped_unclusterable"] == 1
    assert report["rollout_failures"] == 10
    [row] = _read_jsonl(out)
    assert row["prompt"] == "能放行吗？"
    assert row["answers"] == ["pass"]
    assert row["counts"] == [6]
    assert row["weights"] == [pytest.approx(1.0)]
    assert row["consistency"] == pytest.approx(1.0)


def test_export_self_train_keeps_hard_cases_only_and_dedupes_by_prompt(
    tmp_path: Path,
) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    _add(
        store,
        request_id="r1",
        prompt="重复的问题",
        task="quality_verdict",
        adapter_version="edge-v1",
    )
    _add(
        store,
        request_id="r2",
        prompt=" 重复的问题 ",
        task="quality_verdict",
        adapter_version="edge-v2",
    )
    _add(store, request_id="r3", prompt="非结构化任务", task="standard_qa")
    _add(store, request_id="r4", prompt="   ", task="process_calc")
    _add(
        store,
        request_id="r5",
        prompt="简单问题",
        task="quality_verdict",
        accepted=True,
        escalation_reason=None,
    )
    generate = FakeGenerate(
        {
            "重复的问题": [
                "结论：放行",
                "合格",
                "放行",
                "合格",
                "放行",
                "合格",
                "放行",
                "合格",
            ]
        }
    )
    out = tmp_path / "self_train.jsonl"

    report = store.export_self_train(out, generate=generate)

    assert report == {
        "path": str(out),
        "records": 5,
        "selected": 4,
        "exported": 1,
        "skipped_empty": 1,
        "skipped_unsupported": 1,
        "skipped_unclusterable": 0,
        "rollout_failures": 0,
        "by_task": {"quality_verdict": 1},
        "cluster_methods": {"verdict_label": 1},
        "low_consistency_rate": 0.0,
        "low_consistency_rate_all": 0.0,
    }
    assert generate.calls == [("重复的问题", "quality_verdict")] * 8
    [row] = _read_jsonl(out)
    assert row["prompt"] == "重复的问题"
    assert row["adapter_version"] == "edge-v2"
    assert not (tmp_path / "self_train.jsonl.tmp").exists()

    first = out.read_bytes()
    store.export_self_train(
        out,
        generate=FakeGenerate(
            {
                "重复的问题": [
                    "结论：放行",
                    "合格",
                    "放行",
                    "合格",
                    "放行",
                    "合格",
                    "放行",
                    "合格",
                ]
            }
        ),
    )
    assert out.read_bytes() == first


def test_unclusterable_number_tie_is_skipped(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    prompt = "两个答案打平的问题"
    _add(store, request_id="r1", prompt=prompt, task="process_calc")
    generate = FakeGenerate({prompt: ["245℃"] * 4 + ["246℃"] * 4})
    out = tmp_path / "self_train.jsonl"

    report = store.export_self_train(out, generate=generate)

    assert report["exported"] == 0
    assert report["skipped_unclusterable"] == 1
    assert report["by_task"] == {}
    assert report["cluster_methods"] == {}
    assert report["low_consistency_rate"] == 0.0
    # The only rolled-out prompt was unclusterable: it counts as low
    # consistency in the all-prompts rate.
    assert report["low_consistency_rate_all"] == 1.0
    assert out.read_text(encoding="utf-8") == ""


def test_low_consistency_rates_cover_exported_and_unclusterable_prompts(
    tmp_path: Path,
) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    answers = {
        # 5 clusterable: 2 majority (max 3 >= kappa 3), 3 split 2-2.
        "高一致 A": ["放行"] * 3 + ["拦截"],
        "高一致 B": ["返修"] * 4,
        "低一致 A": ["放行", "放行", "拦截", "拦截"],
        "低一致 B": ["合格", "合格", "不合格", "不合格"],
        "低一致 C": ["放行", "拦截", "返修", "返修"],
        # 3 unclusterable: no answer parses as a verdict.
        "不可聚类 A": ["不知道"] * 4,
        "不可聚类 B": ["无法判断"] * 4,
        "不可聚类 C": ["", "  ", "……", "??? "],
    }
    for index, prompt in enumerate(answers):
        _add(store, request_id=f"r{index}", prompt=prompt, task="quality_verdict")
    out = tmp_path / "self_train.jsonl"

    report = store.export_self_train(
        out, generate=FakeGenerate(answers), n=4, kappa=3
    )

    assert report["exported"] == 5
    assert report["skipped_unclusterable"] == 3
    # Exported rows only: 3 of the 5 exported rows are below the kappa gate.
    assert report["low_consistency_rate"] == pytest.approx(3 / 5)
    # All rolled-out prompts: the 3 unclusterable prompts count as low.
    assert report["low_consistency_rate_all"] == pytest.approx(6 / 8)


def test_low_consistency_rate_all_ignores_prompts_without_rollouts(
    tmp_path: Path,
) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    _add(store, request_id="r1", prompt="低一致的问题", task="quality_verdict")
    _add(store, request_id="r2", prompt="不支持的任务", task="standard_qa")
    _add(store, request_id="r3", prompt="   ", task="quality_verdict")
    out = tmp_path / "self_train.jsonl"

    report = store.export_self_train(
        out,
        generate=FakeGenerate({"低一致的问题": ["放行", "放行", "拦截", "拦截"]}),
        n=4,
        kappa=3,
    )

    assert report["skipped_unsupported"] == 1
    assert report["skipped_empty"] == 1
    assert report["low_consistency_rate"] == 1.0
    # Empty/unsupported prompts were never rolled out, so they are not part
    # of the consistency denominator.
    assert report["low_consistency_rate_all"] == 1.0


def test_reference_rollouts_add_aligned_counts_and_consistency(
    tmp_path: Path,
) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    prompt = "能放行吗？"
    _add(store, request_id="r1", prompt=prompt, task="quality_verdict")
    generate = FakeGenerate({prompt: ["合格"] * 6 + ["拦截"] * 2})
    reference_generate = FakeGenerate(
        {prompt: ["结论：放行"] * 6 + ["返修"] * 2}
    )
    out = tmp_path / "self_train.jsonl"

    report = store.export_self_train(
        out, generate=generate, reference_generate=reference_generate
    )

    assert report["rollout_failures"] == 0
    assert generate.calls == [(prompt, "quality_verdict")] * 8
    assert reference_generate.calls == [(prompt, "quality_verdict")] * 8
    [row] = _read_jsonl(out)
    assert row["answers"] == ["pass", "fail"]
    assert row["counts"] == [6, 2]
    # ``rework`` exists only in the reference clusters, so it contributes
    # nothing to the target-aligned counts.
    assert row["reference_counts"] == [6, 0]
    g_ref_major = math.exp(-((1 - 6 / 8) ** 2) / (2 * 0.5**2))
    assert row["reference_consistency"] == pytest.approx(g_ref_major)
    assert row["consistency"] == pytest.approx(
        math.exp(-((1 - 6 / 8) ** 2) / (2 * 0.5**2))
    )


def test_reference_failures_are_counted_and_skipped(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    prompt = "能放行吗？"
    _add(store, request_id="r1", prompt=prompt, task="quality_verdict")
    generate = FakeGenerate({prompt: ["合格"] * 8})
    reference_generate = FakeGenerate(
        {prompt: [RuntimeError("ref down"), "  ", *(["放行"] * 6)]}
    )
    out = tmp_path / "self_train.jsonl"

    report = store.export_self_train(
        out, generate=generate, reference_generate=reference_generate
    )

    assert report["rollout_failures"] == 2
    [row] = _read_jsonl(out)
    assert row["reference_counts"] == [6]
    assert row["reference_consistency"] == pytest.approx(1.0)


def test_max_new_tokens_is_forwarded_to_budget_aware_callbacks(
    tmp_path: Path,
) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    prompt = "能放行吗？"
    _add(store, request_id="r1", prompt=prompt, task="quality_verdict")
    seen: list[int] = []

    def generate(prompt: str, task: str, *, max_new_tokens: int) -> str:
        seen.append(max_new_tokens)
        return "放行"

    store.export_self_train(
        tmp_path / "self_train.jsonl",
        generate=generate,
        n=4,
        max_new_tokens=256,
    )

    assert seen == [256] * 4


def test_export_self_train_validates_rollout_budget(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    generate = FakeGenerate({})

    with pytest.raises(ValueError, match="n must be"):
        store.export_self_train(tmp_path / "a.jsonl", generate=generate, n=0)
    with pytest.raises(ValueError, match="max_new_tokens must be"):
        store.export_self_train(
            tmp_path / "b.jsonl", generate=generate, max_new_tokens=0
        )


def test_export_sft_output_and_report_are_unchanged(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    _add(
        store,
        request_id="r1",
        prompt="问题A",
        task="process_calc",
        escalation_reason=None,
        edge_text="端侧A",
    )
    out = tmp_path / "sft.jsonl"

    report = store.export_sft(out)

    assert report == {
        "path": str(out),
        "records": 1,
        "selected": 1,
        "exported": 1,
        "skipped_empty": 0,
        "by_task": {"process_calc": 1},
        "hard_only": True,
    }
    expected = (
        '{"messages": [{"role": "system", "content": "你是生产线工艺与质量助手。'
        '回答必须给出可执行步骤，涉及数值时说明计算口径。"}, '
        '{"role": "user", "content": "问题A"}, '
        '{"role": "assistant", "content": "端侧A"}], '
        '"meta": {"request_id": "r1", "task": "process_calc", "source": "edge", '
        '"escalation_reason": null, "confidence": 0.4, "accepted": false, '
        '"corrected": false, "adapter_version": "edge-v1"}}\n'
    )
    assert out.read_text(encoding="utf-8") == expected


