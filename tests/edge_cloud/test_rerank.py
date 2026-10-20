"""n-best memory reranking — external-behavior tests (issue #38).

Spec: ``docs/specs/jitrl-memory-hotfix.md`` §Implementation Decisions 3,
Testing Decisions「重排缝」.

The tests drive the public rerank seam only: ``Candidate``,
``normalize_answer``, ``memory_score`` and ``rerank``.  Memory hits are built
from ``MemoryEntry``/``MemoryHit`` so the tests exercise the same objects the
router passes after retrieval.
"""

from __future__ import annotations

import pytest

from edge_cloud.memory import ExperienceMemory, MemoryEntry, MemoryHit
from edge_cloud.rerank import Candidate, memory_score, normalize_answer, rerank

TASK = "fault_diagnosis"
VERDICT_TASK = "quality_verdict"
NUMBER_TASK = "process_calc"


def _candidate(text: str, logprob: float) -> Candidate:
    return Candidate(text=text, logprob=logprob)


def _hit(
    candidates: list[str],
    return_value: float,
    *,
    task: str = TASK,
    entry_id: str = "mem-1",
    score: float = 0.9,
) -> MemoryHit:
    entry = MemoryEntry(
        entry_id=entry_id,
        prompt="E104 真空报警怎么处理",
        task=task,
        candidates=candidates,
        return_value=return_value,
        accepted=return_value > 0,
        corrected=return_value >= 1.0,
        adapter_version="edge-v1",
        tenant="default",
        created_at=1_000.0,
    )
    return MemoryHit(entry=entry, score=score)


def test_candidate_defaults_to_a_completed_generation() -> None:
    candidate = Candidate(text="检查真空泵", logprob=-0.25)

    assert candidate.text == "检查真空泵"
    assert candidate.logprob == -0.25
    assert candidate.finish_reason == "stop"


def test_free_text_normalization_strips_casefolds_and_collapses_whitespace() -> None:
    assert normalize_answer("  Check   Vacuum PUMP ", TASK) == "check vacuum pump"
    assert normalize_answer(" 检查   真空泵 ", TASK) == "检查 真空泵"


def test_verdict_normalization_returns_canonical_labels() -> None:
    assert normalize_answer("判定：返修。", VERDICT_TASK) == "rework"
    assert normalize_answer("拦截", VERDICT_TASK) == "fail"
    assert normalize_answer("不合格", VERDICT_TASK) == "fail"
    assert normalize_answer("合格", VERDICT_TASK) == "pass"
    assert normalize_answer("放行", VERDICT_TASK) == "pass"
    assert normalize_answer("PASS", VERDICT_TASK) == "pass"


def test_verdict_normalization_prefers_bukhege_over_hege() -> None:
    assert normalize_answer("不合格", VERDICT_TASK) == "fail"
    assert normalize_answer("判定：不合格", VERDICT_TASK) == "fail"


def test_number_normalization_returns_value_and_canonical_unit() -> None:
    assert normalize_answer("245℃", NUMBER_TASK) == (245.0, "℃")
    assert normalize_answer("245 摄氏度", NUMBER_TASK) == (245.0, "℃")
    assert normalize_answer("245°C", NUMBER_TASK) == (245.0, "℃")
    assert normalize_answer("245", NUMBER_TASK) == (245.0, "")


def test_unparseable_structured_answers_have_no_normalized_form() -> None:
    assert normalize_answer("", TASK) is None
    assert normalize_answer("看情况", VERDICT_TASK) is None
    assert normalize_answer("没有数值", NUMBER_TASK) is None


def test_memory_score_matches_free_text_after_normalization() -> None:
    hits = [_hit(["  CHECK   vacuum pump "], 1.0)]

    assert memory_score("Check Vacuum Pump", hits, TASK) == 1.0
    assert memory_score("Check Vacuum Pump", hits, TASK) == memory_score(
        Candidate(text="Check Vacuum Pump", logprob=-0.5), hits, TASK
    )


def test_memory_score_sums_verdict_synonym_support() -> None:
    hits = [
        _hit(["拦截"], 1.0, task=VERDICT_TASK, entry_id="mem-a"),
        _hit(["不合格"], 0.5, task=VERDICT_TASK, entry_id="mem-b"),
    ]

    assert memory_score("不合格", hits, VERDICT_TASK) == 1.5


def test_memory_score_matches_numbers_within_unit_and_tolerance() -> None:
    hits = [
        _hit(["245 摄氏度"], 1.0, task=NUMBER_TASK, entry_id="mem-a"),
        _hit(["245°C"], 0.5, task=NUMBER_TASK, entry_id="mem-b"),
    ]

    assert memory_score("245.1℃", hits, NUMBER_TASK) == 1.5
    assert memory_score("250℃", hits, NUMBER_TASK) == 0.0


def test_memory_score_counts_only_one_return_direction() -> None:
    supported = [_hit(["合格"], 1.0, task=VERDICT_TASK, entry_id="mem-a")]
    contradicted = [
        _hit(["合格"], 1.0, task=VERDICT_TASK, entry_id="mem-a"),
        _hit(["合格"], -1.0, task=VERDICT_TASK, entry_id="mem-b"),
    ]

    assert memory_score("合格", supported, VERDICT_TASK) == 1.0
    assert memory_score("合格", contradicted, VERDICT_TASK) == 0.0


def test_memory_score_ignores_unrelated_or_unparseable_answers() -> None:
    hits = [_hit(["清理吸嘴"], -1.0), _hit([""], 1.0)]

    assert memory_score("检查真空泵", hits, TASK) == 0.0
    assert memory_score("", hits, TASK) == 0.0


def test_rerank_without_hits_matches_the_logprob_baseline_exactly() -> None:
    candidates = [
        _candidate("检查真空泵", -0.30),
        _candidate("清理吸嘴", -0.10),
        _candidate("联系维修", -0.20),
    ]

    result = rerank(candidates, [])

    assert result.index == 1
    assert result.text == "清理吸嘴"
    assert result.scores == [-0.30, -0.10, -0.20]
    assert result.memory_score == 0.0
    assert result.support == 0
    assert result.reranked is False
    assert result.reason == "no_memory"


def test_rerank_rewrites_top1_with_enough_positive_support() -> None:
    candidates = [
        _candidate("检查真空泵", -0.10),
        _candidate("先泄压再清理吸嘴", -0.60),
    ]
    hits = [
        _hit(["  先泄压再清理吸嘴  "], 1.0, entry_id="mem-a"),
        _hit(["先泄压再清理吸嘴"], 0.5, entry_id="mem-b"),
    ]

    result = rerank(candidates, hits)

    assert result.index == 1
    assert result.text == "先泄压再清理吸嘴"
    assert result.reranked is True
    assert result.reason == "reranked"
    assert result.memory_score == 1.5
    assert result.support == 2
    assert result.scores[0] == -0.10
    assert result.scores[1] == pytest.approx(-0.60 + 0.5 * 1.5)


def test_rerank_keeps_the_baseline_when_support_is_insufficient() -> None:
    candidates = [
        _candidate("检查真空泵", -0.10),
        _candidate("清理吸嘴", -0.20),
    ]
    hits = [_hit(["清理吸嘴"], 1.0, entry_id="mem-a")]

    result = rerank(candidates, hits)

    assert result.index == 0
    assert result.text == "检查真空泵"
    assert result.reranked is False
    assert result.reason == "insufficient_support"
    assert result.memory_score == 0.0
    assert result.support == 0
    assert result.scores == [-0.10, pytest.approx(0.30)]


def test_rerank_does_not_rewrite_on_negative_returns() -> None:
    candidates = [
        _candidate("检查真空泵", -0.10),
        _candidate("清理吸嘴", -0.20),
    ]
    hits = [
        _hit(["清理吸嘴"], -1.0, entry_id="mem-a"),
        _hit(["清理吸嘴"], -1.0, entry_id="mem-b"),
    ]

    result = rerank(candidates, hits)

    assert result.index == 0
    assert result.reranked is False
    assert result.reason == "low_return"
    assert result.memory_score == 0.0
    assert result.support == 0


def test_rerank_keeps_the_baseline_when_memory_also_supports_it() -> None:
    candidates = [
        _candidate("检查真空泵", -0.10),
        _candidate("清理吸嘴", -0.60),
    ]
    hits = [
        _hit(["检查真空泵"], 1.0, entry_id="mem-a"),
        _hit([" 检查真空泵 "], 0.5, entry_id="mem-b"),
    ]

    result = rerank(candidates, hits)

    assert result.index == 0
    assert result.reranked is False
    assert result.reason == "top1_unchanged"
    assert result.memory_score == 1.5
    assert result.support == 2


def test_rerank_reports_no_support_when_hits_do_not_match_any_candidate() -> None:
    candidates = [_candidate("检查真空泵", -0.10), _candidate("清理吸嘴", -0.20)]
    hits = [_hit(["联系维修"], 1.0, entry_id="mem-a")]

    result = rerank(candidates, hits)

    assert result.index == 0
    assert result.reranked is False
    assert result.reason == "no_support"
    assert result.memory_score == 0.0
    assert result.support == 0


def test_rerank_matches_verdict_synonyms_before_rewriting() -> None:
    candidates = [
        _candidate("合格", -0.10),
        _candidate("不合格", -0.40),
    ]
    hits = [
        _hit(["拦截"], 1.0, task=VERDICT_TASK, entry_id="mem-a"),
        _hit(["不合格"], 0.5, task=VERDICT_TASK, entry_id="mem-b"),
    ]

    result = rerank(candidates, hits)

    assert result.index == 1
    assert result.text == "不合格"
    assert result.reranked is True
    assert result.memory_score == 1.5
    assert result.support == 2


def test_rerank_matches_number_values_with_unit_and_tolerance() -> None:
    candidates = [
        _candidate("250℃", -0.10),
        _candidate("245.1℃", -0.40),
    ]
    hits = [
        _hit(["245 摄氏度"], 1.0, task=NUMBER_TASK, entry_id="mem-a"),
        _hit(["245°C"], 0.5, task=NUMBER_TASK, entry_id="mem-b"),
    ]

    result = rerank(candidates, hits)

    assert result.index == 1
    assert result.text == "245.1℃"
    assert result.reranked is True
    assert result.memory_score == 1.5


def test_rerank_requires_average_return_above_min_return() -> None:
    candidates = [
        _candidate("检查真空泵", -0.10),
        _candidate("清理吸嘴", -0.30),
    ]
    hits = [
        _hit(["清理吸嘴"], 0.5, entry_id="mem-a"),
        _hit(["清理吸嘴"], 0.5, entry_id="mem-b"),
    ]

    blocked = rerank(candidates, hits, min_return=0.75)
    allowed = rerank(candidates, hits, min_return=0.40)

    assert blocked.index == 0
    assert blocked.reranked is False
    assert blocked.reason == "low_return"
    assert allowed.index == 1
    assert allowed.reranked is True


def test_rerank_ties_are_deterministic_and_keep_the_lowest_index() -> None:
    candidates = [_candidate("检查真空泵", -0.50), _candidate("清理吸嘴", -0.50)]
    hits = [
        _hit(["检查真空泵"], 1.0, entry_id="mem-a"),
        _hit(["检查真空泵"], 1.0, entry_id="mem-b"),
        _hit(["清理吸嘴"], 1.0, entry_id="mem-c"),
        _hit(["清理吸嘴"], 1.0, entry_id="mem-d"),
    ]

    first = rerank(candidates, hits)
    second = rerank(candidates, hits)

    assert first == second
    assert first.index == 0
    assert first.reranked is False
    assert first.reason == "top1_unchanged"
    assert first.scores[0] == pytest.approx(first.scores[1])


def test_rerank_consumes_hits_returned_by_experience_memory() -> None:
    memory = ExperienceMemory()
    for entry_id, answer, return_value in (
        ("mem-a", "拦截", 1.0),
        ("mem-b", "不合格", 0.5),
    ):
        memory.add(
            MemoryEntry(
                entry_id=entry_id,
                prompt="能放行吗？",
                task=VERDICT_TASK,
                candidates=[answer],
                return_value=return_value,
                accepted=True,
                corrected=return_value >= 1.0,
                adapter_version="edge-v1",
                tenant="default",
                created_at=1_000.0,
            )
        )
    hits = memory.retrieve(
        "能放行吗？",
        task=VERDICT_TASK,
        tenant="default",
        adapter_version="edge-v1",
        now=1_050.0,
    )
    candidates = [_candidate("合格", -0.10), _candidate("不合格", -0.40)]

    result = rerank(candidates, hits)

    assert len(hits) == 2
    assert result.index == 1
    assert result.text == "不合格"
    assert result.reranked is True
    assert result.support == 2


def test_rerank_rejects_an_empty_candidate_list() -> None:
    with pytest.raises(ValueError):
        rerank([], [])
