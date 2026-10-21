"""PyroDash Stage 0 runner — pure-function tests (issue #41).

Stage 0 is a diagnostic runner, so the tests pin its external contract:
GSM8K answer extraction/format, numeric matching, character metrics,
RESTRAIN grading reuse, per-arm summaries, and the go/no-go gate tree.
No torch, no transformers, no dataset download.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "exp14_pyrodash_stage0.py"
_SPEC = importlib.util.spec_from_file_location("exp14_pyrodash_stage0", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
stage0 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = stage0
_SPEC.loader.exec_module(stage0)


def make_record(
    *,
    arm: str,
    prompt_id: str,
    correct: bool,
    format_valid: bool = True,
    empty: bool = False,
    truncated: bool = False,
    dataset: str = stage0.DATASET_GSM8K,
    task: str = "math",
    n_new_tokens: int = 10,
    output: str = "#### 1",
    expected: str | None = None,
    keyword_coverage: float | None = None,
) -> stage0.GenerationRecord:
    return stage0.GenerationRecord(
        arm=arm,
        dataset=dataset,
        prompt_id=prompt_id,
        task=task,
        prompt="q",
        output=output,
        gold="1",
        expected=expected,
        expected_keywords=[],
        correct=correct,
        format_valid=format_valid,
        empty=empty,
        truncated=truncated,
        hit_eos=not truncated,
        n_new_tokens=n_new_tokens,
        distinct3=1.0,
        char_distinct3=1.0,
        repetition4=0.0,
        keyword_coverage=keyword_coverage,
    )


# ---------------------------------------------------------------------------
# GSM8K parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Reasoning...\n#### 18", "18"),
        ("#### 1,000", "1000"),
        ("no marker but 42", "42"),
        ("#### -3.5", "-3.5"),
        ("", ""),
    ],
)
def test_extract_gsm8k_final(text, expected):
    assert stage0.extract_gsm8k_final(text) == expected


def test_gsm8k_format_valid_requires_marker_and_number():
    assert stage0.gsm8k_format_valid("...\n#### 18")
    assert not stage0.gsm8k_format_valid("... final is 18")
    assert not stage0.gsm8k_format_valid("#### answer")


@pytest.mark.parametrize(
    ("predicted", "gold", "expected"),
    [
        ("#### 18", "18", True),
        ("#### 18.0", "18", True),
        ("#### 1,000", "1000", True),
        ("#### 18", "19", False),
        ("#### 0", "0", True),
    ],
)
def test_answers_match_is_numeric(predicted, gold, expected):
    assert stage0.answers_match(predicted, gold) is expected


# ---------------------------------------------------------------------------
# Text metrics
# ---------------------------------------------------------------------------


def test_distinct_and_repetition_ngrams():
    # The gate metric is word-level: legitimate math/LaTeX answers repeat
    # characters but not word 3-grams.
    assert stage0.distinct_n("a b c d", 3) == 1.0
    assert stage0.distinct_n("a a a a", 3) == 0.5
    assert stage0.distinct_n("", 3) == 0.0
    assert stage0.char_distinct_n("abcd", 3) == 1.0
    assert stage0.char_distinct_n("aaaa", 3) == 0.5
    assert stage0.repetition_rate("aaaa", 3) == 0.5
    assert stage0.repetition_rate("ab", 3) == 0.0
    # CJK characters count as individual words, so a repeated phrase is caught.
    assert stage0.distinct_n("焊点间距与焊点间距", 3) < 1.0


def test_keyword_coverage_is_fraction_of_hits():
    text = "峰值 235–245℃，217℃ 以上维持 60–90 秒。"
    keywords = ["235", "245", "217", "60", "90"]
    assert stage0.keyword_coverage(text, keywords) == 1.0
    assert stage0.keyword_coverage("只有 235", keywords) == 0.2
    assert stage0.keyword_coverage("", keywords) == 0.0
    assert stage0.keyword_coverage("anything", []) == 0.0


# ---------------------------------------------------------------------------
# Production QA grading
# ---------------------------------------------------------------------------


def test_structured_qa_grading_reuses_restrain_extractor():
    assert stage0._resolve_qa_answer("判定：返修。", "quality_verdict", "判定：返修。")
    assert not stage0._resolve_qa_answer("判定：放行。", "quality_verdict", "判定：返修。")
    assert stage0._resolve_qa_answer("答案是 30000", "process_calc", "30000")
    assert not stage0._resolve_qa_answer("答案是 3000", "process_calc", "30000")
    # Multi-step answers show the formula first; the last number is the result.
    assert stage0._resolve_qa_answer(
        "DPPM = (6 / 200) × 1,000,000 = 30,000\n\n30000",
        "process_calc",
        "30000",
    )
    assert not stage0._resolve_qa_answer(
        "DPPM = (6 / 200) × 1,000,000 = 30,000\n\n300",
        "process_calc",
        "30000",
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return f"SYS:{messages[0]['content']}|USER:{messages[1]['content']}"


def test_format_prompt_adds_task_suffix():
    tok = FakeTokenizer()
    verdict = stage0.format_prompt(
        tok,
        stage0.PromptRecord(
            id="v",
            dataset=stage0.DATASET_QA,
            task="quality_verdict",
            prompt="能否放行？",
            expected="判定：返修。",
        ),
    )
    assert verdict.startswith("SYS:")
    assert "能否放行？" in verdict
    assert "判定：<结论>" in verdict
    # The prompt must not spell out a label, otherwise a copying model is
    # parsed as if it had answered that label.
    assert "放行/返修/不合格" not in verdict

    number = stage0.format_prompt(
        tok,
        stage0.PromptRecord(
            id="n",
            dataset=stage0.DATASET_QA,
            task="process_calc",
            prompt="DPPM 是多少？",
            expected="30000",
        ),
    )
    assert "End with the numeric answer" in number

    math = stage0.format_prompt(
        tok,
        stage0.PromptRecord(
            id="m",
            dataset=stage0.DATASET_GSM8K,
            task="math",
            prompt="1+1?",
            gold="2",
        ),
    )
    assert "#### <number>" in math


# ---------------------------------------------------------------------------
# Fixture and summaries
# ---------------------------------------------------------------------------


def test_generation_records_roundtrip_for_report_rebuild(tmp_path):
    row = make_record(arm="raw06", prompt_id="1", correct=True).to_dict()
    path = tmp_path / "generations.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    loaded = stage0.load_generation_records(path)
    assert len(loaded) == 1
    assert loaded[0].arm == "raw06"
    assert loaded[0].char_distinct3 == 1.0


def test_regrade_recomputes_structured_qa_from_stored_output():
    record = make_record(
        arm="ctl4b",
        prompt_id="qa-04",
        correct=False,
        dataset=stage0.DATASET_QA,
        task="process_calc",
        output="DPPM = (6 / 200) × 1,000,000 = 30,000\n\n30000",
        expected="30000",
    )
    stage0.regrade_records([record])
    assert record.correct is True
    assert record.format_valid is True


def test_production_qa_fixture_has_sixteen_rows():
    rows = stage0.load_production_qa()
    assert len(rows) == 16
    assert {row.dataset for row in rows} == {stage0.DATASET_QA}
    tasks = {row.task for row in rows}
    assert {"quality_verdict", "process_calc", "fault_diagnosis", "safety"} <= tasks
    assert any(row.expected == "30000" for row in rows)


def test_summarize_reports_accuracy_and_degeneracy():
    rows = [
        make_record(arm="a", prompt_id="1", correct=True),
        make_record(
            arm="a",
            prompt_id="2",
            correct=False,
            format_valid=False,
            truncated=True,
            n_new_tokens=64,
        ),
    ]
    summary = stage0.summarize(rows, n_boot=100, seed=0)
    assert summary["n"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["format_valid_rate"] == 0.5
    assert summary["truncation_rate"] == 0.5
    assert summary["mean_new_tokens"] == 37.0
    assert summary["accuracy_ci95"][0] <= 0.5 <= summary["accuracy_ci95"][1]


# ---------------------------------------------------------------------------
# Gate tree
# ---------------------------------------------------------------------------


def _gate_fixture(
    *,
    raw_correct: int,
    shadow_correct: int,
    control_correct: int,
    n: int = 10,
    raw_format: bool = True,
    shadow_format: bool = True,
) -> tuple[dict, dict]:
    raw = [
        make_record(
            arm=stage0.ARM_RAW,
            prompt_id=str(i),
            correct=i < raw_correct,
            format_valid=raw_format,
        )
        for i in range(n)
    ]
    shadow = [
        make_record(
            arm=stage0.ARM_SHADOW,
            prompt_id=str(i),
            correct=i < shadow_correct,
            format_valid=shadow_format,
        )
        for i in range(n)
    ]
    control = [
        make_record(
            arm=stage0.ARM_CONTROL,
            prompt_id=str(i),
            correct=i < control_correct,
        )
        for i in range(n)
    ]
    records = {
        stage0.ARM_RAW: raw,
        stage0.ARM_SHADOW: shadow,
        stage0.ARM_CONTROL: control,
    }
    summaries = {
        arm: {
            stage0.DATASET_GSM8K: stage0.summarize(rows, n_boot=50, seed=0)
        }
        for arm, rows in records.items()
    }
    return summaries, records


def test_gate_verdict_incomplete_when_an_arm_is_missing():
    summaries, records = _gate_fixture(
        raw_correct=5, shadow_correct=4, control_correct=9
    )
    records.pop(stage0.ARM_SHADOW)
    gates = stage0.evaluate_gates(summaries, records)
    assert gates["verdict"] == "incomplete"
    assert stage0.ARM_SHADOW in gates["reason"]


def test_gate_verdict_go_when_shadow_viable_and_control_works():
    summaries, records = _gate_fixture(
        raw_correct=5, shadow_correct=4, control_correct=9
    )
    gates = stage0.evaluate_gates(summaries, records)
    assert gates["verdict"] == "go"
    assert gates["shadow_viability"]["accuracy_pass"]
    assert gates["shadow_viability"]["format_pass"]


def test_gate_verdict_mechanism_only_when_shadow_too_weak():
    summaries, records = _gate_fixture(
        raw_correct=5, shadow_correct=1, control_correct=9
    )
    gates = stage0.evaluate_gates(summaries, records)
    assert gates["verdict"] == "mechanism_only"
    assert not gates["shadow_viability"]["accuracy_pass"]


def test_gate_verdict_no_go_control_failed():
    summaries, records = _gate_fixture(
        raw_correct=5, shadow_correct=4, control_correct=1
    )
    gates = stage0.evaluate_gates(summaries, records)
    assert gates["verdict"] == "no_go_control_failed"


def test_control_fails_when_a_correct_answer_is_lost_to_format():
    summaries, records = _gate_fixture(
        raw_correct=5, shadow_correct=4, control_correct=9
    )
    records[stage0.ARM_CONTROL][0] = make_record(
        arm=stage0.ARM_CONTROL,
        prompt_id="0",
        correct=True,
        format_valid=False,
    )
    gates = stage0.evaluate_gates(summaries, records)
    assert gates["control"]["correct_lost_to_format"] == 1
    assert gates["verdict"] == "no_go_control_failed"


def test_gate_verdict_no_go_when_no_product_arm_passes():
    summaries, records = _gate_fixture(
        raw_correct=0, shadow_correct=0, control_correct=9
    )
    gates = stage0.evaluate_gates(summaries, records)
    assert gates["verdict"] == "no_go"
    assert not gates["product"][stage0.ARM_RAW]["accuracy_in_range"]


def test_teacher_solvable_hard_counts_only_wrong_product_prompts():
    summaries, records = _gate_fixture(
        raw_correct=5, shadow_correct=4, control_correct=7, n=10
    )
    gates = stage0.evaluate_gates(summaries, records)
    teacher = gates["teacher_solvable_hard"][stage0.ARM_RAW]
    # raw wrong on prompts 5..9; control correct on 0..6 -> 2 of 5.
    assert teacher["wrong"] == 5
    assert teacher["solvable"] == 2
    assert teacher["rate"] == pytest.approx(0.4)
    assert not teacher["pass"]
