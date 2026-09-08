"""OCR content scoring through the frozen image/target public boundary."""

from dataclasses import replace

import pytest

from shadow_lab.mage_w4_search.ocr import (
    EngineIdentity,
    OCRConfig,
    RawSegment,
    TargetInventory,
    match_text,
    recognize,
)


def config(engine="PaddleOCR"):
    return OCRConfig(EngineIdentity(engine, "fixture-1", (("weights", "a" * 64),)))


def receipt(texts):
    return recognize(
        b"fixed-image-fixture",
        config(),
        lambda image, settings: tuple(
            RawSegment(text, (0.0, index * 30.0, 40.0, index * 30.0 + 10.0))
            for index, text in enumerate(texts)
        ),
    )


def test_wrong_character_is_scored_from_recognition_not_requested_answer():
    recognized = receipt(["OPFN"])
    score = match_text(recognized, TargetInventory(("OPEN",)))
    assert score.status == "ok"
    assert score.character_correctness == 0.75
    assert score.cer == 0.25
    assert score.full_inventory_correct is False
    assert score.substitutions == 1
    assert score.deletions == score.insertions == 0
    assert recognized.raw_segments[0].text == "OPFN"


@pytest.mark.parametrize(
    ("targets", "observed", "correctness", "deletions", "insertions"),
    [
        (("OPEN",), (), 0.0, 4, 0),
        (("OPEN",), ("OPEN", "SALE"), 0.0, 0, 4),
        (("OPEN", "SALE"), ("SALE", "OPEN"), 1.0, 0, 0),
        (("OPEN", "OPEN"), ("OPEN",), 0.5, 4, 0),
        (("A",), ("AAAAAAAA",), 0.0, 0, 7),
    ],
)
def test_every_target_and_recognized_segment_is_used_at_most_once(
    targets, observed, correctness, deletions, insertions
):
    score = match_text(receipt(observed), TargetInventory(targets))
    assert score.character_correctness == correctness
    assert score.deletions == deletions
    assert score.insertions == insertions
    used = [item.recognized_index for item in score.matches if item.recognized_index is not None]
    assert len(used) == len(set(used)) == len(observed)
    targets_used = [item.target_index for item in score.matches if item.target_index is not None]
    assert len(targets_used) == len(set(targets_used)) == len(targets)


def test_matching_is_global_minimum_and_handles_many_segments():
    # Greedy A→AA (cost 1) leaves AB→A (cost 1); global A→A and AB→AA costs 1.
    score = match_text(receipt(("AA", "A")), TargetInventory(("A", "AB")))
    assert score.substitutions + score.deletions + score.insertions == 1
    many = tuple(f"text{index:02d}" for index in range(30))
    assert match_text(receipt(many[::-1]), TargetInventory(many)).full_inventory_correct


def test_geometry_merges_fragments_without_access_to_answers():
    image = b"image-with-fragments"
    raw = (
        RawSegment("OP", (0, 0, 20, 10)),
        RawSegment("FN", (21, 0, 41, 10)),
        RawSegment("SALE", (0, 30, 40, 40)),
    )
    recognized = recognize(image, config(), lambda data, frozen: raw)
    assert tuple(row.text for row in recognized.segments) == ("OPFN", "SALE")
    assert recognized.segments[0].raw_indices == (0, 1)
    wrong = match_text(recognized, TargetInventory(("OPEN", "SALE")))
    right = match_text(recognized, TargetInventory(("OPFN", "SALE")))
    assert wrong.character_correctness == 0.875
    assert right.character_correctness == 1.0
    assert wrong.recognition_sha256 == right.recognition_sha256
    assert recognized.raw_segments == raw
    assert recognize(image, config(), lambda data, frozen: raw).segments == recognized.segments


@pytest.mark.parametrize(
    ("target", "observed", "correctness"),
    [
        ("é", "e\u0301", 1.0),
        ("A B", " A\n\u00a0 B ", 1.0),
        ("ABC!", "abc!", 0.25),
        ("A", "Ａ", 0.0),
        ("傳", "传", 0.0),
        ("!", "！", 0.0),
    ],
)
def test_only_canonical_encoding_and_frozen_layout_whitespace_are_normalized(
    target, observed, correctness
):
    assert (
        match_text(receipt((observed,)), TargetInventory((target,))).character_correctness
        == correctness
    )


def test_no_target_successful_empty_and_failed_engine_have_distinct_results():
    empty = receipt(())
    assert match_text(empty, TargetInventory(("OPEN",))).character_correctness == 0.0
    na = match_text(empty, TargetInventory((), complete=False))
    assert na.status == "not_applicable"
    assert na.character_correctness is na.cer is na.full_inventory_correct is None

    def broken(image, frozen):
        raise RuntimeError("engine unavailable")

    failed = recognize(b"image", config(), broken)
    assert failed.status == "engine_failure"
    assert "engine unavailable" in failed.error
    score = match_text(failed, TargetInventory(("OPEN",)))
    assert score.status == "engine_failure"
    assert score.character_correctness is score.cer is score.full_inventory_correct is None
    assert not score.matches
    with pytest.raises(ValueError, match="nonempty"):
        TargetInventory(("\t ",))


def test_malformed_output_and_answer_specific_settings_cannot_be_valid_scores():
    malformed = recognize(b"image", config(), lambda data, frozen: ["invented segment"])
    assert malformed.status == "engine_failure"
    with pytest.raises(ValueError, match="target"):
        OCRConfig(config().identity, settings_json='{"recognizer": {"target_text": "OPEN"}}')
    with pytest.raises(ValueError, match="SHA-256"):
        EngineIdentity("PaddleOCR", "3.0.0", (("weights", "latest"),))
    with pytest.raises(ValueError, match="box"):
        RawSegment("OPEN", (0, 0, 40, 0))


def test_calibration_needs_actual_human_transcriptions_and_cannot_freeze_fixtures():
    from shadow_lab.mage_w4_search.ocr import (
        CalibrationPlan,
        HumanTranscription,
        compare_ocr,
        freeze_scorer,
    )

    paddle = receipt(("OPFN",))
    easy = recognize(
        b"fixed-image-fixture",
        config("EasyOCR"),
        lambda image, frozen: (RawSegment("OPEN", (0, 0, 40, 10)),),
    )
    plan = CalibrationPlan(
        image_sha256=(paddle.image_sha256,),
        configs=(config(), config("EasyOCR")),
        generation_receipt_sha256=("b" * 64,),
        registered_at_utc="2026-01-01T00:00:00+00:00",
        maximum_mean_error=0.25,
    )
    with pytest.raises(ValueError, match="human transcription"):
        compare_ocr(plan, (), (paddle, easy), execution_kind="fixture")
    with pytest.raises(ValueError, match="human"):
        HumanTranscription(paddle.image_sha256, ("OPEN",), "prompt_target", "rater", "c" * 64)

    # Actual image says OPFN although the requested target was OPEN. Tool
    # selection must reward Paddle for reading the actual image correctly.
    actual = HumanTranscription(
        paddle.image_sha256,
        ("OPFN",),
        "human_image_transcription",
        "rater",
        "c" * 64,
    )
    report = compare_ocr(plan, (actual,), (paddle, easy), execution_kind="fixture")
    assert report.selected_engine == "PaddleOCR"
    assert report.engines[0].mean_normalized_edit_error == 0.0
    assert report.engines[1].mean_normalized_edit_error == 0.25
    assert report.engines[0].images[0].matches[0].recognized == "OPFN"
    with pytest.raises(ValueError, match="GB10"):
        freeze_scorer(report)


def test_selection_keeps_empty_actual_text_failures_cost_and_missing_evidence_explicit():
    from shadow_lab.mage_w4_search.ocr import CalibrationPlan, HumanTranscription, compare_ocr

    def broken(image, frozen):
        raise RuntimeError("GPU worker failed")

    paddle = recognize(b"no visible text", config(), broken)
    easy = recognize(
        b"no visible text",
        config("EasyOCR"),
        lambda image, frozen: (RawSegment("HI", (0, 0, 20, 10)),),
    )
    actual = HumanTranscription(
        paddle.image_sha256,
        (),
        "human_image_transcription",
        "rater",
        "c" * 64,
    )
    plan = CalibrationPlan(
        (paddle.image_sha256,),
        (config(), config("EasyOCR")),
        ("b" * 64,),
        "2026-01-01T00:00:00+00:00",
        maximum_mean_error=0.5,
    )
    report = compare_ocr(plan, (actual,), (paddle, easy), execution_kind="fixture")
    assert report.selected_engine is None
    assert report.engines[0].failure_count == 1
    assert report.engines[0].mean_normalized_edit_error is None
    assert report.engines[1].images[0].actual_characters == 0
    assert report.engines[1].mean_normalized_edit_error == 2.0
    assert report.engines[1].images[0].matches[0].insertions == 2
    with pytest.raises(ValueError, match="observations"):
        compare_ocr(plan, (actual,), (paddle,), execution_kind="fixture")
    with pytest.raises(ValueError, match="observations"):
        compare_ocr(plan, (actual,), (paddle, easy, easy), execution_kind="fixture")
    with pytest.raises(ValueError, match="preregistration"):
        compare_ocr(
            replace(plan, registered_at_utc="2100-01-01T00:00:00+00:00"),
            (actual,),
            (paddle, easy),
            execution_kind="fixture",
        )
    with pytest.raises(ValueError, match="GB10"):
        compare_ocr(plan, (actual,), (paddle, easy), execution_kind="gb10_measured")


def test_identical_accuracy_and_cost_use_registered_order_not_arrival_order():
    from shadow_lab.mage_w4_search.ocr import CalibrationPlan, HumanTranscription, compare_ocr

    paddle = receipt(("OPEN",))
    easy = recognize(
        b"fixed-image-fixture",
        config("EasyOCR"),
        lambda image, frozen: (RawSegment("OPEN", (0, 0, 40, 10)),),
    )
    actual = HumanTranscription(
        paddle.image_sha256, ("OPEN",), "human_image_transcription", "rater", "c" * 64
    )
    plan = CalibrationPlan(
        (paddle.image_sha256,),
        (config(), config("EasyOCR")),
        ("b" * 64,),
        "2026-01-01T00:00:00+00:00",
        maximum_mean_error=0.25,
    )
    equal_costs = ((config().sha256, 2.0), (config("EasyOCR").sha256, 2.0))
    report = compare_ocr(
        plan,
        (actual,),
        (easy, paddle),
        execution_kind="fixture",
        occupied_seconds_by_config=equal_costs,
    )
    assert report.selected_engine == "PaddleOCR"
    cheaper_easy = ((config().sha256, 2.0), (config("EasyOCR").sha256, 1.0))
    report = compare_ocr(
        plan,
        (actual,),
        (paddle, easy),
        execution_kind="fixture",
        occupied_seconds_by_config=cheaper_easy,
    )
    assert report.selected_engine == "EasyOCR"


def test_geometry_does_not_merge_duplicate_boxes_rotated_text_or_different_lines():
    raw = (
        RawSegment("OPEN", (0, 0, 40, 10)),
        RawSegment("OPEN", (0, 0, 40, 10)),
        RawSegment("SALE", (41, 0, 81, 10), orientation_degrees=90.0),
        RawSegment("HERE", (0, 30, 40, 40)),
    )
    recognized = recognize(b"image", config(), lambda image, frozen: raw)
    assert len(recognized.segments) == 4
    score = match_text(recognized, TargetInventory(("OPEN", "SALE", "HERE")))
    assert score.insertions == 4


def test_reloaded_recognition_cannot_replace_target_blind_merged_text():
    recognized = receipt(("OPFN",))
    with pytest.raises(ValueError, match="geometry"):
        replace(recognized, segments=(replace(recognized.segments[0], text="OPEN"),))
