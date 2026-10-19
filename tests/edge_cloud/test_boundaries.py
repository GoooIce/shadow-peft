from __future__ import annotations

import pytest

from edge_cloud.boundaries import detect_boundary


@pytest.mark.parametrize(
    "text",
    [
        "已检查阀门。",
        "压力是多少？",
        "立即停机！",
        "结果待定……",
    ],
)
def test_chinese_sentence_punctuation_is_a_boundary(text: str) -> None:
    assert detect_boundary(text) == "sentence"


@pytest.mark.parametrize(
    "text",
    [
        "Check the valve.",
        "Is the pressure stable?",
        "Stop now!",
    ],
)
def test_english_sentence_punctuation_is_a_boundary(text: str) -> None:
    assert detect_boundary(text) == "sentence"


@pytest.mark.parametrize(
    "text",
    [
        "他说：“可以。”",
        'He said "done."',
        "结论见附录（见上。）",
    ],
)
def test_closing_quote_or_bracket_after_punctuation_is_a_boundary(text: str) -> None:
    assert detect_boundary(text) == "sentence"


@pytest.mark.parametrize(
    "text",
    [
        "检查完成。  ",
        "The valve is closed.\t",
    ],
)
def test_trailing_horizontal_space_does_not_hide_a_sentence_end(text: str) -> None:
    assert detect_boundary(text) == "sentence"


@pytest.mark.parametrize(
    "text",
    [
        "Step one continues\n",
        "still typing\r\n",
        "no punctuation yet\n  ",
    ],
)
def test_line_break_is_a_newline_boundary(text: str) -> None:
    assert detect_boundary(text) == "newline"


@pytest.mark.parametrize(
    "text",
    [
        "1. ",
        "2、",
        "3)",
        "   -  ",
        "•\t",
        "二、",
        "4.",
    ],
)
def test_step_marker_at_the_edge_is_a_step_boundary(text: str) -> None:
    assert detect_boundary(text) == "step"


@pytest.mark.parametrize(
    "text",
    [
        "1. 检查阀门",
        "1. 压力 245 kPa",
        "- check the gauge",
    ],
)
def test_step_marker_with_content_is_not_a_boundary(text: str) -> None:
    assert detect_boundary(text) is None


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ('{"temperature": 245,', "json_field"),
        ('{"temperature": 245}', "json_field"),
        ('{"samples": [1, 2, 3]', "json_field"),
        ('{"temperature": 245,\n', "json_field"),
    ],
)
def test_json_field_end_is_a_json_boundary(text: str, kind: str) -> None:
    assert detect_boundary(text, expected_format="json") == kind


def test_json_braces_only_count_when_json_is_expected() -> None:
    assert detect_boundary('{"temperature": 245}') is None
    assert detect_boundary("if x: y = {1, 2}") is None


@pytest.mark.parametrize(
    "text",
    [
        "我们需要检查",
        "The pressure is",
        "温度 245",
        "3.14",
        "版本 v1.2.3",
        "a/b",
        "结果如下：",
    ],
)
def test_half_sentence_is_not_a_boundary(text: str) -> None:
    assert detect_boundary(text) is None


@pytest.mark.parametrize("text", ["", "   ", "\n", " \t\n\t"])
def test_empty_or_whitespace_only_text_has_no_boundary(text: str) -> None:
    assert detect_boundary(text) is None


def test_trailing_edge_decides_the_kind() -> None:
    # A sentence followed by a line break ends on the line break: both are
    # valid cut points, and the boundary kind describes the trailing edge.
    assert detect_boundary("检查完成。\n") == "newline"
    # JSON stays structural even when the generator emits a newline after a field.
    assert detect_boundary('{"a": 1,\n', expected_format="json") == "json_field"
