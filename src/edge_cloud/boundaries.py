"""Semantic boundary detection for token-level handoff (语义边界检测).

The edge model streams tokens; a handoff may only cut the generated answer at a
point where the cloud can continue without repairing a half-finished span. This
module answers one question about the **trailing edge** of the generated text:
"does it end on a semantic boundary, and which kind?" It is a pure function: it
only looks at text, never at a tokenizer's special tokens.

Boundary kinds and priority (first match wins):

1. ``"json_field"`` — only when ``expected_format == "json"`` and the text ends
   with ``,``, ``}`` or ``]``: a JSON field or value just closed. JSON is
   structural, so this outranks line breaks and sentence punctuation.
2. ``"step"`` — the last line is nothing but a numbered or bulleted step marker
   (``1.``, ``2、``, ``3)``, ``-``, ``*``, ``•``, ``二、``): the next step's
   content has not started yet, so the cut sits between steps.
3. ``"sentence"`` — the text ends with sentence punctuation (``. ! ? 。！？…``),
   optionally followed by closing quotes or brackets (``”』）)]`` …).
4. ``"newline"`` — the text ends with a line break (``\\n`` or ``\\r``).

Only the trailing edge is examined, and trailing spaces/tabs are ignored, so
``"检查完成。 "`` is a sentence boundary while ``"检查完成。\\n"`` is a newline
boundary. A half sentence (no trailing punctuation or marker) returns ``None``;
so do decimal numbers such as ``"3.14"`` because they do not end in a period.
"""

from __future__ import annotations

import re

# Sentence-final punctuation, Chinese and English.
_SENTENCE_END = frozenset(".!?。！？…")

# Characters that may legally follow the final punctuation mark.
_CLOSERS = frozenset("\"'”’」』）)]】")

# A line that is only a numbered / bulleted step marker.
_STEP_MARKER = re.compile(
    r"^[ \t]*(?:\d{1,3}[.、．)]|[一二三四五六七八九十]+[、.．)]|[-*•·])\s*$"
)

_JSON_FIELD_END = frozenset(",}]")


def detect_boundary(text: str, *, expected_format: str | None = None) -> str | None:
    """Return the boundary kind at the end of ``text``, or ``None``.

    Args:
        text: The text generated so far (the would-be partial trace).
        expected_format: The task's expected answer format. ``"json"`` enables
            structural JSON field detection; other values do not change the
            plain-text rules.

    Returns:
        One of ``"json_field"``, ``"step"``, ``"sentence"``, ``"newline"``, or
        ``None`` when the text stops mid-span. Empty or whitespace-only text has
        no boundary.
    """
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    if not text.strip():
        return None

    # Trailing horizontal whitespace never changes the meaning of the edge.
    edge = text.rstrip(" \t")

    if expected_format == "json":
        structural = edge.rstrip()
        if structural and structural[-1] in _JSON_FIELD_END:
            return "json_field"

    last_line = edge.rsplit("\n", 1)[-1].rstrip("\r")
    if _STEP_MARKER.match(last_line):
        return "step"

    index = len(edge) - 1
    while index >= 0 and edge[index] in _CLOSERS:
        index -= 1
    if index >= 0 and edge[index] in _SENTENCE_END:
        return "sentence"

    if edge.endswith(("\n", "\r")):
        return "newline"

    return None
