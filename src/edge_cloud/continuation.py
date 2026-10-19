"""Pure text helpers for the edge-cloud continuation contract.

The cloud may answer a handoff request in one of two ways:

* *continuation*: the returned text is only the remaining part after the edge
  prefix, so the caller appends it;
* *correction*: the returned text is a full replacement answer that already
  contains (and may fix) the prefix, so the caller must not append.

Both helpers only operate on text; they never call a model and never keep state.
"""

from __future__ import annotations


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def looks_like_duplicate_prefix(prefix: str, text: str) -> bool:
    """Return whether *text* starts by restating the whole *prefix*.

    Whitespace layout is normalized first, so a re-wrapped or re-indented prefix
    still counts as a duplicate. An empty prefix never matches.
    """

    normalized_prefix = _collapse_whitespace(prefix)
    if not normalized_prefix:
        return False
    return _collapse_whitespace(text).startswith(normalized_prefix)


def assemble_continuation(prefix: str, continuation: str, *, corrected: bool = False) -> str:
    """Join an edge prefix and a cloud continuation into the final answer.

    With ``corrected=True`` the cloud returned a full replacement answer, so the
    continuation is returned unchanged and the prefix is discarded. Otherwise the
    continuation is appended verbatim, except when it already restates the prefix
    (``looks_like_duplicate_prefix``): then it is already the complete answer and
    is returned as-is instead of duplicating the prefix.
    """

    if corrected or not prefix:
        return continuation
    if not continuation:
        return prefix
    if looks_like_duplicate_prefix(prefix, continuation):
        return continuation
    return prefix + continuation
