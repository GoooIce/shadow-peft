"""Request router: metadata guardrails -> keyword matching -> sticky -> L0 fallback.

Pure-Python and backend-agnostic. The routing precedence is:

1. **metadata rule** -- first matching ``RoutingRule`` forces a specialist
   (overrides everything, including session stickiness).
2. **session sticky** -- if ``sticky_sessions`` is on and the session already
   has a bound specialist, reuse it.
3. **keyword** -- case-insensitive substring match against each specialist's
   ``keywords``; highest ``priority`` wins on ties.
4. **default fallback** -- ``default_specialist`` (L0).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ShadowHarnessConfig
from .session import Session


@dataclass
class RouteDecision:
    """The outcome of a routing decision.

    Attributes
    ----------
    specialist:
        Name of the specialist chosen.
    reason:
        One of ``"metadata_rule"``, ``"session_sticky"``, ``"keyword"``,
        ``"default_fallback"``.
    matched:
        Optional detail about what matched (e.g. the keyword hit). Useful for
        logging and debugging.
    """

    specialist: str
    reason: str
    matched: str | None = None


@dataclass
class _SpecialistMatch:
    """Internal helper: a specialist name plus the keyword that hit."""

    specialist: str
    keyword: str


class Router:
    """Deterministic, metadata/keyword-based specialist router."""

    def __init__(self, config: ShadowHarnessConfig) -> None:
        self._config = config

    def route(
        self,
        query: str,
        metadata: dict[str, str] | None = None,
        session: Session | None = None,
    ) -> RouteDecision:
        """Decide which specialist should handle ``query``.

        Parameters
        ----------
        query:
            The user's request text.
        metadata:
            Optional request metadata for guardrail matching, e.g.
            ``{"domain": "medical"}``.
        session:
            Optional conversation session for sticky routing.
        """
        # 1. Metadata guardrail -- highest priority, overrides stickiness.
        if metadata:
            for rule in self._config.rules:
                if metadata.get(rule.field) == rule.value:
                    return RouteDecision(rule.specialist, "metadata_rule", rule.field)

        # 2. Session sticky -- keep the same specialist mid-conversation.
        if self._config.sticky_sessions and session is not None and session.specialist is not None:
            return RouteDecision(session.specialist, "session_sticky")

        # 3. Keyword matching -- case-insensitive substring, priority breaks ties.
        query_lower = query.lower()
        best: tuple[int, _SpecialistMatch] | None = None
        for s in self._config.specialists:
            for kw in s.keywords:
                if kw and kw.lower() in query_lower:
                    cand = (s.priority, _SpecialistMatch(s.name, kw))
                    if best is None or cand[0] > best[0]:
                        best = cand
        if best is not None:
            match = best[1]
            return RouteDecision(match.specialist, "keyword", match.keyword)

        # 4. Default fallback (L0).
        return RouteDecision(self._config.default_specialist, "default_fallback")
