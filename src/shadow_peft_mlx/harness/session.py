"""Per-conversation state: sticky routing + cross-task history isolation.

Mirror of ``shadow_peft/harness/session.py`` -- pure-Python, backend-agnostic.
Keep both files in sync.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class Session:
    """A single conversation tracked across multiple requests."""

    session_id: str = ""
    specialist: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.session_id:
            self.session_id = uuid.uuid4().hex[:12]

    def bind(self, specialist: str) -> None:
        """Bind (or re-bind) this session to a specialist."""
        self.specialist = specialist

    def add_turn(self, role: str, content: str) -> None:
        """Append a turn to the conversation history."""
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")
        self.history.append({"role": role, "content": content})

    def reset(self) -> None:
        """Clear binding and history -- start a fresh conversation in-place."""
        self.specialist = None
        self.history = []
