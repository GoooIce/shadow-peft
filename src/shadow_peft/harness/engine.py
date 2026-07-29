"""Top-level harness engine: route -> ensure adapter -> generate.

Torch-specific. Composes the pure-Python ``Router``/``Session`` with the
torch ``AdapterRegistry`` to serve multiple shadow adapters from one base
model. Generation goes through HuggingFace ``model.generate``, so the existing
``_ShadowLayerWrapper`` per-layer injection and shadow KV-cache machinery stay
fully in control.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import ShadowHarnessConfig
from .registry import AdapterRegistry
from .router import RouteDecision, Router
from .session import Session

logger = logging.getLogger(__name__)


class ShadowEngine:
    """Route requests to specialists and generate via a single shared model.

    Parameters
    ----------
    model:
        A live ``ShadowForCausalLM`` (torch). One base model serves all
        specialists; adapters are swapped in place.
    tokenizer:
        A HuggingFace tokenizer used for chat-template rendering and encoding.
    config:
        Harness configuration describing the specialists, routing rules, and
        the L0 fallback.
    """

    def __init__(self, model, tokenizer, config: ShadowHarnessConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.router = Router(config)
        self.registry = AdapterRegistry(model, config)
        # Sessions keyed by session_id for resume-by-id convenience.
        self._sessions: dict[str, Session] = {}

    # ------------------------------------------------------------------ #
    # Session management
    # ------------------------------------------------------------------ #
    def session(self, session_id: str | None = None) -> Session:
        """Create a new session, or resume one by id if it exists."""
        if session_id is not None and session_id in self._sessions:
            return self._sessions[session_id]
        sess = Session(session_id=session_id or "")
        self._sessions[sess.session_id] = sess
        return sess

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(
        self,
        query: str,
        *,
        metadata: dict[str, str] | None = None,
        session: Session | None = None,
        max_new_tokens: int = 128,
        **gen_kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        """Route ``query`` to a specialist and generate a response.

        Parameters
        ----------
        query:
            The user's request text.
        metadata:
            Optional request metadata for guardrail routing.
        session:
            Optional conversation session for sticky routing and multi-turn
            history. When provided, the generated turn is appended to the
            session history.
        max_new_tokens, **gen_kwargs:
            Forwarded to ``model.generate``.

        Returns
        -------
        tuple of ``(text, info)`` where ``info`` carries the routing decision
        (``specialist``, ``reason``, ``matched``).
        """
        import torch

        # 1. Route.
        decision: RouteDecision = self.router.route(query, metadata, session)

        # 2. Ensure the right adapter is live (swap + reset shadow cache).
        self.registry.ensure(decision.specialist)

        # 3. Apply the specialist's inference mode.
        spec = self.config.specialist(decision.specialist)
        self.model.set_inference_mode(spec.inference_mode)

        # 4. Bind session + assemble prompt with session-isolated history.
        if session is not None:
            session.bind(decision.specialist)
        history = session.history if session is not None else []
        prompt = self._render_prompt(query, history)

        # 5. Generate (shadow per-layer injection happens inside the HF loop).
        device = self.model.device
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # Move tensors to device (works for both BatchEncoding and plain dict).
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        else:
            inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **gen_kwargs,
            )
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = out[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # 6. Record the turn into the session.
        if session is not None:
            session.add_turn("user", query)
            session.add_turn("assistant", text)

        info = {
            "specialist": decision.specialist,
            "reason": decision.reason,
            "matched": decision.matched,
            "inference_mode": spec.inference_mode,
            "session_id": session.session_id if session is not None else None,
        }
        return text, info

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _render_prompt(self, query: str, history: list[dict[str, str]]) -> str:
        """Render a prompt string from the query plus session history.

        Uses ``tokenizer.apply_chat_template`` when available (the common case
        for instruct models). Falls back to a minimal concatenation for toy
        tokenizers without a chat template (e.g. in tests).
        """
        messages = [*history, {"role": "user", "content": query}]
        apply = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply):
            try:
                return apply(messages, add_generation_prompt=True, tokenize=False)
            except Exception:
                # Tokenizer has a chat_template attr but it failed; fall through.
                pass
        # Minimal fallback: concatenate turns with role markers.
        parts = []
        for msg in messages:
            parts.append(f"{msg['role']}: {msg['content']}")
        return "\n".join(parts)
