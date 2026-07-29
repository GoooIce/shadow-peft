"""Top-level harness engine (MLX): route -> ensure adapter -> generate.

MLX counterpart of ``shadow_peft/harness/engine.py``. Composes the pure-Python
``Router``/``Session`` with the MLX ``AdapterRegistryMlx`` to serve multiple
shadow adapters from one base model.

Key MLX differences from torch:
- Generation uses ``mlx_lm.generate.generate_step`` (a token generator), not
  HF ``model.generate``. Generation runs through ``peft_model`` (returns raw
  logits), not the task wrapper (returns a dataclass).
- A per-request KV-cache is created via ``make_prompt_cache(peft)`` and passed
  as ``prompt_cache=`` (NOT ``cache=``).
- There is no ``.device`` on MLX modules; arrays live on the global default
  device.
- The shadow per-layer injection and the internally-managed shadow KV-cache
  stay fully in control (same as torch).
"""

from __future__ import annotations

import logging
from typing import Any

from .config import ShadowHarnessConfig
from .registry_mlx import AdapterRegistryMlx
from .router import RouteDecision, Router
from .session import Session

logger = logging.getLogger(__name__)


class ShadowEngineMlx:
    """Route requests to specialists and generate via a single shared MLX model.

    Parameters
    ----------
    model:
        A live ``shadow_peft_mlx.ShadowForCausalLM``. One base model serves all
        specialists; adapters are swapped in place.
    tokenizer:
        An ``mlx_lm`` tokenizer (``TokenizerWrapper``) used for chat-template
        rendering, encoding, and decoding.
    config:
        Harness configuration describing the specialists, routing rules, and
        the L0 fallback.
    """

    def __init__(self, model, tokenizer, config: ShadowHarnessConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.router = Router(config)
        self.registry = AdapterRegistryMlx(model, config)
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
        max_tokens: int = 256,
        temp: float = 0.0,
        top_p: float = 0.0,
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
            history. When provided, the generated turn is appended.
        max_tokens:
            Maximum number of tokens to generate.
        temp, top_p:
            Sampling parameters (``temp=0`` -> greedy). Forwarded to
            ``mlx_lm.sample_utils.make_sampler``.
        **gen_kwargs:
            Forwarded to ``generate_step`` (e.g. ``prompt_cache=``).

        Returns
        -------
        tuple of ``(text, info)`` where ``info`` carries the routing decision.
        """
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

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

        # 5. Encode + generate through peft_model (returns raw logits).
        prompt_ids = self._encode(prompt)
        sampler = make_sampler(temp=temp, top_p=top_p)
        # Fresh per-request cache; shadow KV-cache is auto-managed by peft_model.
        gen_kwargs.setdefault("prompt_cache", make_prompt_cache(self.model.peft_model))

        generated: list[int] = []
        eos_ids = self._eos_token_ids()
        for token, _ in generate_step(
            mx.array(prompt_ids),
            self.model.peft_model,
            max_tokens=max_tokens,
            sampler=sampler,
            **gen_kwargs,
        ):
            generated.append(int(token))
            if eos_ids and int(token) in eos_ids:
                break

        text = self.tokenizer.decode(generated)

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

        Uses ``tokenizer.apply_chat_template`` when available; falls back to a
        minimal concatenation for toy tokenizers without a chat template.
        """
        messages = [*history, {"role": "user", "content": query}]
        apply = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply):
            try:
                return apply(messages, add_generation_prompt=True, tokenize=False)
            except Exception:
                pass
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        return "\n".join(parts)

    def _encode(self, text: str) -> list[int]:
        """Encode text to a list of token ids, using chat-template-aware encoding."""
        encode = getattr(self.tokenizer, "encode", None)
        if callable(encode):
            try:
                return encode(text, add_special_tokens=False)
            except TypeError:
                return encode(text)
        # Last-resort fallback for stub tokenizers in tests.
        return [(ord(c) % 120 + 4) for c in text[:48]] or [1]

    def _eos_token_ids(self) -> set[int]:
        """Return the set of EOS token ids (mlx_lm exposes ``eos_token_ids``)."""
        ids = getattr(self.tokenizer, "eos_token_ids", None)
        if ids is not None:
            try:
                return set(int(i) for i in ids)
            except TypeError:
                return set()
        single = getattr(self.tokenizer, "eos_token_id", None)
        return {int(single)} if single is not None else set()
