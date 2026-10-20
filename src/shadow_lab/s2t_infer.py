"""S2T-Local shadow-only inference with an optional selector re-rank.

The wrapper serves a ``ShadowForCausalLM`` in ``shadow_only`` mode without ever
running the base forward. Each step recomputes the **initial shadow hidden**
(``ShadowPeftModel._compute_initial_shadow_hidden``), projects it through
``shadow_lm_head``, keeps the top-K candidates, and optionally lets a trained
:class:`~shadow_lab.s2t_selector.SelectorHead` pick one of them. Without a
head (or without candidates) it falls back to the shadow top-1, so the tokens
are exactly the greedy ``shadow_only`` tokens.

Guarantees (spec §Implementation Decisions 5, Testing Decisions「导出与推理缝」):

- ``h_t`` is the same initial shadow hidden that produced ``shadow_logits``;
  the base hidden state is never used.
- The selector only re-ranks the shadow candidate set, so emitted tokens stay
  inside the shadow vocabulary and inside that step's top-K.
- No base forward runs during generation; only the shadow backbone and the
  frozen shadow LM head are called. A base forward is not required even when
  the selector is missing.
- The wrapped model's ``inference_mode`` and train/eval state are restored;
  ``shadow_lm_head`` / ``lm_head`` are read-only, so ``export_shadow`` output
  is unaffected and the selector remains a standalone adapter file.
- Greedy decoding is deterministic; ``use_cache=True`` uses the shadow
  backbone KV-cache and produces the same tokens as full-prefix recompute.

The selector is loaded with :func:`shadow_lab.s2t_selector.load_selector` and
never enters the ShadowPEFT state dict: it is an additional module with its own
``selector.safetensors`` / ``selector_config.json``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from shadow_lab.s2t_selector import (
    DEFAULT_K,
    SelectorHead,
    load_selector,
    select_candidate,
)

DEFAULT_MAX_NEW_TOKENS = 32
DEFAULT_LATENCY_BUDGET = 0.20


@dataclass(frozen=True, slots=True)
class SelectionStep:
    """One generated position: candidates, greedy pick and final pick.

    ``candidates`` are the shadow top-K token ids (descending logit order) and
    ``greedy_token`` is the shadow top-1 fallback. ``selector_scores`` is
    ``None`` whenever the selector did not run (no head, no candidates, or
    greedy mode). ``selected_token`` always belongs to ``candidates``.
    """

    position: int
    candidates: tuple[int, ...]
    candidate_logits: tuple[float, ...]
    greedy_token: int
    selected_token: int
    used_selector: bool
    selector_scores: tuple[float, ...] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "candidates": list(self.candidates),
            "candidate_logits": list(self.candidate_logits),
            "greedy_token": self.greedy_token,
            "selected_token": self.selected_token,
            "used_selector": self.used_selector,
            "selector_scores": (
                None if self.selector_scores is None else list(self.selector_scores)
            ),
        }


@dataclass(frozen=True, slots=True)
class ShadowGeneration:
    """Generated tokens plus per-step provenance and wall-clock timing.

    ``elapsed_seconds`` covers the whole model loop (shadow forward + selector
    scoring) but excludes tokenization and text decoding. ``steps`` has one
    entry per generated token, including a terminal EOS token.
    """

    prompt: str
    prompt_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    text: str
    steps: tuple[SelectionStep, ...]
    elapsed_seconds: float
    use_cache: bool

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def seconds_per_token(self) -> float:
        if not self.token_ids:
            raise ValueError("seconds_per_token needs at least one generated token")
        return self.elapsed_seconds / len(self.token_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "prompt_ids": list(self.prompt_ids),
            "token_ids": list(self.token_ids),
            "text": self.text,
            "steps": [step.to_dict() for step in self.steps],
            "n_tokens": self.n_tokens,
            "elapsed_seconds": self.elapsed_seconds,
            "seconds_per_token": None if not self.token_ids else self.seconds_per_token,
            "use_cache": self.use_cache,
        }


@dataclass(frozen=True, slots=True)
class SelectionBreakdown:
    """Token-level hit/selection failure split (``delta_hit`` / ``delta_sel``)."""

    n_tokens: int
    hit_count: int
    agree_count: int
    greedy_agree_count: int

    @property
    def hit_at_k(self) -> float:
        return self.hit_count / self.n_tokens

    @property
    def agree_at_1(self) -> float:
        return self.agree_count / self.n_tokens

    @property
    def greedy_agree_at_1(self) -> float:
        return self.greedy_agree_count / self.n_tokens

    @property
    def delta_hit(self) -> float:
        """Fraction of positions where the base top-1 is outside the candidates."""

        return (self.n_tokens - self.hit_count) / self.n_tokens

    @property
    def delta_sel(self) -> float:
        """Fraction of positions where the base top-1 is a candidate but not chosen."""

        return (self.hit_count - self.agree_count) / self.n_tokens

    @property
    def selection_accuracy_on_hits(self) -> float | None:
        if not self.hit_count:
            return None
        return self.agree_count / self.hit_count

    @property
    def selector_gain(self) -> float:
        return self.agree_at_1 - self.greedy_agree_at_1

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_tokens": self.n_tokens,
            "hit_count": self.hit_count,
            "agree_count": self.agree_count,
            "greedy_agree_count": self.greedy_agree_count,
            "hit_at_k": self.hit_at_k,
            "agree_at_1": self.agree_at_1,
            "greedy_agree_at_1": self.greedy_agree_at_1,
            "delta_hit": self.delta_hit,
            "delta_sel": self.delta_sel,
            "selection_accuracy_on_hits": self.selection_accuracy_on_hits,
            "selector_gain": self.selector_gain,
        }


def selection_breakdown(
    steps: Sequence[SelectionStep],
    base_top1: Sequence[int],
) -> SelectionBreakdown:
    """Compare step records with the base top-1 of the same context.

    ``delta_hit`` is the irreducible loss (base top-1 not among the shadow
    candidates); ``delta_sel`` is the avoidable loss (candidate present but the
    selector picked another one). ``greedy_agree_at_1`` uses the recorded
    shadow top-1, so a selector's gain is measured on identical prefixes.
    """

    step_list = list(steps)
    base_list = list(base_top1)
    if not step_list:
        raise ValueError("selection_breakdown needs at least one step")
    if len(step_list) != len(base_list):
        raise ValueError(
            f"steps and base_top1 must have the same length, got "
            f"{len(step_list)} and {len(base_list)}"
        )
    hits = agrees = greedy_agrees = 0
    for index, (step, base) in enumerate(zip(step_list, base_list, strict=True)):
        if isinstance(base, bool) or not isinstance(base, int) or base < 0:
            raise ValueError(f"base_top1[{index}] must be a non-negative int, got {base!r}")
        if step.selected_token not in step.candidates:
            raise ValueError(
                f"steps[{index}].selected_token {step.selected_token} is not in its "
                "candidate set"
            )
        hit = base in step.candidates
        hits += int(hit)
        agrees += int(step.selected_token == base)
        greedy_agrees += int(step.greedy_token == base)
    return SelectionBreakdown(
        n_tokens=len(step_list),
        hit_count=hits,
        agree_count=agrees,
        greedy_agree_count=greedy_agrees,
    )


@dataclass(frozen=True, slots=True)
class LatencyComparison:
    """Selector vs greedy wall-clock per token, with the 20% deploy gate."""

    selector_seconds: float
    selector_tokens: int
    greedy_seconds: float
    greedy_tokens: int
    budget: float

    @property
    def selector_seconds_per_token(self) -> float:
        return self.selector_seconds / self.selector_tokens

    @property
    def greedy_seconds_per_token(self) -> float:
        return self.greedy_seconds / self.greedy_tokens

    @property
    def selector_over_greedy(self) -> float:
        return self.selector_seconds_per_token / self.greedy_seconds_per_token

    @property
    def overhead(self) -> float:
        return self.selector_over_greedy - 1.0

    @property
    def deployable(self) -> bool:
        """False when selector latency is more than ``budget`` above greedy."""

        return self.overhead <= self.budget + 1e-12

    def to_dict(self) -> dict[str, Any]:
        return {
            "selector_seconds": self.selector_seconds,
            "n_selector_tokens": self.selector_tokens,
            "selector_seconds_per_token": self.selector_seconds_per_token,
            "greedy_seconds": self.greedy_seconds,
            "n_greedy_tokens": self.greedy_tokens,
            "greedy_seconds_per_token": self.greedy_seconds_per_token,
            "selector_over_greedy": self.selector_over_greedy,
            "overhead": self.overhead,
            "budget": self.budget,
            "deployable": self.deployable,
        }


def latency_comparison(
    *,
    selector_seconds: float,
    selector_tokens: int,
    greedy_seconds: float,
    greedy_tokens: int,
    budget: float = DEFAULT_LATENCY_BUDGET,
) -> LatencyComparison:
    """Build a :class:`LatencyComparison` from aggregate timing measurements.

    The deploy gate is engineering policy (spec §Implementation Decisions 6):
    more than a 20% per-token latency increase means no deployment by default.
    """

    selector_tokens = _positive_int(selector_tokens, name="selector_tokens")
    greedy_tokens = _positive_int(greedy_tokens, name="greedy_tokens")
    for name, value in (
        ("selector_seconds", selector_seconds),
        ("greedy_seconds", greedy_seconds),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a non-negative number, got {value!r}")
        if value < 0 or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a non-negative finite number, got {value!r}")
    if greedy_seconds == 0:
        raise ValueError("greedy_seconds must be positive to compare latency")
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget < 0:
        raise ValueError(f"budget must be a non-negative number, got {budget!r}")
    return LatencyComparison(
        selector_seconds=float(selector_seconds),
        selector_tokens=selector_tokens,
        greedy_seconds=float(greedy_seconds),
        greedy_tokens=greedy_tokens,
        budget=float(budget),
    )


class SelectorShadowModel:
    """Serve a ``shadow_only`` model with an optional candidate selector.

    Parameters
    ----------
    model:
        A ``ShadowForCausalLM`` whose ``inference_mode`` is ``"shadow_only"``.
        Its base path is never invoked.
    tokenizer:
        HF tokenizer (or a stub exposing ``__call__`` and optionally
        ``decode``/``eos_token_id``) used to encode prompts and decode tokens.
    selector:
        A :class:`SelectorHead`, a path to a selector directory written by
        :func:`save_selector`, or ``None`` for pure shadow-greedy serving.
    k:
        Candidate width. ``k`` larger than the vocabulary is truncated.
    use_selector:
        Default selection policy. ``True`` with no head still falls back to
        greedy, so the same wrapper can serve both modes.
    use_cache:
        Default decode mode. ``False`` recomputes the full prefix each step;
        ``True`` reuses the shadow backbone KV-cache.
    device:
        Override device for the selector head. Defaults to the model device.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        selector: SelectorHead | str | Path | None = None,
        *,
        k: int = DEFAULT_K,
        use_selector: bool = True,
        use_cache: bool = False,
        device: str | torch.device | None = None,
    ) -> None:
        mode = getattr(model, "inference_mode", None)
        if mode != "shadow_only":
            raise ValueError(
                f"SelectorShadowModel needs a shadow_only model, got inference_mode={mode!r}"
            )
        peft_model = getattr(model, "peft_model", None)
        shadow_lm_head = getattr(model, "shadow_lm_head", None)
        if peft_model is None or shadow_lm_head is None:
            raise ValueError(
                "model must be a ShadowForCausalLM with peft_model and shadow_lm_head"
            )
        weight = getattr(shadow_lm_head, "weight", None)
        if weight is None or weight.ndim != 2:
            raise ValueError("shadow_lm_head must expose a 2-D weight table")
        self.model = model
        self.peft_model = peft_model
        self.shadow_lm_head = shadow_lm_head
        self.tokenizer = tokenizer
        self.k = _positive_int(k, name="k")
        if not isinstance(use_selector, bool):
            raise ValueError(f"use_selector must be a bool, got {use_selector!r}")
        if not isinstance(use_cache, bool):
            raise ValueError(f"use_cache must be a bool, got {use_cache!r}")
        self.use_selector = use_selector
        self.use_cache = use_cache
        self.device = torch.device(
            device if device is not None else getattr(model, "device", "cpu")
        )
        self.hidden_size = int(weight.shape[1])
        self.vocab_size = int(weight.shape[0])
        self.selector = self._resolve_selector(selector)

    def _resolve_selector(
        self, selector: SelectorHead | str | Path | None
    ) -> Any:
        if selector is None:
            return None
        if isinstance(selector, (str, Path)):
            selector = load_selector(selector, device=self.device)
        if not callable(getattr(selector, "score", None)):
            raise TypeError("selector must expose a score(h_t, candidate_ids, candidate_weight) method")
        hidden_size = int(getattr(selector, "hidden_size", self.hidden_size))
        candidate_dim = int(getattr(selector, "candidate_dim", self.hidden_size))
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"selector hidden_size {hidden_size} does not match the shadow hidden "
                f"size {self.hidden_size}"
            )
        if candidate_dim != self.hidden_size:
            raise ValueError(
                f"selector candidate_dim {candidate_dim} does not match the shadow LM "
                f"head width {self.hidden_size}"
            )
        if isinstance(selector, torch.nn.Module):
            selector.to(self.device)
            selector.eval()
        return selector

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        use_selector: bool | None = None,
        use_cache: bool | None = None,
    ) -> ShadowGeneration:
        """Greedy-decode ``max_new_tokens`` tokens for one prompt.

        ``use_selector``/``use_cache`` override the constructor defaults.
        Generation stops early when the selected token is
        ``tokenizer.eos_token_id`` (that token is included). No base forward is
        run; only the shadow backbone and ``shadow_lm_head`` are called.
        """
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")
        max_new_tokens = _positive_int(max_new_tokens, name="max_new_tokens")
        selector_enabled = self.use_selector if use_selector is None else use_selector
        if not isinstance(selector_enabled, bool):
            raise ValueError(f"use_selector must be a bool, got {selector_enabled!r}")
        cache_enabled = self.use_cache if use_cache is None else use_cache
        if not isinstance(cache_enabled, bool):
            raise ValueError(f"use_cache must be a bool, got {cache_enabled!r}")

        input_ids, attention_mask = self._encode(prompt)
        prompt_ids = tuple(int(token) for token in input_ids[0].tolist())
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        model_was_training = bool(getattr(self.model, "training", False))
        selector_was_training = (
            bool(self.selector.training)
            if isinstance(self.selector, torch.nn.Module)
            else None
        )
        self.model.eval()
        if isinstance(self.selector, torch.nn.Module):
            self.selector.eval()
        steps: list[SelectionStep] = []
        token_ids: list[int] = []
        cache_active = cache_enabled
        if cache_enabled:
            # Fresh shadow cache; the first call prefills and adopts it.
            self.peft_model._resolve_shadow_cache(True, None)
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                for position in range(max_new_tokens):
                    if cache_active and position > 0 and not self._shadow_cache_ready():
                        # The backbone did not materialize a cache; fall back to
                        # full-prefix recompute for the remaining steps.
                        cache_active = False
                    if cache_active:
                        step_ids = input_ids if position == 0 else input_ids[:, -1:]
                        position_ids = (
                            None
                            if position == 0
                            else torch.tensor(
                                [[input_ids.shape[1] - 1]],
                                dtype=torch.long,
                                device=input_ids.device,
                            )
                        )
                        shadow_cache = self.peft_model._shadow_cache
                    else:
                        step_ids = input_ids
                        position_ids = None
                        shadow_cache = None
                    hidden = self.peft_model._compute_initial_shadow_hidden(
                        input_ids=step_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        inputs_embeds=None,
                        shadow_cache=shadow_cache,
                    )
                    if cache_active and position == 0 and not self._shadow_cache_ready():
                        cache_active = False
                    shadow_logits = self.shadow_lm_head(hidden)[0, -1].float()
                    step = self._select(
                        position=position,
                        h_t=hidden[0, -1],
                        shadow_logits=shadow_logits,
                        selector_enabled=selector_enabled,
                    )
                    steps.append(step)
                    token_ids.append(step.selected_token)
                    next_token = torch.tensor(
                        [[step.selected_token]],
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    )
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones_like(next_token)],
                        dim=1,
                    )
                    if eos_token_id is not None and step.selected_token == eos_token_id:
                        break
        finally:
            elapsed = time.perf_counter() - started
            if cache_enabled:
                self.peft_model._resolve_shadow_cache(False, None)
            self.model.train(model_was_training)
            if isinstance(self.selector, torch.nn.Module) and selector_was_training is not None:
                self.selector.train(selector_was_training)

        return ShadowGeneration(
            prompt=prompt,
            prompt_ids=prompt_ids,
            token_ids=tuple(token_ids),
            text=self._decode(token_ids),
            steps=tuple(steps),
            elapsed_seconds=elapsed,
            use_cache=cache_active,
        )

    def _shadow_cache_ready(self) -> bool:
        """Whether the shadow backbone materialized a non-empty KV-cache."""

        cache = self.peft_model._shadow_cache
        if cache is None:
            return False
        get_length = getattr(cache, "get_seq_length", None)
        if not callable(get_length):
            return True
        try:
            return int(get_length()) > 0
        except (TypeError, ValueError):
            return False

    def _select(
        self,
        *,
        position: int,
        h_t: torch.Tensor,
        shadow_logits: torch.Tensor,
        selector_enabled: bool,
    ) -> SelectionStep:
        effective_k = min(self.k, int(shadow_logits.shape[-1]))
        top = torch.topk(shadow_logits, k=effective_k)
        candidate_ids = top.indices
        candidate_logits = top.values
        if candidate_ids.numel():
            greedy_token = int(candidate_ids[0])
        else:
            greedy_token = int(shadow_logits.argmax())
        selected_token = greedy_token
        used_selector = False
        selector_scores: tuple[float, ...] | None = None
        if selector_enabled and self.selector is not None and candidate_ids.numel():
            scores = self.selector.score(
                h_t,
                candidate_ids.unsqueeze(0),
                self.shadow_lm_head.weight,
            )
            if scores.shape != (1, candidate_ids.shape[-1]):
                raise RuntimeError(
                    "selector.score returned shape "
                    f"{tuple(scores.shape)}, expected {(1, int(candidate_ids.shape[-1]))}"
                )
            selected_token = int(select_candidate(scores, candidate_ids.unsqueeze(0))[0])
            selector_scores = tuple(
                float(value) for value in scores[0].detach().float().cpu().tolist()
            )
            used_selector = True
        return SelectionStep(
            position=position,
            candidates=tuple(int(token) for token in candidate_ids.tolist()),
            candidate_logits=tuple(float(value) for value in candidate_logits.tolist()),
            greedy_token=greedy_token,
            selected_token=selected_token,
            used_selector=used_selector,
            selector_scores=selector_scores,
        )

    def _encode(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.tokenizer(prompt, return_tensors="pt")
        input_ids = batch["input_ids"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"tokenizer must return one id row, got shape {tuple(input_ids.shape)}"
            )
        if int(input_ids.shape[-1]) == 0:
            raise ValueError("prompt tokenized to zero tokens")
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} does not match "
                f"input_ids shape {tuple(input_ids.shape)}"
            )
        return (
            input_ids.to(self.device),
            attention_mask.to(self.device),
        )

    def _decode(self, token_ids: Sequence[int]) -> str:
        decode = getattr(self.tokenizer, "decode", None)
        if not callable(decode):
            return ""
        return str(decode(list(token_ids), skip_special_tokens=True))


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


__all__ = [
    "DEFAULT_LATENCY_BUDGET",
    "DEFAULT_MAX_NEW_TOKENS",
    "LatencyComparison",
    "SelectionBreakdown",
    "SelectionStep",
    "SelectorShadowModel",
    "ShadowGeneration",
    "latency_comparison",
    "selection_breakdown",
]
