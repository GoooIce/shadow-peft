"""S2T-Local selector head: re-rank the shadow model's own top-K candidates.

The shadow path is cheap enough to serve alone, but the diagnostic
(``docs/exp11-s2t-diagnostic.md``) shows its top-8 candidates cover the base
top-1 89.8% of the time while its own top-1 agrees only 72.1%.  A selector
head learns to pick the base-preferred candidate without running the base
model at inference time.

Design constraints (spec issue #3, §Implementation Decisions 3/4):

- ``score(c) = MLP([h_t ; e_c])`` with ``e_c`` read from the shadow LM head
  weight table (``shadow_lm_head.weight[c]``, or ``lm_head.weight[c]`` for an
  exported shadow-only model).  The head never touches the shape or values of
  either LM head.
- Candidates are the shadow model's top-K logits (K=8).  The selector can
  only re-rank inside that set; it cannot emit an out-of-vocabulary token.
- **Shadow hidden caveat:** ``h_t`` collected by :func:`collect_selector_examples`
  is the *initial* shadow hidden state returned by
  ``ShadowPeftModel.forward_with_shadow`` — the same hidden state that produced
  ``shadow_logits`` via ``shadow_lm_head``.  It is not the base hidden state and
  not the injected/updated ``base_shadow`` deployment hidden.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file
from torch import nn

IGNORE_INDEX = -100
DEFAULT_K = 8
DEFAULT_BETA = 0.1
DEFAULT_MLP_HIDDEN = 64
SELECTOR_WEIGHTS_FILENAME = "selector.safetensors"
SELECTOR_CONFIG_FILENAME = "selector_config.json"
SELECTOR_VERSION = "s2t-selector/v1"


class SelectorHead(nn.Module):
    """Shared small MLP scoring one (hidden state, candidate embedding) pair.

    ``score(c) = MLP([h_t ; e_c])``.  The head owns only the MLP; candidate
    vectors are passed in at call time so the (large) LM head weight table is
    never copied, frozen-in, or modified.

    Parameters
    ----------
    hidden_size:
        Width of ``h_t``.
    candidate_dim:
        Width of ``e_c``.  Defaults to ``hidden_size`` (the usual case when
        the LM head input width matches the backbone hidden width).
    mlp_hidden:
        Hidden width of the two-layer MLP; the default 64 keeps the Qwen3-4B
        case (hidden 2560, candidate input 5120) at ~328k parameters, inside
        the spec's "few hundred thousand" budget.
    """

    def __init__(
        self,
        hidden_size: int,
        candidate_dim: int | None = None,
        mlp_hidden: int = DEFAULT_MLP_HIDDEN,
    ) -> None:
        super().__init__()
        hidden_size = _positive_int(hidden_size, name="hidden_size")
        candidate_dim = (
            hidden_size if candidate_dim is None else _positive_int(candidate_dim, name="candidate_dim")
        )
        mlp_hidden = _positive_int(mlp_hidden, name="mlp_hidden")
        self.hidden_size = hidden_size
        self.candidate_dim = candidate_dim
        self.mlp_hidden = mlp_hidden
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size + candidate_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, h_t: torch.Tensor, candidate_embeds: torch.Tensor) -> torch.Tensor:
        """Score candidates from a hidden state and candidate vectors.

        ``h_t`` is ``[..., hidden_size]`` and ``candidate_embeds`` is
        ``[..., num_candidates, candidate_dim]``; the result is
        ``[..., num_candidates]``.  Leading dimensions must broadcast.
        """
        if candidate_embeds.shape[-1] != self.candidate_dim:
            raise ValueError(
                f"candidate_embeds last dim {candidate_embeds.shape[-1]} != "
                f"candidate_dim {self.candidate_dim}"
            )
        if h_t.shape[-1] != self.hidden_size:
            raise ValueError(
                f"h_t last dim {h_t.shape[-1]} != hidden_size {self.hidden_size}"
            )
        param_dtype = self.mlp[0].weight.dtype
        h = h_t.to(param_dtype)
        embeds = candidate_embeds.to(param_dtype)
        h_expanded = h.unsqueeze(-2).expand(*embeds.shape[:-1], self.hidden_size)
        return self.mlp(torch.cat([h_expanded, embeds], dim=-1)).squeeze(-1)

    def score(
        self,
        h_t: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Score ``candidate_ids`` using rows of ``candidate_weight``.

        ``candidate_weight`` is the LM head weight table (``[vocab, dim]``),
        e.g. ``model.shadow_lm_head.weight`` or an exported model's
        ``lm_head.weight``.  Only the selected rows are read; the table is
        never modified.
        """
        if candidate_weight.ndim != 2:
            raise ValueError(f"candidate_weight must be 2-D, got {tuple(candidate_weight.shape)}")
        if candidate_weight.shape[1] != self.candidate_dim:
            raise ValueError(
                f"candidate_weight width {candidate_weight.shape[1]} != "
                f"candidate_dim {self.candidate_dim}"
            )
        return self.forward(h_t, candidate_weight[candidate_ids])


def candidate_ce(
    scores: torch.Tensor,
    target_idx: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy over candidate scores, with ``IGNORE_INDEX`` rows dropped.

    ``scores`` is ``[n, num_candidates]``; ``target_idx`` holds candidate
    positions (0..K-1), where ``IGNORE_INDEX`` marks a row that must not
    contribute (e.g. the base top-1 is outside the candidate set).  Rows are
    dropped from the denominator, so ``beta=0`` training on an all-miss batch
    yields exactly zero rather than NaN.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D, got {tuple(scores.shape)}")
    target_idx = target_idx.to(device=scores.device, dtype=torch.long)
    if target_idx.shape != (scores.shape[0],):
        raise ValueError(
            f"target_idx shape {tuple(target_idx.shape)} != ({scores.shape[0]},)"
        )
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction must be mean/sum/none, got {reduction!r}")
    valid = target_idx != IGNORE_INDEX
    if not bool(valid.any()):
        zero = scores.sum() * 0.0
        if reduction == "none":
            return scores.new_zeros(scores.shape[0])
        return zero
    if reduction == "mean":
        return F.cross_entropy(scores[valid], target_idx[valid], reduction="mean")
    if reduction == "sum":
        return F.cross_entropy(scores[valid], target_idx[valid], reduction="sum")
    safe_target = target_idx.clamp(min=0)
    return F.cross_entropy(scores, safe_target, reduction="none") * valid


def select_candidate(scores: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
    """Argmax inside the candidate set, returning the original token ids."""
    if scores.shape != candidate_ids.shape:
        raise ValueError(
            f"scores shape {tuple(scores.shape)} != candidate_ids shape {tuple(candidate_ids.shape)}"
        )
    best = scores.argmax(dim=-1, keepdim=True)
    return candidate_ids.gather(-1, best).squeeze(-1)


@dataclass(frozen=True, slots=True)
class SelectorLoss:
    """Decomposed training loss: ``total = sel + beta * reg``."""

    total: torch.Tensor
    sel: torch.Tensor
    reg: torch.Tensor
    n_sel: int
    n_reg: int


def selector_loss(
    scores: torch.Tensor,
    candidate_ids: torch.Tensor,
    base_top1: torch.Tensor,
    shadow_top1: torch.Tensor | None = None,
    *,
    beta: float = DEFAULT_BETA,
) -> SelectorLoss:
    """S2T-Local training loss over candidate scores.

    ``L_sel`` is a candidate-internal cross-entropy whose target is the base
    top-1, computed **only** for positions whose base top-1 is in that
    position's candidate set.  Missing positions are dropped from the mean, so
    a hit failure never masquerades as a selection error.

    ``L_reg`` is a candidate-internal cross-entropy whose target is the shadow
    top-1 (the unmodified shadow preference), computed for every supplied
    non-padding position.  ``beta=0`` makes the loss pure ``L_sel``.

    ``shadow_top1`` defaults to ``candidate_ids[:, 0]`` (topk is sorted).
    Rows whose shadow top-1 is somehow absent from the candidate set are
    dropped from ``L_reg`` as well.
    """
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta!r}")
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D, got {tuple(scores.shape)}")
    if candidate_ids.shape != scores.shape:
        raise ValueError(
            f"candidate_ids shape {tuple(candidate_ids.shape)} != scores shape {tuple(scores.shape)}"
        )
    candidate_ids = candidate_ids.to(device=scores.device)
    base_top1 = base_top1.to(device=scores.device, dtype=torch.long)
    if shadow_top1 is None:
        shadow_top1 = candidate_ids[:, 0]
    shadow_top1 = shadow_top1.to(device=scores.device, dtype=torch.long)

    sel_target = _candidate_targets(candidate_ids, base_top1)
    reg_target = _candidate_targets(candidate_ids, shadow_top1)
    sel = candidate_ce(scores, sel_target, reduction="mean")
    reg = candidate_ce(scores, reg_target, reduction="mean")
    total = sel + beta * reg
    return SelectorLoss(
        total=total,
        sel=sel,
        reg=reg,
        n_sel=int((sel_target != IGNORE_INDEX).sum().item()),
        n_reg=int((reg_target != IGNORE_INDEX).sum().item()),
    )


def _candidate_targets(candidate_ids: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Map vocabulary token ids to candidate slots, or ``IGNORE_INDEX``."""
    matches = candidate_ids == token_ids.unsqueeze(-1)
    hit = matches.any(dim=-1)
    slot = matches.float().argmax(dim=-1)
    return torch.where(hit, slot, torch.full_like(slot, IGNORE_INDEX)).long()


@dataclass(frozen=True, slots=True)
class SelectorExample:
    """One generated token's selector training signal.

    ``h_t`` is the **initial shadow hidden state** (the one that produced
    ``shadow_logits`` via ``shadow_lm_head``), stored on CPU in float32.
    ``candidate_ids``/``candidate_logits`` are the shadow top-K for that
    position; ``hit`` is ``base_top1 in candidate_ids``.
    """

    prompt_id: str
    task: str
    position: int
    h_t: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_logits: torch.Tensor
    base_top1: int
    shadow_top1: int
    hit: bool


@dataclass(frozen=True, slots=True)
class _PromptSpec:
    prompt_id: str
    text: str
    task: str


def _normalize_prompts(
    prompts: Sequence[str | Mapping[str, Any]],
    *,
    task_key: str,
) -> list[_PromptSpec]:
    if isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence):
        raise TypeError("prompts must be a sequence of strings or mappings")
    specs: list[_PromptSpec] = []
    for index, item in enumerate(prompts):
        if isinstance(item, str):
            specs.append(_PromptSpec(prompt_id=f"p{index}", text=item, task="unspecified"))
        elif isinstance(item, Mapping):
            text = item.get("prompt")
            if not isinstance(text, str):
                raise ValueError(f"prompt[{index}] needs a string 'prompt' field")
            raw_task = item.get(task_key)
            raw_id = item.get("prompt_id")
            specs.append(
                _PromptSpec(
                    prompt_id=f"p{index}" if raw_id is None else str(raw_id),
                    text=text,
                    task="unspecified" if raw_task is None or raw_task == "" else str(raw_task),
                )
            )
        else:
            raise TypeError(
                f"prompt[{index}] must be a string or mapping, got {type(item).__name__}"
            )
    return specs


def collect_selector_examples(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str | Mapping[str, Any]],
    *,
    max_new_tokens: int,
    k: int = DEFAULT_K,
    seed: int = 0,
    task_key: str = "task",
) -> list[SelectorExample]:
    """Collect per-token selector examples from greedy ``base_shadow`` decoding.

    Generation advances with the **base** top-1 token, exactly like the
    diagnostic, while each recorded position keeps the shadow top-K
    candidates.  ``h_t`` is the *initial* shadow hidden returned by
    ``ShadowPeftModel.forward_with_shadow``: the hidden state that produced
    ``shadow_logits`` through ``shadow_lm_head``.  It is **not** the base
    hidden and **not** the injected ``base_shadow`` hidden; do not substitute
    one for the other when training the selector.

    The model must be a ``ShadowForCausalLM`` in ``base_shadow`` mode.  No
    gradients are tracked (collection runs under ``torch.inference_mode``);
    the caller's train/eval state is restored on exit.  Empty prompts are
    skipped.  Generation stops after ``max_new_tokens`` or when the base emits
    ``tokenizer.eos_token_id`` (that EOS token is recorded).  All returned
    tensors are detached CPU float32 (ids are long), so the examples are safe
    to store and move to a training device later.
    """
    max_new_tokens = _positive_int(max_new_tokens, name="max_new_tokens")
    k = _positive_int(k, name="k")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    mode = getattr(model, "inference_mode", None)
    if mode != "base_shadow":
        raise ValueError(f"collect_selector_examples needs a base_shadow model, got {mode!r}")
    peft_model = getattr(model, "peft_model", None)
    shadow_head = getattr(model, "shadow_lm_head", None)
    if peft_model is None or shadow_head is None:
        raise ValueError("model must be a ShadowForCausalLM with peft_model and shadow_lm_head")

    specs = _normalize_prompts(prompts, task_key=task_key)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device_of(model)
    was_training = bool(getattr(model, "training", False))
    model.eval()
    examples: list[SelectorExample] = []
    try:
        with torch.inference_mode():
            for spec in specs:
                if not spec.text.strip():
                    continue
                batch = tokenizer(spec.text, return_tensors="pt")
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                attention_mask = attention_mask.to(device)
                if int(input_ids.shape[-1]) == 0:
                    continue
                eos_token_id = getattr(tokenizer, "eos_token_id", None)
                for position in range(max_new_tokens):
                    outputs, shadow_hidden = peft_model.forward_with_shadow(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                    base_logits = outputs.logits[0, -1].float()
                    shadow_logits = shadow_head(shadow_hidden)[0, -1].float()
                    effective_k = min(k, int(shadow_logits.shape[-1]))
                    top = torch.topk(shadow_logits, k=effective_k)
                    base_top1 = int(base_logits.argmax().item())
                    candidate_ids = top.indices.detach().to("cpu", torch.long)
                    examples.append(
                        SelectorExample(
                            prompt_id=spec.prompt_id,
                            task=spec.task,
                            position=position,
                            h_t=shadow_hidden[0, -1].detach().to("cpu", torch.float32),
                            candidate_ids=candidate_ids,
                            candidate_logits=top.values.detach().to("cpu", torch.float32),
                            base_top1=base_top1,
                            shadow_top1=int(shadow_logits.argmax().item()),
                            hit=bool((candidate_ids == base_top1).any().item()),
                        )
                    )
                    if eos_token_id is not None and base_top1 == eos_token_id:
                        break
                    if position + 1 >= max_new_tokens:
                        break
                    input_ids = torch.cat(
                        [
                            input_ids,
                            torch.tensor([[base_top1]], dtype=input_ids.dtype, device=device),
                        ],
                        dim=1,
                    )
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(
                                (1, 1), dtype=attention_mask.dtype, device=device
                            ),
                        ],
                        dim=1,
                    )
    finally:
        model.train(was_training)
    return examples


def _device_of(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def split_selector_examples(
    examples: Sequence[SelectorExample],
    *,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[list[SelectorExample], list[SelectorExample]]:
    """Split examples by prompt id, keeping every token of a prompt on one side.

    The validation unit is the prompt, never the token, so no prompt's
    trajectory can leak across the split.  With fewer than two prompts (or
    ``val_ratio=0``) everything stays in train and validation is empty.
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio!r}")
    prompt_ids = list(dict.fromkeys(example.prompt_id for example in examples))
    if val_ratio == 0.0 or len(prompt_ids) < 2:
        return list(examples), []
    n_val = max(1, int(round(len(prompt_ids) * val_ratio)))
    n_val = min(n_val, len(prompt_ids) - 1)
    shuffled = list(prompt_ids)
    random.Random(seed).shuffle(shuffled)
    val_ids = set(shuffled[:n_val])
    train = [example for example in examples if example.prompt_id not in val_ids]
    val = [example for example in examples if example.prompt_id in val_ids]
    return train, val


@dataclass(frozen=True, slots=True)
class SelectorTrainResult:
    """Trained head plus its step-loss history and prompt-grouped split sizes."""

    head: SelectorHead
    history: tuple[float, ...]
    val_loss: float | None
    steps: int
    n_train_examples: int
    n_val_examples: int
    n_train_prompts: int
    n_val_prompts: int


def train_selector(
    examples: Sequence[SelectorExample],
    *,
    candidate_weight: torch.Tensor,
    epochs: int = 1,
    steps: int | None = None,
    lr: float = 1e-3,
    beta: float = DEFAULT_BETA,
    seed: int = 0,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
    val_ratio: float = 0.1,
    mlp_hidden: int = DEFAULT_MLP_HIDDEN,
    head: SelectorHead | None = None,
) -> SelectorTrainResult:
    """Train :class:`SelectorHead` with Adam on candidate-internal CE.

    ``candidate_weight`` is the LM head weight table used to look up ``e_c``
    (e.g. ``model.shadow_lm_head.weight``); it is read-only and detached.
    Examples are split by ``prompt_id`` before training, so validation tokens
    never share a prompt with training tokens.

    ``steps`` (when given) overrides ``epochs`` and runs exactly that many
    optimizer steps, cycling the train set if needed.  Otherwise the head runs
    ``epochs`` passes over the train split.  ``beta`` weights ``L_reg``
    (shadow-preference regularizer); ``beta=0`` is pure ``L_sel``.

    Returns the head in eval mode, the per-step training losses, and the
    validation loss over the whole val split (``None`` when val is empty).
    Reseeding torch and the shuffle RNG makes fixed-seed runs bit-identical.
    """
    epochs = _positive_int(epochs, name="epochs")
    if steps is not None:
        steps = _positive_int(steps, name="steps")
    batch_size = _positive_int(batch_size, name="batch_size")
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr!r}")
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta!r}")
    if not examples:
        raise ValueError("examples is empty; collect selector examples first")

    weight = candidate_weight.detach()
    if weight.ndim != 2:
        raise ValueError(f"candidate_weight must be 2-D, got {tuple(weight.shape)}")
    hidden_size = int(examples[0].h_t.shape[-1])
    num_candidates = int(examples[0].candidate_ids.shape[-1])
    for index, example in enumerate(examples):
        if int(example.h_t.shape[-1]) != hidden_size:
            raise ValueError(f"examples[{index}].h_t has an inconsistent hidden size")
        if int(example.candidate_ids.shape[-1]) != num_candidates:
            raise ValueError(
                f"examples[{index}] has {example.candidate_ids.shape[-1]} candidates; "
                f"expected {num_candidates}"
            )
    torch.manual_seed(seed)
    if head is None:
        if int(weight.shape[1]) != hidden_size:
            raise ValueError(
                f"candidate_weight width {weight.shape[1]} does not match h_t width {hidden_size}"
            )
        head = SelectorHead(hidden_size, mlp_hidden=mlp_hidden)
    elif head.hidden_size != hidden_size or head.candidate_dim != int(weight.shape[1]):
        raise ValueError(
            f"head shape (hidden_size={head.hidden_size}, candidate_dim={head.candidate_dim}) "
            f"does not match (h_t={hidden_size}, weight={weight.shape[1]})"
        )
    target_device = torch.device(device)
    head = head.to(target_device)
    head.train()
    weight = weight.to(target_device)

    train_examples, val_examples = split_selector_examples(
        examples, val_ratio=val_ratio, seed=seed
    )
    train_prompt_ids = {example.prompt_id for example in train_examples}
    val_prompt_ids = {example.prompt_id for example in val_examples}
    if train_prompt_ids & val_prompt_ids:
        raise AssertionError("prompt split leaked between train and val")

    train_tensors = _stack_examples(train_examples, device=target_device)
    target_steps = (
        steps
        if steps is not None
        else epochs * ((len(train_examples) + batch_size - 1) // batch_size)
    )
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    rng = random.Random(seed)
    history: list[float] = []
    completed = 0
    epoch = 0
    while completed < target_steps:
        order = list(range(len(train_examples)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            if completed >= target_steps:
                break
            index = torch.tensor(
                order[start : start + batch_size], dtype=torch.long, device=target_device
            )
            scores = _score_batch(head, train_tensors, index, weight)
            loss = _batch_loss(scores, train_tensors, index, beta).total
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach().cpu()))
            completed += 1
        epoch += 1
        if steps is None and epoch >= epochs:
            break

    head.eval()
    val_loss: float | None = None
    if val_examples:
        val_tensors = _stack_examples(val_examples, device=target_device)
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(val_examples), batch_size):
                index = torch.arange(
                    start,
                    min(start + batch_size, len(val_examples)),
                    dtype=torch.long,
                    device=target_device,
                )
                scores = _score_batch(head, val_tensors, index, weight)
                batch_loss = _batch_loss(scores, val_tensors, index, beta).total
                total += float(batch_loss.cpu()) * index.numel()
        val_loss = total / len(val_examples)

    return SelectorTrainResult(
        head=head,
        history=tuple(history),
        val_loss=val_loss,
        steps=completed,
        n_train_examples=len(train_examples),
        n_val_examples=len(val_examples),
        n_train_prompts=len(train_prompt_ids),
        n_val_prompts=len(val_prompt_ids),
    )


@dataclass(frozen=True, slots=True)
class _StackedExamples:
    h_t: torch.Tensor
    candidate_ids: torch.Tensor
    base_top1: torch.Tensor
    shadow_top1: torch.Tensor


def _stack_examples(
    examples: Sequence[SelectorExample], *, device: torch.device
) -> _StackedExamples:
    return _StackedExamples(
        h_t=torch.stack([example.h_t.float() for example in examples]).to(device),
        candidate_ids=torch.stack([example.candidate_ids.long() for example in examples]).to(
            device
        ),
        base_top1=torch.tensor(
            [example.base_top1 for example in examples], dtype=torch.long, device=device
        ),
        shadow_top1=torch.tensor(
            [example.shadow_top1 for example in examples], dtype=torch.long, device=device
        ),
    )


def _score_batch(
    head: SelectorHead,
    tensors: _StackedExamples,
    index: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    candidate_ids = tensors.candidate_ids[index]
    return head(tensors.h_t[index], weight[candidate_ids])


def _batch_loss(
    scores: torch.Tensor,
    tensors: _StackedExamples,
    index: torch.Tensor,
    beta: float,
) -> SelectorLoss:
    return selector_loss(
        scores,
        tensors.candidate_ids[index],
        tensors.base_top1[index],
        tensors.shadow_top1[index],
        beta=beta,
    )


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def save_selector(
    head: SelectorHead,
    out_dir: str | Path,
    *,
    k: int | None = None,
) -> dict[str, str]:
    """Write ``selector.safetensors`` + ``selector_config.json`` into ``out_dir``.

    The selector head is stored as a standalone adapter beside the ShadowPEFT
    checkpoint: it never enters ``shadow_adapter.safetensors``, the shadow
    backbone, or either LM head, so ``export_shadow`` semantics are unchanged.
    ``k`` records the fixed candidate width; when omitted it falls back to
    ``head.k`` (set by :func:`load_selector`) or :data:`DEFAULT_K`.
    """
    if k is None:
        k = int(getattr(head, "k", DEFAULT_K))
    k = _positive_int(k, name="k")
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in head.state_dict().items()
    }
    weights_path = out / SELECTOR_WEIGHTS_FILENAME
    config_path = out / SELECTOR_CONFIG_FILENAME
    safetensors_save_file(state, str(weights_path))
    config = {
        "version": SELECTOR_VERSION,
        "hidden_size": int(head.hidden_size),
        "candidate_dim": int(head.candidate_dim),
        "mlp_hidden": int(head.mlp_hidden),
        "k": int(k),
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    head.k = int(k)
    return {"weights": str(weights_path), "config": str(config_path)}


def load_selector(
    out_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> SelectorHead:
    """Load a selector head written by :func:`save_selector`.

    Restores the architecture from ``selector_config.json`` and the MLP
    weights from ``selector.safetensors``; trained scoring is bit-identical to
    the saved head (no re-initialization, no dtype change).
    """
    out = Path(out_dir).expanduser()
    config_path = out / SELECTOR_CONFIG_FILENAME
    weights_path = out / SELECTOR_WEIGHTS_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing selector config: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing selector weights: {weights_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    version = config.get("version")
    if version != SELECTOR_VERSION:
        raise ValueError(
            f"Unsupported selector config version {version!r}; expected {SELECTOR_VERSION!r}"
        )
    head = SelectorHead(
        hidden_size=_config_positive_int(config, "hidden_size"),
        candidate_dim=_config_positive_int(config, "candidate_dim"),
        mlp_hidden=_config_positive_int(config, "mlp_hidden"),
    )
    state = safetensors_load_file(str(weights_path), device=str(device))
    head.load_state_dict(state, strict=True)
    head.to(device)
    head.eval()
    head.k = _config_positive_int(config, "k")
    return head


def _config_positive_int(config: Mapping[str, Any], key: str) -> int:
    if key not in config:
        raise ValueError(f"selector config is missing {key!r}")
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"selector config {key!r} must be a positive integer, got {value!r}")
    return value


__all__ = [
    "DEFAULT_BETA",
    "DEFAULT_K",
    "DEFAULT_MLP_HIDDEN",
    "IGNORE_INDEX",
    "SELECTOR_CONFIG_FILENAME",
    "SELECTOR_VERSION",
    "SELECTOR_WEIGHTS_FILENAME",
    "SelectorExample",
    "SelectorHead",
    "SelectorLoss",
    "SelectorTrainResult",
    "candidate_ce",
    "collect_selector_examples",
    "load_selector",
    "save_selector",
    "select_candidate",
    "selector_loss",
    "split_selector_examples",
    "train_selector",
]
