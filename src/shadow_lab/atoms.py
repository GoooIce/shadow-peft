"""Shared atoms for the shadow-peft experiment lab (torch side).

One home for the helpers every experiment script used to re-declare:
device pick, teacher-data loading, batch packing, chunked perplexity,
and the wikitext-2 test-text loader (with the parquet workaround).
"""

from __future__ import annotations

import json
import math
import random

import torch

# <|endoftext|> for the Qwen3 family. Used as the pad id in distillation
# batches; any masked pad id would do, this one keeps parity with the
# original E1/E2 runs.
QWEN3_ENDOFTEXT_ID = 151643


def pick_device() -> str:
    """cuda > mps > cpu — the device every torch experiment wants."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_teacher_data(path: str, min_len: int = 32) -> list[list[int]]:
    """Read teacher-distillation JSONL (one {"ids": [...]} per line)."""
    seqs = []
    with open(path) as f:
        for line in f:
            ids = json.loads(line)["ids"]
            if len(ids) >= min_len:
                seqs.append(ids)
    return seqs


def make_batch(
    seqs: list[list[int]],
    rng: random.Random,
    batch_size: int,
    seq_len: int,
    device: str,
    pad_id: int = QWEN3_ENDOFTEXT_ID,
) -> dict[str, torch.Tensor]:
    """Sample batch_size crops of seq_len tokens, right-padded, labels masked."""
    picked = [rng.choice(seqs) for _ in range(batch_size)]
    cropped = [s[:seq_len] for s in picked]
    maxlen = max(len(s) for s in cropped)
    input_ids = torch.full((batch_size, maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((batch_size, maxlen), dtype=torch.long)
    for i, s in enumerate(cropped):
        input_ids[i, : len(s)] = torch.tensor(s)
        attn[i, : len(s)] = 1
    labels = input_ids.clone()
    labels[attn == 0] = -100
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attn.to(device),
        "labels": labels.to(device),
    }


def bos_ids(tokenizer, text: str) -> list[int]:
    """Encode with a guaranteed leading BOS (Gemma4 ships add_bos_token=False)."""
    ids = tokenizer.encode(text)
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None and (not ids or ids[0] != bos):
        ids = [bos] + ids
    return ids


def _axis_windows(n_ids: int, chunks: int, chunk_len: int, context_len: int):
    """The eval-axis window walk shared by ``eval_ppl`` and ``eval_kld``.

    Yields ``(start, ctx, keep_from)`` per window: ``ctx`` is where the input
    tensor begins (context carry), ``keep_from`` the crop index that drops the
    carried context from the scored span (None when the whole input scores).
    The limit formula intentionally yields ``chunks - 1`` full windows for
    long texts (parity with the original E1/E2 scripts — window starts and
    count must stay comparable); the guard below only rescues the exact-fit
    case and never fires in that regime.
    """
    limit = min(n_ids, chunks * chunk_len) - chunk_len - 1
    if limit <= 0 and n_ids >= chunk_len + 1:
        limit = 1  # exact-fit input: force the single full window
    for start in range(0, limit, chunk_len):
        ctx = max(0, start - context_len)
        keep_from = (start - ctx - 1) if (context_len > 0 and start > 0) else None
        yield start, ctx, keep_from


@torch.no_grad()
def eval_ppl(
    model,
    tokenizer,
    text: str,
    *,
    chunks: int = 8,
    chunk_len: int = 512,
    device,
    context_len: int = 0,
) -> float:
    """Chunked perplexity over the first `chunks` windows of `text`.

    Works for any causal LM returning `.logits` (HF base model or
    ShadowForCausalLM). Restores the model's prior train/eval mode.

    ``context_len > 0`` carries preceding tokens as context and scores only
    the window itself (strided-LM convention). Instruction-tuned models
    resist cold mid-stream windows — their token likelihoods inflate ~30x
    without leading context — so Gemma-it evaluation uses context_len=512.

    BOS: Gemma4's tokenizer ships ``add_bos_token=False`` (Gemma 1-3 default
    to True) and a BOS-less sequence un-anchors the model — plain-text ppl
    inflates ~700x and generation degenerates. Prepend BOS when the tokenizer
    has one and did not add it; no-op for tokenizers that add it themselves
    (Gemma 1-3, Llama) or have none (Qwen).
    """
    ids = bos_ids(tokenizer, text)
    was_training = model.training
    model.eval()
    total_nll, total_tok = 0.0, 0
    for _start, ctx, keep_from in _axis_windows(len(ids), chunks, chunk_len, context_len):
        x = torch.tensor(ids[ctx : _start + chunk_len + 1], device=device)[None]
        logits = model(input_ids=x).logits.float()
        logp = logits[:, :-1] - torch.logsumexp(logits[:, :-1], -1, keepdim=True)
        tgt = x[:, 1:]
        if keep_from is not None:
            # Score only the window tail, not the carried context. logp entry
            # j predicts global position ctx+1+j — crop BOTH tensors to the
            # same span or the gather pairs predictions with unrelated tokens.
            logp = logp[:, keep_from:]
            tgt = tgt[:, keep_from:]
        total_nll += -logp.gather(-1, tgt.unsqueeze(-1)).sum().item()
        total_tok += tgt.numel()
    if was_training:
        model.train()
    if total_tok == 0:
        raise ValueError(
            f"eval_ppl: text is too short to evaluate — need at least chunk_len + 1 = {chunk_len + 1} tokens, got {len(ids)}"
        )
    return math.exp(total_nll / total_tok)


@torch.no_grad()
def eval_kld(
    ref_model,
    model,
    tokenizer,
    text: str,
    *,
    chunks: int = 8,
    chunk_len: int = 512,
    device,
    context_len: int = 0,
) -> float:
    """Mean KL(P_ref || P_model) in nats/token over the same windows as
    ``eval_ppl`` (shared walk, BOS anchoring and context-carry crop), so a
    KLD axis point scores exactly the spans the ppl axis scores.

    Used by the exp1-grace arms as KLD(bf16 base || quant+shadow): a recovery
    arm improves the axis iff it moves KLD toward 0 relative to the
    quant-only baseline. Restores both models' prior train/eval modes.
    """
    ids = bos_ids(tokenizer, text)
    were_training = (ref_model.training, model.training)
    ref_model.eval()
    model.eval()
    total_kl, total_tok = 0.0, 0
    for _start, ctx, keep_from in _axis_windows(len(ids), chunks, chunk_len, context_len):
        x = torch.tensor(ids[ctx : _start + chunk_len + 1], device=device)[None]
        ref_logits = ref_model(input_ids=x).logits[:, :-1].float()
        logits = model(input_ids=x).logits[:, :-1].float()
        if keep_from is not None:
            ref_logits = ref_logits[:, keep_from:]
            logits = logits[:, keep_from:]
        ref_logp = ref_logits - torch.logsumexp(ref_logits, -1, keepdim=True)
        logp = logits - torch.logsumexp(logits, -1, keepdim=True)
        kl = (ref_logp.exp() * (ref_logp - logp)).sum(-1)
        total_kl += kl.sum().item()
        total_tok += kl.numel()
    for m, was in zip((ref_model, model), were_training, strict=True):
        if was:
            m.train()
    if total_tok == 0:
        raise ValueError(
            f"eval_kld: text is too short to evaluate — need at least chunk_len + 1 = {chunk_len + 1} tokens, got {len(ids)}"
        )
    return total_kl / total_tok


def despace_wikitext(text: str) -> str:
    """Undo wikitext-2-raw's space-around-punctuation pretokenization.

    The 2016 corpus format ("film , television", "@-@", " = header = ") is
    out-of-distribution for 2026-era tokenizers — Gemma4 scores it ~30-50x
    worse than natural text at every sequence length. All tiers are evaluated
    on the same despaced text, so comparisons stay exact; the raw-text ppl is
    reported alongside for transparency.
    """
    import re

    t = re.sub(r"\s+([,.!?;:])", r"\1", text)
    t = re.sub(r"([(\"\[;:$])\s+", r"\1", t)
    t = t.replace("@-@", "-").replace("= ", "").replace(" =", "")
    return t


def load_wikitext2_test_text(max_chars: int = 200_000) -> str:
    """wikitext-2-raw-v1 test split joined into one string.

    Fetches the parquet shard directly: load_dataset("wikitext", ...) is
    broken on recent huggingface_hub (single-segment repo id).
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    parquet = hf_hub_download(
        "wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset"
    )
    ds = load_dataset("parquet", data_files=parquet, split="train")
    return "\n".join(line for line in ds["text"] if line.strip())[:max_chars]
