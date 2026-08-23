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


@torch.no_grad()
def eval_ppl(
    model,
    tokenizer,
    text: str,
    *,
    chunks: int = 8,
    chunk_len: int = 512,
    device: str,
) -> float:
    """Chunked perplexity over the first `chunks` windows of `text`.

    Works for any causal LM returning `.logits` (HF base model or
    ShadowForCausalLM). Restores the model's prior train/eval mode.
    """
    ids = tokenizer.encode(text)
    was_training = model.training
    model.eval()
    total_nll, total_tok = 0.0, 0
    # The limit formula intentionally yields `chunks - 1` full windows for
    # long texts (parity with the original E1/E2 scripts — window starts
    # and count must stay comparable); the guard below only rescues the
    # exact-fit case and never fires in that regime.
    limit = min(len(ids), chunks * chunk_len) - chunk_len - 1
    if limit <= 0 and len(ids) >= chunk_len + 1:
        limit = 1  # exact-fit input: force the single full window
    for start in range(0, limit, chunk_len):
        x = torch.tensor(ids[start : start + chunk_len + 1], device=device)[None]
        logits = model(input_ids=x).logits.float()
        logp = logits[:, :-1] - torch.logsumexp(logits[:, :-1], -1, keepdim=True)
        tgt = x[:, 1:]
        total_nll += -logp.gather(-1, tgt.unsqueeze(-1)).sum().item()
        total_tok += tgt.numel()
    if was_training:
        model.train()
    if total_tok == 0:
        raise ValueError(
            f"eval_ppl: text is too short to evaluate — need at least chunk_len + 1 = {chunk_len + 1} tokens, got {len(ids)}"
        )
    return math.exp(total_nll / total_tok)


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
