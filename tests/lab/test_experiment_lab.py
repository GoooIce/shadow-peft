from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

# The experiment lab is a package inside experiment/.

from shadow_lab.atoms import (  # noqa: E402
    eval_kld,
    eval_kld_ids,
    eval_ppl,
    eval_ppl_ids,
    make_batch,
)
from shadow_lab.eval import (  # noqa: E402
    GSM8KDataCollator,
    MMLUDataCollator,
    _build_position_ids,
    _clean_generated_text,
    compute_mmlu_reward,
    extract_answer_from_text,
    extract_gsm8k_answer_from_text,
    extract_gsm8k_boxed_answer,
)
from shadow_lab.trainer import TrainConfig, _summary_args_dump  # noqa: E402

# ---------------------------------------------------------------------------
# lab.atoms.make_batch
# ---------------------------------------------------------------------------

def test_make_batch_pads_masks_and_labels():
    seqs = [list(range(10))]  # both sampled rows crop to seq_len=8
    import random

    rng = random.Random(0)
    batch = make_batch(seqs, rng, batch_size=2, seq_len=8, device="cpu", pad_id=999)
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] == 8
    assert int(batch["attention_mask"].sum()) == 2 * 8
    # pad positions masked out of the loss
    labels = batch["labels"]
    assert (labels[batch["attention_mask"] == 0] == -100).all()
    assert (labels[batch["attention_mask"] == 1] != -100).all()


def test_make_batch_pads_short_sequences_with_pad_id():
    seqs = [[1, 2, 3]]
    import random

    rng = random.Random(0)
    batch = make_batch(seqs, rng, batch_size=3, seq_len=8, device="cpu", pad_id=999)
    # rows 1 and 2 sampled the same 3-token sequence; the rest is pad
    assert (batch["input_ids"][:, 3:] == 999).all()
    assert (batch["attention_mask"][:, 3:] == 0).all()


# ---------------------------------------------------------------------------
# lab.atoms.eval_ppl (numerical parity against an independent computation)
# ---------------------------------------------------------------------------

class _FixedTokenizer:
    """Minimal tokenizer stand-in: encode() returns a fixed id sequence."""

    def __init__(self, ids):
        self._ids = list(ids)

    def encode(self, text):
        return list(self._ids)


def test_eval_ppl_matches_reference_computation():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(cfg)
    ids = list(range(20))
    tokenizer = _FixedTokenizer(ids)

    ppl = eval_ppl(model, tokenizer, "ignored", chunks=2, chunk_len=8, device="cpu")

    # Independent reference over the same single 8-token window.
    x = torch.tensor([ids[:9]])
    with torch.no_grad():
        logits = model(input_ids=x).logits.float()
    shifted = logits[:, :-1]
    targets = x[:, 1:]
    logp = shifted - torch.logsumexp(shifted, -1, keepdim=True)
    nll = -logp.gather(-1, targets.unsqueeze(-1)).sum().item()
    expected = torch.exp(torch.tensor(nll / targets.numel())).item()

    assert ppl == pytest.approx(expected, rel=1e-5)


def test_eval_ppl_restores_training_mode():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(cfg)
    model.train()
    eval_ppl(model, _FixedTokenizer(list(range(20))), "ignored", chunks=2, chunk_len=8, device="cpu")
    assert model.training is True
    model.eval()
    eval_ppl(model, _FixedTokenizer(list(range(20))), "ignored", chunks=2, chunk_len=8, device="cpu")
    assert model.training is False


def test_eval_ppl_exact_fit_input_evaluates_single_window():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=1024,
    )
    model = LlamaForCausalLM(cfg)
    ids = [i % 32 for i in range(513)]  # exactly chunk_len + 1 tokens
    tokenizer = _FixedTokenizer(ids)

    ppl = eval_ppl(model, tokenizer, "ignored", chunks=1, chunk_len=512, device="cpu")

    assert math.isfinite(ppl)
    assert ppl > 0

    # Independent reference over the single 512-token window.
    x = torch.tensor([ids[:513]])
    with torch.no_grad():
        logits = model(input_ids=x).logits.float()
    shifted = logits[:, :-1]
    targets = x[:, 1:]
    logp = shifted - torch.logsumexp(shifted, -1, keepdim=True)
    nll = -logp.gather(-1, targets.unsqueeze(-1)).sum().item()
    expected = torch.exp(torch.tensor(nll / targets.numel())).item()

    assert ppl == pytest.approx(expected, rel=1e-5)


def test_eval_ppl_too_short_text_raises_clear_error():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(cfg)
    tokenizer = _FixedTokenizer(list(range(20)))  # shorter than chunk_len + 1 = 33

    with pytest.raises(ValueError, match="too short"):
        eval_ppl(model, tokenizer, "ignored", chunks=2, chunk_len=32, device="cpu")


# ---------------------------------------------------------------------------
# lab.atoms.eval_kld (numerical parity against an independent computation)
# ---------------------------------------------------------------------------


def _tiny_llama_for_eval(vocab_size: int = 32, seed: int = 0):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(cfg)


def test_eval_kld_zero_for_identical_models():
    model = _tiny_llama_for_eval()
    tokenizer = _FixedTokenizer(list(range(20)))
    kld = eval_kld(model, model, tokenizer, "ignored", chunks=2, chunk_len=8, device="cpu")
    assert kld == pytest.approx(0.0, abs=1e-10)


def test_eval_kld_matches_reference_computation():
    ref = _tiny_llama_for_eval(seed=0)
    model = _tiny_llama_for_eval(seed=1234)  # different init from the ref
    ids = list(range(20))
    tokenizer = _FixedTokenizer(ids)

    kld = eval_kld(ref, model, tokenizer, "ignored", chunks=2, chunk_len=8, device="cpu")

    # Independent KL over the same single 8-token window.
    x = torch.tensor([ids[:9]])
    with torch.no_grad():
        rl = ref(input_ids=x).logits[:, :-1].float()
        ml = model(input_ids=x).logits[:, :-1].float()
    rlogp = rl - torch.logsumexp(rl, -1, keepdim=True)
    mlogp = ml - torch.logsumexp(ml, -1, keepdim=True)
    expected = (rlogp.exp() * (rlogp - mlogp)).sum(-1).mean().item()
    assert kld == pytest.approx(expected, rel=1e-5)
    assert kld > 0


def test_eval_kld_restores_training_mode():
    ref = _tiny_llama_for_eval()
    model = _tiny_llama_for_eval()
    tokenizer = _FixedTokenizer(list(range(20)))
    ref.train()
    model.eval()
    eval_kld(ref, model, tokenizer, "ignored", chunks=2, chunk_len=8, device="cpu")
    assert ref.training is True
    assert model.training is False


# ---------------------------------------------------------------------------
# lab.atoms.eval_ppl_ids / eval_kld_ids (raw-id streams: parity with the
# text-level entries — held-out self-distillation pools must not round-trip
# through decode/re-encode)
# ---------------------------------------------------------------------------


def test_eval_ppl_ids_matches_text_level():
    model = _tiny_llama_for_eval()
    ids = list(range(20))
    tokenizer = _FixedTokenizer(ids)
    a = eval_ppl(model, tokenizer, "ignored", chunks=2, chunk_len=8, device="cpu")
    b = eval_ppl_ids(model, ids, chunks=2, chunk_len=8, device="cpu")
    assert b == pytest.approx(a, rel=1e-10)


def test_eval_kld_ids_matches_text_level():
    ref = _tiny_llama_for_eval(seed=0)
    model = _tiny_llama_for_eval(seed=1234)
    ids = list(range(20))
    tokenizer = _FixedTokenizer(ids)
    a = eval_kld(ref, model, tokenizer, "ignored", chunks=2, chunk_len=8, device="cpu")
    b = eval_kld_ids(ref, model, ids, chunks=2, chunk_len=8, device="cpu")
    assert b == pytest.approx(a, rel=1e-10)


def test_eval_ppl_ids_too_short_raises():
    model = _tiny_llama_for_eval()
    with pytest.raises(ValueError, match="too short"):
        eval_ppl_ids(model, list(range(8)), chunks=2, chunk_len=8, device="cpu")


# ---------------------------------------------------------------------------
# lab.eval: position ids
# ---------------------------------------------------------------------------

def test_position_ids_right_padding():
    attn = torch.tensor([[1, 1, 1, 0, 0]])
    pos = _build_position_ids(attn, "right")
    assert pos.tolist() == [[0, 1, 2, 3, 4]]


def test_position_ids_left_padding():
    attn = torch.tensor([[0, 0, 1, 1, 1]])
    pos = _build_position_ids(attn, "left")
    assert pos.tolist() == [[0, 0, 0, 1, 2]]


# ---------------------------------------------------------------------------
# lab.eval: generated-text cleaning + answer extraction (the single home)
# ---------------------------------------------------------------------------

def test_clean_generated_text_strips_thinking_and_tags():
    raw = "<think>reasoning...</think> The   answer is <b>B</b> <|im_end|>"
    assert _clean_generated_text(raw) == "The answer is B"


def test_extract_answer_from_text_letter_priority():
    assert extract_answer_from_text("I choose C") == "C"
    assert extract_answer_from_text("answer: a") == "A"
    assert extract_answer_from_text("no letters here") is None
    assert extract_answer_from_text("") is None


def test_extract_gsm8k_answer_from_text_standard_format():
    text = "<think>calc</think> Because 2+2=4.\n#### 4"
    assert extract_gsm8k_answer_from_text(text) == "4"


def test_extract_gsm8k_boxed_answer_strategies():
    assert extract_gsm8k_boxed_answer(r"The answer is $\boxed{42}$") == "42"
    assert extract_gsm8k_boxed_answer(r"Final Answer: $\boxed{1,234}$") == "1234"
    assert extract_gsm8k_boxed_answer("blah\n#### 7\n") == "7"
    assert extract_gsm8k_boxed_answer("first 3 then 5") == "5"
    assert extract_gsm8k_boxed_answer("") == ""


def test_compute_mmlu_reward_boundaries():
    rewards = compute_mmlu_reward(
        ["the answer is B", "the answer is C", "gibberish"],
        ["B", "B", "B"],
    )
    assert rewards == [1.0, -0.5, -1.0]  # correct, wrong, invalid-format


# ---------------------------------------------------------------------------
# lab.eval: collators (with a minimal tokenizer stub)
# ---------------------------------------------------------------------------

class _TokStub:
    pad_token_id = 7
    padding_side = "right"


def test_mmlu_data_collator_preserves_answers_and_prompts():
    collator = MMLUDataCollator(_TokStub())
    batch = collator(
        [
            {
                "input_ids": [1, 2, 3],
                "labels": [1, 2, 3],
                "answer_letter": "A",
                "answer_idx": 0,
                "prompt_input_ids": [1, 2],
                "prompt_attention_mask": [1, 1],
            },
            {
                "input_ids": [4, 5],
                "labels": [4, 5],
                "answer_letter": "C",
                "answer_idx": 2,
            },
        ]
    )
    assert batch["answer_letters"] == ["A", "C"]
    assert batch["answer_indices"] == [0, 2]
    assert batch["input_ids"].shape == (2, 3)
    assert (batch["input_ids"][1, 2] == 7)  # padded
    assert batch["position_ids"].tolist() == [[0, 1, 2], [0, 1, 2]]
    assert batch["prompt_input_ids"].shape == (2, 2)
    assert batch["prompt_input_ids"][1].tolist() == [4, 5, 7][:2]


def test_gsm8k_data_collator_preserves_gold_answers():
    collator = GSM8KDataCollator(_TokStub())
    batch = collator(
        [
            {
                "input_ids": [1, 2, 3],
                "labels": [-100, -100, 3],
                "gold_answer": "42",
                "prompt_input_ids": [1, 2],
            }
        ]
    )
    assert batch["gold_answers"] == ["42"]
    assert batch["labels"].shape == (1, 3)
    assert batch["position_ids"].shape == (1, 3)
    assert batch["prompt_attention_mask"].shape == (1, 2)


# ---------------------------------------------------------------------------
# lab.trainer: TrainConfig validation
# ---------------------------------------------------------------------------

def test_train_config_rejects_bad_bits_and_loss_mode():
    with pytest.raises(ValueError, match="bits"):
        TrainConfig(bits=3)
    with pytest.raises(ValueError, match="loss_mode"):
        TrainConfig(loss_mode="nope")


def test_train_config_profile_resolution():
    cfg1 = TrainConfig(bits=1, loss_mode="ce")
    assert cfg1.profile.tag == "e1"
    assert cfg1.profile.dtype is torch.float16
    assert cfg1.resolved_save_dir().as_posix().endswith("runs/e1_ce")

    cfg2 = TrainConfig(bits=2, loss_mode="ce_kl")
    assert cfg2.profile.tag == "e2"
    assert cfg2.profile.dtype is torch.bfloat16
    assert cfg2.profile.refine_iters == 8


def test_summary_args_dump_stringifies_torch_dtype():
    # TrainConfig accepts torch.dtype; summary.json must still serialize.
    cfg = TrainConfig(bits=1, loss_mode="ce", dtype=torch.float16)
    args_dump = _summary_args_dump(cfg)
    dumped = json.dumps({"args": args_dump})  # must not raise
    assert args_dump["dtype"] == "torch.float16"
    assert "torch.float16" in dumped
    # JSON-safe fields pass through unchanged.
    assert args_dump["bits"] == 1
    assert args_dump["loss_mode"] == "ce"
