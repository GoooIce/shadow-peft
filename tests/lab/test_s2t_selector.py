"""S2T-Local selector head — external-behavior tests (issue #27).

The selector re-ranks the shadow model's own top-K candidates with a small
shared MLP ``score(c) = MLP([h_t; e_c])``. These tests use synthetic tensors
and a tiny Llama ShadowPEFT wrapper on CPU; no real checkpoint is loaded.

Anchors:
- candidate CE is finite and falls as the target logit grows; ignored rows do
  not contribute.
- ``L_sel`` is hit-only (target = base top-1); ``L_reg`` always targets the
  shadow top-1 with weight ``beta``; ``beta=0`` is pure ``L_sel``.
- selection only ever emits ids from the shadow candidate set.
- ``collect_selector_examples`` stores the *initial shadow* hidden state (the
  one that produced ``shadow_logits``), never the base hidden.
- save/load round-trips bit-identically; fixed-seed training is deterministic.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from shadow_lab.s2t_selector import (  # noqa: E402
    DEFAULT_BETA,
    DEFAULT_MLP_HIDDEN,
    IGNORE_INDEX,
    SELECTOR_CONFIG_FILENAME,
    SELECTOR_VERSION,
    SELECTOR_WEIGHTS_FILENAME,
    SelectorExample,
    SelectorHead,
    candidate_ce,
    collect_selector_examples,
    load_selector,
    save_selector,
    select_candidate,
    selector_loss,
    train_selector,
)

# ---------------------------------------------------------------------------
# SelectorHead: shared MLP over [h_t ; e_c]
# ---------------------------------------------------------------------------


def test_head_scores_every_candidate_with_a_finite_scalar_per_slot():
    torch.manual_seed(0)
    head = SelectorHead(hidden_size=16, mlp_hidden=32)
    h_t = torch.randn(3, 16)
    candidate_embeds = torch.randn(3, 5, 16)

    scores = head(h_t, candidate_embeds)

    assert scores.shape == (3, 5)
    assert torch.isfinite(scores).all()


def test_score_reads_candidate_vectors_from_the_lm_head_weight_table():
    torch.manual_seed(1)
    head = SelectorHead(hidden_size=16, mlp_hidden=32)
    h_t = torch.randn(2, 16)
    weight = torch.randn(50, 16)
    candidate_ids = torch.tensor([[3, 7, 9], [11, 2, 5]])

    scores = head.score(h_t, candidate_ids, weight)

    assert torch.equal(scores, head(h_t, weight[candidate_ids]))


def test_default_head_stays_under_a_million_parameters_for_qwen3_4b():
    head = SelectorHead(hidden_size=2560)

    parameters = sum(parameter.numel() for parameter in head.parameters())

    assert DEFAULT_MLP_HIDDEN == 64
    assert head.mlp_hidden == DEFAULT_MLP_HIDDEN
    # 5120 -> 64 -> 1 MLP: 5120*64 + 64 + 64*1 + 1 = 327,809.
    assert parameters == 327_809
    assert parameters < 1_000_000


def test_head_does_not_mutate_the_candidate_weight_table():
    torch.manual_seed(2)
    head = SelectorHead(hidden_size=8, mlp_hidden=16)
    weight = torch.randn(20, 8)
    before = weight.clone()
    h_t = torch.randn(2, 8, requires_grad=True)

    head.score(h_t, torch.tensor([[1, 2], [3, 4]]), weight).sum().backward()

    assert torch.equal(weight, before)
    assert h_t.grad is not None


# ---------------------------------------------------------------------------
# candidate CE
# ---------------------------------------------------------------------------


def test_candidate_ce_is_finite_and_falls_as_the_target_logit_grows():
    scores = torch.zeros(1, 3)
    target = torch.tensor([1])

    base = candidate_ce(scores, target)
    boosted = candidate_ce(scores + torch.tensor([[0.0, 2.0, 0.0]]), target)

    assert float(base) == pytest.approx(math.log(3.0))
    assert math.isfinite(float(base))
    assert float(boosted) < float(base)


def test_candidate_ce_ignored_targets_do_not_contribute():
    scores = torch.tensor([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0]])

    ignored = candidate_ce(scores, torch.tensor([IGNORE_INDEX, IGNORE_INDEX]))
    mixed = candidate_ce(scores, torch.tensor([IGNORE_INDEX, 0]))

    assert float(ignored) == 0.0
    assert float(mixed) == pytest.approx(float(candidate_ce(scores[1:], torch.tensor([0]))))


# ---------------------------------------------------------------------------
# candidate-only argmax never leaves the shadow candidate set
# ---------------------------------------------------------------------------


def test_select_candidate_only_returns_tokens_from_the_candidate_set():
    candidate_ids = torch.tensor([[10, 42, 77], [3, 9, 200]])
    scores = torch.tensor([[0.0, 5.0, 1.0], [7.0, 0.0, 1.0]])

    selected = select_candidate(scores, candidate_ids)

    assert selected.tolist() == [42, 3]
    for row, token in zip(candidate_ids.tolist(), selected.tolist(), strict=True):
        assert token in row


# ---------------------------------------------------------------------------
# selector loss: hit-only L_sel + shadow-preference L_reg
# ---------------------------------------------------------------------------


def test_sel_loss_only_counts_positions_whose_base_top1_is_a_candidate():
    scores = torch.zeros(2, 3)
    candidate_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # row 0: base top-1 is candidate slot 1 (a hit); row 1: token 9 is absent.
    loss = selector_loss(
        scores,
        candidate_ids,
        base_top1=torch.tensor([2, 9]),
        shadow_top1=torch.tensor([1, 4]),
        beta=0.0,
    )

    assert loss.n_sel == 1
    assert loss.n_reg == 2
    assert float(loss.sel) == pytest.approx(math.log(3.0))
    assert float(loss.reg) == pytest.approx(math.log(3.0))
    assert float(loss.total) == pytest.approx(math.log(3.0))


def test_sel_loss_gives_zero_when_the_target_moves_out_of_the_candidates():
    scores = torch.zeros(1, 3)
    candidate_ids = torch.tensor([[1, 2, 3]])

    loss = selector_loss(
        scores,
        candidate_ids,
        base_top1=torch.tensor([7]),
        shadow_top1=torch.tensor([1]),
        beta=0.0,
    )

    assert loss.n_sel == 0
    assert float(loss.sel) == 0.0
    assert float(loss.total) == 0.0


def test_a_miss_position_does_not_dilute_the_sel_denominator():
    scores = torch.zeros(2, 3)
    candidate_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    loss = selector_loss(
        scores,
        candidate_ids,
        base_top1=torch.tensor([2, 9]),
        shadow_top1=torch.tensor([1, 4]),
        beta=0.0,
    )

    # Averaging over both rows would give ln(3)/2; hit-only keeps ln(3).
    assert float(loss.sel) == pytest.approx(math.log(3.0))


def test_reg_target_is_the_shadow_top1_not_the_base_top1():
    scores = torch.tensor([[0.0, math.log(2.0), 0.0]])
    candidate_ids = torch.tensor([[10, 20, 30]])
    # Sparse candidate ids: shadow top-1 is token 20 == candidate slot 1.
    loss = selector_loss(
        scores,
        candidate_ids,
        base_top1=torch.tensor([10]),
        shadow_top1=torch.tensor([20]),
        beta=DEFAULT_BETA,
    )

    assert float(loss.reg) == pytest.approx(math.log(2.0))
    assert float(loss.sel) == pytest.approx(math.log(1.0 + 2.0 + 1.0) - math.log(1.0))
    assert float(loss.total) == pytest.approx(float(loss.sel) + DEFAULT_BETA * math.log(2.0))


def test_beta_zero_is_pure_sel_and_beta_adds_the_regularizer():
    scores = torch.zeros(2, 3)
    candidate_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    base_top1 = torch.tensor([2, 9])
    shadow_top1 = torch.tensor([1, 4])

    pure = selector_loss(scores, candidate_ids, base_top1, shadow_top1, beta=0.0)
    regularized = selector_loss(scores, candidate_ids, base_top1, shadow_top1, beta=DEFAULT_BETA)

    assert float(pure.total) == pytest.approx(float(pure.sel))
    assert float(regularized.total - pure.total) == pytest.approx(
        DEFAULT_BETA * float(regularized.reg)
    )
    assert float(regularized.total) > float(pure.total)


def test_negative_beta_is_rejected():
    with pytest.raises(ValueError, match="beta"):
        selector_loss(
            torch.zeros(1, 2),
            torch.tensor([[1, 2]]),
            torch.tensor([1]),
            torch.tensor([1]),
            beta=-0.1,
        )


# ---------------------------------------------------------------------------
# collect_selector_examples: initial shadow hidden + per-token candidates
# ---------------------------------------------------------------------------


class _StaticTokenizer:
    """Map each prompt text to a fixed id row; no real tokenizer needed."""

    eos_token_id = None

    def __init__(self, table: dict[str, list[int]]) -> None:
        self._table = table

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([self._table[text]], dtype=torch.long)}


def _tiny_base_shadow(vocab: int = 96, hidden: int = 32, seed: int = 0):
    from transformers import LlamaConfig, LlamaForCausalLM

    from shadow_peft import ShadowConfig, ShadowForCausalLM, get_shadow_model

    torch.manual_seed(seed)
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=vocab,
            hidden_size=hidden,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
        )
    )
    shadow_config = ShadowConfig(
        num_shadow_layers=1,
        injection_hidden_size=8,
        gate_hidden_size=10,
        alpha=0.1,
        dropout=0.0,
    )
    wrapper = ShadowForCausalLM(
        get_shadow_model(base, shadow_config), inference_mode="base_shadow"
    )
    return wrapper.eval()


_TINY_PROMPTS = [
    {"prompt": "a b c", "task": "fault", "prompt_id": "p0"},
    {"prompt": "d e", "task": "calc", "prompt_id": "p1"},
]
_TINY_TABLE = {"a b c": [2, 5, 7, 9], "d e": [3, 6, 11]}


def test_collect_stores_shadow_hidden_candidates_and_metadata():
    wrapper = _tiny_base_shadow()
    tokenizer = _StaticTokenizer(_TINY_TABLE)

    examples = collect_selector_examples(
        wrapper, tokenizer, _TINY_PROMPTS, max_new_tokens=4, k=8, seed=0
    )

    assert len(examples) == 8
    assert [ex.prompt_id for ex in examples] == ["p0"] * 4 + ["p1"] * 4
    assert [ex.task for ex in examples] == ["fault"] * 4 + ["calc"] * 4
    assert [ex.position for ex in examples] == [0, 1, 2, 3, 0, 1, 2, 3]
    for ex in examples:
        assert ex.h_t.shape == (wrapper.config.hidden_size,)
        assert torch.isfinite(ex.h_t).all()
        assert ex.candidate_ids.shape == (8,)
        assert len(torch.unique(ex.candidate_ids)) == 8
        assert bool(((ex.candidate_ids >= 0) & (ex.candidate_ids < 96)).all())
        assert torch.isfinite(ex.candidate_logits).all()
        assert torch.all(ex.candidate_logits[:-1] >= ex.candidate_logits[1:])
        assert int(ex.shadow_top1) == int(ex.candidate_ids[0])
        assert ex.hit == bool((ex.candidate_ids == ex.base_top1).any())

    # h_t is the initial shadow hidden that produced shadow_logits, not the base one.
    ids = torch.tensor([_TINY_TABLE["a b c"]], dtype=torch.long)
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        _, shadow_hidden = wrapper.peft_model.forward_with_shadow(
            input_ids=ids, attention_mask=mask, use_cache=False
        )
        base_out = wrapper(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            use_cache=False,
        )
    assert torch.equal(examples[0].h_t, shadow_hidden[0, -1].float())
    assert not torch.equal(examples[0].h_t, base_out.hidden_states[-1][0, -1].float())


def test_collect_restores_train_mode_and_leaves_no_gradients():
    wrapper = _tiny_base_shadow()
    wrapper.train()
    tokenizer = _StaticTokenizer(_TINY_TABLE)

    examples = collect_selector_examples(
        wrapper, tokenizer, _TINY_PROMPTS[:1], max_new_tokens=2, k=8, seed=0
    )

    assert len(examples) == 2
    assert wrapper.training is True
    assert torch.is_grad_enabled() is True
    assert all(p.grad is None for p in wrapper.parameters())


def test_collect_with_a_fixed_seed_is_deterministic():
    wrapper = _tiny_base_shadow()
    tokenizer = _StaticTokenizer(_TINY_TABLE)

    first = collect_selector_examples(wrapper, tokenizer, _TINY_PROMPTS, max_new_tokens=3)
    second = collect_selector_examples(wrapper, tokenizer, _TINY_PROMPTS, max_new_tokens=3)

    assert len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a.h_t, b.h_t)
        assert torch.equal(a.candidate_ids, b.candidate_ids)
        assert torch.equal(a.candidate_logits, b.candidate_logits)
        assert a.base_top1 == b.base_top1


def test_collect_rejects_a_shadow_only_model():
    wrapper = _tiny_base_shadow()
    wrapper.set_inference_mode("shadow_only")

    with pytest.raises(ValueError, match="base_shadow"):
        collect_selector_examples(
            wrapper, _StaticTokenizer(_TINY_TABLE), _TINY_PROMPTS[:1], max_new_tokens=1
        )


# ---------------------------------------------------------------------------
# train_selector: prompt-grouped split, Adam, deterministic fixed-seed run
# ---------------------------------------------------------------------------


def _synthetic_selector_examples(
    *,
    n_prompts: int = 4,
    positions: int = 3,
    hidden: int = 8,
    k: int = 4,
    vocab: int = 40,
    seed: int = 3,
) -> list[SelectorExample]:
    generator = torch.Generator().manual_seed(seed)
    examples: list[SelectorExample] = []
    for prompt in range(n_prompts):
        for position in range(positions):
            candidate_ids = torch.randperm(vocab, generator=generator)[:k]
            candidate_list = candidate_ids.tolist()
            if (prompt + position) % 2 == 0:
                base_top1 = int(candidate_ids[1])
                hit = True
            else:
                base_top1 = int(torch.randint(0, vocab, (1,), generator=generator))
                while base_top1 in candidate_list:
                    base_top1 = (base_top1 + 1) % vocab
                hit = False
            examples.append(
                SelectorExample(
                    prompt_id=f"p{prompt}",
                    task="synthetic",
                    position=position,
                    h_t=torch.randn(hidden, generator=generator),
                    candidate_ids=candidate_ids,
                    candidate_logits=torch.linspace(1.0, 0.0, k),
                    base_top1=base_top1,
                    shadow_top1=int(candidate_ids[0]),
                    hit=hit,
                )
            )
    return examples


@pytest.mark.parametrize("beta", [0.0, DEFAULT_BETA])
def test_train_selector_is_deterministic_and_splits_by_prompt(beta):
    examples = _synthetic_selector_examples()
    candidate_weight = torch.randn(40, 8)

    kwargs = dict(
        candidate_weight=candidate_weight,
        epochs=1,
        steps=2,
        lr=0.01,
        beta=beta,
        seed=0,
        device="cpu",
        batch_size=2,
        val_ratio=0.5,
    )
    first = train_selector(examples, **kwargs)
    second = train_selector(examples, **kwargs)

    assert first.steps == 2
    assert len(first.history) == 2
    assert all(math.isfinite(loss) for loss in first.history)
    assert first.val_loss is not None and math.isfinite(first.val_loss)
    assert first.n_train_examples == 6
    assert first.n_val_examples == 6
    assert first.n_train_prompts == 2
    assert first.n_val_prompts == 2
    assert first.head.training is False
    assert first.head.hidden_size == 8

    assert list(first.history) == list(second.history)
    for (name, left), (other, right) in zip(
        first.head.state_dict().items(), second.head.state_dict().items(), strict=True
    ):
        assert name == other
        assert torch.equal(left, right)


def test_train_selector_accepts_an_explicit_mlp_hidden_and_saves_it(tmp_path):
    examples = _synthetic_selector_examples()
    candidate_weight = torch.randn(40, 8)

    result = train_selector(
        examples,
        candidate_weight=candidate_weight,
        steps=1,
        mlp_hidden=32,
        val_ratio=0.5,
        seed=0,
        device="cpu",
    )
    paths = save_selector(result.head, tmp_path)

    config = json.loads(Path(paths["config"]).read_text(encoding="utf-8"))
    assert result.head.mlp_hidden == 32
    assert config["mlp_hidden"] == 32
    assert load_selector(tmp_path).mlp_hidden == 32


def test_train_selector_with_a_single_prompt_has_no_validation_split():
    examples = _synthetic_selector_examples(n_prompts=1, positions=4)
    candidate_weight = torch.randn(40, 8)

    result = train_selector(
        examples,
        candidate_weight=candidate_weight,
        epochs=1,
        lr=0.05,
        seed=0,
        batch_size=4,
        val_ratio=0.5,
    )

    assert result.steps == 1
    assert result.val_loss is None
    assert result.n_val_examples == 0
    assert result.n_val_prompts == 0
    assert result.n_train_examples == 4


# ---------------------------------------------------------------------------
# save / load: selector.safetensors + selector_config.json, bit-identical scores
# ---------------------------------------------------------------------------


def test_save_load_round_trip_is_bit_identical_and_writes_only_selector_files(tmp_path):
    torch.manual_seed(0)
    head = SelectorHead(hidden_size=16, mlp_hidden=32)
    candidate_weight = torch.randn(30, 16)
    h_t = torch.randn(2, 16)
    candidate_ids = torch.tensor([[1, 4, 9], [3, 5, 7]])
    expected = head.score(h_t, candidate_ids, candidate_weight)
    weight_before = candidate_weight.clone()

    paths = save_selector(head, tmp_path, k=8)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        SELECTOR_WEIGHTS_FILENAME,
        SELECTOR_CONFIG_FILENAME,
    ]
    assert paths["weights"].endswith(SELECTOR_WEIGHTS_FILENAME)
    config = json.loads((tmp_path / SELECTOR_CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert config == {
        "version": SELECTOR_VERSION,
        "hidden_size": 16,
        "candidate_dim": 16,
        "mlp_hidden": 32,
        "k": 8,
    }

    loaded = load_selector(tmp_path)

    assert isinstance(loaded, SelectorHead)
    assert loaded.training is False
    assert torch.equal(loaded.score(h_t, candidate_ids, candidate_weight), expected)
    assert torch.equal(candidate_weight, weight_before)


def test_load_selector_round_trips_the_saved_k(tmp_path):
    head = SelectorHead(hidden_size=8, mlp_hidden=16)
    save_selector(head, tmp_path, k=4)

    loaded = load_selector(tmp_path)
    save_selector(loaded, tmp_path / "again")

    config = json.loads(
        (tmp_path / "again" / SELECTOR_CONFIG_FILENAME).read_text(encoding="utf-8")
    )
    assert config["k"] == 4


def test_load_selector_rejects_an_unknown_config_version(tmp_path):
    head = SelectorHead(hidden_size=8, mlp_hidden=16)
    save_selector(head, tmp_path)
    config_path = tmp_path / SELECTOR_CONFIG_FILENAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["version"] = "s2t-selector/v0"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="version"):
        load_selector(tmp_path)
