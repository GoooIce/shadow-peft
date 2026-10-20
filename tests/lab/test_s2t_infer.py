"""S2T-Local shadow-only inference — external-behavior tests (issue #28).

The wrapper runs the shadow backbone alone, re-ranks the shadow top-K with an
optional selector head, and never touches the base forward. These tests use a
tiny randomly initialized Llama ShadowPEFT wrapper on CPU; no real checkpoint
is loaded.

Anchors:
- generation only emits tokens from the shadow candidate set (shadow vocab);
- the selector changes the picked token, and a missing head falls back to the
  same greedy tokens as ``shadow_only`` top-1;
- a base-forward counter stays at zero for the whole generation;
- saving/loading the selector preserves the generated tokens bit-identically;
- inference leaves ``shadow_lm_head``/``lm_head`` and ``export_shadow`` alone;
- decoder-cached and full-prefix decoding agree;
- the evaluation metrics (Agree@1, delta_hit/delta_sel) and the 20% latency
  deploy gate are computed from step records;
- the CLI parses ``--help`` without loading a model.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from shadow_lab.s2t_infer import (  # noqa: E402
    DEFAULT_LATENCY_BUDGET,
    SelectionStep,
    SelectorShadowModel,
    latency_comparison,
    selection_breakdown,
)
from shadow_lab.s2t_selector import SelectorHead, save_selector  # noqa: E402
from shadow_peft import ShadowConfig, ShadowForCausalLM, get_shadow_model  # noqa: E402

VOCAB = 128
HIDDEN = 32


class _StubTokenizer:
    """Prompt-name -> fixed token ids; deterministic decode for grading."""

    eos_token_id = None

    def __init__(self) -> None:
        self._encodings = {
            "alpha": [1, 2, 3],
            "beta": [4, 5, 6],
            "gamma": [7],
        }

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        ids = torch.tensor([self._encodings[text]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return " ".join(str(int(token)) for token in token_ids)


class _LastCandidateSelector:
    """Deterministic selector that always prefers the last candidate slot."""

    def __init__(self) -> None:
        self.hidden_size = HIDDEN
        self.candidate_dim = HIDDEN

    def score(self, h_t, candidate_ids, candidate_weight):
        slots = torch.arange(
            candidate_ids.shape[-1], dtype=torch.float, device=candidate_ids.device
        )
        return slots.expand_as(candidate_ids).to(torch.float)


def _tiny_shadow_only(seed: int = 0) -> ShadowForCausalLM:
    torch.manual_seed(seed)
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=VOCAB,
            hidden_size=HIDDEN,
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
        get_shadow_model(base, shadow_config), inference_mode="shadow_only"
    )
    return wrapper.eval()


# ---------------------------------------------------------------------------
# Candidate-only generation
# ---------------------------------------------------------------------------


def test_generation_chooses_only_from_the_shadow_candidate_set():
    model = _tiny_shadow_only()
    runner = SelectorShadowModel(
        model,
        _StubTokenizer(),
        selector=_LastCandidateSelector(),
        k=8,
    )

    generation = runner.generate("alpha", max_new_tokens=3)

    assert len(generation.token_ids) == 3
    assert len(generation.steps) == 3
    assert generation.text == " ".join(str(token) for token in generation.token_ids)
    for step, token in zip(generation.steps, generation.token_ids, strict=True):
        assert len(step.candidates) == 8
        assert step.selected_token == token
        assert token == step.candidates[-1]
        assert token in step.candidates
        assert 0 <= token < VOCAB
        assert step.used_selector is True


def _greedy_oracle(model, tokenizer, prompt: str, n_tokens: int) -> tuple[int, ...]:
    """Greedy oracle through the default ``shadow_only`` forward (public path)."""

    batch = tokenizer(prompt)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    tokens: list[int] = []
    with torch.inference_mode():
        for _ in range(n_tokens):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            token = int(logits[0, -1].argmax())
            tokens.append(token)
            next_token = torch.tensor([[token]], dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
    return tuple(tokens)


def test_missing_selector_falls_back_to_the_shadow_greedy_tokens():
    model = _tiny_shadow_only()
    tokenizer = _StubTokenizer()
    runner = SelectorShadowModel(model, tokenizer, selector=None, k=8)

    fallback = runner.generate("alpha", max_new_tokens=3, use_selector=True)
    greedy = runner.generate("alpha", max_new_tokens=3, use_selector=False)
    oracle = _greedy_oracle(model, tokenizer, "alpha", 3)

    assert fallback.token_ids == greedy.token_ids == oracle
    for step in fallback.steps:
        assert step.used_selector is False
        assert step.selector_scores is None
        assert step.selected_token == step.greedy_token == step.candidates[0]


def test_selector_can_override_the_shadow_top1():
    model = _tiny_shadow_only()
    runner = SelectorShadowModel(
        model,
        _StubTokenizer(),
        selector=_LastCandidateSelector(),
        k=8,
    )

    greedy = runner.generate("alpha", max_new_tokens=3, use_selector=False)
    selected = runner.generate("alpha", max_new_tokens=3, use_selector=True)

    assert selected.token_ids != greedy.token_ids
    for step in selected.steps:
        assert step.used_selector is True
        assert step.selected_token == step.candidates[-1]
        assert step.selected_token != step.greedy_token
        assert step.selector_scores is not None


@pytest.mark.parametrize("use_cache", [False, True])
def test_generation_never_calls_the_base_forward(monkeypatch, use_cache):
    model = _tiny_shadow_only()
    base_forward = model.peft_model.base_model.forward
    calls: list[tuple[Any, ...]] = []

    def counting_forward(*args, **kwargs):
        calls.append(args)
        return base_forward(*args, **kwargs)

    monkeypatch.setattr(model.peft_model.base_model, "forward", counting_forward)
    runner = SelectorShadowModel(
        model,
        _StubTokenizer(),
        selector=_LastCandidateSelector(),
        k=8,
        use_cache=use_cache,
    )

    generation = runner.generate("alpha", max_new_tokens=3)

    assert len(generation.token_ids) == 3
    assert calls == []


@pytest.mark.parametrize("with_selector", [False, True])
def test_cached_and_full_prefix_decoding_agree(with_selector):
    model = _tiny_shadow_only()
    selector = _LastCandidateSelector() if with_selector else None
    runner = SelectorShadowModel(model, _StubTokenizer(), selector=selector, k=8)

    full = runner.generate("alpha", max_new_tokens=3, use_cache=False)
    cached = runner.generate("alpha", max_new_tokens=3, use_cache=True)

    assert full.token_ids == cached.token_ids
    assert full.n_tokens == cached.n_tokens
    for left, right in zip(full.steps, cached.steps, strict=True):
        assert left.candidates == right.candidates
        assert left.selected_token == right.selected_token
        assert left.greedy_token == right.greedy_token


def test_generate_restores_the_model_training_mode_and_inference_mode():
    model = _tiny_shadow_only()
    model.train()
    runner = SelectorShadowModel(model, _StubTokenizer(), k=8)

    runner.generate("alpha", max_new_tokens=2)

    assert model.training is True
    assert model.inference_mode == "shadow_only"


def test_wrapper_rejects_a_base_shadow_model():
    model = _tiny_shadow_only()
    model.set_inference_mode("base_shadow")

    with pytest.raises(ValueError, match="shadow_only"):
        SelectorShadowModel(model, _StubTokenizer(), k=8)


def test_save_load_selector_keeps_scores_and_tokens_bit_identical(tmp_path):
    torch.manual_seed(7)
    head = SelectorHead(hidden_size=HIDDEN, mlp_hidden=16)
    save_selector(head, tmp_path, k=8)
    model = _tiny_shadow_only()
    tokenizer = _StubTokenizer()
    candidate_weight = model.shadow_lm_head.weight
    h_t = torch.randn(HIDDEN)
    candidate_ids = torch.tensor([[3, 5, 7, 11, 13, 17, 19, 23]])
    expected_scores = head.score(h_t, candidate_ids, candidate_weight)

    direct = SelectorShadowModel(model, tokenizer, selector=head, k=8)
    loaded = SelectorShadowModel(model, tokenizer, selector=tmp_path, k=8)

    direct_generation = direct.generate("alpha", max_new_tokens=3)
    loaded_generation = loaded.generate("alpha", max_new_tokens=3)

    assert loaded_generation.token_ids == direct_generation.token_ids
    for left, right in zip(
        direct_generation.steps, loaded_generation.steps, strict=True
    ):
        assert left.selected_token == right.selected_token
        assert left.selector_scores == right.selector_scores
    assert torch.equal(
        loaded.selector.score(h_t, candidate_ids, candidate_weight), expected_scores
    )


def test_inference_does_not_touch_either_lm_head_or_export():
    model = _tiny_shadow_only()
    base_weight = model.lm_head.weight
    shadow_weight = model.shadow_lm_head.weight
    base_before = base_weight.detach().clone()
    shadow_before = shadow_weight.detach().clone()
    shadow_shape = tuple(shadow_weight.shape)
    runner = SelectorShadowModel(
        model, _StubTokenizer(), selector=_LastCandidateSelector(), k=8
    )

    runner.generate("alpha", max_new_tokens=3)

    assert torch.equal(base_weight, base_before)
    assert torch.equal(shadow_weight, shadow_before)
    assert tuple(shadow_weight.shape) == shadow_shape == (VOCAB, HIDDEN)

    exported = model.peft_model.export_shadow()
    exported_head = exported.get_output_embeddings()
    assert tuple(exported_head.weight.shape) == shadow_shape
    assert torch.equal(exported_head.weight.detach().cpu(), base_before.cpu())
    assert tuple(model.shadow_lm_head.weight.shape) == shadow_shape


def test_generation_stops_after_the_eos_token():
    model = _tiny_shadow_only()
    with torch.no_grad():
        model.shadow_lm_head.weight[5].fill_(100.0)
    tokenizer = _StubTokenizer()
    tokenizer.eos_token_id = 5
    runner = SelectorShadowModel(model, tokenizer, k=8)

    generation = runner.generate("alpha", max_new_tokens=4)

    assert generation.token_ids == (5,)
    assert generation.n_tokens == 1
    assert generation.steps[0].selected_token == 5


# ---------------------------------------------------------------------------
# Hit/selection failure decomposition and the deploy latency gate
# ---------------------------------------------------------------------------


def _step(
    position: int,
    candidates: tuple[int, ...],
    *,
    greedy: int,
    selected: int,
    used_selector: bool = False,
) -> SelectionStep:
    return SelectionStep(
        position=position,
        candidates=candidates,
        candidate_logits=tuple(float(-index) for index in range(len(candidates))),
        greedy_token=greedy,
        selected_token=selected,
        used_selector=used_selector,
        selector_scores=None,
    )


def test_breakdown_splits_hit_failures_from_selection_failures():
    steps = [
        _step(0, (1, 2), greedy=1, selected=2, used_selector=True),
        _step(1, (3, 4), greedy=3, selected=3),
        _step(2, (5, 6), greedy=5, selected=6, used_selector=True),
        _step(3, (7, 8), greedy=7, selected=7),
    ]

    breakdown = selection_breakdown(steps, base_top1=[2, 9, 5, 7])

    assert breakdown.n_tokens == 4
    assert breakdown.hit_count == 3
    assert breakdown.agree_count == 2
    assert breakdown.greedy_agree_count == 2
    assert breakdown.hit_at_k == pytest.approx(0.75)
    assert breakdown.agree_at_1 == pytest.approx(0.5)
    assert breakdown.greedy_agree_at_1 == pytest.approx(0.5)
    assert breakdown.delta_hit == pytest.approx(0.25)
    assert breakdown.delta_sel == pytest.approx(0.25)
    assert breakdown.selection_accuracy_on_hits == pytest.approx(2 / 3)
    assert breakdown.selector_gain == pytest.approx(0.0)
    assert breakdown.to_dict()["delta_sel"] == pytest.approx(0.25)


def test_breakdown_with_no_hits_has_no_conditional_selection_accuracy():
    steps = [
        _step(0, (1, 2), greedy=1, selected=2),
        _step(1, (3, 4), greedy=3, selected=4),
    ]

    breakdown = selection_breakdown(steps, base_top1=[90, 91])

    assert breakdown.hit_count == 0
    assert breakdown.hit_at_k == 0.0
    assert breakdown.delta_hit == 1.0
    assert breakdown.delta_sel == 0.0
    assert breakdown.selection_accuracy_on_hits is None


def test_breakdown_rejects_empty_or_mismatched_inputs():
    with pytest.raises(ValueError, match="at least one"):
        selection_breakdown([], base_top1=[])
    with pytest.raises(ValueError, match="same length"):
        selection_breakdown(
            [_step(0, (1, 2), greedy=1, selected=1)],
            base_top1=[1, 2],
        )


def test_latency_gate_flags_only_more_than_twenty_percent_overhead():
    at_budget = latency_comparison(
        selector_seconds=1.2, selector_tokens=10, greedy_seconds=1.0, greedy_tokens=10
    )
    over_budget = latency_comparison(
        selector_seconds=1.21, selector_tokens=10, greedy_seconds=1.0, greedy_tokens=10
    )
    no_selector = latency_comparison(
        selector_seconds=1.0, selector_tokens=10, greedy_seconds=1.0, greedy_tokens=10
    )

    assert pytest.approx(0.20) == DEFAULT_LATENCY_BUDGET
    assert at_budget.deployable is True
    assert at_budget.selector_over_greedy == pytest.approx(1.2)
    assert at_budget.overhead == pytest.approx(0.2)
    assert over_budget.deployable is False
    assert over_budget.overhead == pytest.approx(0.21)
    assert no_selector.deployable is True
    assert no_selector.overhead == pytest.approx(0.0)
    assert over_budget.to_dict()["deployable"] is False


def test_latency_comparison_rejects_empty_or_negative_measurements():
    with pytest.raises(ValueError, match="tokens"):
        latency_comparison(
            selector_seconds=1.0, selector_tokens=0, greedy_seconds=1.0, greedy_tokens=10
        )
    with pytest.raises(ValueError, match="non-negative"):
        latency_comparison(
            selector_seconds=-1.0, selector_tokens=10, greedy_seconds=1.0, greedy_tokens=10
        )


# ---------------------------------------------------------------------------
# Evaluation CLI (parse only; no model is loaded)
# ---------------------------------------------------------------------------

EVAL_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "exp11_s2t_eval.py"


def test_cli_help_lists_modes_and_selector_flags():
    completed = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT), "--help"],
        cwd=EVAL_SCRIPT.parents[1],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    for flag in (
        "--model",
        "--shadow-ckpt",
        "--shadow-model-path",
        "--selector",
        "--prompts",
        "--mode",
        "--max-new-tokens",
        "--k",
        "--out",
        "--limit",
        "--device",
        "--dtype",
        "--seed",
        "--use-cache",
        "--skip-greedy-baseline",
        "--skip-base-reference",
    ):
        assert flag in completed.stdout
    for mode in ("shadow_greedy", "shadow_selector", "base_upper"):
        assert mode in completed.stdout


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("exp11_s2t_eval", EVAL_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_prompt_loader_keeps_task_expected_and_prompt_id(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt": "check the batch",
                        "task": "quality_verdict",
                        "expected": "pass",
                        "prompt_id": "q0",
                    }
                ),
                "",
                json.dumps({"prompt": "compute the yield"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    prompts = _load_eval_module().load_prompts(path)

    assert [prompt["prompt_id"] for prompt in prompts] == ["q0", "p1"]
    assert prompts[0]["task"] == "quality_verdict"
    assert prompts[0]["expected"] == "pass"
    assert prompts[1]["task"] is None
    assert "expected" not in prompts[1]


def test_eval_prompt_loader_rejects_a_row_without_a_prompt(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"task": "quality_verdict"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="prompt"):
        _load_eval_module().load_prompts(path)


def _base_shadow_greedy_oracle(model, tokenizer, prompt, max_new_tokens):
    """Manual base_shadow greedy loop, independent of HF generate()."""

    model.set_inference_mode("base_shadow")
    batch = tokenizer(prompt)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    eos = getattr(model.generation_config, "eos_token_id", None)
    tokens: list[int] = []
    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                token = int(logits[0, -1].argmax())
                tokens.append(token)
                if eos is not None and token == eos:
                    break
                next_token = torch.tensor([[token]], dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token)], dim=1
                )
    finally:
        model.set_inference_mode("shadow_only")
    return tuple(tokens)


def test_eval_base_upper_generates_on_base_shadow_and_restores_the_mode():
    model = _tiny_shadow_only()
    tokenizer = _StubTokenizer()
    module = _load_eval_module()
    expected = _base_shadow_greedy_oracle(model, tokenizer, "alpha", 3)

    token_ids, text, elapsed = module._base_shadow_generate(
        model, tokenizer, "alpha", max_new_tokens=3, torch_module=torch
    )

    assert token_ids == expected
    assert text == " ".join(str(token) for token in token_ids)
    assert elapsed >= 0.0
    assert model.inference_mode == "shadow_only"


def test_eval_base_reference_matches_per_prefix_base_shadow_logits():
    model = _tiny_shadow_only()
    tokenizer = _StubTokenizer()
    module = _load_eval_module()
    token_ids = _base_shadow_greedy_oracle(model, tokenizer, "alpha", 3)
    prompt_ids = tuple(tokenizer("alpha")["input_ids"][0].tolist())

    reference = module._reference_top1(
        model, prompt_ids, token_ids, torch_module=torch
    )

    manual: list[int] = []
    model.set_inference_mode("base_shadow")
    try:
        with torch.inference_mode():
            for index in range(len(token_ids)):
                prefix = torch.tensor([list(prompt_ids) + list(token_ids[:index])])
                logits = model(
                    input_ids=prefix, attention_mask=torch.ones_like(prefix)
                ).logits
                manual.append(int(logits[0, -1].argmax()))
    finally:
        model.set_inference_mode("shadow_only")
    assert reference == manual
