"""Shadow-only recovery pipeline tests (issue #46).

The suite pins the public seams of ``shadow_lab.shadow_only`` without loading
the 0.6B/4B checkpoints: tiny random Llama models, a deterministic stub
tokenizer, and injected generation/loader callables for the heavy paths.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from shadow_lab.shadow_only import (  # noqa: E402
    IGNORE_INDEX,
    ShadowOnlyTrainConfig,
    ShadowOnlyTrainStats,
    TrainableScopeReport,
    build_sft_rows,
    evaluate_shadow_only,
    export_roundtrip_check,
    format_is_valid,
    load_exported_shadow,
    numeric_answer_match,
    parse_gsm8k_answer,
    repetition_rate,
    set_trainable_scope,
    summarize_generations,
    train_shadow_only_sft,
)

VOCAB = 128


class StubTokenizer:
    """One id per character; a chat template that marks assistant turns."""

    pad_token_id = 0
    eos_token_id = 2
    chat_template = "stub"
    padding_side = "right"

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **_: object,
    ):
        text = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "<|assistant|>"
        if not tokenize:
            return text
        return self(text)

    def __call__(self, text, add_special_tokens: bool = True, **_: object):
        del add_special_tokens
        return {"input_ids": [3 + (ord(char) % (VOCAB - 4)) for char in text]}

    def decode(self, ids, skip_special_tokens: bool = True):
        del skip_special_tokens
        return "".join(chr(33 + (int(token) % 90)) for token in ids)

    def save_pretrained(self, path):
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
        (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        return tuple(str(item) for item in sorted(directory.iterdir()))


def tiny_model(
    *, seed: int = 0, hidden_size: int = 16, num_layers: int = 1
) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=VOCAB,
            hidden_size=hidden_size,
            intermediate_size=32,
            num_hidden_layers=num_layers,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
    )


class TinyExported(torch.nn.Module):
    """Stand-in for the projection-wrapper export shape."""

    def __init__(self) -> None:
        super().__init__()
        self.shadow_model = torch.nn.Module()
        self.shadow_model.embed_tokens = torch.nn.Embedding(10, 4)
        self.shadow_model.layer = torch.nn.Linear(4, 4)
        self.shadow_hidden_projection = torch.nn.Linear(4, 4, bias=False)
        self.lm_head = torch.nn.Linear(4, 10, bias=False)


# ---------------------------------------------------------------------------
# Config and scope
# ---------------------------------------------------------------------------


def test_train_config_defaults_and_validation():
    config = ShadowOnlyTrainConfig()
    assert config.train_scope == "backbone"
    assert config.device == "cpu"
    assert config.steps >= 1

    with pytest.raises(ValueError, match="train_scope"):
        ShadowOnlyTrainConfig(train_scope="everything")
    with pytest.raises(ValueError, match="steps"):
        ShadowOnlyTrainConfig(steps=0)
    with pytest.raises(ValueError, match="dtype"):
        ShadowOnlyTrainConfig(dtype="int4")


@pytest.mark.parametrize(
    ("scope", "expected_trainable"),
    [
        ("backbone", 60),
        ("head", 56),
        ("all", 116),
        ("none", 0),
    ],
)
def test_set_trainable_scope_counts(scope, expected_trainable):
    model = TinyExported()
    report = set_trainable_scope(model, scope)

    assert isinstance(report, TrainableScopeReport)
    assert report.scope == scope
    assert report.total_parameters == 116
    assert report.trainable_parameters == expected_trainable
    assert isinstance(report.as_dict(), dict)


def test_set_trainable_scope_freezes_projection_and_head_for_backbone():
    model = TinyExported()
    set_trainable_scope(model, "backbone")

    assert all(p.requires_grad for p in model.shadow_model.parameters())
    assert not any(p.requires_grad for p in model.shadow_hidden_projection.parameters())
    assert not any(p.requires_grad for p in model.lm_head.parameters())


def test_set_trainable_scope_all_then_none_roundtrip():
    model = TinyExported()
    assert set_trainable_scope(model, "all").trainable_parameters == 116
    assert set_trainable_scope(model, "none").trainable_parameters == 0
    assert not any(p.requires_grad for p in model.parameters())


# ---------------------------------------------------------------------------
# SFT row construction
# ---------------------------------------------------------------------------


def test_build_sft_rows_masks_prompt_and_keeps_answer_labels():
    tokenizer = StubTokenizer()
    rows = build_sft_rows(
        [{"prompt": "2+2?", "answer": "4"}], tokenizer, max_length=64
    )

    assert len(rows) == 1
    row = rows[0]
    assert len(row["input_ids"]) == len(row["labels"]) == len(row["attention_mask"])
    assert all(mask == 1 for mask in row["attention_mask"])
    assert row["truncated"] is False

    first_label = next(index for index, label in enumerate(row["labels"]) if label != IGNORE_INDEX)
    assert first_label > 0
    assert row["labels"][:first_label] == [IGNORE_INDEX] * first_label
    assert row["labels"][first_label:] == row["input_ids"][first_label:]
    # The prompt text itself must be inside the masked prefix.
    assert row["input_ids"][:first_label] == tokenizer.apply_chat_template(
        [{"role": "user", "content": "2+2?"}],
        add_generation_prompt=True,
    )["input_ids"]


def test_build_sft_rows_accepts_messages_and_masks_only_last_answer():
    tokenizer = StubTokenizer()
    rows = build_sft_rows(
        [
            {
                "messages": [
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "ok"},
                ]
            }
        ],
        tokenizer,
        max_length=128,
    )

    row = rows[0]
    first_label = next(index for index, label in enumerate(row["labels"]) if label != IGNORE_INDEX)
    assert row["labels"][:first_label] == [IGNORE_INDEX] * first_label
    assert row["input_ids"][first_label:] == tokenizer("ok")["input_ids"]


def test_build_sft_rows_truncates_from_the_left_and_keeps_answer():
    tokenizer = StubTokenizer()
    answer = "abcdefghij"
    rows = build_sft_rows(
        [{"prompt": "q", "answer": answer}], tokenizer, max_length=10
    )

    row = rows[0]
    assert row["truncated"] is True
    assert len(row["input_ids"]) == 10
    assert row["input_ids"] == tokenizer(answer)["input_ids"]
    assert row["labels"] == row["input_ids"]


def test_build_sft_rows_rejects_rows_without_an_answer():
    tokenizer = StubTokenizer()
    with pytest.raises(ValueError, match="answer"):
        build_sft_rows([{"prompt": "q", "answer": ""}], tokenizer, max_length=32)
    with pytest.raises(ValueError, match="messages"):
        build_sft_rows(
            [{"messages": [{"role": "user", "content": "q"}]}],
            tokenizer,
            max_length=32,
        )


def test_build_sft_rows_rejects_oversized_max_length():
    with pytest.raises(ValueError, match="max_length"):
        build_sft_rows([{"prompt": "q", "answer": "a"}], StubTokenizer(), max_length=1)


# ---------------------------------------------------------------------------
# Shadow-only SFT loop
# ---------------------------------------------------------------------------


SFT_ROWS = [
    {"prompt": f"question {index}", "answer": f"answer {index}"}
    for index in range(4)
]


def _train_config(**overrides) -> ShadowOnlyTrainConfig:
    values = {
        "train_scope": "all",
        "lr": 1e-3,
        "steps": 2,
        "batch_size": 2,
        "max_seq_len": 64,
        "seed": 7,
        "dtype": "float32",
        "device": "cpu",
        "log_every": 1,
    }
    values.update(overrides)
    return ShadowOnlyTrainConfig(**values)


def test_train_shadow_only_sft_returns_deterministic_stats():
    tokenizer = StubTokenizer()
    model = tiny_model(seed=3)
    stats = train_shadow_only_sft(model, tokenizer, SFT_ROWS, _train_config())

    assert isinstance(stats, ShadowOnlyTrainStats)
    assert stats.steps == 2
    assert stats.rows == 4
    assert stats.tokens > 0
    assert len(stats.losses) == 2
    assert all(math.isfinite(loss) for loss in stats.losses)
    assert stats.train_scope == "all"
    assert stats.trainable_parameters == stats.total_parameters
    assert stats.seconds >= 0.0
    payload = stats.as_dict()
    assert payload["steps"] == 2
    assert payload["tokens"] == stats.tokens
    assert isinstance(payload["losses"], list)
    # The loop leaves the model in eval mode for the follow-up generation.
    assert model.training is False


def test_train_shadow_only_sft_is_reproducible_for_a_fixed_seed():
    tokenizer = StubTokenizer()
    first = train_shadow_only_sft(tiny_model(seed=11), tokenizer, SFT_ROWS, _train_config())
    second = train_shadow_only_sft(tiny_model(seed=11), tokenizer, SFT_ROWS, _train_config())

    assert first.tokens == second.tokens
    assert first.losses == pytest.approx(second.losses, abs=1e-7)


def test_train_shadow_only_sft_backbone_scope_freezes_the_head():
    model = tiny_model(seed=5)
    stats = train_shadow_only_sft(
        model, StubTokenizer(), SFT_ROWS, _train_config(train_scope="backbone")
    )

    assert stats.trainable_parameters < stats.total_parameters
    assert not any(p.requires_grad for p in model.lm_head.parameters())


def test_train_shadow_only_sft_rejects_frozen_or_empty_inputs():
    with pytest.raises(ValueError, match="没有启用任何参数"):
        train_shadow_only_sft(
            tiny_model(), StubTokenizer(), SFT_ROWS, _train_config(train_scope="none")
        )
    with pytest.raises(ValueError, match="rows"):
        train_shadow_only_sft(
            tiny_model(), StubTokenizer(), [], _train_config()
        )


# ---------------------------------------------------------------------------
# GSM8K grading and generation summaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Reasoning...\n#### 18", "18"),
        ("#### 1,000", "1000"),
        ("no marker but 42", "42"),
        ("#### -3.5", "-3.5"),
        ("", ""),
    ],
)
def test_parse_gsm8k_answer(text, expected):
    assert parse_gsm8k_answer(text) == expected


def test_format_is_valid_requires_the_marker_and_a_number():
    assert format_is_valid("...\n#### 18")
    assert format_is_valid("...\n#### 1,000")
    assert not format_is_valid("... final is 18")
    assert not format_is_valid("#### answer")
    assert not format_is_valid("")


@pytest.mark.parametrize(
    ("predicted", "gold", "expected"),
    [
        ("#### 18", "18", True),
        ("#### 18.0", "18", True),
        ("#### 1,000", "1000", True),
        ("#### 18", "19", False),
        ("#### 0", "0", True),
    ],
)
def test_numeric_answer_match(predicted, gold, expected):
    assert numeric_answer_match(predicted, gold) is expected


def test_repetition_rate_is_character_ngram_repeat_fraction():
    assert repetition_rate("aaaa", 3) == pytest.approx(0.5)
    assert repetition_rate("ab", 3) == 0.0
    assert repetition_rate("", 3) == 0.0


def test_summarize_generations_recomputes_every_gate_metric():
    records = [
        {
            "prompt": "q1",
            "answer": "18",
            "output": "two plus sixteen\n#### 18",
            "truncated": False,
            "empty": False,
            "n_new_tokens": 10,
        },
        {
            "prompt": "q2",
            "answer": "5",
            "output": "I think it is 6\n#### 6",
            "truncated": True,
            "empty": False,
            "n_new_tokens": 32,
        },
        {
            "prompt": "q3",
            "answer": "3",
            "output": "",
            "truncated": False,
            "empty": True,
            "n_new_tokens": 0,
        },
    ]
    summary = summarize_generations(records, max_new_tokens=32)

    assert summary["n"] == 3
    assert summary["accuracy"] == pytest.approx(1 / 3)
    assert summary["format_rate"] == pytest.approx(2 / 3)
    assert summary["truncation_rate"] == pytest.approx(1 / 3)
    assert summary["empty_rate"] == pytest.approx(1 / 3)
    assert summary["mean_new_tokens"] == pytest.approx(42 / 3)
    assert 0.0 <= summary["mean_distinct3"] <= 1.0
    assert 0.0 <= summary["mean_repetition4"] <= 1.0
    assert summary["max_new_tokens"] == 32


def test_evaluate_shadow_only_grades_through_the_generation_seam():
    examples = [
        {"prompt": "q1", "answer": "18"},
        {"prompt": "q2", "answer": "5"},
    ]
    canned = {
        "q1": {"text": "#### 18", "truncated": False, "n_new_tokens": 2},
        "q2": {"text": "#### 7", "truncated": True, "n_new_tokens": 8},
    }

    def fake_generate(model, tokenizer, prompts, *, max_new_tokens, batch_size, device):
        assert model is None and tokenizer is None
        assert (max_new_tokens, batch_size, device) == (8, 2, "cpu")
        return [dict(canned[prompt]) for prompt in prompts]

    result = evaluate_shadow_only(
        None,
        None,
        examples,
        max_new_tokens=8,
        batch_size=2,
        device="cpu",
        generate_fn=fake_generate,
    )

    assert result["summary"]["n"] == 2
    assert result["summary"]["accuracy"] == pytest.approx(0.5)
    assert result["summary"]["format_rate"] == pytest.approx(1.0)
    assert result["summary"]["truncation_rate"] == pytest.approx(0.5)
    assert [row["correct"] for row in result["generations"]] == [True, False]
    assert result["generations"][0]["prompt"] == "q1"


def test_evaluate_shadow_only_rejects_mismatched_generation_counts():
    def fake_generate(*args, **kwargs):
        return []

    with pytest.raises(ValueError, match="生成"):
        evaluate_shadow_only(
            None,
            None,
            [{"prompt": "q", "answer": "1"}],
            max_new_tokens=4,
            generate_fn=fake_generate,
        )


# ---------------------------------------------------------------------------
# Export round-trip
# ---------------------------------------------------------------------------


class FakeRoundtripModel(torch.nn.Module):
    """Deterministic logits from token ids; saving writes a marker file."""

    def __init__(self, *, bump: float = 0.0) -> None:
        super().__init__()
        self.bump = float(bump)
        self.config = type("Config", (), {"vocab_size": 6})()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        batch, length = input_ids.shape
        base = torch.arange(length * 6, dtype=torch.float32).reshape(1, length, 6)
        logits = base.expand(batch, -1, -1) + self.bump + self.anchor * 0.0
        return type("Output", (), {"logits": logits})()

    def save_pretrained(self, path):
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.safetensors").write_text("marker", encoding="utf-8")
        (directory / "config.json").write_text("{}", encoding="utf-8")


def _roundtrip_loader(bump: float):
    def load(path, dtype=None):
        del path, dtype
        return FakeRoundtripModel(bump=bump)

    return load


class FakeBufferModel(FakeRoundtripModel):
    """Fake model with a non-persistent-style floating buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("rotary", torch.ones(2, dtype=torch.float32))


def test_export_roundtrip_check_casts_non_persistent_buffers(tmp_path):
    loaded = FakeBufferModel()

    def loader(path, dtype=None):
        del path, dtype
        return loaded

    report = export_roundtrip_check(
        FakeBufferModel(),
        StubTokenizer(),
        prompt="hello",
        save_dir=tmp_path / "buffers",
        dtype="bfloat16",
        model_loader=loader,
    )

    assert report["passed"] is True
    assert loaded.rotary.dtype == torch.bfloat16


def test_export_roundtrip_check_reports_schema_and_passes(tmp_path):
    model = FakeRoundtripModel()
    save_dir = tmp_path / "roundtrip"
    report = export_roundtrip_check(
        model,
        StubTokenizer(),
        prompt="hello",
        save_dir=save_dir,
        model_loader=_roundtrip_loader(0.0),
    )

    assert report["passed"] is True
    assert report["allclose"] is True
    assert report["max_abs_diff"] == 0.0
    assert report["vocab_size"] == 6
    assert report["n_tokens"] == 5
    assert report["error"] is None
    assert "tokenizer.json" in report["tokenizer_files"]
    assert "tokenizer_config.json" in report["tokenizer_files"]
    assert "model.safetensors" in report["model_files"]
    assert report["save_dir"] == str(save_dir)
    assert (save_dir / "tokenizer.json").exists()


def test_export_roundtrip_check_reports_a_mismatch(tmp_path):
    report = export_roundtrip_check(
        FakeRoundtripModel(),
        StubTokenizer(),
        prompt="hello",
        save_dir=tmp_path / "mismatch",
        model_loader=_roundtrip_loader(0.5),
    )

    assert report["passed"] is False
    assert report["allclose"] is False
    assert report["max_abs_diff"] == pytest.approx(0.5)
    assert report["error"] is None


def test_export_roundtrip_check_reports_loader_errors(tmp_path):
    def broken_loader(path, dtype=None):
        raise RuntimeError("cannot reload")

    report = export_roundtrip_check(
        FakeRoundtripModel(),
        StubTokenizer(),
        prompt="hello",
        save_dir=tmp_path / "broken",
        model_loader=broken_loader,
    )

    assert report["passed"] is False
    assert "cannot reload" in report["error"]


# ---------------------------------------------------------------------------
# Export loading (miniature base + H4B-style projection wrapper)
# ---------------------------------------------------------------------------


def _write_tiny_shadow_checkpoint(tmp_path):
    from shadow_peft import (
        AutoModelForCausalLMWithHiddenProjection,
        ShadowConfig,
        get_shadow_model,
    )

    base = tiny_model(seed=0, hidden_size=16, num_layers=2)
    base.save_pretrained(tmp_path / "base")

    torch.manual_seed(0)
    shadow_full = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=VOCAB,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
    )
    wrapper = AutoModelForCausalLMWithHiddenProjection.wrap(
        shadow_model=shadow_full,
        shadow_hidden_projection=torch.nn.Linear(8, 16, bias=False),
        lm_head=torch.nn.Linear(16, VOCAB, bias=False),
        init_optimal_projection=False,
    )
    wrapper.save_pretrained(tmp_path / "shadow-model")

    adapter = get_shadow_model(
        base,
        ShadowConfig(
            num_shadow_layers=1,
            injection_hidden_size=4,
            gate_hidden_size=4,
            alpha=0.1,
            dropout=0.0,
        ),
        shadow_model=wrapper,
    )
    adapter.save_pretrained(tmp_path / "ckpt")
    return tmp_path / "base", tmp_path / "shadow-model", tmp_path / "ckpt"


def test_load_exported_shadow_returns_a_standalone_model(tmp_path):
    base_path, shadow_path, ckpt_path = _write_tiny_shadow_checkpoint(tmp_path)

    exported, tokenizer = load_exported_shadow(
        str(base_path),
        str(shadow_path),
        str(ckpt_path),
        device="cpu",
        dtype="float32",
        tokenizer_loader=lambda path: StubTokenizer(),
    )

    assert isinstance(tokenizer, StubTokenizer)
    assert type(exported).__name__ == "AutoModelForCausalLMWithHiddenProjection"
    assert exported.training is False
    with torch.inference_mode():
        logits = exported(input_ids=torch.tensor([[1, 2, 3]])).logits
    assert tuple(logits.shape) == (1, 3, VOCAB)


def test_load_exported_shadow_trains_through_the_projection_wrapper(tmp_path):
    base_path, shadow_path, ckpt_path = _write_tiny_shadow_checkpoint(tmp_path)
    exported, tokenizer = load_exported_shadow(
        str(base_path),
        str(shadow_path),
        str(ckpt_path),
        device="cpu",
        dtype="float32",
        tokenizer_loader=lambda path: StubTokenizer(),
    )

    stats = train_shadow_only_sft(
        exported,
        tokenizer,
        [{"prompt": "2+2?", "answer": "4"}],
        ShadowOnlyTrainConfig(
            train_scope="backbone",
            lr=1e-3,
            steps=1,
            batch_size=1,
            max_seq_len=32,
            seed=0,
            dtype="float32",
            device="cpu",
        ),
    )

    assert stats.steps == 1
    assert stats.trainable_parameters < stats.total_parameters
    assert math.isfinite(stats.loss_last)


def test_load_exported_shadow_rejects_missing_paths(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_exported_shadow(
            str(tmp_path / "missing-base"),
            str(tmp_path / "missing-shadow"),
            str(tmp_path / "missing-ckpt"),
            device="cpu",
            dtype="float32",
            tokenizer_loader=lambda path: StubTokenizer(),
        )


# ---------------------------------------------------------------------------
# exp16 script seams
# ---------------------------------------------------------------------------

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "exp16_shadow_only.py"
_SPEC = importlib.util.spec_from_file_location("exp16_shadow_only", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
exp16 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = exp16
_SPEC.loader.exec_module(exp16)


def test_exp16_parser_exposes_all_commands():
    parser = exp16.build_parser()
    for command in ("diagnose", "train", "eval", "run"):
        args = parser.parse_args([command, "--out", "/tmp/exp16", "--smoke"])
        assert args.command == command
        assert args.smoke is True
        if command in ("train", "run"):
            assert args.train_scope == "backbone"


def test_exp16_gate_summary_does_not_gate_recovery_on_the_baseline():
    gates = exp16._gate_summary(
        {"passed": True},
        {"format_rate": 0.0},
        {
            "format_rate": 0.9,
            "truncation_rate": 0.1,
            "empty_rate": 0.0,
        },
    )

    assert gates["roundtrip_passed"] is True
    assert gates["diagnose_format_ok"] is False
    assert gates["eval_format_ok"] is True
    assert gates["recovery_ready"] is True
    assert gates["accuracy_gate"].startswith("n/a")


def test_exp16_load_jsonl_rows_honors_limit(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"prompt": "q1", "answer": "1"},
                {"prompt": "q2", "answer": "2"},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert exp16.load_jsonl_rows(path) == [
        {"prompt": "q1", "answer": "1"},
        {"prompt": "q2", "answer": "2"},
    ]
    assert exp16.load_jsonl_rows(path, limit=1) == [{"prompt": "q1", "answer": "1"}]
