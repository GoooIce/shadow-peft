"""On-policy rollout data pipeline (issue #19).

The seams under test are the module-level data functions only: prompt
loading, prompt-level splitting, rollout generation through the public
``ShadowForCausalLM.generate`` path, and the JSONL round-trip. Tests run on
CPU with a tiny random Llama and finish in seconds.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from tokenizers import Tokenizer  # noqa: E402
from tokenizers import models as tk_models  # noqa: E402
from tokenizers import pre_tokenizers as tk_pre  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast  # noqa: E402

from shadow_lab.onpolicy import (  # noqa: E402
    RolloutConfig,
    generate_rollouts,
    load_prompts,
    load_rollouts,
    split_prompts,
    write_rollouts,
)
from shadow_peft import ShadowConfig, ShadowForCausalLM, get_shadow_model  # noqa: E402

VOCAB = 96


@pytest.fixture(scope="module")
def tiny_bundle():
    """Random-init tiny Llama + a matching WordLevel tokenizer, on CPU."""
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    base = LlamaForCausalLM(cfg)
    # The tokenizer deliberately has no EOS: random init cannot produce a
    # controlled EOS, and disabling early stop keeps these tests non-flaky.
    base.generation_config.eos_token_id = None
    base.generation_config.pad_token_id = 0
    shadow_cfg = ShadowConfig(
        num_shadow_layers=1,
        injection_hidden_size=8,
        gate_hidden_size=10,
        alpha=0.1,
        dropout=0.0,
    )
    model = ShadowForCausalLM(
        get_shadow_model(base, shadow_cfg),
        shadow_loss_weight=0.05,
        inference_mode="base_shadow",
    )
    model.eval()
    tk = Tokenizer(
        tk_models.WordLevel(
            vocab={f"w{i}": i for i in range(VOCAB)},
            unk_token="w0",
        )
    )
    tk.pre_tokenizer = tk_pre.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        pad_token="w0",
        bos_token="w2",
    )
    return model, tokenizer


def _write_jsonl(path, rows: list[dict | str]) -> None:
    lines = []
    for row in rows:
        lines.append(row if isinstance(row, str) else json.dumps(row, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_prompts_skips_empty_missing_and_malformed(tmp_path):
    path = tmp_path / "prompts.jsonl"
    _write_jsonl(
        path,
        [
            {"prompt": "alpha", "task": "qa", "source": "feedback", "confidence": 0.2},
            {"prompt": "   ", "task": "qa"},
            {"no_prompt": True},
            "{not-json",
            "",
            {"prompt": "beta", "task": "process_calc", "source": "cloud"},
        ],
    )

    loaded = load_prompts(path)

    assert [row["prompt"] for row in loaded] == ["alpha", "beta"]
    assert loaded[0]["task"] == "qa"
    assert loaded[1]["source"] == "cloud"


def test_load_prompts_dedupes_by_prompt_keeping_highest_confidence(tmp_path):
    path = tmp_path / "prompts.jsonl"
    _write_jsonl(
        path,
        [
            {"prompt": "alpha", "confidence": 0.2, "request_id": "low"},
            {"prompt": "beta", "confidence": 0.5, "request_id": "beta"},
            {"prompt": "alpha", "confidence": 0.9, "request_id": "high"},
            {"prompt": "alpha", "confidence": 0.9, "request_id": "high-tie-last"},
        ],
    )

    loaded = load_prompts(path)

    # First-seen order is kept; the representative row is the highest-confidence
    # (last wins on ties) so the export metadata survives deduplication.
    assert [row["prompt"] for row in loaded] == ["alpha", "beta"]
    assert loaded[0]["request_id"] == "high-tie-last"
    assert loaded[0]["confidence"] == 0.9


def test_load_prompts_keeps_export_prompts_metadata(tmp_path):
    path = tmp_path / "prompts.jsonl"
    row = {
        "prompt": "alpha",
        "task": "quality_verdict",
        "source": "feedback",
        "confidence": 0.8,
        "request_id": "r-1",
        "escalation_reason": "low_confidence",
        "adapter_version": "v7",
        "is_hard_case": True,
        "accepted": False,
        "corrected": True,
    }
    _write_jsonl(path, [row])

    assert load_prompts(path) == [row]


def test_split_prompts_groups_duplicates_and_is_deterministic():
    prompts = [{"prompt": f"p{i}", "confidence": 0.5} for i in range(8)]
    prompts.append({"prompt": "p0", "confidence": 0.1})

    train, val = split_prompts(prompts, val_ratio=0.25, seed=7)

    train_keys = {row["prompt"] for row in train}
    val_keys = {row["prompt"] for row in val}
    assert train_keys.isdisjoint(val_keys)
    assert train_keys | val_keys == {f"p{i}" for i in range(8)}
    assert len(val_keys) == 2
    # Deterministic under the same seed and prompt order.
    assert (train, val) == split_prompts(prompts, val_ratio=0.25, seed=7)


def test_split_prompts_zero_val_ratio_keeps_all_prompts_in_train():
    prompts = [{"prompt": f"p{i}"} for i in range(4)]

    train, val = split_prompts(prompts, val_ratio=0.0, seed=3)

    assert [row["prompt"] for row in train] == ["p0", "p1", "p2", "p3"]
    assert val == []


def test_generate_rollouts_is_bitwise_reproducible_and_masks_prompt_labels(tiny_bundle):
    model, tokenizer = tiny_bundle
    config = RolloutConfig(
        num_rollouts=2,
        max_new_tokens=4,
        max_rollout_tokens=4,
        seed=11,
        val_ratio=0.0,
        adapter_version="v-test",
    )
    prompts = [{"prompt": "w5 w6 w7", "task": "qa", "source": "feedback"}]

    first = generate_rollouts(model, tokenizer, prompts, config)
    second = generate_rollouts(model, tokenizer, prompts, config)

    assert isinstance(first, list)
    assert first == second
    assert len(first) == 2

    prompt_ids = tokenizer("w5 w6 w7")["input_ids"]
    for row in first:
        assert row["prompt"] == "w5 w6 w7"
        assert row["task"] == "qa"
        assert row["source"] == "feedback"
        assert row["split"] == "train"
        assert row["rollout_mode"] == "base_shadow"
        assert row["adapter_version"] == "v-test"
        assert row["seed"] == 11
        assert row["prompt_ids"] == prompt_ids
        assert 1 <= len(row["rollout_ids"]) <= 4
        assert row["finish_reason"] == "length"
        assert row["labels"] == [-100] * len(prompt_ids) + row["rollout_ids"]

    assert first.stats["attempts"] == 2
    assert first.stats["kept"] == 2
    assert first.stats["discarded"] == 0
    assert first.stats["discard_rate"] == 0.0


def test_generate_rollouts_keeps_all_rollouts_of_a_prompt_on_one_side(tiny_bundle):
    model, tokenizer = tiny_bundle
    prompts = [{"prompt": f"w{i} w{i + 1}", "source": "feedback"} for i in range(1, 7)]
    config = RolloutConfig(
        num_rollouts=2,
        max_new_tokens=1,
        max_rollout_tokens=1,
        seed=3,
        val_ratio=0.5,
    )

    rows = generate_rollouts(model, tokenizer, prompts, config)

    by_prompt: dict[str, set[str]] = {}
    for row in rows:
        by_prompt.setdefault(row["prompt"], set()).add(row["split"])
    assert len(by_prompt) == 6
    assert all(len(splits) == 1 for splits in by_prompt.values())
    assert {next(iter(splits)) for splits in by_prompt.values()} == {"train", "val"}
    assert all(sum(1 for row in rows if row["prompt"] == prompt) == 2 for prompt in by_prompt)


def test_generate_rollouts_supports_shadow_only_and_restores_inference_mode(tiny_bundle):
    model, tokenizer = tiny_bundle
    model.set_inference_mode("base_shadow")
    config = RolloutConfig(
        rollout_mode="shadow_only",
        num_rollouts=1,
        max_new_tokens=2,
        max_rollout_tokens=2,
        seed=5,
        val_ratio=0.0,
    )

    rows = generate_rollouts(model, tokenizer, [{"prompt": "w5 w6"}], config)

    assert len(rows) == 1
    assert rows[0]["rollout_mode"] == "shadow_only"
    assert model.inference_mode == "base_shadow"


def test_generate_rollouts_discards_empty_immediate_eos_and_too_long(tiny_bundle, monkeypatch):
    model, tokenizer = tiny_bundle
    monkeypatch.setattr(tokenizer, "eos_token_id", 1)
    eos = 1
    # One attempt per queued completion. [5, eos, 7, 8] trims at EOS to two
    # tokens and survives; [5, 6, 7, 8] exceeds max_rollout_tokens.
    queue = [[], [eos], [5, 6, 7, 8], [5, eos, 7, 8]]
    calls: list[dict] = []

    def fake_generate(*, input_ids, attention_mask=None, **kwargs):
        calls.append({**kwargs, "inference_mode": model.inference_mode})
        new_tokens = queue.pop(0)
        if not new_tokens:
            return input_ids.clone()
        return torch.cat(
            [input_ids, torch.tensor([new_tokens], dtype=torch.long)],
            dim=1,
        )

    monkeypatch.setattr(model, "generate", fake_generate)
    config = RolloutConfig(
        num_rollouts=4,
        max_new_tokens=4,
        max_rollout_tokens=3,
        seed=9,
        val_ratio=0.0,
    )

    rows = generate_rollouts(model, tokenizer, [{"prompt": "w5 w6"}], config)

    assert len(rows) == 1
    row = rows[0]
    assert row["rollout_ids"] == [5, eos]
    assert row["finish_reason"] == "eos"
    assert row["labels"][: len(row["prompt_ids"])] == [-100] * len(row["prompt_ids"])
    assert row["labels"][len(row["prompt_ids"]) :] == [5, eos]

    assert rows.stats["attempts"] == 4
    assert rows.stats["kept"] == 1
    assert rows.stats["discarded"] == 3
    assert rows.stats["discard_rate"] == 0.75
    assert rows.stats["by_reason"] == {"empty": 1, "immediate_eos": 1, "too_long": 1}

    assert all(call["do_sample"] is True for call in calls)
    assert all(call["temperature"] == 1.0 and call["top_p"] == 1.0 for call in calls)
    assert all(call["max_new_tokens"] == 4 and call["use_cache"] is False for call in calls)
    assert all(call["inference_mode"] == "base_shadow" for call in calls)


def test_generate_rollouts_truncates_long_prompts_to_max_prompt_tokens(tiny_bundle, monkeypatch):
    model, tokenizer = tiny_bundle

    def fake_generate(*, input_ids, attention_mask=None, **kwargs):
        return torch.cat([input_ids, torch.tensor([[8]], dtype=torch.long)], dim=1)

    monkeypatch.setattr(model, "generate", fake_generate)
    config = RolloutConfig(
        num_rollouts=1,
        max_new_tokens=1,
        max_rollout_tokens=1,
        max_prompt_tokens=3,
        seed=2,
        val_ratio=0.0,
    )

    rows = generate_rollouts(model, tokenizer, [{"prompt": "w3 w4 w5 w6 w7"}], config)

    assert len(rows) == 1
    assert rows[0]["prompt"] == "w3 w4 w5 w6 w7"
    assert rows[0]["prompt_ids"] == [5, 6, 7]
    assert rows[0]["labels"] == [-100, -100, -100, 8]
    assert rows.stats["truncated_prompts"] == 1


def test_rollout_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="rollout_mode"):
        RolloutConfig(rollout_mode="base")
    with pytest.raises(ValueError, match="num_rollouts"):
        RolloutConfig(num_rollouts=0)
    with pytest.raises(ValueError, match="val_ratio"):
        RolloutConfig(val_ratio=1.0)


def test_write_and_load_rollouts_roundtrip_is_stable(tmp_path):
    rows = [
        {
            "prompt": "b",
            "task": "qa",
            "source": "feedback",
            "split": "val",
            "prompt_ids": [5],
            "rollout_ids": [3, 1],
            "labels": [-100, 3, 1],
            "finish_reason": "eos",
            "rollout_mode": "base_shadow",
            "adapter_version": "v1",
            "seed": 4,
        },
        {
            "prompt": "a",
            "task": "qa",
            "source": "feedback",
            "split": "train",
            "prompt_ids": [6, 7],
            "rollout_ids": [2],
            "labels": [-100, -100, 2],
            "finish_reason": "length",
            "rollout_mode": "base_shadow",
            "adapter_version": "v1",
            "seed": 5,
        },
    ]
    first = tmp_path / "rollouts.jsonl"
    second = tmp_path / "copy.jsonl"

    write_rollouts(rows, first)
    write_rollouts(rows, second)

    assert first.read_bytes() == second.read_bytes()
    assert load_rollouts(first) == rows
    assert not (tmp_path / "rollouts.jsonl.tmp").exists()
    for line in first.read_text(encoding="utf-8").splitlines():
        assert list(json.loads(line)) == sorted(json.loads(line))


def test_generated_rows_survive_jsonl_roundtrip(tiny_bundle, tmp_path):
    model, tokenizer = tiny_bundle
    config = RolloutConfig(
        num_rollouts=1,
        max_new_tokens=1,
        max_rollout_tokens=1,
        seed=1,
        val_ratio=0.0,
    )
    rows = generate_rollouts(model, tokenizer, [{"prompt": "w5 w6"}], config)
    path = tmp_path / "generated.jsonl"

    write_rollouts(rows, path)

    assert load_rollouts(path) == rows
