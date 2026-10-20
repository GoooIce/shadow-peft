"""S2T-Local diagnostic runner (``shadow_lab.s2t_diagnose``) tests.

The runner is exercised on a tiny randomly initialized Llama shadow model so
the tests need no downloads and stay deterministic. The tokenizer is a stub
that maps prompt names to fixed token ids; generation is greedy on the base
path and the shadow distribution is read from the initial shadow hidden.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from shadow_lab.s2t import CSV_FIELDS, SCHEMA_VERSION, summarize_hits, token_hit  # noqa: E402
from shadow_lab.s2t_diagnose import diagnose, run_diagnostic  # noqa: E402
from shadow_peft import (  # noqa: E402
    ShadowConfig,
    ShadowForCausalLM,
    get_shadow_model,
)


class _StubTokenizer:
    """Prompt-name -> fixed token ids; no real vocabulary needed."""

    def __init__(self, *, eos_token_id: int | None = None) -> None:
        self.eos_token_id = eos_token_id
        self._encodings = {
            "alpha": [1, 2, 3],
            "beta": [4, 5, 6],
            "gamma": [7],
        }

    def __call__(self, text: str, return_tensors: str = "pt"):
        ids = torch.tensor([self._encodings[text]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _tiny_shadow_model(seed: int = 0) -> ShadowForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    base = LlamaForCausalLM(config)
    shadow_config = ShadowConfig(
        num_shadow_layers=1,
        injection_hidden_size=8,
        gate_hidden_size=10,
        alpha=0.1,
        dropout=0.0,
    )
    wrapper = ShadowForCausalLM(
        get_shadow_model(base, shadow_config),
        inference_mode="base_shadow",
        shadow_loss_weight=0.05,
    )
    return wrapper.eval()


def _expected_steps(
    model: ShadowForCausalLM,
    tokenizer: _StubTokenizer,
    prompt: str,
    *,
    n_steps: int,
    k: int,
) -> list[dict]:
    """Independent full-prefix greedy oracle (no cache, no runner code)."""

    batch = tokenizer(prompt)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    steps: list[dict] = []
    for _ in range(n_steps):
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        base_logits = outputs.logits[0, -1].float()
        shadow_logits = outputs.shadow_logits[0, -1].float()
        steps.append(
            {
                "base_top1": int(base_logits.argmax()),
                "base_topk": torch.topk(base_logits, k=k).indices.tolist(),
                "shadow_topk_ids": torch.topk(shadow_logits, k=k).indices.tolist(),
                "shadow_logits": shadow_logits.tolist(),
            }
        )
        next_id = torch.tensor([[steps[-1]["base_top1"]]], dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_id)], dim=1)
    return steps


def test_diagnose_records_greedy_steps_and_matches_pure_metrics():
    model = _tiny_shadow_model()
    tokenizer = _StubTokenizer()
    prompts = [{"prompt": "alpha", "task": "fault"}]

    records = diagnose(model, tokenizer, prompts, max_new_tokens=2, k=4, seed=0)

    assert len(records) == 2
    expected = _expected_steps(model, tokenizer, "alpha", n_steps=2, k=4)
    required = {
        "prompt_id",
        "task",
        "position",
        "base_top1",
        "base_topk",
        "shadow_topk_ids",
        "shadow_topk_logprobs",
        "base_top1_rank_in_shadow",
        "shadow_hidden",
    }
    for position, (record, step) in enumerate(zip(records, expected, strict=True)):
        assert required <= set(record)
        assert record["prompt_id"] == "p0"
        assert record["task"] == "fault"
        assert record["position"] == position
        assert record["shadow_hidden"] == "initial"
        assert record["base_top1"] == step["base_top1"]
        assert record["base_topk"] == step["base_topk"]
        assert record["shadow_topk_ids"] == step["shadow_topk_ids"]
        assert len(record["shadow_topk_ids"]) == 4
        assert len(record["shadow_topk_logprobs"]) == 4
        assert record["shadow_topk_logprobs"] == sorted(
            record["shadow_topk_logprobs"], reverse=True
        )
        expected_hit = token_hit(step["shadow_logits"], step["base_top1"], k=4)
        assert record["base_top1_rank_in_shadow"] == expected_hit.rank

    summary = summarize_hits(records, ks=4)
    expected_hits = [token_hit(step["shadow_logits"], step["base_top1"], k=4) for step in expected]
    assert summary["hit_at_4"] == pytest.approx(
        sum(hit.hit for hit in expected_hits) / len(expected_hits)
    )
    assert summary["agree_at_1"] == pytest.approx(
        sum(hit.agree for hit in expected_hits) / len(expected_hits)
    )


def test_diagnose_stops_at_eos_and_records_the_eos_token():
    model = _tiny_shadow_model()
    tokenizer = _StubTokenizer()
    first = _expected_steps(model, tokenizer, "alpha", n_steps=1, k=8)[0]
    tokenizer.eos_token_id = first["base_top1"]

    records = diagnose(model, tokenizer, ["alpha"], max_new_tokens=8, k=8)

    assert len(records) == 1
    assert records[0]["base_top1"] == first["base_top1"]


def test_diagnose_stops_at_max_new_tokens_per_prompt():
    model = _tiny_shadow_model()
    tokenizer = _StubTokenizer()
    prompts = [{"prompt": "alpha", "task": "fault"}, {"prompt": "beta", "task": "calc"}]

    records = diagnose(model, tokenizer, prompts, max_new_tokens=2, k=4)

    assert len(records) == 4
    assert [(record["prompt_id"], record["position"]) for record in records] == [
        ("p0", 0),
        ("p0", 1),
        ("p1", 0),
        ("p1", 1),
    ]
    assert [record["task"] for record in records] == [
        "fault",
        "fault",
        "calc",
        "calc",
    ]


def test_diagnose_runs_in_inference_mode_and_restores_train_state():
    model = _tiny_shadow_model(seed=1).train()
    tokenizer = _StubTokenizer()
    observed: list[tuple[bool, bool]] = []

    def _observe(module, _args, _kwargs):
        observed.append((module.training, torch.is_inference_mode_enabled()))

    handle = model.register_forward_pre_hook(_observe, with_kwargs=True)
    try:
        records = diagnose(model, tokenizer, ["alpha"], max_new_tokens=1, k=4)
    finally:
        handle.remove()

    assert records
    assert observed
    assert all(not training for training, _ in observed)
    assert all(inference for _, inference in observed)
    assert model.training is True
    assert not torch.is_inference_mode_enabled()


def test_diagnose_rejects_shadow_only_models():
    model = _tiny_shadow_model()
    model.set_inference_mode("shadow_only")

    with pytest.raises(ValueError, match="base_shadow"):
        diagnose(model, _StubTokenizer(), ["alpha"], max_new_tokens=1, k=4)


def test_use_cache_matches_the_full_prefix_path():
    model = _tiny_shadow_model()
    tokenizer = _StubTokenizer()
    prompts = [{"prompt": "alpha", "task": "fault"}, {"prompt": "beta", "task": "calc"}]

    cached = diagnose(model, tokenizer, prompts, max_new_tokens=3, k=4, use_cache=True)
    full = diagnose(model, tokenizer, prompts, max_new_tokens=3, k=4, use_cache=False)

    assert len(cached) == len(full) == 6
    for cached_record, full_record in zip(cached, full, strict=True):
        assert cached_record["base_top1"] == full_record["base_top1"]
        assert cached_record["base_topk"] == full_record["base_topk"]
        assert cached_record["shadow_topk_ids"] == full_record["shadow_topk_ids"]
        assert cached_record["base_top1_rank_in_shadow"] == full_record["base_top1_rank_in_shadow"]
        assert cached_record["shadow_topk_logprobs"] == pytest.approx(
            full_record["shadow_topk_logprobs"], abs=1e-4
        )


def test_run_diagnostic_defaults_to_k16_and_reports_the_full_hit_curve():
    model = _tiny_shadow_model()
    tokenizer = _StubTokenizer()

    result = run_diagnostic(
        model,
        tokenizer,
        [{"prompt": "alpha", "task": "fault"}, {"prompt": "beta", "task": "calc"}],
        max_new_tokens=3,
        n_boot=20,
    )

    report = result.report
    assert report["candidate_k"] == 16
    assert all(len(record["shadow_topk_ids"]) == 16 for record in result.records)
    hits = [report["hit_at_1"], report["hit_at_4"], report["hit_at_8"], report["hit_at_16"]]
    assert hits == sorted(hits)
    summary = summarize_hits(result.records, ks=(1, 4, 8, 16))
    assert report["hit_at_16"] == pytest.approx(summary["hit_at_16"])
    assert report["hit_at_8"] == pytest.approx(summary["hit_at_8"])
    assert not any("capped by this capture width" in note for note in report["notes"])


def test_report_rank_all_uses_every_exact_token_rank():
    model = _tiny_shadow_model()
    tokenizer = _StubTokenizer()

    result = run_diagnostic(
        model,
        tokenizer,
        [{"prompt": "alpha", "task": "fault"}],
        max_new_tokens=8,
        k=16,
        n_boot=20,
    )

    report = result.report
    exact_ranks = [record["base_top1_rank_in_shadow"] for record in result.records]
    assert all(isinstance(rank, int) and rank >= 1 for rank in exact_ranks)
    assert report["n_rank_known_all"] == len(exact_ranks) == report["n_tokens"]
    assert report["n_rank_known"] <= report["n_rank_known_all"]
    assert report["rank_all_p99"] is not None
    assert report["rank_all_p99"] >= report["rank_hit_p99"]


def test_hit_at_8_is_identical_when_capture_widens_from_k8_to_k16():
    model = _tiny_shadow_model()
    tokenizer = _StubTokenizer()
    prompts = [{"prompt": "alpha", "task": "fault"}, {"prompt": "beta", "task": "calc"}]

    narrow = diagnose(model, tokenizer, prompts, max_new_tokens=3, k=8)
    wide = diagnose(model, tokenizer, prompts, max_new_tokens=3, k=16)

    narrow_summary = summarize_hits(narrow, ks=(1, 4, 8, 16))
    wide_summary = summarize_hits(wide, ks=(1, 4, 8, 16))
    assert wide_summary["hit_at_1"] == pytest.approx(narrow_summary["hit_at_1"])
    assert wide_summary["hit_at_4"] == pytest.approx(narrow_summary["hit_at_4"])
    assert wide_summary["hit_at_8"] == pytest.approx(narrow_summary["hit_at_8"])
    assert (
        wide_summary["hit_at_1"]
        <= wide_summary["hit_at_4"]
        <= wide_summary["hit_at_8"]
        <= wide_summary["hit_at_16"]
    )


def test_cli_k_default_is_16():
    script = Path(__file__).resolve().parents[2] / "scripts" / "exp11_s2t_diagnose.py"
    spec = importlib.util.spec_from_file_location("exp11_s2t_diagnose", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args(
        ["--model", "m", "--shadow-ckpt", "c", "--prompts", "p"]
    )

    assert args.k == 16


def test_run_diagnostic_exports_stable_json_jsonl_and_csv(tmp_path):
    model = _tiny_shadow_model()
    tokenizer = _StubTokenizer()
    prompts = [
        {"prompt": "alpha", "task": "fault"},
        {"prompt": "   ", "task": "skipped"},
        {"prompt": "beta", "task": "calc"},
    ]

    result = run_diagnostic(
        model,
        tokenizer,
        prompts,
        out=tmp_path,
        max_new_tokens=2,
        k=4,
        seed=3,
        n_boot=50,
    )

    report = result.report
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["shadow_hidden"] == "initial"
    assert "initial shadow hidden" in report["shadow_hidden_note"]
    assert report["candidate_k"] == 4
    assert report["n_tokens"] == 4
    assert report["n_prompts"] == 2
    assert report["skipped_prompts"] == ["p1"]
    assert set(report["by_task"]) == {"calc", "fault"}
    assert set(report["by_position_bucket"]) == {"0", "1-3"}
    assert report["go_no_go"] in {"go", "pilot", "no_go"}
    assert report["ci_n_boot"] == 50
    assert report["ci_seed"] == 3

    json_path = Path(result.paths["json"])
    jsonl_path = Path(result.paths["jsonl"])
    csv_path = Path(result.paths["csv"])
    assert json.loads(json_path.read_text(encoding="utf-8")) == report

    token_records = [
        json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(token_records) == report["n_tokens"]
    assert [record["prompt_id"] for record in token_records] == ["p0", "p0", "p2", "p2"]
    assert all(record["shadow_hidden"] == "initial" for record in token_records)
    summary = summarize_hits(token_records, ks=4)
    assert report["hit_at_4"] == pytest.approx(summary["hit_at_4"])
    assert report["agree_at_1"] == pytest.approx(summary["agree_at_1"])

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert tuple(reader.fieldnames) == CSV_FIELDS
    assert len(rows) == 1 + len(report["by_task"]) + len(report["by_position_bucket"])
    overall = next(row for row in rows if row["scope"] == "overall")
    assert int(overall["n_tokens"]) == report["n_tokens"]
    assert overall["go_no_go"] == report["go_no_go"]
    assert sum(int(row["n_tokens"]) for row in rows if row["scope"] == "task") == report["n_tokens"]
    assert (
        sum(int(row["n_tokens"]) for row in rows if row["scope"] == "position_bucket")
        == report["n_tokens"]
    )


def test_cli_help_lists_the_required_flags():
    script = Path(__file__).resolve().parents[2] / "scripts" / "exp11_s2t_diagnose.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    for flag in (
        "--model",
        "--shadow-ckpt",
        "--shadow-model-path",
        "--prompts",
        "--max-new-tokens",
        "--k",
        "--seed",
        "--out",
        "--limit",
        "--device",
        "--dtype",
        "--use-cache",
    ):
        assert flag in completed.stdout
