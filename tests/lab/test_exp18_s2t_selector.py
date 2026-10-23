"""exp18 selector-scale script — data-source tests.

The corpus loader is exercised with a tiny local parquet set, so no real
exp7-corpus shard or Qwen model is loaded. Anchors:

- train and held-out prompts are disjoint documents selected by subset;
- documents are truncated to ``--prompt-tokens`` and blank/short rows drop out;
- a corpus run requires an explicit held-out slice, which is what exp17 later
  evaluates.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
from shadow_lab.real_eval import EvalExample  # noqa: E402


def _load_script() -> Any:
    path = Path(__file__).resolve().parents[2] / "scripts" / "exp18_s2t_selector.py"
    spec = importlib.util.spec_from_file_location("exp18_s2t_selector", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _WordTokenizer:
    """Split on whitespace; enough to test truncation and decoding."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, Any]:
        del add_special_tokens
        return {"input_ids": text.split()}

    def decode(self, ids: list[str], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(ids)


def _write_corpus(root: Path, rows: list[tuple[str, str, str]]) -> None:
    shard = root / "shard-00001.parquet"
    table = pa.table(
        {
            "subset": [row[0] for row in rows],
            "uid": [row[1] for row in rows],
            "text": [row[2] for row in rows],
        }
    )
    pq.write_table(table, shard)
    manifest = {
        "dataset": "toy",
        "shards": [
            {"subset": "en_qa", "path": str(shard), "docs": len(rows)},
        ],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_load_corpus_examples_truncates_and_keeps_train_eval_disjoint(
    tmp_path: Path,
) -> None:
    script = _load_script()
    rows = [
        ("en_qa", f"u{index}", " ".join(f"w{index}_{pos}" for pos in range(12)))
        for index in range(6)
    ]
    rows.append(("zh_qa", "zh1", "中文 文档 内容 测试"))
    _write_corpus(tmp_path, rows)

    train, held_out = script._load_corpus_examples(
        tmp_path,
        ["en_qa"],
        _WordTokenizer(),
        train_limit=3,
        eval_limit=2,
        prompt_tokens=4,
        seed=0,
    )

    assert len(train) == 3
    assert len(held_out) == 2
    assert all(isinstance(example, EvalExample) for example in train + held_out)
    assert {example.id for example in train}.isdisjoint(
        {example.id for example in held_out}
    )
    assert all(example.task == "corpus_en_qa" for example in train + held_out)
    assert all(len(example.prompt.split()) == 4 for example in train + held_out)

    # The zh_qa row was filtered out by --corpus-subsets.
    assert all("zh1" not in example.id for example in train + held_out)


def test_load_corpus_examples_rejects_a_slice_that_does_not_fit(
    tmp_path: Path,
) -> None:
    script = _load_script()
    _write_corpus(
        tmp_path,
        [("en_qa", f"u{index}", "alpha beta gamma") for index in range(2)],
    )

    with pytest.raises(SystemExit, match="need 5"):
        script._load_corpus_examples(
            tmp_path,
            ["en_qa"],
            _WordTokenizer(),
            train_limit=3,
            eval_limit=2,
            prompt_tokens=4,
            seed=0,
        )


def test_smoke_shrinks_the_corpus_eval_slice() -> None:
    script = _load_script()
    args = script.build_parser().parse_args(
        [
            "--model", "base",
            "--shadow-model-dir", "shadow",
            "--source", "exp7-corpus",
            "--eval-limit", "500",
            "--train-limit", "2000",
            "--smoke",
        ]
    )
    script._resolve_smoke(args)
    assert args.train_limit == 4
    assert args.eval_limit == 2
    assert args.epochs == 2
