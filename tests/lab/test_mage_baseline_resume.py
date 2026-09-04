from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "run_mage_flow_bf16_baseline.py"
SPEC = importlib.util.spec_from_file_location("run_mage_flow_bf16_baseline", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_resume_accepts_only_a_hash_verified_frozen_prefix(tmp_path: Path) -> None:
    prompt_rows = [{"prompt": "a cat", "seed": 11, "sample_id": "sample"}]
    image_path = tmp_path / "gen_0000_seed_11.png"
    image_path.write_bytes(b"png")
    record = {
        "index": 0,
        "prompt": "a cat",
        "prompt_metadata": {"seed": 11, "sample_id": "sample"},
        "seed": 11,
        "image": image_path.name,
        "sha256": RUNNER.sha256_file(image_path),
        "deterministic": True,
    }
    records_path = tmp_path / "samples.jsonl"
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    args = SimpleNamespace(offset=0, seed_field="seed", seed=42, prompt_field="prompt")

    assert RUNNER.load_verified_resume_prefix(
        records_path, tmp_path, prompt_rows, args
    ) == [record]

    record["seed"] = 12
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen input"):
        RUNNER.load_verified_resume_prefix(records_path, tmp_path, prompt_rows, args)
