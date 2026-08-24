"""Direction-① runbook tests: the instance scripts' logic runs locally.

`smoke_12b --tiny` is the exact code path the 5090 runs against the real 12B
(no download); the training/eval CLIs are import- and argparse-checked.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "experiment"))
sys.path.insert(0, str(_REPO / "experiment" / "dir1"))


def test_budget_ledger_and_stage_timer(tmp_path: Path):
    from dir1.ops import BudgetLedger, StageTimer

    ledger = BudgetLedger(tmp_path / "budget.jsonl", rate_per_hour=2.88)
    with StageTimer(ledger, "unit"):
        pass
    ledger.add("manual", 1.0)
    summary = ledger.summary()
    assert summary["total_cny"] == pytest.approx(2.88, abs=0.01)
    assert not summary["over_budget"]
    entries = [json.loads(l) for l in (tmp_path / "budget.jsonl").read_text().splitlines()]
    assert [e["stage"] for e in entries] == ["unit", "manual"]


def test_smoke_tiny_end_to_end():
    from smoke_12b import run_smoke

    assert run_smoke(model_id=None, tiny=True, device=torch.device("cpu"))


@pytest.mark.parametrize(
    "script", ["train_tier.py", "eval_grid.py", "smoke_12b.py"], ids=["train", "eval", "smoke"]
)
def test_runbook_cli_parses(script):
    proc = subprocess.run(
        [sys.executable, str(_REPO / "experiment" / "dir1" / script), "--help"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_gen_teacher_data_cli_parses():
    proc = subprocess.run(
        [sys.executable, str(_REPO / "experiment" / "gen_teacher_data_torch.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "--vocab" in proc.stdout and "--eos" in proc.stdout
