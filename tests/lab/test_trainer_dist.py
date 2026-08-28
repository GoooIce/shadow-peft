"""DDP seams of shadow_lab.trainer via a real torchrun launch (gloo, CPU).

Spawns `torchrun --standalone --nproc-per-node=2` on a tiny worker script, so
RANK/WORLD_SIZE/LOCAL_RANK arrive exactly as in production launches. End-to-end
DDP equivalence is verified on the GB10 pair (SHADOW_DIST_SAME_DATA smoke, see
.scratch/experiment-redesign/specs/exp1-ppl-floor.md).

Hang note: do NOT run the workers inside pytest via torch.multiprocessing —
both spawn (re-imports pytest main) and fork+queue were observed to hang; a
subprocess torchrun is robust everywhere.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

WORKER = '''
import json, random, sys
sys.path.insert(0, {src!r})
import torch
from shadow_lab.trainer import _maybe_init_distributed, _sync_grads

ddp = _maybe_init_distributed()
assert ddp is not None, "torchrun env missing"
rank, world, local = ddp
p = torch.nn.Parameter(torch.arange(4, dtype=torch.float32) * (rank + 1))
p.grad = torch.full((4,), float(rank + 1))
none_p = torch.nn.Parameter(torch.zeros(2))
none_p.grad = None
_sync_grads([p, none_p], world)
seeded = [random.Random(7 + rank).random() for _ in range(2)]
with open({out!r} + str(rank) + ".json", "w") as f:
    json.dump({{"rank": rank, "world": world, "local": local,
               "grad": p.grad.tolist(), "seeded": seeded}}, f)
'''


def test_no_torchrun_env_returns_none(monkeypatch):
    sys.path.insert(0, str(REPO / "src"))
    from shadow_lab.trainer import _maybe_init_distributed
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert _maybe_init_distributed() is None


@pytest.mark.skipif(sys.platform == "darwin", reason="local sandbox blocks listen sockets")
def test_torchrun_env_parsing_and_grad_averaging(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(WORKER.format(src=str(REPO / "src"), out=str(tmp_path / "out")))
    torchrun = REPO / ".venv" / "bin" / "torchrun"
    if not torchrun.exists():
        pytest.skip("no project venv torchrun")
    import subprocess

    env = {**os.environ, "DIST_BACKEND": "gloo", "GLOO_SOCKET_IFNAME": "lo",
           "PYTHONUNBUFFERED": "1"}
    proc = subprocess.run(
        [str(torchrun), "--standalone", "--nnodes=1", "--nproc-per-node=2", str(worker)],
        capture_output=True, text=True, timeout=120, env=env, cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    results = [json.loads((tmp_path / f"out{r}.json").read_text()) for r in (0, 1)]
    by_rank = {r["rank"]: r for r in results}
    assert set(by_rank) == {0, 1}
    for rank, r in by_rank.items():
        assert r["world"] == 2
        assert r["local"] == rank  # single node: local rank == global rank
        assert r["grad"] == [1.5, 1.5, 1.5, 1.5]
    assert by_rank[0]["seeded"] != by_rank[1]["seeded"]
