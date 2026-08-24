"""Instance-side ops helpers for the direction-① runbooks.

Discipline (spec 铁律): the AutoDL token arrives via the ``AUTODL_TOKEN`` env
var set inside the instance — credentials are never written to any produced
file, and every stage logs its wall time for budget accounting
(~80 GPU-hour ≈ ¥230 ceiling; overspend => stop and report).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

WECHAT_URL = "https://www.autodl.com/api/v1/wechat/message/send"


def wechat_notify(title: str, content: str) -> bool:
    """Send a WeChat notification via AutoDL (50/day, 5/min limits)."""
    token = os.environ.get("AUTODL_TOKEN")
    if not token:
        print("wechat_notify: AUTODL_TOKEN not set, skipping", flush=True)
        return False
    try:
        import urllib.request

        body = json.dumps({"title": title, "name": "dir1", "content": content}).encode()
        req = urllib.request.Request(
            WECHAT_URL,
            data=body,
            headers={"Authorization": token, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            ok = json.loads(resp.read()).get("code") == "Success"
        print(f"wechat_notify: sent={ok}", flush=True)
        return ok
    except Exception as exc:  # never let a notification failure kill a run
        print(f"wechat_notify failed: {exc}", flush=True)
        return False


def power_off_now() -> None:
    """Shut the instance down (billing stops at the second of shutdown)."""
    print("powering off instance NOW", flush=True)
    os.system("/usr/bin/shutdown")


class BudgetLedger:
    """Append-only stage ledger; ¥ = hours × rate (snapshot-verified ¥2.88/h)."""

    def __init__(self, path: str | Path, rate_per_hour: float = 2.88) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rate = rate_per_hour
        self.total_hours = 0.0

    def add(self, stage: str, hours: float, note: str = "") -> dict:
        entry = {
            "stage": stage,
            "hours": round(hours, 4),
            "cny": round(hours * self.rate, 2),
            "note": note,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self.total_hours += hours
        return entry

    def summary(self) -> dict:
        return {
            "total_hours": round(self.total_hours, 3),
            "total_cny": round(self.total_hours * self.rate, 2),
            "budget_hours": 80,
            "over_budget": self.total_hours > 80,
        }


class StageTimer:
    """Context manager pairing a wall-clock span with a BudgetLedger."""

    def __init__(self, ledger: BudgetLedger, stage: str, note: str = "") -> None:
        self.ledger = ledger
        self.stage = stage
        self.note = note
        self.t0 = 0.0

    def __enter__(self) -> "StageTimer":
        self.t0 = time.time()
        print(f"=== stage start: {self.stage} ===", flush=True)
        return self

    def __exit__(self, *exc) -> None:
        hours = (time.time() - self.t0) / 3600.0
        entry = self.ledger.add(self.stage, hours, self.note)
        print(
            f"=== stage done: {self.stage} "
            f"{hours*3600:.0f}s ≈ ¥{entry['cny']} (cum ¥{self.ledger.total_hours * self.ledger.rate:.2f}) ===",
            flush=True,
        )
