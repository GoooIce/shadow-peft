"""Private, persistent paired ballots for a protocol-development pilot."""

from __future__ import annotations

import json
import secrets
import threading
import time
from pathlib import Path

from .mage_qad_campaign import atomic_json


class PilotBallots:
    def __init__(self, root: Path, data: dict):
        self.root, self.data = root, data
        self.private = root / "private"
        self.private.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = threading.RLock()

    def session(self, token: str | None):
        with self.lock:
            if token and len(token) == 48 and all(c in "0123456789abcdef" for c in token):
                path = self.private / f"{token}.json"
                if path.is_file():
                    return json.loads(path.read_text())
            token = secrets.token_hex(24)
            order = list(range(8))
            secrets.SystemRandom().shuffle(order)
            session: dict = dict(
                token=token,
                created_at=time.time(),
                votes={},
                submitted_at=None,
                revealed_at=None,
                cards=[],
            )
            for index in order:
                arms = ["image", "auxiliary"]
                secrets.SystemRandom().shuffle(arms)
                session["cards"].append(
                    dict(
                        id=secrets.token_hex(12),
                        index=index,
                        assets={
                            side: dict(id=secrets.token_hex(16), arm=arm)
                            for side, arm in zip(("left", "right"), arms, strict=True)
                        },
                    )
                )
            self.save(session)
            return session

    def save(self, session):
        atomic_json(self.private / f"{session['token']}.json", session)

    def public(self, session):
        return dict(
            submitted=bool(session["submitted_at"]),
            votes=session["votes"],
            cards=[
                dict(
                    id=card["id"],
                    prompt=self.data["validation"][card["index"]]["prompt"],
                    **{side: "/asset/" + asset["id"] for side, asset in card["assets"].items()},
                )
                for card in session["cards"]
            ],
        )

    def vote(self, session, card_id, choice):
        if session["submitted_at"]:
            raise ValueError("投票已经锁定，不能修改")
        if card_id not in {c["id"] for c in session["cards"]}:
            raise ValueError("未知题目")
        if choice not in {"left", "right", "tie", "both_bad", "unsure"}:
            raise ValueError("未知选项")
        session["votes"][card_id] = choice
        self.save(session)

    def submit(self, session):
        if len(session["votes"]) != 8:
            raise ValueError("请完成全部 8 题；无法判断可以选择跳过")
        if not session["submitted_at"]:
            session["submitted_at"] = time.time()
            self.save(session)

    def reveal(self, session):
        if not session["submitted_at"]:
            raise ValueError("提交并锁定投票后才能揭盲")
        if not session["revealed_at"]:
            session["revealed_at"] = time.time()
            self.save(session)
        counts = dict(image=0, auxiliary=0, tie=0, both_bad=0, unsure=0)
        for card in session["cards"]:
            choice = session["votes"][card["id"]]
            counts[card["assets"][choice]["arm"] if choice in {"left", "right"} else choice] += 1
        return dict(
            counts=counts,
            mapping=[
                dict(id=c["id"], **{side: asset["arm"] for side, asset in c["assets"].items()})
                for c in session["cards"]
            ],
        )

    def image_path(self, session, asset_id):
        for card in session["cards"]:
            for asset in card["assets"].values():
                if asset["id"] == asset_id:
                    return (
                        self.root / asset["arm"] / "images/validation" / f"{card['index']:02d}.png"
                    )
        raise ValueError("未知图片")
