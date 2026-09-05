import json

import pytest

from shadow_lab.mage_pilot_ballot import PilotBallots


def test_blind_ballot_persists_and_locks_before_revealing(tmp_path):
    store = PilotBallots(tmp_path, dict(validation=[dict(prompt=f"prompt {i}") for i in range(8)]))
    session = store.session(None)
    public = store.public(session)
    encoded = json.dumps(public)
    assert "auxiliary" not in encoded and "image" not in encoded and "index" not in encoded
    assert "token" not in encoded
    with pytest.raises(ValueError, match="提交"):
        store.reveal(session)
    with pytest.raises(ValueError, match="完成全部"):
        store.submit(session)
    for card in session["cards"]:
        store.vote(session, card["id"], "left")
        assert store.image_path(session, card["assets"]["left"]["id"]).name.endswith(".png")
    other = store.session(None)
    with pytest.raises(ValueError, match="未知图片"):
        store.image_path(other, session["cards"][0]["assets"]["left"]["id"])
    restored = store.session(session["token"])
    assert restored == session
    store.submit(restored)
    with pytest.raises(ValueError, match="锁定"):
        store.vote(restored, restored["cards"][0]["id"], "right")
    revealed = store.reveal(restored)
    assert revealed["counts"]["image"] + revealed["counts"]["auxiliary"] == 8
    assert all(r["left"] != r["right"] for r in revealed["mapping"])
    assert store.session(session["token"])["revealed_at"]
