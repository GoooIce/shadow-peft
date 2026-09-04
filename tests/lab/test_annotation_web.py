from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from shadow_lab.annotation import AnnotationStore
from shadow_lab.annotation.web import make_handler

PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00payload"
)


def _id() -> str:
    return str(uuid.uuid4())


def _server(tmp_path: Path):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    assets = []
    for index, arm in enumerate(("secret-bf16", "secret-nvfp4")):
        path = asset_root / f"secret-{index}.png"
        path.write_bytes(PNG_1X1 + bytes([index]))
        data = path.read_bytes()
        assets.append(
            {
                "asset_id": _id(),
                "sha256": hashlib.sha256(data).hexdigest(),
                "mime_type": "image/png",
                "width": 1,
                "height": 1,
                "byte_length": len(data),
                "storage_key": path.name,
                "arm_id": arm,
            }
        )
    experiment_id = _id()
    manifest = {
        "schema_version": "shadow-ab/0.1",
        "experiment": {
            "experiment_id": experiment_id,
            "protocol_sha256": "",
            "dataset_id": "demo",
            "split": "human_dev",
            "guideline_version": "1.0.0",
            "target_votes": 1,
        },
        "assets": assets,
        "comparisons": [
            {
                "comparison_id": _id(),
                "prompt_id": _id(),
                "prompt_text": "A test prompt",
                "seed": 1,
                "width": 1,
                "height": 1,
                "generation_protocol_sha256": "a" * 64,
                "candidate_asset_ids_unordered": [item["asset_id"] for item in assets],
            }
        ],
    }
    protocol = json.loads(json.dumps(manifest))
    protocol["experiment"].pop("protocol_sha256")
    manifest["experiment"]["protocol_sha256"] = hashlib.sha256(
        json.dumps(
            protocol,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    store = AnnotationStore(tmp_path / "annotation.sqlite3", asset_root)
    now = datetime(2026, 9, 4, tzinfo=timezone.utc)
    owner_token = store.create_user(_id(), "Owner", "experiment_owner", now=now)
    token = store.create_user(_id(), "Rater", "annotator", now=now)
    reviewer_token = store.create_user(_id(), "Reviewer", "reviewer", now=now)
    store.import_manifest(manifest, now=now)
    store.transition_experiment(store.authenticate(owner_token), experiment_id, "frozen", now=now)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, experiment_id))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    return {
        "server": server,
        "thread": thread,
        "token": token,
        "reviewer_token": reviewer_token,
        "owner_token": owner_token,
        "store": store,
        "experiment_id": experiment_id,
        "now": now,
    }


def _request(server, path: str, token: str | None = None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return urllib.request.urlopen(
        urllib.request.Request(f"http://127.0.0.1:{server.server_port}{path}", headers=headers)
    )


def test_http_adapter_serves_ui_and_only_blind_assignment_dto(tmp_path: Path) -> None:
    context = _server(tmp_path)
    server = context["server"]
    thread = context["thread"]
    token = context["token"]
    try:
        html = _request(server, "/").read().decode()
        assert "图像质量盲测" in html
        assert 'script src="/app.js"' in html
        with pytest.raises(urllib.error.HTTPError) as unauthorized:
            _request(server, "/api/next")
        assert unauthorized.value.code == 401

        me = json.load(_request(server, "/api/me", token))
        assignment = json.load(_request(server, "/api/next", token))["assignment"]

        assert me["role"] == "annotator"
        encoded = json.dumps(assignment)
        assert "secret-bf16" not in encoded
        assert "secret-nvfp4" not in encoded
        assert "secret-0.png" not in encoded
        assert assignment["left_media_url"].endswith("/media/left")

        answer_request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/api/answers",
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            data=json.dumps(
                {
                    "assignment_id": assignment["assignment_id"],
                    "lease_token": assignment["lease_token"],
                    "idempotency_key": _id(),
                    "session_id": _id(),
                    "decision": "tie",
                    "reason_codes": ["visual_quality"],
                    "reason_target": "both",
                }
            ).encode(),
        )
        receipt = json.load(urllib.request.urlopen(answer_request))
        progress = json.load(_request(server, "/api/progress", token))
        assert receipt["event_type"] == "answer"
        assert receipt["decision"] == "tie"
        assert progress["answered"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_http_adapter_dispatches_reviewer_queue_and_submission(tmp_path: Path) -> None:
    context = _server(tmp_path)
    server = context["server"]
    try:
        assignment = json.load(_request(server, "/api/next", context["token"]))["assignment"]
        answer_request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/api/answers",
            method="POST",
            headers={
                "Authorization": f"Bearer {context['token']}",
                "Content-Type": "application/json",
            },
            data=json.dumps(
                {
                    "assignment_id": assignment["assignment_id"],
                    "lease_token": assignment["lease_token"],
                    "idempotency_key": _id(),
                    "session_id": _id(),
                    "decision": "left",
                    "flagged": True,
                }
            ).encode(),
        )
        urllib.request.urlopen(answer_request).read()
        context["store"].transition_experiment(
            context["store"].authenticate(context["owner_token"]),
            context["experiment_id"],
            "answers_frozen",
            now=context["now"],
        )

        review = json.load(_request(server, "/api/next", context["reviewer_token"]))["assignment"]
        assert review["queue_reasons"] == ["flagged"]
        assert "arm" not in json.dumps(review)
        review_request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/api/reviews",
            method="POST",
            headers={
                "Authorization": f"Bearer {context['reviewer_token']}",
                "Content-Type": "application/json",
            },
            data=json.dumps(
                {
                    "review_assignment_id": review["review_assignment_id"],
                    "lease_token": review["lease_token"],
                    "idempotency_key": _id(),
                    "session_id": _id(),
                    "decision": "tie",
                }
            ).encode(),
        )
        receipt = json.load(urllib.request.urlopen(review_request))
        progress = json.load(_request(server, "/api/progress", context["reviewer_token"]))
        assert receipt["event_type"] == "adjudication"
        assert progress["adjudicated"] == 1
        assert progress["remaining"] == 0
    finally:
        server.shutdown()
        server.server_close()
        context["thread"].join()


def test_annotation_import_does_not_load_torch_or_model_runtime() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import shadow_lab.annotation; "
            "assert 'torch' not in sys.modules; assert 'mage_flow' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
