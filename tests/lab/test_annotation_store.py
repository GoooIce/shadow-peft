from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from shadow_lab.annotation import AnnotationStore, AnswerCommand, ReviewCommand
from shadow_lab.annotation.errors import ConflictError, ValidationError
from shadow_lab.annotation.exports import to_csv, to_jsonl

PNG_1X1 = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)


def _id() -> str:
    return str(uuid.uuid4())


def _manifest(asset_root: Path, *, target_votes: int = 1) -> dict:
    first = asset_root / "opaque-a.png"
    second = asset_root / "opaque-b.png"
    first.write_bytes(PNG_1X1 + b"a")
    second.write_bytes(PNG_1X1 + b"b")
    assets = []
    for path, arm in ((first, "bf16"), (second, "nvfp4-es")):
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
    manifest = {
        "schema_version": "shadow-ab/0.1",
        "experiment": {
            "experiment_id": _id(),
            "protocol_sha256": "",
            "dataset_id": "human-dev-v1",
            "split": "human_dev",
            "guideline_version": "1.0.0",
            "target_votes": target_votes,
        },
        "assets": assets,
        "comparisons": [
            {
                "comparison_id": _id(),
                "prompt_id": _id(),
                "prompt_text": "A red cube on a blue table",
                "seed": 42,
                "width": 1,
                "height": 1,
                "generation_protocol_sha256": "1" * 64,
                "candidate_asset_ids_unordered": [
                    assets[1]["asset_id"],
                    assets[0]["asset_id"],
                ],
                "qc_kind": "none",
                "has_text_target": False,
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
    return manifest


def _ready_store(tmp_path: Path, *, target_votes: int = 1):
    assets = tmp_path / "assets"
    assets.mkdir()
    store = AnnotationStore(tmp_path / "annotation.sqlite3", assets, lease_seconds=60)
    owner_id = _id()
    annotator_id = _id()
    owner_token = store.create_user(owner_id, "Owner", "experiment_owner", now=NOW)
    annotator_token = store.create_user(annotator_id, "Rater", "annotator", now=NOW)
    manifest = _manifest(assets, target_votes=target_votes)
    experiment_id = store.import_manifest(manifest, now=NOW)
    owner = store.authenticate(owner_token)
    annotator = store.authenticate(annotator_token)
    store.transition_experiment(owner, experiment_id, "frozen", now=NOW)
    return store, experiment_id, annotator, manifest


def test_blind_claim_media_idempotent_answer_and_append_only_undo(tmp_path: Path) -> None:
    store, experiment_id, annotator, manifest = _ready_store(tmp_path)

    assignment = store.claim_next(annotator, experiment_id, now=NOW)

    assert assignment is not None
    exposed = json.dumps(asdict(assignment))
    assert "bf16" not in exposed
    assert "nvfp4" not in exposed
    assert "opaque-a.png" not in exposed
    assert assignment.left_media_url.endswith("/media/left")
    left_bytes, mime_type = store.read_media(
        annotator,
        assignment.assignment_id,
        "left",
        assignment.lease_token,
    )
    assert left_bytes in {
        (tmp_path / "assets" / "opaque-a.png").read_bytes(),
        (tmp_path / "assets" / "opaque-b.png").read_bytes(),
    }
    assert mime_type == "image/png"

    session_id = _id()
    command = AnswerCommand(
        assignment_id=assignment.assignment_id,
        lease_token=assignment.lease_token,
        idempotency_key=_id(),
        decision="left",
        reason_codes=("visual_quality",),
        reason_target="left",
        client_version="test",
    )
    first = store.submit_answer(
        annotator,
        command,
        session_id=session_id,
        now=NOW + timedelta(seconds=1),
    )
    replay = store.submit_answer(
        annotator,
        command,
        session_id=session_id,
        now=NOW + timedelta(seconds=2),
    )
    assert first.event_id == replay.event_id
    assert replay.idempotent_replay is True
    with pytest.raises(ConflictError, match="revealed"):
        store.export_experiment(
            store.authenticate(
                store.create_user(_id(), "Second Owner", "experiment_owner", now=NOW)
            ),
            experiment_id,
            revealed=True,
        )
    with pytest.raises(ConflictError, match="different payload"):
        store.submit_answer(
            annotator,
            AnswerCommand(**{**asdict(command), "decision": "right"}),
            session_id=session_id,
            now=NOW + timedelta(seconds=2),
        )

    withdrawn = store.undo_answer(
        annotator,
        assignment.assignment_id,
        _id(),
        session_id=session_id,
        now=NOW + timedelta(seconds=3),
    )
    assert withdrawn.event_type == "withdraw"
    assert withdrawn.decision is None
    progress = store.progress(annotator, experiment_id)
    assert progress.answered == 0
    assert progress.saved_events == 2
    reclaimed = store.claim_next(
        annotator,
        experiment_id,
        now=NOW + timedelta(seconds=4),
    )
    assert reclaimed is not None
    assert reclaimed.assignment_id == assignment.assignment_id
    assert reclaimed.left_media_url == assignment.left_media_url
    assert manifest["experiment"]["protocol_sha256"]


def test_blind_and_revealed_exports_keep_storage_paths_server_only(tmp_path: Path) -> None:
    store, experiment_id, annotator, _ = _ready_store(tmp_path)
    assignment = store.claim_next(annotator, experiment_id, now=NOW)
    assert assignment is not None
    store.submit_answer(
        annotator,
        AnswerCommand(
            assignment_id=assignment.assignment_id,
            lease_token=assignment.lease_token,
            idempotency_key=_id(),
            decision="right",
        ),
        session_id=_id(),
        now=NOW + timedelta(seconds=1),
    )
    owner_token = store.create_user(_id(), "Export Owner", "experiment_owner", now=NOW)
    owner = store.authenticate(owner_token)

    blind = store.export_experiment(owner, experiment_id)

    encoded = json.dumps(blind)
    assert blind["export_mode"] == "blind"
    assert "bf16" not in encoded
    assert "nvfp4" not in encoded
    assert "opaque-a.png" not in encoded
    assert blind["answers"][0]["canonical_decision"] in {
        "candidate_a",
        "candidate_b",
    }
    assert to_jsonl(blind).count("\n") == 2
    assert "left_arm_id" in to_csv(blind).splitlines()[0]

    store.transition_experiment(owner, experiment_id, "answers_frozen", now=NOW)
    store.transition_experiment(owner, experiment_id, "revealed", now=NOW)
    revealed = store.export_experiment(owner, experiment_id, revealed=True)

    assert revealed["export_mode"] == "revealed"
    assert revealed["answers"][0]["reveal"]["arm_decision"] in {
        "bf16",
        "nvfp4-es",
    }
    assert "opaque-a.png" not in json.dumps(revealed)


def test_manifest_rejects_asset_root_escape(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(PNG_1X1)
    (assets / "escape.png").symlink_to(outside)
    store = AnnotationStore(tmp_path / "annotation.sqlite3", assets)
    manifest = _manifest(assets)
    manifest["assets"][0]["storage_key"] = "escape.png"
    protocol = json.loads(json.dumps(manifest))
    protocol["experiment"].pop("protocol_sha256")
    manifest["experiment"]["protocol_sha256"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    with pytest.raises(ValidationError, match="controlled root"):
        store.import_manifest(manifest, now=NOW)


def test_concurrent_claim_does_not_exceed_target_votes(tmp_path: Path) -> None:
    store, experiment_id, first, _ = _ready_store(tmp_path, target_votes=1)
    second_id = _id()
    second_token = store.create_user(second_id, "Second", "annotator", now=NOW)
    second = store.authenticate(second_token)
    barrier = threading.Barrier(3)
    results = []

    def claim(principal) -> None:
        barrier.wait()
        results.append(store.claim_next(principal, experiment_id, now=NOW))

    threads = [
        threading.Thread(target=claim, args=(first,)),
        threading.Thread(target=claim, args=(second,)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sum(result is not None for result in results) == 1


def test_flagged_answer_requires_blind_append_only_review_before_reveal(
    tmp_path: Path,
) -> None:
    store, experiment_id, annotator, _ = _ready_store(tmp_path)
    assignment = store.claim_next(annotator, experiment_id, now=NOW)
    assert assignment is not None
    store.submit_answer(
        annotator,
        AnswerCommand(
            assignment_id=assignment.assignment_id,
            lease_token=assignment.lease_token,
            idempotency_key=_id(),
            decision="left",
            flagged=True,
        ),
        session_id=_id(),
        now=NOW + timedelta(seconds=1),
    )
    owner = store.authenticate(
        store.create_user(_id(), "Review Owner", "experiment_owner", now=NOW)
    )
    reviewer = store.authenticate(store.create_user(_id(), "Reviewer", "reviewer", now=NOW))
    store.transition_experiment(
        owner,
        experiment_id,
        "answers_frozen",
        now=NOW + timedelta(seconds=2),
    )

    with pytest.raises(ConflictError, match="review queue"):
        store.transition_experiment(owner, experiment_id, "revealed", now=NOW)
    review = store.claim_review(reviewer, experiment_id, now=NOW + timedelta(seconds=3))
    assert review is not None
    assert review.queue_reasons == ("flagged",)
    encoded = json.dumps(asdict(review))
    assert "bf16" not in encoded
    assert "nvfp4" not in encoded
    media, mime_type = store.read_review_media(
        reviewer,
        review.review_assignment_id,
        "left",
        review.lease_token,
    )
    assert media.startswith(PNG_1X1)
    assert mime_type == "image/png"

    command = ReviewCommand(
        review_assignment_id=review.review_assignment_id,
        lease_token=review.lease_token,
        idempotency_key=_id(),
        decision="right",
        comment="Independent blind adjudication",
        client_version="test",
    )
    review_session_id = _id()
    receipt = store.submit_review(
        reviewer,
        command,
        session_id=review_session_id,
        now=NOW + timedelta(seconds=4),
    )
    replay = store.submit_review(
        reviewer,
        command,
        session_id=review_session_id,
        now=NOW + timedelta(seconds=5),
    )
    assert replay.event_id == receipt.event_id
    assert replay.idempotent_replay is True
    progress = store.review_progress(reviewer, experiment_id)
    assert (progress.queued, progress.adjudicated, progress.remaining) == (1, 1, 0)

    blind = store.export_experiment(owner, experiment_id)
    assert blind["review_event_count"] == 1
    assert blind["reviews"][0]["canonical_decision"] in {
        "candidate_a",
        "candidate_b",
    }
    assert "bf16" not in json.dumps(blind)
    assert "nvfp4" not in json.dumps(blind)

    store.undo_review(
        reviewer,
        review.review_assignment_id,
        _id(),
        session_id=_id(),
        now=NOW + timedelta(seconds=6),
    )
    with pytest.raises(ConflictError, match="review queue"):
        store.transition_experiment(owner, experiment_id, "revealed", now=NOW)
    replacement = store.authenticate(
        store.create_user(_id(), "Replacement Reviewer", "reviewer", now=NOW)
    )
    reclaimed = store.claim_review(
        replacement,
        experiment_id,
        now=NOW + timedelta(seconds=70),
    )
    assert reclaimed is not None
    assert reclaimed.left_media_url == review.left_media_url
    store.submit_review(
        replacement,
        ReviewCommand(
            review_assignment_id=reclaimed.review_assignment_id,
            lease_token=reclaimed.lease_token,
            idempotency_key=_id(),
            decision="tie",
        ),
        session_id=_id(),
        now=NOW + timedelta(seconds=71),
    )
    store.transition_experiment(
        owner,
        experiment_id,
        "revealed",
        now=NOW + timedelta(seconds=72),
    )
    revealed = store.export_experiment(owner, experiment_id, revealed=True)
    assert revealed["review_event_count"] == 3
    assert revealed["reviews"][-1]["reveal"]["arm_decision"] == "tie"


def test_disagreement_enters_review_queue_without_exposing_votes(tmp_path: Path) -> None:
    store, experiment_id, first, manifest = _ready_store(tmp_path, target_votes=2)
    second = store.authenticate(store.create_user(_id(), "Second", "annotator", now=NOW))
    assignments = [
        store.claim_next(first, experiment_id, now=NOW),
        store.claim_next(second, experiment_id, now=NOW),
    ]
    candidates = sorted(manifest["comparisons"][0]["candidate_asset_ids_unordered"])
    for index, (principal, assignment) in enumerate(zip((first, second), assignments, strict=True)):
        assert assignment is not None
        side_hash = hashlib.sha256(
            (f"{experiment_id}:{assignment.comparison_id}:{principal.user_id}:1").encode()
        ).hexdigest()
        left_candidate = 0 if int(side_hash[-1], 16) % 2 == 0 else 1
        desired_candidate = index
        decision = "left" if left_candidate == desired_candidate else "right"
        assert candidates[desired_candidate]
        store.submit_answer(
            principal,
            AnswerCommand(
                assignment_id=assignment.assignment_id,
                lease_token=assignment.lease_token,
                idempotency_key=_id(),
                decision=decision,
            ),
            session_id=_id(),
            now=NOW + timedelta(seconds=1),
        )
    owner = store.authenticate(
        store.create_user(_id(), "Review Owner", "experiment_owner", now=NOW)
    )
    reviewer = store.authenticate(store.create_user(_id(), "Reviewer", "reviewer", now=NOW))
    store.transition_experiment(owner, experiment_id, "answers_frozen", now=NOW)

    review = store.claim_review(reviewer, experiment_id, now=NOW)

    assert review is not None
    assert review.queue_reasons == ("disagreement",)
    assert not hasattr(review, "answers")
