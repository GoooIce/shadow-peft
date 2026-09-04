"""SQLite implementation behind the local blind-annotation interface."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import struct
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .errors import AuthorizationError, ConflictError, NotFoundError, ValidationError
from .models import (
    DECISIONS,
    REASON_CODES,
    REASON_TARGETS,
    ROLES,
    AnswerCommand,
    AnswerReceipt,
    BlindAssignment,
    BlindReview,
    Principal,
    Progress,
    ReviewCommand,
    ReviewProgress,
    ReviewReceipt,
)

SCHEMA_VERSION = "shadow-ab/0.1"
EXPERIMENT_TRANSITIONS = {
    "draft": "frozen",
    "frozen": "collecting",
    "collecting": "answers_frozen",
    "answers_frozen": "revealed",
    "revealed": "archived",
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('annotator','reviewer','experiment_owner')),
    token_sha256 TEXT NOT NULL UNIQUE,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0,1)),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    protocol_sha256 TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    split TEXT NOT NULL,
    guideline_version TEXT NOT NULL,
    target_votes INTEGER NOT NULL CHECK (target_votes > 0),
    status TEXT NOT NULL CHECK (
        status IN ('draft','frozen','collecting','answers_frozen','revealed','archived')
    ),
    manifest_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    frozen_at TEXT
);
CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    sha256 TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    byte_length INTEGER NOT NULL,
    storage_key TEXT NOT NULL,
    arm_id TEXT NOT NULL,
    UNIQUE (experiment_id, sha256)
);
CREATE TABLE IF NOT EXISTS comparisons (
    comparison_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    prompt_id TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    seed INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    generation_protocol_sha256 TEXT NOT NULL,
    asset_a_id TEXT NOT NULL REFERENCES assets(asset_id),
    asset_b_id TEXT NOT NULL REFERENCES assets(asset_id),
    qc_kind TEXT NOT NULL DEFAULT 'none',
    qc_group_id TEXT,
    has_text_target INTEGER NOT NULL DEFAULT 0 CHECK (has_text_target IN (0,1)),
    UNIQUE (experiment_id, prompt_id, seed, generation_protocol_sha256, asset_a_id, asset_b_id)
);
CREATE TABLE IF NOT EXISTS assignments (
    assignment_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    comparison_id TEXT NOT NULL REFERENCES comparisons(comparison_id),
    annotator_id TEXT NOT NULL REFERENCES users(user_id),
    presentation_version INTEGER NOT NULL,
    left_asset_id TEXT NOT NULL REFERENCES assets(asset_id),
    right_asset_id TEXT NOT NULL REFERENCES assets(asset_id),
    lease_token_sha256 TEXT NOT NULL,
    leased_at TEXT NOT NULL,
    lease_expires_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('leased','answered','withdrawn')),
    UNIQUE (experiment_id, comparison_id, annotator_id)
);
CREATE TABLE IF NOT EXISTS answer_events (
    event_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    assignment_id TEXT NOT NULL REFERENCES assignments(assignment_id),
    session_id TEXT NOT NULL,
    annotator_id TEXT NOT NULL REFERENCES users(user_id),
    event_type TEXT NOT NULL CHECK (event_type IN ('answer','withdraw')),
    decision TEXT CHECK (decision IN ('left','tie','right','skip')),
    reason_codes_json TEXT NOT NULL,
    reason_target TEXT NOT NULL,
    comment TEXT,
    flagged INTEGER NOT NULL CHECK (flagged IN (0,1)),
    guideline_version TEXT NOT NULL,
    client_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    received_at TEXT NOT NULL,
    supersedes_event_id TEXT REFERENCES answer_events(event_id),
    UNIQUE (annotator_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS answer_events_assignment_idx
    ON answer_events(assignment_id, received_at, event_id);
CREATE TABLE IF NOT EXISTS review_assignments (
    review_assignment_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    comparison_id TEXT NOT NULL REFERENCES comparisons(comparison_id),
    reviewer_id TEXT NOT NULL REFERENCES users(user_id),
    presentation_version INTEGER NOT NULL,
    left_asset_id TEXT NOT NULL REFERENCES assets(asset_id),
    right_asset_id TEXT NOT NULL REFERENCES assets(asset_id),
    lease_token_sha256 TEXT NOT NULL,
    leased_at TEXT NOT NULL,
    lease_expires_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('leased','adjudicated','withdrawn')),
    UNIQUE (experiment_id, comparison_id)
);
CREATE TABLE IF NOT EXISTS review_events (
    event_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    review_assignment_id TEXT NOT NULL REFERENCES review_assignments(review_assignment_id),
    session_id TEXT NOT NULL,
    reviewer_id TEXT NOT NULL REFERENCES users(user_id),
    event_type TEXT NOT NULL CHECK (event_type IN ('adjudication','withdraw')),
    decision TEXT CHECK (decision IN ('left','tie','right','skip')),
    comment TEXT,
    guideline_version TEXT NOT NULL,
    client_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    received_at TEXT NOT NULL,
    supersedes_event_id TEXT REFERENCES review_events(event_id),
    UNIQUE (reviewer_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS review_events_assignment_idx
    ON review_events(review_assignment_id, received_at, event_id);
CREATE TABLE IF NOT EXISTS audit_events (
    audit_id TEXT PRIMARY KEY,
    actor_id TEXT,
    action TEXT NOT NULL,
    object_type TEXT NOT NULL,
    object_id TEXT NOT NULL,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _utc(value: datetime | None = None) -> datetime:
    result = value or datetime.now(timezone.utc)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValidationError("timestamps must be timezone-aware")
    return result.astimezone(timezone.utc)


def _timestamp(value: datetime | None = None) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _require_uuid(value: str, field: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (TypeError, ValueError, AttributeError) as error:
        raise ValidationError(f"{field} must be a UUID") from error


def _require_sha256(value: str, field: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValidationError(f"{field} must be a lowercase SHA-256")
    return value


def _probe_image(data: bytes) -> tuple[str, int, int]:
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        return "image/png", width, height
    if data.startswith(b"\xff\xd8"):
        cursor = 2
        sof_markers = {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }
        while cursor + 4 <= len(data):
            if data[cursor] != 0xFF:
                cursor += 1
                continue
            marker = data[cursor + 1]
            cursor += 2
            if marker in {0xD8, 0xD9}:
                continue
            if cursor + 2 > len(data):
                break
            length = int.from_bytes(data[cursor : cursor + 2], "big")
            if length < 2 or cursor + length > len(data):
                break
            if marker in sof_markers and length >= 7:
                height = int.from_bytes(data[cursor + 3 : cursor + 5], "big")
                width = int.from_bytes(data[cursor + 5 : cursor + 7], "big")
                return "image/jpeg", width, height
            cursor += length
    raise ValidationError("asset is not a supported PNG or JPEG image")


class AnnotationStore:
    """Deep module for blind assignment, immutable answers and controlled media."""

    def __init__(
        self,
        database_path: str | Path,
        asset_root: str | Path,
        *,
        lease_seconds: int = 900,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.database_path = Path(database_path).expanduser().resolve()
        self.asset_root = Path(asset_root).expanduser().resolve()
        self.asset_root.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.lease_seconds = lease_seconds
        with self._connect() as connection:
            connection.executescript(SCHEMA_SQL)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.database_path,
            timeout=30,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()

    @staticmethod
    def _audit(
        connection: sqlite3.Connection,
        *,
        actor_id: str | None,
        action: str,
        object_type: str,
        object_id: str,
        details: dict,
        now: datetime,
    ) -> None:
        connection.execute(
            "INSERT INTO audit_events VALUES (?,?,?,?,?,?,?)",
            (
                str(uuid.uuid4()),
                actor_id,
                action,
                object_type,
                object_id,
                json.dumps(details, ensure_ascii=False, sort_keys=True),
                _timestamp(now),
            ),
        )

    def create_user(
        self,
        user_id: str,
        display_name: str,
        role: str,
        *,
        now: datetime | None = None,
    ) -> str:
        user_id = _require_uuid(user_id, "user_id")
        display_name = display_name.strip()
        if not display_name or len(display_name) > 100 or role not in ROLES:
            raise ValidationError("invalid display name or role")
        token = secrets.token_urlsafe(32)
        current = _utc(now)
        try:
            with self._transaction() as connection:
                connection.execute(
                    "INSERT INTO users VALUES (?,?,?,?,?,?)",
                    (
                        user_id,
                        display_name,
                        role,
                        _sha256_bytes(token.encode()),
                        1,
                        _timestamp(current),
                    ),
                )
                self._audit(
                    connection,
                    actor_id=None,
                    action="user.created",
                    object_type="user",
                    object_id=user_id,
                    details={"role": role},
                    now=current,
                )
        except sqlite3.IntegrityError as error:
            raise ConflictError("user already exists") from error
        return token

    def authenticate(self, token: str) -> Principal:
        token_hash = _sha256_bytes(token.encode())
        with self._connect() as connection:
            row = connection.execute(
                "SELECT user_id,display_name,role FROM users WHERE token_sha256=? AND active=1",
                (token_hash,),
            ).fetchone()
        if row is None:
            raise AuthorizationError("invalid annotation token")
        return Principal(row["user_id"], row["display_name"], row["role"])

    def _asset_path(self, storage_key: str) -> Path:
        if not isinstance(storage_key, str) or not storage_key or os.path.isabs(storage_key):
            raise ValidationError("asset storage_key must be relative")
        candidate = (self.asset_root / storage_key).resolve(strict=True)
        if not candidate.is_relative_to(self.asset_root) or not candidate.is_file():
            raise ValidationError("asset storage_key escapes the controlled root")
        return candidate

    def import_manifest(self, manifest: dict, *, now: datetime | None = None) -> str:
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValidationError(f"manifest schema_version must be {SCHEMA_VERSION}")
        experiment = manifest.get("experiment")
        assets = manifest.get("assets")
        comparisons = manifest.get("comparisons")
        if not isinstance(experiment, dict) or not isinstance(assets, list) or not assets:
            raise ValidationError("manifest requires experiment and non-empty assets")
        if not isinstance(comparisons, list) or not comparisons:
            raise ValidationError("manifest requires non-empty comparisons")
        experiment_id = _require_uuid(experiment.get("experiment_id"), "experiment_id")
        protocol_sha256 = _require_sha256(experiment.get("protocol_sha256"), "protocol_sha256")
        protocol_value = json.loads(json.dumps(manifest))
        protocol_value["experiment"].pop("protocol_sha256", None)
        if _sha256_bytes(_canonical(protocol_value)) != protocol_sha256:
            raise ValidationError("manifest protocol_sha256 differs from its content")
        target_votes = experiment.get("target_votes")
        if not isinstance(target_votes, int) or isinstance(target_votes, bool) or target_votes <= 0:
            raise ValidationError("target_votes must be a positive integer")
        for field in ("dataset_id", "split", "guideline_version"):
            if not isinstance(experiment.get(field), str) or not experiment[field].strip():
                raise ValidationError(f"experiment.{field} must be non-empty")

        checked_assets: dict[str, tuple[dict, Path]] = {}
        for asset in assets:
            if not isinstance(asset, dict):
                raise ValidationError("asset records must be objects")
            asset_id = _require_uuid(asset.get("asset_id"), "asset_id")
            if asset_id in checked_assets:
                raise ValidationError("asset_id values must be unique")
            path = self._asset_path(asset.get("storage_key"))
            data = path.read_bytes()
            mime_type, width, height = _probe_image(data)
            expected = {
                "sha256": _sha256_bytes(data),
                "mime_type": mime_type,
                "width": width,
                "height": height,
                "byte_length": len(data),
            }
            for field, value in expected.items():
                if asset.get(field) != value:
                    raise ValidationError(f"asset {asset_id} {field} differs from file")
            if not isinstance(asset.get("arm_id"), str) or not asset["arm_id"].strip():
                raise ValidationError("asset arm_id must be non-empty and server-only")
            checked_assets[asset_id] = (asset, path)

        checked_comparisons = []
        comparison_ids = set()
        for comparison in comparisons:
            if not isinstance(comparison, dict):
                raise ValidationError("comparison records must be objects")
            comparison_id = _require_uuid(comparison.get("comparison_id"), "comparison_id")
            if comparison_id in comparison_ids:
                raise ValidationError("comparison_id values must be unique")
            comparison_ids.add(comparison_id)
            candidate_ids = comparison.get("candidate_asset_ids_unordered")
            if (
                not isinstance(candidate_ids, list)
                or len(candidate_ids) != 2
                or len(set(candidate_ids)) != 2
                or any(asset_id not in checked_assets for asset_id in candidate_ids)
            ):
                raise ValidationError("comparison must reference two distinct manifest assets")
            candidate_ids = sorted(candidate_ids)
            for field in ("prompt_id", "prompt_text"):
                if not isinstance(comparison.get(field), str) or not comparison[field]:
                    raise ValidationError(f"comparison.{field} must be non-empty")
            for field in ("seed", "width", "height"):
                if not isinstance(comparison.get(field), int) or isinstance(
                    comparison[field], bool
                ):
                    raise ValidationError(f"comparison.{field} must be an integer")
            _require_sha256(
                comparison.get("generation_protocol_sha256"),
                "generation_protocol_sha256",
            )
            if any(
                checked_assets[asset_id][0][dimension] != comparison[dimension]
                for asset_id in candidate_ids
                for dimension in ("width", "height")
            ):
                raise ValidationError("comparison and asset dimensions differ")
            qc_kind = comparison.get("qc_kind", "none")
            if qc_kind not in {"none", "repeat", "sentinel"}:
                raise ValidationError("invalid comparison qc_kind")
            checked_comparisons.append((comparison, comparison_id, candidate_ids, qc_kind))

        current = _utc(now)
        try:
            with self._transaction() as connection:
                connection.execute(
                    "INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        experiment_id,
                        protocol_sha256,
                        experiment["dataset_id"],
                        experiment["split"],
                        experiment["guideline_version"],
                        target_votes,
                        "draft",
                        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
                        _timestamp(current),
                        None,
                    ),
                )
                for asset_id, (asset, _) in checked_assets.items():
                    connection.execute(
                        "INSERT INTO assets VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            asset_id,
                            experiment_id,
                            asset["sha256"],
                            asset["mime_type"],
                            asset["width"],
                            asset["height"],
                            asset["byte_length"],
                            asset["storage_key"],
                            asset["arm_id"],
                        ),
                    )
                for comparison, comparison_id, candidate_ids, qc_kind in checked_comparisons:
                    connection.execute(
                        "INSERT INTO comparisons VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            comparison_id,
                            experiment_id,
                            comparison["prompt_id"],
                            comparison["prompt_text"],
                            comparison["seed"],
                            comparison["width"],
                            comparison["height"],
                            comparison["generation_protocol_sha256"],
                            candidate_ids[0],
                            candidate_ids[1],
                            qc_kind,
                            comparison.get("qc_group_id"),
                            int(bool(comparison.get("has_text_target", False))),
                        ),
                    )
                self._audit(
                    connection,
                    actor_id=None,
                    action="experiment.imported",
                    object_type="experiment",
                    object_id=experiment_id,
                    details={
                        "asset_count": len(checked_assets),
                        "comparison_count": len(checked_comparisons),
                        "protocol_sha256": protocol_sha256,
                    },
                    now=current,
                )
        except sqlite3.IntegrityError as error:
            raise ConflictError("manifest conflicts with existing records") from error
        return experiment_id

    @staticmethod
    def _require_owner(principal: Principal) -> None:
        if principal.role != "experiment_owner":
            raise AuthorizationError("experiment owner role required")

    @staticmethod
    def _review_queue(
        connection: sqlite3.Connection,
        experiment_id: str,
        *,
        pending_only: bool = False,
    ) -> list[sqlite3.Row]:
        pending_clause = ""
        if pending_only:
            pending_clause = """
              AND NOT EXISTS (
                SELECT 1
                FROM review_assignments review_assignment
                JOIN review_events review_event
                  ON review_event.review_assignment_id=
                     review_assignment.review_assignment_id
                WHERE review_assignment.comparison_id=c.comparison_id
                  AND review_event.event_type='adjudication'
                  AND NOT EXISTS (
                    SELECT 1 FROM review_events child
                    WHERE child.supersedes_event_id=review_event.event_id
                  )
              )
            """
        return connection.execute(
            f"""
            WITH current_answers AS (
              SELECT assignment.comparison_id,event.flagged,
                     CASE event.decision
                       WHEN 'left' THEN assignment.left_asset_id
                       WHEN 'right' THEN assignment.right_asset_id
                       ELSE event.decision
                     END AS canonical_decision
              FROM answer_events event
              JOIN assignments assignment
                ON assignment.assignment_id=event.assignment_id
              WHERE assignment.experiment_id=?
                AND event.event_type='answer'
                AND NOT EXISTS (
                  SELECT 1 FROM answer_events child
                  WHERE child.supersedes_event_id=event.event_id
                )
            ), queue AS (
              SELECT comparison_id,
                     MAX(flagged) AS has_flagged_answer,
                     COUNT(DISTINCT CASE
                       WHEN canonical_decision != 'skip' THEN canonical_decision
                     END) > 1 AS has_disagreement
              FROM current_answers
              GROUP BY comparison_id
              HAVING MAX(flagged)=1 OR COUNT(DISTINCT CASE
                WHEN canonical_decision != 'skip' THEN canonical_decision
              END) > 1
            )
            SELECT c.*,e.guideline_version,queue.has_flagged_answer,
                   queue.has_disagreement
            FROM queue
            JOIN comparisons c ON c.comparison_id=queue.comparison_id
            JOIN experiments e ON e.experiment_id=c.experiment_id
            WHERE c.experiment_id=? {pending_clause}
            ORDER BY queue.has_flagged_answer DESC,queue.has_disagreement DESC,
                     c.comparison_id
            """,
            (experiment_id, experiment_id),
        ).fetchall()

    def transition_experiment(
        self,
        principal: Principal,
        experiment_id: str,
        target_status: str,
        *,
        now: datetime | None = None,
    ) -> None:
        self._require_owner(principal)
        experiment_id = _require_uuid(experiment_id, "experiment_id")
        current = _utc(now)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT status FROM experiments WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError("experiment not found")
            if EXPERIMENT_TRANSITIONS.get(row["status"]) != target_status:
                raise ConflictError(
                    f"invalid experiment transition: {row['status']} -> {target_status}"
                )
            if target_status == "revealed" and self._review_queue(
                connection,
                experiment_id,
                pending_only=True,
            ):
                raise ConflictError("review queue must be adjudicated before reveal")
            connection.execute(
                "UPDATE experiments SET status=?, frozen_at=COALESCE(frozen_at,?) "
                "WHERE experiment_id=?",
                (target_status, _timestamp(current), experiment_id),
            )
            self._audit(
                connection,
                actor_id=principal.user_id,
                action=f"experiment.{target_status}",
                object_type="experiment",
                object_id=experiment_id,
                details={"from": row["status"], "to": target_status},
                now=current,
            )

    @staticmethod
    def _assignment_from_row(row: sqlite3.Row, lease_token: str) -> BlindAssignment:
        assignment_id = row["assignment_id"]
        return BlindAssignment(
            assignment_id=assignment_id,
            experiment_id=row["experiment_id"],
            comparison_id=row["comparison_id"],
            prompt_text=row["prompt_text"],
            seed=row["seed"],
            width=row["width"],
            height=row["height"],
            left_media_url=f"/api/assignments/{assignment_id}/media/left",
            right_media_url=f"/api/assignments/{assignment_id}/media/right",
            lease_token=lease_token,
            lease_expires_at=row["lease_expires_at"],
            guideline_version=row["guideline_version"],
            has_text_target=bool(row["has_text_target"]),
        )

    def claim_next(
        self,
        principal: Principal,
        experiment_id: str,
        *,
        now: datetime | None = None,
    ) -> BlindAssignment | None:
        if principal.role != "annotator":
            raise AuthorizationError("annotator role required")
        experiment_id = _require_uuid(experiment_id, "experiment_id")
        current = _utc(now)
        expires = current + timedelta(seconds=self.lease_seconds)
        lease_token = secrets.token_urlsafe(32)
        lease_hash = _sha256_bytes(lease_token.encode())
        with self._transaction() as connection:
            experiment = connection.execute(
                "SELECT status,target_votes FROM experiments WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            if experiment is None:
                raise NotFoundError("experiment not found")
            if experiment["status"] not in {"frozen", "collecting"}:
                raise ConflictError("experiment is not accepting annotations")
            if experiment["status"] == "frozen":
                connection.execute(
                    "UPDATE experiments SET status='collecting' WHERE experiment_id=?",
                    (experiment_id,),
                )

            existing = connection.execute(
                """
                SELECT a.*,c.prompt_text,c.seed,c.width,c.height,c.has_text_target,
                       e.guideline_version
                FROM assignments a
                JOIN comparisons c ON c.comparison_id=a.comparison_id
                JOIN experiments e ON e.experiment_id=a.experiment_id
                WHERE a.experiment_id=? AND a.annotator_id=?
                  AND NOT EXISTS (
                    SELECT 1 FROM answer_events head
                    WHERE head.assignment_id=a.assignment_id
                      AND NOT EXISTS (
                        SELECT 1 FROM answer_events child
                        WHERE child.supersedes_event_id=head.event_id
                      )
                      AND head.event_type='answer'
                  )
                  AND (
                    SELECT COUNT(*) FROM assignments other
                    WHERE other.comparison_id=a.comparison_id
                      AND other.assignment_id != a.assignment_id
                      AND (
                        other.lease_expires_at > ? OR EXISTS (
                          SELECT 1 FROM answer_events other_head
                          WHERE other_head.assignment_id=other.assignment_id
                            AND NOT EXISTS (
                              SELECT 1 FROM answer_events other_child
                              WHERE other_child.supersedes_event_id=other_head.event_id
                            )
                            AND other_head.event_type='answer'
                        )
                      )
                  ) < ?
                ORDER BY a.leased_at DESC LIMIT 1
                """,
                (
                    experiment_id,
                    principal.user_id,
                    _timestamp(current),
                    experiment["target_votes"],
                ),
            ).fetchone()
            if existing is not None:
                connection.execute(
                    "UPDATE assignments SET lease_token_sha256=?,leased_at=?,"
                    "lease_expires_at=?,status='leased' WHERE assignment_id=?",
                    (
                        lease_hash,
                        _timestamp(current),
                        _timestamp(expires),
                        existing["assignment_id"],
                    ),
                )
                existing = dict(existing)
                existing["lease_expires_at"] = _timestamp(expires)
                return self._assignment_from_row(existing, lease_token)

            comparison = connection.execute(
                """
                SELECT c.*,e.guideline_version
                FROM comparisons c
                JOIN experiments e ON e.experiment_id=c.experiment_id
                WHERE c.experiment_id=?
                  AND NOT EXISTS (
                    SELECT 1 FROM assignments mine
                    WHERE mine.comparison_id=c.comparison_id AND mine.annotator_id=?
                  )
                  AND (
                    SELECT COUNT(*) FROM assignments a
                    WHERE a.comparison_id=c.comparison_id
                      AND (
                        a.lease_expires_at > ? OR EXISTS (
                          SELECT 1 FROM answer_events head
                          WHERE head.assignment_id=a.assignment_id
                            AND NOT EXISTS (
                              SELECT 1 FROM answer_events child
                              WHERE child.supersedes_event_id=head.event_id
                            )
                            AND head.event_type='answer'
                        )
                      )
                  ) < ?
                ORDER BY CASE c.qc_kind WHEN 'sentinel' THEN 0 WHEN 'repeat' THEN 1 ELSE 2 END,
                         c.comparison_id
                LIMIT 1
                """,
                (
                    experiment_id,
                    principal.user_id,
                    _timestamp(current),
                    experiment["target_votes"],
                ),
            ).fetchone()
            if comparison is None:
                return None
            presentation_version = 1
            side_hash = _sha256_bytes(
                (
                    f"{experiment_id}:{comparison['comparison_id']}:"
                    f"{principal.user_id}:{presentation_version}"
                ).encode()
            )
            assets = (comparison["asset_a_id"], comparison["asset_b_id"])
            left, right = assets if int(side_hash[-1], 16) % 2 == 0 else assets[::-1]
            assignment_id = str(
                uuid.uuid5(
                    uuid.UUID(experiment_id),
                    f"{comparison['comparison_id']}:{principal.user_id}:{presentation_version}",
                )
            )
            connection.execute(
                "INSERT INTO assignments VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    assignment_id,
                    experiment_id,
                    comparison["comparison_id"],
                    principal.user_id,
                    presentation_version,
                    left,
                    right,
                    lease_hash,
                    _timestamp(current),
                    _timestamp(expires),
                    "leased",
                ),
            )
            self._audit(
                connection,
                actor_id=principal.user_id,
                action="assignment.claimed",
                object_type="assignment",
                object_id=assignment_id,
                details={"presentation_version": presentation_version},
                now=current,
            )
            row = dict(comparison)
            row.update(
                {
                    "assignment_id": assignment_id,
                    "experiment_id": experiment_id,
                    "lease_expires_at": _timestamp(expires),
                    "left_asset_id": left,
                    "right_asset_id": right,
                }
            )
            return self._assignment_from_row(row, lease_token)

    @staticmethod
    def _validate_answer(command: AnswerCommand) -> tuple[str, str | None]:
        if command.decision not in DECISIONS:
            raise ValidationError("invalid answer decision")
        if len(set(command.reason_codes)) != len(command.reason_codes) or any(
            reason not in REASON_CODES for reason in command.reason_codes
        ):
            raise ValidationError("invalid or repeated reason code")
        if command.reason_target not in REASON_TARGETS:
            raise ValidationError("invalid reason target")
        comment = command.comment.strip() if command.comment is not None else None
        if comment == "":
            comment = None
        if comment is not None and len(comment) > 500:
            raise ValidationError("answer comment exceeds 500 characters")
        if "other" in command.reason_codes and comment is None:
            raise ValidationError("the other reason requires a comment")
        if not isinstance(command.flagged, bool):
            raise ValidationError("flagged must be a boolean")
        if not isinstance(command.client_version, str) or len(command.client_version) > 100:
            raise ValidationError("client version is too long")
        return command.decision, comment

    @staticmethod
    def _receipt(row: sqlite3.Row, *, replay: bool) -> AnswerReceipt:
        return AnswerReceipt(
            event_id=row["event_id"],
            assignment_id=row["assignment_id"],
            decision=row["decision"],
            event_type=row["event_type"],
            saved_at=row["received_at"],
            idempotent_replay=replay,
        )

    def submit_answer(
        self,
        principal: Principal,
        command: AnswerCommand,
        *,
        session_id: str,
        now: datetime | None = None,
    ) -> AnswerReceipt:
        if principal.role != "annotator":
            raise AuthorizationError("annotator role required")
        decision, comment = self._validate_answer(command)
        assignment_id = _require_uuid(command.assignment_id, "assignment_id")
        session_id = _require_uuid(session_id, "session_id")
        _require_uuid(command.idempotency_key, "idempotency_key")
        current = _utc(now)
        payload = {
            **asdict(command),
            "reason_codes": sorted(command.reason_codes),
            "comment": comment,
            "session_id": session_id,
        }
        payload_hash = _sha256_bytes(_canonical(payload))
        with self._transaction() as connection:
            replay = connection.execute(
                "SELECT * FROM answer_events WHERE annotator_id=? AND idempotency_key=?",
                (principal.user_id, command.idempotency_key),
            ).fetchone()
            if replay is not None:
                if replay["payload_sha256"] != payload_hash:
                    raise ConflictError("idempotency key was reused with a different payload")
                return self._receipt(replay, replay=True)
            assignment = connection.execute(
                """
                SELECT a.*,e.status AS experiment_status,e.guideline_version,
                       c.has_text_target
                FROM assignments a
                JOIN experiments e ON e.experiment_id=a.experiment_id
                JOIN comparisons c ON c.comparison_id=a.comparison_id
                WHERE a.assignment_id=? AND a.annotator_id=?
                """,
                (assignment_id, principal.user_id),
            ).fetchone()
            if assignment is None:
                raise NotFoundError("assignment not found")
            if assignment["experiment_status"] != "collecting":
                raise ConflictError("experiment is not accepting answers")
            if assignment["status"] != "leased":
                raise ConflictError("assignment does not have an active lease")
            if assignment["lease_expires_at"] <= _timestamp(current):
                raise ConflictError("assignment lease expired")
            if not secrets.compare_digest(
                assignment["lease_token_sha256"],
                _sha256_bytes(command.lease_token.encode()),
            ):
                raise AuthorizationError("invalid assignment lease token")
            if assignment["guideline_version"] == "":
                raise ConflictError("assignment guideline is invalid")
            reason_target = "none" if decision == "skip" else command.reason_target
            head = connection.execute(
                """
                SELECT event_id,event_type FROM answer_events event
                WHERE assignment_id=? AND NOT EXISTS (
                    SELECT 1 FROM answer_events child
                    WHERE child.supersedes_event_id=event.event_id
                )
                """,
                (assignment_id,),
            ).fetchone()
            if head is not None and head["event_type"] == "answer":
                raise ConflictError("answer must be undone before replacement")
            event_id = str(uuid.uuid4())
            received_at = _timestamp(current)
            connection.execute(
                "INSERT INTO answer_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    event_id,
                    command.idempotency_key,
                    payload_hash,
                    assignment_id,
                    session_id,
                    principal.user_id,
                    "answer",
                    decision,
                    json.dumps(sorted(command.reason_codes)),
                    reason_target,
                    comment,
                    int(command.flagged),
                    assignment["guideline_version"],
                    command.client_version,
                    received_at,
                    received_at,
                    head["event_id"] if head is not None else None,
                ),
            )
            connection.execute(
                "UPDATE assignments SET status='answered' WHERE assignment_id=?",
                (assignment_id,),
            )
            self._audit(
                connection,
                actor_id=principal.user_id,
                action="answer.saved",
                object_type="answer_event",
                object_id=event_id,
                details={"assignment_id": assignment_id, "decision": decision},
                now=current,
            )
            row = connection.execute(
                "SELECT * FROM answer_events WHERE event_id=?", (event_id,)
            ).fetchone()
            return self._receipt(row, replay=False)

    def undo_answer(
        self,
        principal: Principal,
        assignment_id: str,
        idempotency_key: str,
        *,
        session_id: str,
        now: datetime | None = None,
    ) -> AnswerReceipt:
        if principal.role != "annotator":
            raise AuthorizationError("annotator role required")
        assignment_id = _require_uuid(assignment_id, "assignment_id")
        idempotency_key = _require_uuid(idempotency_key, "idempotency_key")
        session_id = _require_uuid(session_id, "session_id")
        current = _utc(now)
        payload_hash = _sha256_bytes(
            _canonical(
                {
                    "assignment_id": assignment_id,
                    "idempotency_key": idempotency_key,
                    "session_id": session_id,
                    "event_type": "withdraw",
                }
            )
        )
        with self._transaction() as connection:
            replay = connection.execute(
                "SELECT * FROM answer_events WHERE annotator_id=? AND idempotency_key=?",
                (principal.user_id, idempotency_key),
            ).fetchone()
            if replay is not None:
                if replay["payload_sha256"] != payload_hash:
                    raise ConflictError("idempotency key was reused with a different payload")
                return self._receipt(replay, replay=True)
            assignment = connection.execute(
                """
                SELECT a.*,e.guideline_version,e.status AS experiment_status
                FROM assignments a JOIN experiments e ON e.experiment_id=a.experiment_id
                WHERE assignment_id=? AND annotator_id=?
                """,
                (assignment_id, principal.user_id),
            ).fetchone()
            if assignment is None:
                raise NotFoundError("assignment not found")
            if assignment["status"] != "answered" or assignment["experiment_status"] in {
                "answers_frozen",
                "revealed",
                "archived",
            }:
                raise ConflictError("assignment has no mutable current answer")
            head = connection.execute(
                """
                SELECT * FROM answer_events event WHERE assignment_id=?
                AND NOT EXISTS (
                    SELECT 1 FROM answer_events child
                    WHERE child.supersedes_event_id=event.event_id
                )
                """,
                (assignment_id,),
            ).fetchone()
            if head is None or head["event_type"] != "answer":
                raise ConflictError("assignment has no current answer")
            event_id = str(uuid.uuid4())
            received_at = _timestamp(current)
            connection.execute(
                "INSERT INTO answer_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    event_id,
                    idempotency_key,
                    payload_hash,
                    assignment_id,
                    session_id,
                    principal.user_id,
                    "withdraw",
                    None,
                    "[]",
                    "none",
                    None,
                    0,
                    assignment["guideline_version"],
                    "undo",
                    received_at,
                    received_at,
                    head["event_id"],
                ),
            )
            connection.execute(
                "UPDATE assignments SET status='withdrawn' WHERE assignment_id=?",
                (assignment_id,),
            )
            self._audit(
                connection,
                actor_id=principal.user_id,
                action="answer.withdrawn",
                object_type="answer_event",
                object_id=event_id,
                details={"assignment_id": assignment_id},
                now=current,
            )
            row = connection.execute(
                "SELECT * FROM answer_events WHERE event_id=?", (event_id,)
            ).fetchone()
            return self._receipt(row, replay=False)

    def read_media(
        self,
        principal: Principal,
        assignment_id: str,
        side: str,
        lease_token: str,
    ) -> tuple[bytes, str]:
        if principal.role != "annotator" or side not in {"left", "right"}:
            raise AuthorizationError("media is not available")
        assignment_id = _require_uuid(assignment_id, "assignment_id")
        column = "left_asset_id" if side == "left" else "right_asset_id"
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT asset.storage_key,asset.mime_type,a.lease_token_sha256
                FROM assignments a JOIN assets asset ON asset.asset_id=a.{column}
                WHERE a.assignment_id=? AND a.annotator_id=?
                """,
                (assignment_id, principal.user_id),
            ).fetchone()
        if row is None or not secrets.compare_digest(
            row["lease_token_sha256"], _sha256_bytes(lease_token.encode())
        ):
            raise AuthorizationError("media assignment or lease token is invalid")
        return self._asset_path(row["storage_key"]).read_bytes(), row["mime_type"]

    def progress(self, principal: Principal, experiment_id: str) -> Progress:
        if principal.role != "annotator":
            raise AuthorizationError("annotator role required")
        experiment_id = _require_uuid(experiment_id, "experiment_id")
        with self._connect() as connection:
            experiment = connection.execute(
                "SELECT target_votes FROM experiments WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            if experiment is None:
                raise NotFoundError("experiment not found")
            assigned = connection.execute(
                "SELECT COUNT(*) FROM assignments WHERE experiment_id=? AND annotator_id=?",
                (experiment_id, principal.user_id),
            ).fetchone()[0]
            answered = connection.execute(
                "SELECT COUNT(*) FROM assignments WHERE experiment_id=? AND annotator_id=? "
                "AND status='answered'",
                (experiment_id, principal.user_id),
            ).fetchone()[0]
            total = connection.execute(
                "SELECT COUNT(*) FROM comparisons WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()[0]
            events = connection.execute(
                "SELECT COUNT(*) FROM answer_events event JOIN assignments a "
                "ON a.assignment_id=event.assignment_id "
                "WHERE a.experiment_id=? AND event.annotator_id=?",
                (experiment_id, principal.user_id),
            ).fetchone()[0]
        return Progress(
            experiment_id=experiment_id,
            assigned=assigned,
            answered=answered,
            remaining=max(total - answered, 0),
            target_votes=experiment["target_votes"],
            saved_events=events,
        )

    @staticmethod
    def _review_from_row(row: dict, lease_token: str) -> BlindReview:
        review_assignment_id = row["review_assignment_id"]
        reasons = []
        if row["has_flagged_answer"]:
            reasons.append("flagged")
        if row["has_disagreement"]:
            reasons.append("disagreement")
        return BlindReview(
            review_assignment_id=review_assignment_id,
            experiment_id=row["experiment_id"],
            comparison_id=row["comparison_id"],
            prompt_text=row["prompt_text"],
            seed=row["seed"],
            width=row["width"],
            height=row["height"],
            left_media_url=f"/api/reviews/{review_assignment_id}/media/left",
            right_media_url=f"/api/reviews/{review_assignment_id}/media/right",
            lease_token=lease_token,
            lease_expires_at=row["lease_expires_at"],
            guideline_version=row["guideline_version"],
            queue_reasons=tuple(reasons),
        )

    def claim_review(
        self,
        principal: Principal,
        experiment_id: str,
        *,
        now: datetime | None = None,
    ) -> BlindReview | None:
        if principal.role != "reviewer":
            raise AuthorizationError("reviewer role required")
        experiment_id = _require_uuid(experiment_id, "experiment_id")
        current = _utc(now)
        expires = current + timedelta(seconds=self.lease_seconds)
        lease_token = secrets.token_urlsafe(32)
        lease_hash = _sha256_bytes(lease_token.encode())
        with self._transaction() as connection:
            experiment = connection.execute(
                "SELECT status FROM experiments WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            if experiment is None:
                raise NotFoundError("experiment not found")
            if experiment["status"] != "answers_frozen":
                raise ConflictError("reviews require frozen answers")

            queued = self._review_queue(connection, experiment_id, pending_only=True)
            for comparison_row in queued:
                comparison = dict(comparison_row)
                review = connection.execute(
                    "SELECT * FROM review_assignments WHERE experiment_id=? AND comparison_id=?",
                    (experiment_id, comparison["comparison_id"]),
                ).fetchone()
                if (
                    review is not None
                    and review["reviewer_id"] != principal.user_id
                    and review["lease_expires_at"] > _timestamp(current)
                ):
                    continue

                presentation_version = 1
                side_hash = _sha256_bytes(
                    (
                        f"review:{experiment_id}:{comparison['comparison_id']}:"
                        f"{principal.user_id}:{presentation_version}"
                    ).encode()
                )
                assets = (comparison["asset_a_id"], comparison["asset_b_id"])
                left, right = assets if int(side_hash[-1], 16) % 2 == 0 else assets[::-1]
                if review is not None:
                    # Preserve the original side presentation when a reviewer
                    # takes over an expired or withdrawn review assignment.
                    presentation_version = review["presentation_version"]
                    left = review["left_asset_id"]
                    right = review["right_asset_id"]
                if review is None:
                    review_assignment_id = str(
                        uuid.uuid5(
                            uuid.UUID(experiment_id),
                            f"review:{comparison['comparison_id']}",
                        )
                    )
                    connection.execute(
                        "INSERT INTO review_assignments VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            review_assignment_id,
                            experiment_id,
                            comparison["comparison_id"],
                            principal.user_id,
                            presentation_version,
                            left,
                            right,
                            lease_hash,
                            _timestamp(current),
                            _timestamp(expires),
                            "leased",
                        ),
                    )
                    action = "review.claimed"
                else:
                    review_assignment_id = review["review_assignment_id"]
                    connection.execute(
                        "UPDATE review_assignments SET reviewer_id=?,"
                        "presentation_version=?,left_asset_id=?,right_asset_id=?,"
                        "lease_token_sha256=?,leased_at=?,lease_expires_at=?,"
                        "status='leased' WHERE review_assignment_id=?",
                        (
                            principal.user_id,
                            presentation_version,
                            left,
                            right,
                            lease_hash,
                            _timestamp(current),
                            _timestamp(expires),
                            review_assignment_id,
                        ),
                    )
                    action = "review.reclaimed"
                self._audit(
                    connection,
                    actor_id=principal.user_id,
                    action=action,
                    object_type="review_assignment",
                    object_id=review_assignment_id,
                    details={"presentation_version": presentation_version},
                    now=current,
                )
                comparison.update(
                    {
                        "review_assignment_id": review_assignment_id,
                        "experiment_id": experiment_id,
                        "lease_expires_at": _timestamp(expires),
                        "left_asset_id": left,
                        "right_asset_id": right,
                    }
                )
                return self._review_from_row(comparison, lease_token)
            return None

    @staticmethod
    def _validate_review(command: ReviewCommand) -> tuple[str, str | None]:
        if command.decision not in DECISIONS:
            raise ValidationError("invalid review decision")
        comment = command.comment.strip() if command.comment is not None else None
        if comment == "":
            comment = None
        if comment is not None and len(comment) > 500:
            raise ValidationError("review comment exceeds 500 characters")
        if not isinstance(command.client_version, str) or len(command.client_version) > 100:
            raise ValidationError("client version is too long")
        return command.decision, comment

    @staticmethod
    def _review_receipt(row: sqlite3.Row, *, replay: bool) -> ReviewReceipt:
        return ReviewReceipt(
            event_id=row["event_id"],
            review_assignment_id=row["review_assignment_id"],
            decision=row["decision"],
            event_type=row["event_type"],
            saved_at=row["received_at"],
            idempotent_replay=replay,
        )

    def submit_review(
        self,
        principal: Principal,
        command: ReviewCommand,
        *,
        session_id: str,
        now: datetime | None = None,
    ) -> ReviewReceipt:
        if principal.role != "reviewer":
            raise AuthorizationError("reviewer role required")
        decision, comment = self._validate_review(command)
        review_assignment_id = _require_uuid(
            command.review_assignment_id,
            "review_assignment_id",
        )
        session_id = _require_uuid(session_id, "session_id")
        _require_uuid(command.idempotency_key, "idempotency_key")
        current = _utc(now)
        payload_hash = _sha256_bytes(
            _canonical(
                {
                    **asdict(command),
                    "comment": comment,
                    "session_id": session_id,
                }
            )
        )
        with self._transaction() as connection:
            replay = connection.execute(
                "SELECT * FROM review_events WHERE reviewer_id=? AND idempotency_key=?",
                (principal.user_id, command.idempotency_key),
            ).fetchone()
            if replay is not None:
                if replay["payload_sha256"] != payload_hash:
                    raise ConflictError("idempotency key was reused with a different payload")
                return self._review_receipt(replay, replay=True)
            assignment = connection.execute(
                """
                SELECT review_assignment.*,experiment.status AS experiment_status,
                       experiment.guideline_version
                FROM review_assignments review_assignment
                JOIN experiments experiment
                  ON experiment.experiment_id=review_assignment.experiment_id
                WHERE review_assignment.review_assignment_id=?
                  AND review_assignment.reviewer_id=?
                """,
                (review_assignment_id, principal.user_id),
            ).fetchone()
            if assignment is None:
                raise NotFoundError("review assignment not found")
            if assignment["experiment_status"] != "answers_frozen":
                raise ConflictError("experiment is not accepting reviews")
            if assignment["status"] != "leased":
                raise ConflictError("review assignment does not have an active lease")
            if assignment["lease_expires_at"] <= _timestamp(current):
                raise ConflictError("review assignment lease expired")
            if not secrets.compare_digest(
                assignment["lease_token_sha256"],
                _sha256_bytes(command.lease_token.encode()),
            ):
                raise AuthorizationError("invalid review lease token")
            head = connection.execute(
                """
                SELECT event_id,event_type FROM review_events event
                WHERE review_assignment_id=? AND NOT EXISTS (
                  SELECT 1 FROM review_events child
                  WHERE child.supersedes_event_id=event.event_id
                )
                """,
                (review_assignment_id,),
            ).fetchone()
            if head is not None and head["event_type"] == "adjudication":
                raise ConflictError("review must be undone before replacement")
            event_id = str(uuid.uuid4())
            received_at = _timestamp(current)
            connection.execute(
                "INSERT INTO review_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    event_id,
                    command.idempotency_key,
                    payload_hash,
                    review_assignment_id,
                    session_id,
                    principal.user_id,
                    "adjudication",
                    decision,
                    comment,
                    assignment["guideline_version"],
                    command.client_version,
                    received_at,
                    received_at,
                    head["event_id"] if head is not None else None,
                ),
            )
            connection.execute(
                "UPDATE review_assignments SET status='adjudicated' WHERE review_assignment_id=?",
                (review_assignment_id,),
            )
            self._audit(
                connection,
                actor_id=principal.user_id,
                action="review.adjudicated",
                object_type="review_event",
                object_id=event_id,
                details={"review_assignment_id": review_assignment_id},
                now=current,
            )
            row = connection.execute(
                "SELECT * FROM review_events WHERE event_id=?",
                (event_id,),
            ).fetchone()
            return self._review_receipt(row, replay=False)

    def undo_review(
        self,
        principal: Principal,
        review_assignment_id: str,
        idempotency_key: str,
        *,
        session_id: str,
        now: datetime | None = None,
    ) -> ReviewReceipt:
        if principal.role != "reviewer":
            raise AuthorizationError("reviewer role required")
        review_assignment_id = _require_uuid(
            review_assignment_id,
            "review_assignment_id",
        )
        idempotency_key = _require_uuid(idempotency_key, "idempotency_key")
        session_id = _require_uuid(session_id, "session_id")
        current = _utc(now)
        payload_hash = _sha256_bytes(
            _canonical(
                {
                    "review_assignment_id": review_assignment_id,
                    "idempotency_key": idempotency_key,
                    "session_id": session_id,
                    "event_type": "withdraw",
                }
            )
        )
        with self._transaction() as connection:
            replay = connection.execute(
                "SELECT * FROM review_events WHERE reviewer_id=? AND idempotency_key=?",
                (principal.user_id, idempotency_key),
            ).fetchone()
            if replay is not None:
                if replay["payload_sha256"] != payload_hash:
                    raise ConflictError("idempotency key was reused with a different payload")
                return self._review_receipt(replay, replay=True)
            assignment = connection.execute(
                """
                SELECT review_assignment.*,experiment.status AS experiment_status,
                       experiment.guideline_version
                FROM review_assignments review_assignment
                JOIN experiments experiment
                  ON experiment.experiment_id=review_assignment.experiment_id
                WHERE review_assignment.review_assignment_id=?
                  AND review_assignment.reviewer_id=?
                """,
                (review_assignment_id, principal.user_id),
            ).fetchone()
            if assignment is None:
                raise NotFoundError("review assignment not found")
            if assignment["experiment_status"] != "answers_frozen":
                raise ConflictError("review decisions are frozen")
            if assignment["status"] != "adjudicated":
                raise ConflictError("review assignment has no current adjudication")
            head = connection.execute(
                """
                SELECT * FROM review_events event WHERE review_assignment_id=?
                AND NOT EXISTS (
                  SELECT 1 FROM review_events child
                  WHERE child.supersedes_event_id=event.event_id
                )
                """,
                (review_assignment_id,),
            ).fetchone()
            if head is None or head["event_type"] != "adjudication":
                raise ConflictError("review assignment has no current adjudication")
            event_id = str(uuid.uuid4())
            received_at = _timestamp(current)
            connection.execute(
                "INSERT INTO review_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    event_id,
                    idempotency_key,
                    payload_hash,
                    review_assignment_id,
                    session_id,
                    principal.user_id,
                    "withdraw",
                    None,
                    None,
                    assignment["guideline_version"],
                    "undo",
                    received_at,
                    received_at,
                    head["event_id"],
                ),
            )
            connection.execute(
                "UPDATE review_assignments SET status='withdrawn' WHERE review_assignment_id=?",
                (review_assignment_id,),
            )
            self._audit(
                connection,
                actor_id=principal.user_id,
                action="review.withdrawn",
                object_type="review_event",
                object_id=event_id,
                details={"review_assignment_id": review_assignment_id},
                now=current,
            )
            row = connection.execute(
                "SELECT * FROM review_events WHERE event_id=?",
                (event_id,),
            ).fetchone()
            return self._review_receipt(row, replay=False)

    def read_review_media(
        self,
        principal: Principal,
        review_assignment_id: str,
        side: str,
        lease_token: str,
    ) -> tuple[bytes, str]:
        if principal.role != "reviewer" or side not in {"left", "right"}:
            raise AuthorizationError("review media is not available")
        review_assignment_id = _require_uuid(
            review_assignment_id,
            "review_assignment_id",
        )
        column = "left_asset_id" if side == "left" else "right_asset_id"
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT asset.storage_key,asset.mime_type,
                       review_assignment.lease_token_sha256
                FROM review_assignments review_assignment
                JOIN assets asset ON asset.asset_id=review_assignment.{column}
                WHERE review_assignment.review_assignment_id=?
                  AND review_assignment.reviewer_id=?
                """,
                (review_assignment_id, principal.user_id),
            ).fetchone()
        if row is None or not secrets.compare_digest(
            row["lease_token_sha256"],
            _sha256_bytes(lease_token.encode()),
        ):
            raise AuthorizationError("review media assignment or lease token is invalid")
        return self._asset_path(row["storage_key"]).read_bytes(), row["mime_type"]

    def review_progress(
        self,
        principal: Principal,
        experiment_id: str,
    ) -> ReviewProgress:
        if principal.role != "reviewer":
            raise AuthorizationError("reviewer role required")
        experiment_id = _require_uuid(experiment_id, "experiment_id")
        with self._connect() as connection:
            experiment = connection.execute(
                "SELECT experiment_id FROM experiments WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            if experiment is None:
                raise NotFoundError("experiment not found")
            queued = len(self._review_queue(connection, experiment_id))
            remaining = len(self._review_queue(connection, experiment_id, pending_only=True))
            saved_events = connection.execute(
                """
                SELECT COUNT(*) FROM review_events event
                JOIN review_assignments assignment
                  ON assignment.review_assignment_id=event.review_assignment_id
                WHERE assignment.experiment_id=? AND event.reviewer_id=?
                """,
                (experiment_id, principal.user_id),
            ).fetchone()[0]
        return ReviewProgress(
            experiment_id=experiment_id,
            queued=queued,
            adjudicated=queued - remaining,
            remaining=remaining,
            saved_events=saved_events,
        )

    def export_experiment(
        self,
        principal: Principal,
        experiment_id: str,
        *,
        revealed: bool = False,
    ) -> dict:
        """Return an auditable export without ever exposing storage keys."""

        self._require_owner(principal)
        experiment_id = _require_uuid(experiment_id, "experiment_id")
        with self._connect() as connection:
            experiment = connection.execute(
                "SELECT * FROM experiments WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            if experiment is None:
                raise NotFoundError("experiment not found")
            if revealed and experiment["status"] not in {"revealed", "archived"}:
                raise ConflictError("de-blinded export requires a revealed experiment")
            rows = connection.execute(
                """
                SELECT event.*,a.comparison_id,a.left_asset_id,a.right_asset_id,
                       c.prompt_id,c.prompt_text,c.seed,c.width,c.height,
                       c.generation_protocol_sha256,c.asset_a_id,c.asset_b_id,c.qc_kind,
                       c.qc_group_id,left_asset.arm_id AS left_arm_id,
                       right_asset.arm_id AS right_arm_id,
                       EXISTS(
                         SELECT 1 FROM answer_events child
                         WHERE child.supersedes_event_id=event.event_id
                       ) AS is_superseded
                FROM answer_events event
                JOIN assignments a ON a.assignment_id=event.assignment_id
                JOIN comparisons c ON c.comparison_id=a.comparison_id
                JOIN assets left_asset ON left_asset.asset_id=a.left_asset_id
                JOIN assets right_asset ON right_asset.asset_id=a.right_asset_id
                WHERE a.experiment_id=?
                ORDER BY event.received_at,event.event_id
                """,
                (experiment_id,),
            ).fetchall()
            review_rows = connection.execute(
                """
                SELECT event.*,assignment.comparison_id,
                       assignment.left_asset_id,assignment.right_asset_id,
                       comparison.prompt_id,comparison.prompt_text,comparison.seed,
                       comparison.asset_a_id,comparison.asset_b_id,
                       left_asset.arm_id AS left_arm_id,
                       right_asset.arm_id AS right_arm_id,
                       EXISTS(
                         SELECT 1 FROM review_events child
                         WHERE child.supersedes_event_id=event.event_id
                       ) AS is_superseded
                FROM review_events event
                JOIN review_assignments assignment
                  ON assignment.review_assignment_id=event.review_assignment_id
                JOIN comparisons comparison
                  ON comparison.comparison_id=assignment.comparison_id
                JOIN assets left_asset
                  ON left_asset.asset_id=assignment.left_asset_id
                JOIN assets right_asset
                  ON right_asset.asset_id=assignment.right_asset_id
                WHERE assignment.experiment_id=?
                ORDER BY event.received_at,event.event_id
                """,
                (experiment_id,),
            ).fetchall()
            comparison_count = connection.execute(
                "SELECT COUNT(*) FROM comparisons WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()[0]
            audit_count = connection.execute(
                """
                SELECT COUNT(*) FROM audit_events
                WHERE object_id=?
                   OR object_id IN (
                     SELECT assignment_id FROM assignments WHERE experiment_id=?
                   )
                   OR object_id IN (
                     SELECT review_assignment_id FROM review_assignments
                     WHERE experiment_id=?
                   )
                """,
                (experiment_id, experiment_id, experiment_id),
            ).fetchone()[0]

        answers = []
        for row in rows:
            left_candidate = (
                "candidate_a" if row["left_asset_id"] == row["asset_a_id"] else "candidate_b"
            )
            right_candidate = (
                "candidate_a" if row["right_asset_id"] == row["asset_a_id"] else "candidate_b"
            )
            if row["decision"] == "left":
                canonical_decision = left_candidate
            elif row["decision"] == "right":
                canonical_decision = right_candidate
            else:
                canonical_decision = row["decision"]
            answer = {
                "event_id": row["event_id"],
                "idempotency_key": row["idempotency_key"],
                "assignment_id": row["assignment_id"],
                "comparison_id": row["comparison_id"],
                "prompt_id": row["prompt_id"],
                "prompt_text": row["prompt_text"],
                "seed": row["seed"],
                "width": row["width"],
                "height": row["height"],
                "generation_protocol_sha256": row["generation_protocol_sha256"],
                "annotator_id": row["annotator_id"],
                "event_type": row["event_type"],
                "presentation_decision": row["decision"],
                "canonical_decision": canonical_decision,
                "left_candidate": left_candidate,
                "right_candidate": right_candidate,
                "reason_codes": json.loads(row["reason_codes_json"]),
                "reason_target": row["reason_target"],
                "comment": row["comment"],
                "flagged": bool(row["flagged"]),
                "guideline_version": row["guideline_version"],
                "client_version": row["client_version"],
                "created_at": row["created_at"],
                "received_at": row["received_at"],
                "supersedes_event_id": row["supersedes_event_id"],
                "is_superseded": bool(row["is_superseded"]),
                "qc_kind": row["qc_kind"],
                "qc_group_id": row["qc_group_id"],
            }
            if revealed:
                arm_decision = {
                    "left": row["left_arm_id"],
                    "right": row["right_arm_id"],
                    "tie": "tie",
                    "skip": "skip",
                    None: None,
                }[row["decision"]]
                answer["reveal"] = {
                    "left_arm_id": row["left_arm_id"],
                    "right_arm_id": row["right_arm_id"],
                    "arm_decision": arm_decision,
                }
            answers.append(answer)
        reviews = []
        for row in review_rows:
            left_candidate = (
                "candidate_a" if row["left_asset_id"] == row["asset_a_id"] else "candidate_b"
            )
            right_candidate = (
                "candidate_a" if row["right_asset_id"] == row["asset_a_id"] else "candidate_b"
            )
            canonical_decision = {
                "left": left_candidate,
                "right": right_candidate,
                "tie": "tie",
                "skip": "skip",
                None: None,
            }[row["decision"]]
            review = {
                "event_id": row["event_id"],
                "idempotency_key": row["idempotency_key"],
                "review_assignment_id": row["review_assignment_id"],
                "comparison_id": row["comparison_id"],
                "prompt_id": row["prompt_id"],
                "prompt_text": row["prompt_text"],
                "seed": row["seed"],
                "reviewer_id": row["reviewer_id"],
                "event_type": row["event_type"],
                "presentation_decision": row["decision"],
                "canonical_decision": canonical_decision,
                "left_candidate": left_candidate,
                "right_candidate": right_candidate,
                "comment": row["comment"],
                "guideline_version": row["guideline_version"],
                "client_version": row["client_version"],
                "created_at": row["created_at"],
                "received_at": row["received_at"],
                "supersedes_event_id": row["supersedes_event_id"],
                "is_superseded": bool(row["is_superseded"]),
            }
            if revealed:
                review["reveal"] = {
                    "left_arm_id": row["left_arm_id"],
                    "right_arm_id": row["right_arm_id"],
                    "arm_decision": {
                        "left": row["left_arm_id"],
                        "right": row["right_arm_id"],
                        "tie": "tie",
                        "skip": "skip",
                        None: None,
                    }[row["decision"]],
                }
            reviews.append(review)
        return {
            "schema_version": SCHEMA_VERSION,
            "export_mode": "revealed" if revealed else "blind",
            "experiment": {
                "experiment_id": experiment_id,
                "protocol_sha256": experiment["protocol_sha256"],
                "dataset_id": experiment["dataset_id"],
                "split": experiment["split"],
                "guideline_version": experiment["guideline_version"],
                "target_votes": experiment["target_votes"],
                "status": experiment["status"],
                "created_at": experiment["created_at"],
                "frozen_at": experiment["frozen_at"],
            },
            "comparison_count": comparison_count,
            "answer_event_count": len(answers),
            "review_event_count": len(reviews),
            "audit_event_count": audit_count,
            "answers": answers,
            "reviews": reviews,
        }
