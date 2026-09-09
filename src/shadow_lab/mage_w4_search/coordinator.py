"""Durable GB10 development rounds with explicitly simulated confirmation.

Model files stay in immutable checkpoints. SQLite owns versions, frozen work,
proposal progress and at-most-once decisions; the single external BudgetLedger
continues to own device occupancy. This coordinator cannot open formal data or
create an independently confirmed center.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .budget import BudgetLedger
from .contracts import (
    ArtifactIdentity,
    CandidateRequest,
    DataManifest,
    digest_json,
    preflight,
)
from .rewards import ScaleReceipt, ScoredBatch, screen_candidates

FULL_STATE_ROLES = frozenset(
    {"quantized_packed", "weight_scales", "quantized_remainder", "layer_manifest"}
)


def _hash(value: str) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


@dataclass(frozen=True)
class FullCenter:
    # The native canonical packed-tensor/remainder identity is intentionally
    # different from an archive-file SHA or this artifact manifest's digest.
    native_payload_sha256: str
    artifacts: tuple[ArtifactIdentity, ...]

    @property
    def digest(self) -> str:
        return self.native_payload_sha256

    @property
    def artifact_manifest_digest(self) -> str:
        return digest_json(
            sorted((item.role, item.sha256, item.size_bytes) for item in self.artifacts)
        )

    def verify_files(self) -> None:
        if (
            not _hash(self.native_payload_sha256)
            or {item.role for item in self.artifacts} != FULL_STATE_ROLES
            or len(self.artifacts) != 4
        ):
            raise ValueError("a full native center requires four unique quantized state roles")
        for item in self.artifacts:
            if (
                not _hash(item.sha256)
                or not item.path
                or type(item.size_bytes) is not int
                or item.size_bytes < 0
            ):
                raise ValueError("complete center artifact identities and paths are required")
            path = Path(item.path)
            before = path.stat()
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            after = path.stat()
            stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if (
                any(getattr(before, field) != getattr(after, field) for field in stable_fields)
                or before.st_size != item.size_bytes
                or digest.hexdigest() != item.sha256
            ):
                raise ValueError(f"center artifact content changed: {item.role}")


@dataclass(frozen=True)
class CandidateAssignment:
    request: CandidateRequest
    manifest: DataManifest
    center: FullCenter


@dataclass(frozen=True)
class DevelopmentConfirmation:
    round_id: str
    candidate_id: str
    candidate_digest: str
    screening_sha256: str
    accepted: bool
    fixture_evidence_sha256: str
    kind: str = "development_mock_confirmation"

    def __post_init__(self) -> None:
        if self.kind != "development_mock_confirmation" or type(self.accepted) is not bool:
            raise ValueError("only explicitly mocked development confirmation is supported")
        if (
            not self.round_id
            or not self.candidate_id
            or not all(
                _hash(value)
                for value in (
                    self.candidate_digest,
                    self.screening_sha256,
                    self.fixture_evidence_sha256,
                )
            )
        ):
            raise ValueError(
                "a mock decision needs frozen candidate, screening and fixture identities"
            )


def _center(value: dict) -> FullCenter:
    return FullCenter(
        value["native_payload_sha256"],
        tuple(ArtifactIdentity(**item) for item in value["artifacts"]),
    )


def _assignment(value: dict) -> CandidateAssignment:
    return CandidateAssignment(
        CandidateRequest.from_dict(value["request"]),
        DataManifest.from_dict(value["manifest"]),
        _center(value["center"]),
    )


class DevelopmentCoordinator:
    def __init__(
        self,
        path: str | Path,
        *,
        q0: FullCenter,
        scales: ScaleReceipt,
        budget: BudgetLedger,
        rng_seed: int,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.budget, self.scales = budget, scales
        if self.path.resolve() == budget.path.resolve():
            raise ValueError("coordinator and budget require distinct databases")
        if type(rng_seed) is not int or rng_seed < 0:
            raise ValueError("a nonnegative frozen RNG seed is required")
        q0.verify_files()
        if scales.status != "ready" or scales.plan.q0_digest != q0.digest:
            raise ValueError("ready fixed-q0 scales are required")
        settings = {
            "protocol": "mage-w4-development-coordinator-v1",
            "rng_seed": rng_seed,
            "q0": asdict(q0),
            "scales_sha256": scales.sha256,
            "budget_path": str(budget.path.resolve()),
            "budget_settings": budget.settings,
        }
        with self._transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS events (event_id INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS rounds (round_id TEXT PRIMARY KEY, status TEXT NOT NULL, registration_digest TEXT NOT NULL, plan TEXT NOT NULL, screening TEXT, final TEXT)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS work (candidate_id TEXT PRIMARY KEY, round_id TEXT NOT NULL, device_id TEXT NOT NULL, planned TEXT NOT NULL, status TEXT NOT NULL, lease_id TEXT, submission_sha256 TEXT, submission TEXT)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS failed_payloads (digest TEXT PRIMARY KEY, round_id TEXT NOT NULL, candidate_id TEXT NOT NULL)"
            )
            existing = db.execute("SELECT payload FROM metadata WHERE key='settings'").fetchone()
            if existing and existing[0] != _json(settings):
                raise ValueError(
                    "coordinator q0/scales/RNG/budget identity differs from saved state"
                )
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('settings', ?)", (_json(settings),))
            initial = {
                "center": asdict(q0),
                "q0": asdict(q0),
                "center_version": 0,
                "group_index": 0,
                "proposal_cursor": 0,
                "completed_groups": 0,
                "consecutive_no_update_groups": 0,
                "active_round": None,
                "status": "ready",
                "center_identity": "development_q0",
                "independently_confirmed": False,
            }
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('state', ?)", (_json(initial),))

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    def _state(self, db) -> dict:
        return json.loads(
            db.execute("SELECT payload FROM metadata WHERE key='state'").fetchone()[0]
        )

    def _save(self, db, state: dict) -> None:
        db.execute("UPDATE metadata SET payload=? WHERE key='state'", (_json(state),))

    def snapshot(self) -> dict:
        with self._transaction() as db:
            state = self._state(db)
            state["events"] = [
                {"kind": row["kind"], **json.loads(row["payload"])}
                for row in db.execute("SELECT * FROM events ORDER BY event_id")
            ]
        _center(state["center"]).verify_files()
        state["budget"] = self.budget.snapshot()
        return state

    def next_mutation_seeds(self, count: int) -> tuple[int, ...]:
        if type(count) is not int or count <= 0:
            raise ValueError("a positive candidate count is required")
        with self._transaction() as db:
            state = self._state(db)
            settings = json.loads(
                db.execute("SELECT payload FROM metadata WHERE key='settings'").fetchone()[0]
            )
        return tuple(
            int(digest_json((settings["rng_seed"], state["proposal_cursor"] + index))[:15], 16)
            for index in range(count)
        )

    def begin_round(self, round_id: str, assignments: tuple[CandidateAssignment, ...]) -> dict:
        if (
            not round_id
            or not assignments
            or len({item.request.candidate_id for item in assignments}) != len(assignments)
        ):
            raise ValueError("a round requires unique preassigned candidates")
        self._require_budget()
        if len({item.center.digest for item in assignments}) != len(assignments):
            raise ValueError("duplicate payload proposals cannot count as distinct candidates")
        registration = [asdict(item) for item in assignments]
        registration_digest = digest_json(registration)
        for item in assignments:
            item.center.verify_files()
        with self._transaction() as db:
            existing = db.execute("SELECT * FROM rounds WHERE round_id=?", (round_id,)).fetchone()
            if existing:
                if existing["registration_digest"] != registration_digest:
                    raise ValueError("round identity reused with changed work")
                return {
                    "round_id": round_id,
                    "status": existing["status"],
                    "registration_digest": registration_digest,
                }
            state = self._state(db)
            if state["active_round"] is not None or state["status"] != "ready":
                raise ValueError("another round is active or development search has stopped")
            settings = json.loads(
                db.execute("SELECT payload FROM metadata WHERE key='settings'").fetchone()[0]
            )
            anchor = assignments[0].request
            current = _center(state["center"])
            q0 = _center(state["q0"])
            current.verify_files()
            for index, item in enumerate(assignments):
                request = item.request
                expected_seed = int(
                    digest_json((settings["rng_seed"], state["proposal_cursor"] + index))[:15], 16
                )
                if (
                    request.center_version != state["center_version"]
                    or request.center_digest != current.digest
                    or request.q0_digest != q0.digest
                    or request.group_id != state["group_index"]
                ):
                    raise ValueError("work refers to a stale center or wrong layer group")
                if (
                    request.candidate_digest != item.center.digest
                    or request.mutation_seed != expected_seed
                ):
                    raise ValueError("candidate payload identity or proposal RNG progress mismatch")
                if request.effective_edits <= 0 or item.center.digest == current.digest:
                    raise ValueError("no-op proposals cannot be evaluated as effective candidates")
                if db.execute(
                    "SELECT 1 FROM failed_payloads WHERE digest=?", (item.center.digest,)
                ).fetchone():
                    raise ValueError("failed payload cannot resit under another candidate identity")
                if request.purpose != "candidate_development" or request.device_id not in {
                    "gooo",
                    "goooinext",
                }:
                    raise ValueError("only preassigned GB10 development work is allowed")
                declared = {
                    artifact.role: (artifact.sha256, artifact.size_bytes)
                    for artifact in request.dependencies
                }
                if any(
                    declared.get(artifact.role) != (artifact.sha256, artifact.size_bytes)
                    for artifact in item.center.artifacts
                ):
                    raise ValueError("request state artifacts differ from the complete checkpoint")
                if {
                    (row.role, row.sha256, row.size_bytes)
                    for row in item.center.artifacts
                    if row.role != "quantized_packed"
                } != {
                    (row.role, row.sha256, row.size_bytes)
                    for row in q0.artifacts
                    if row.role != "quantized_packed"
                }:
                    raise ValueError("candidate changed frozen scales/remainder/layer manifest")
                if (
                    request.samples != anchor.samples
                    or request.manifest_digest != anchor.manifest_digest
                ):
                    raise ValueError("same batch requires identical prompt and seed identities")
                if {
                    key: value for key, value in request.runtime_identity.items() if key != "device"
                } != {
                    key: value for key, value in anchor.runtime_identity.items() if key != "device"
                }:
                    raise ValueError("same batch requires identical non-device runtime identities")
                checked = preflight(
                    replace(request, lease_id=None),
                    item.manifest,
                    expected_center_version=state["center_version"],
                )
                if checked.status != "ready_for_reservation":
                    raise ValueError(f"invalid frozen assignment: {checked.reasons}")
                if db.execute(
                    "SELECT 1 FROM work WHERE candidate_id=?", (request.candidate_id,)
                ).fetchone():
                    raise ValueError("candidate identity cannot be reused in another round")
            plan = {
                "candidate_order": [item.request.candidate_id for item in assignments],
                "center_version": state["center_version"],
                "group_index": state["group_index"],
                "center_digest": current.digest,
                "q0_digest": q0.digest,
            }
            db.execute(
                "INSERT INTO rounds VALUES (?, 'collecting', ?, ?, NULL, NULL)",
                (round_id, registration_digest, _json(plan)),
            )
            for item in assignments:
                db.execute(
                    "INSERT INTO work VALUES (?, ?, ?, ?, 'pending', NULL, NULL, NULL)",
                    (
                        item.request.candidate_id,
                        round_id,
                        item.request.device_id,
                        _json(asdict(item)),
                    ),
                )
            state["active_round"] = round_id
            state["proposal_cursor"] += len(assignments)
            self._save(db, state)
            db.execute(
                "INSERT INTO events(kind,payload) VALUES ('round_registered', ?)",
                (_json({"round_id": round_id, **plan}),),
            )
            return {
                "round_id": round_id,
                "status": "collecting",
                "registration_digest": registration_digest,
            }

    def claim(
        self,
        round_id: str,
        candidate_id: str,
        device_id: str,
        *,
        reserve_seconds: float,
        expected_request_digest: str | None = None,
        require_unclaimed: bool = False,
    ) -> CandidateRequest:
        self._require_budget()
        with self._transaction() as db:
            state = self._state(db)
            row = db.execute(
                "SELECT * FROM work WHERE candidate_id=? AND round_id=?", (candidate_id, round_id)
            ).fetchone()
            if row is None or row["device_id"] != device_id:
                raise ValueError("candidate can only be claimed by its assigned device")
            if state["active_round"] != round_id or row["status"] not in {"pending", "running"}:
                raise ValueError("candidate work is no longer claimable")
            item = _assignment(json.loads(row["planned"]))
            if (
                expected_request_digest is not None
                and item.request.request_digest != expected_request_digest
            ):
                raise ValueError("prepared request differs from the registered candidate")
            if require_unclaimed and row["lease_id"] is not None:
                raise ValueError("candidate already has a lease; recover without relaunching")
            request = replace(item.request, lease_id=row["lease_id"])
            checked = preflight(
                request,
                item.manifest,
                expected_center_version=state["center_version"],
                ledger=self.budget,
                reserve_seconds=reserve_seconds if request.lease_id is None else None,
            )
            if checked.status != "authorized":
                raise ValueError(f"device claim refused: {checked.reasons}")
            lease_id = checked.lease["lease_id"]
            db.execute(
                "UPDATE work SET status='running', lease_id=? WHERE candidate_id=?",
                (lease_id, candidate_id),
            )
            return replace(item.request, lease_id=lease_id)

    def work_request(self, round_id: str, candidate_id: str) -> CandidateRequest:
        """Read the original claim for transport recovery; never reserve or start work."""
        with self._transaction() as db:
            row = db.execute(
                "SELECT planned, lease_id FROM work WHERE round_id=? AND candidate_id=?",
                (round_id, candidate_id),
            ).fetchone()
            if row is None:
                raise ValueError("unknown registered candidate")
            request = _assignment(json.loads(row["planned"])).request
            return replace(request, lease_id=row["lease_id"])

    def reconcile_claim(
        self, round_id: str, candidate_id: str, lease_id: str, expected_request_digest: str
    ) -> CandidateRequest:
        """Attach an original committed reservation after a lost RPC reply; never reserve or launch."""
        with self._transaction() as db:
            state = self._state(db)
            row = db.execute(
                "SELECT * FROM work WHERE round_id=? AND candidate_id=?", (round_id, candidate_id)
            ).fetchone()
            if (
                row is None
                or state["active_round"] != round_id
                or row["status"] not in {"pending", "running"}
            ):
                raise ValueError("only an active uncompleted candidate can reconcile its claim")
            item = _assignment(json.loads(row["planned"]))
            if item.request.request_digest != expected_request_digest or row["lease_id"] not in {
                None,
                lease_id,
            }:
                raise ValueError("recovery differs from the original candidate request/lease")
            lease = self.budget.assert_active(lease_id)
            if (lease.request_id, lease.ticket_id, lease.device_id, lease.run_id) != (
                item.request.request_id,
                item.request.ticket_id,
                item.request.device_id,
                item.request.run_id,
            ):
                raise ValueError("recovery lease belongs to another original request")
            request = replace(item.request, lease_id=lease_id)
            checked = preflight(
                request,
                item.manifest,
                expected_center_version=state["center_version"],
                ledger=self.budget,
            )
            if checked.status != "authorized":
                raise ValueError(f"original lease cannot be recovered: {checked.reasons}")
            if row["lease_id"] is None:
                db.execute(
                    "UPDATE work SET status='running',lease_id=? WHERE candidate_id=?",
                    (lease_id, candidate_id),
                )
                db.execute(
                    "INSERT INTO events(kind,payload) VALUES ('claim_reconciled_without_launch',?)",
                    (
                        _json(
                            {
                                "round_id": round_id,
                                "candidate_id": candidate_id,
                                "lease_id": lease_id,
                                "request_digest": expected_request_digest,
                            }
                        ),
                    ),
                )
            return request

    def screen(self, round_id: str) -> dict:
        with self._transaction() as db:
            row = db.execute("SELECT * FROM rounds WHERE round_id=?", (round_id,)).fetchone()
            if row is None or row["screening"] is None:
                raise ValueError("round results are incomplete")
            return json.loads(row["screening"])

    def submit(
        self,
        round_id: str,
        candidate: ScoredBatch,
        current_reference: ScoredBatch,
        q0_reference: ScoredBatch,
    ) -> dict:
        candidate_id = candidate.request.candidate_id
        submission_identity = digest_json(
            {
                "candidate": candidate.sha256,
                "current": current_reference.sha256,
                "q0": q0_reference.sha256,
            }
        )
        with self._transaction() as db:
            row = db.execute(
                "SELECT * FROM work WHERE candidate_id=? AND round_id=?", (candidate_id, round_id)
            ).fetchone()
            if row is None:
                raise ValueError("unknown candidate submission")
            if row["submission_sha256"] is not None:
                if row["submission_sha256"] != submission_identity:
                    raise ValueError("duplicate candidate message changed its result")
                return {"status": "duplicate", "submission_sha256": submission_identity}
            state = self._state(db)
            if state["active_round"] != round_id or row["status"] != "running":
                raise ValueError("late result from inactive or failed round")
            planned = _assignment(json.loads(row["planned"]))
            if (
                candidate.request.request_digest != planned.request.request_digest
                or candidate.request.lease_id != row["lease_id"]
            ):
                raise ValueError("submitted request differs from claimed frozen candidate")
            if candidate.request.center_version != state["center_version"]:
                raise ValueError("late result from stale center version")
            usage = self.budget.snapshot()
            lease = next(
                (item for item in usage["leases"] if item["lease_id"] == row["lease_id"]), None
            )
            if lease is None or (lease["device_id"], lease["ticket_id"], lease["run_id"]) != (
                candidate.request.device_id,
                candidate.request.ticket_id,
                candidate.request.run_id,
            ):
                raise ValueError("result has no matching device occupancy record")
            ended = lease["stopped_at"] if lease["stopped_at"] is not None else usage["at"]
            if ended > lease["hard_deadline"]:
                raise ValueError("result exceeded its device hard deadline")
            screened = screen_candidates(
                self.scales,
                (candidate,),
                {candidate_id: current_reference},
                {candidate_id: q0_reference},
                candidate_order=(candidate_id,),
            )
            if screened.center_digest != _center(state["center"]).digest:
                raise ValueError("submitted center identity changed")
            payload = {
                "candidate": asdict(candidate),
                "current": asdict(current_reference),
                "q0": asdict(q0_reference),
                "screened_candidate": asdict(screened.candidates[0]),
            }
            db.execute(
                "UPDATE work SET status='received', submission_sha256=?, submission=? WHERE candidate_id=?",
                (submission_identity, _json(payload), candidate_id),
            )
            remaining = db.execute(
                "SELECT COUNT(*) FROM work WHERE round_id=? AND status!='received'", (round_id,)
            ).fetchone()[0]
            if not remaining:
                round_row = db.execute(
                    "SELECT * FROM rounds WHERE round_id=?", (round_id,)
                ).fetchone()
                plan = json.loads(round_row["plan"])
                received = {
                    item["candidate_id"]: json.loads(item["submission"])["screened_candidate"]
                    for item in db.execute("SELECT * FROM work WHERE round_id=?", (round_id,))
                }
                results = [received[identity] for identity in plan["candidate_order"]]
                feasible = [item for item in results if item["feasible"]]
                winner = (
                    max(feasible, key=lambda item: item["current"]["composite_improvement"])
                    if feasible
                    else None
                )
                final_screen = {
                    "status": "best_feasible_candidate" if winner else "no_feasible_candidate",
                    "selected_candidate_id": winner["candidate_id"] if winner else None,
                    "candidates": results,
                    "candidate_order": plan["candidate_order"],
                    "scales_sha256": self.scales.sha256,
                    "center_version": plan["center_version"],
                    "center_digest": plan["center_digest"],
                    "q0_digest": plan["q0_digest"],
                    "execution_kind": self.scales.execution_kind,
                    "stage": "development_screening",
                    "independently_confirmed": False,
                }
                final_screen["sha256"] = digest_json(final_screen)
                db.execute(
                    "UPDATE rounds SET status='screened', screening=? WHERE round_id=?",
                    (_json(final_screen), round_id),
                )
            return {
                "status": "received",
                "submission_sha256": submission_identity,
                "remaining": remaining,
            }

    def finish_round(
        self, round_id: str, confirmation: DevelopmentConfirmation | None = None
    ) -> dict:
        with self._transaction() as db:
            row = db.execute("SELECT * FROM rounds WHERE round_id=?", (round_id,)).fetchone()
            if row is None:
                raise ValueError("unknown round")
            decision_identity = digest_json(asdict(confirmation)) if confirmation else None
            if row["final"] is not None:
                previous = json.loads(row["final"])
                if previous.get("confirmation_sha256") != decision_identity:
                    raise ValueError("round already closed with a different decision")
                return previous
            state = self._state(db)
            if row["status"] != "screened" or state["active_round"] != round_id:
                raise ValueError("round results are incomplete or no longer active")
            _center(state["q0"]).verify_files()
            _center(state["center"]).verify_files()
            screened = json.loads(row["screening"])
            winner_id = screened["selected_candidate_id"]
            accepted = False
            if winner_id is None:
                if confirmation is not None:
                    raise ValueError("no feasible candidate can receive confirmation")
                decision = "no_feasible_candidate"
            else:
                winner = next(
                    item for item in screened["candidates"] if item["candidate_id"] == winner_id
                )
                if confirmation is None or (
                    confirmation.round_id,
                    confirmation.candidate_id,
                    confirmation.candidate_digest,
                    confirmation.screening_sha256,
                ) != (
                    round_id,
                    winner_id,
                    winner["candidate_digest"],
                    screened["sha256"],
                ):
                    raise ValueError(
                        "only this batch's frozen winner can receive one mock confirmation"
                    )
                accepted = confirmation.accepted
                if accepted and winner["current"]["composite_improvement"] <= 0:
                    raise ValueError(
                        "mock positive-gain acceptance cannot accept a nonpositive winner"
                    )
                decision = "development_accept" if accepted else "confirmation_failed"
                if not accepted:
                    db.execute(
                        "INSERT INTO failed_payloads VALUES (?, ?, ?)",
                        (winner["candidate_digest"], round_id, winner_id),
                    )
                if accepted:
                    planned = _assignment(
                        json.loads(
                            db.execute(
                                "SELECT planned FROM work WHERE candidate_id=?", (winner_id,)
                            ).fetchone()[0]
                        )
                    )
                    planned.center.verify_files()
                    state["center"] = asdict(planned.center)
                    state["center_version"] += 1
                    state["center_identity"] = "development_mock_accepted"
            state["completed_groups"] += 1
            state["consecutive_no_update_groups"] = (
                0 if accepted else state["consecutive_no_update_groups"] + 1
            )
            state["group_index"] = (state["group_index"] + 1) % 13
            state["active_round"] = None
            if state["consecutive_no_update_groups"] >= 26:
                state["status"] = "stopped"
                state["stop_reason"] = "two_complete_sweeps_without_update"
            final = {
                "round_id": round_id,
                "decision": decision,
                "confirmation_sha256": decision_identity,
                "screening_sha256": screened["sha256"],
                "accepted_candidate_id": winner_id if accepted else None,
                "center_version": state["center_version"],
                "center_digest": _center(state["center"]).digest,
                "q0_digest": _center(state["q0"]).digest,
                "next_group": state["group_index"],
                "independently_confirmed": False,
                "scope": "development_mock_only",
            }
            self._save(db, state)
            db.execute(
                "UPDATE rounds SET status='finished', final=? WHERE round_id=?",
                (_json(final), round_id),
            )
            db.execute(
                "INSERT INTO events(kind,payload) VALUES ('round_finished', ?)", (_json(final),)
            )
            return final

    def _require_budget(self) -> None:
        usage = self.budget.snapshot()
        exhausted = usage["used_seconds"] >= self.budget.settings["total_seconds"] or any(
            item["used_seconds"] >= self.budget.settings["per_device_seconds"]
            for item in usage["devices"].values()
        )
        if exhausted:
            # This separate durable transaction records the stop even though
            # the caller is about to receive a refusal exception.
            with self._transaction() as db:
                state = self._state(db)
                state["status"] = "stopped"
                state["stop_reason"] = "device_budget_exhausted"
                self._save(db, state)
            raise ValueError("development device budget is exhausted")

    def abort_round(self, round_id: str, *, reason: str) -> dict:
        """Record technical failure without advancing a group or releasing a GPU.

        Only the owning supervisor may acknowledge its process has stopped and
        release BudgetLedger occupancy. Disconnecting/aborting never does so.
        """
        if not reason:
            raise ValueError("technical abort needs a reason")
        with self._transaction() as db:
            row = db.execute("SELECT * FROM rounds WHERE round_id=?", (round_id,)).fetchone()
            if row is None:
                raise ValueError("unknown round")
            if row["status"] == "aborted":
                return json.loads(row["final"])
            state = self._state(db)
            if row["final"] is not None or state["active_round"] != round_id:
                raise ValueError("finished/inactive round cannot be aborted")
            state["active_round"] = None
            final = {
                "round_id": round_id,
                "decision": "technical_failure",
                "reason": reason,
                "advanced_group": False,
                "center_version": state["center_version"],
                "center_digest": _center(state["center"]).digest,
                "q0_digest": _center(state["q0"]).digest,
                "next_group": state["group_index"],
                "independently_confirmed": False,
                "scope": "development_mock_only",
            }
            self._save(db, state)
            db.execute(
                "UPDATE rounds SET status='aborted', final=? WHERE round_id=?",
                (_json(final), round_id),
            )
            db.execute(
                "UPDATE work SET status='aborted' WHERE round_id=? AND status!='received'",
                (round_id,),
            )
            db.execute(
                "INSERT INTO events(kind,payload) VALUES ('round_aborted', ?)", (_json(final),)
            )
            return final
