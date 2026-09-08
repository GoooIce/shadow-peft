"""Conservative joint evidence bounds over independently sampled source clusters.

See ticket-07-api.md for the derivation, required audits and power limitations.
No ordinary pointwise 95% interval path is exposed.
"""

from __future__ import annotations

import json
import math
import random
import re
import sqlite3
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from .contracts import RequestedSample, digest_json

AXES = ("clip", "hps", "ocr")
WEIGHTS = {"clip": 0.5, "hps": 0.25, "ocr": 0.25}
METHOD = "bounded-source-cluster-hoeffding-bonferroni/v1"


@dataclass(frozen=True)
class SliceSpec:
    slice_id: str
    purpose: str
    samples: tuple[RequestedSample, ...]
    ocr_prompt_ids: tuple[str, ...]
    data_digest: str


@dataclass(frozen=True)
class JointProtocol:
    protocol_id: str
    q0_digest: str
    scales: dict[str, float]
    score_bounds: dict[str, tuple[float, float]]
    confirmation_alphas: tuple[float, ...]
    min_source_clusters: int
    min_ocr_source_clusters: int
    independence_audit_sha256: str
    bounds_audit_sha256: str
    slices: tuple[SliceSpec, ...]
    execution_kind: str = "synthetic"
    formal_freeze_sha256: str | None = None
    scorer_digest: str = ""
    formal_authorization_id: str | None = None
    applicability_sha256: str | None = None

    @property
    def digest(self):
        return digest_json(asdict(self))

    def validate(self) -> tuple[str, ...]:
        reasons = []
        if not self.protocol_id or self.execution_kind not in {
            "synthetic",
            "development-replay",
            "formal",
        }:
            reasons.append("invalid-protocol-identity")
        if any(
            not _hash(value)
            for value in (
                self.q0_digest,
                self.scorer_digest,
                self.independence_audit_sha256,
                self.bounds_audit_sha256,
            )
        ):
            reasons.append("missing-frozen-identity-or-audit")
        if self.execution_kind == "formal" and (
            not _hash(self.formal_freeze_sha256)
            or not _hash(self.applicability_sha256)
            or not self.formal_authorization_id
        ):
            reasons.append("formal-protocol-not-authorized-or-supported")
        if set(self.scales) != set(AXES) or any(
            not _number(x) or x <= 0 for x in self.scales.values()
        ):
            reasons.append("invalid-fixed-scales")
        if set(self.score_bounds) != set(AXES):
            reasons.append("missing-score-support")
        else:
            for bound in self.score_bounds.values():
                if len(bound) != 2 or not all(_number(x) for x in bound) or bound[0] >= bound[1]:
                    reasons.append("invalid-score-support")
            if tuple(self.score_bounds["ocr"]) != (0, 1):
                reasons.append("ocr-support-must-be-zero-one")
        if not self.confirmation_alphas or any(
            not _number(a) or not 0 < a <= 0.05 for a in self.confirmation_alphas
        ):
            reasons.append("invalid-confirmation-risk-allocation")
        elif math.fsum(self.confirmation_alphas) > 0.05 + 1e-15:
            reasons.append("confirmation-risk-exceeds-five-percent")
        if any(
            type(n) is not int or n < 2
            for n in (self.min_source_clusters, self.min_ocr_source_clusters)
        ):
            reasons.append("invalid-source-count-floor")
        ids, groups, prompt_contents, prompt_ids = set(), set(), set(), set()
        for part in self.slices:
            if (
                not part.slice_id
                or part.slice_id in ids
                or part.purpose not in {"confirmation", "final"}
                or not _hash(part.data_digest)
            ):
                reasons.append("invalid-or-duplicate-slice")
            ids.add(part.slice_id)
            part_groups = {sample.source_group for sample in part.samples}
            if groups & part_groups:
                reasons.append("source-group-overlap-between-slices")
            groups.update(part_groups)
            part_contents = {sample.prompt_digest for sample in part.samples}
            part_prompt_ids = {sample.prompt_id for sample in part.samples}
            if prompt_contents & part_contents or prompt_ids & part_prompt_ids:
                reasons.append("prompt-content-overlap-between-slices")
            prompt_contents.update(part_contents)
            prompt_ids.update(part_prompt_ids)
            sample_ids = [sample.sample_id for sample in part.samples]
            if not sample_ids or len(set(sample_ids)) != len(sample_ids):
                reasons.append("empty-or-duplicate-frozen-sample")
            prompt_identity = {}
            for sample in part.samples:
                identity = (sample.prompt_digest, sample.source_group)
                if (
                    not sample.prompt_id
                    or not sample.source_group
                    or not _hash(sample.prompt_digest)
                ):
                    reasons.append("invalid-frozen-sample-identity")
                if type(sample.seed) is not int or not 0 <= sample.seed < 2**32:
                    reasons.append("invalid-frozen-generation-seed")
                if (
                    sample.prompt_id in prompt_identity
                    and prompt_identity[sample.prompt_id] != identity
                ):
                    reasons.append("inconsistent-prompt-identity")
                prompt_identity[sample.prompt_id] = identity
            if (
                not part.ocr_prompt_ids
                or len(set(part.ocr_prompt_ids)) != len(part.ocr_prompt_ids)
                or not set(part.ocr_prompt_ids) <= set(prompt_identity)
            ):
                reasons.append("missing-or-invalid-frozen-ocr-subset")
        return tuple(dict.fromkeys(reasons))


@dataclass(frozen=True)
class PairedRow:
    sample_id: str
    candidate: dict[str, float | None]
    current: dict[str, float | None] | None
    q0: dict[str, float | None]


@dataclass(frozen=True)
class PairedEvidence:
    candidate_id: str
    candidate_digest: str
    current_digest: str
    q0_digest: str
    center_version: int
    slice_id: str
    scorer_digest: str
    rows: tuple[PairedRow, ...]

    @property
    def digest(self):
        value = asdict(self)
        value["rows"] = sorted(value["rows"], key=lambda row: row["sample_id"])
        return digest_json(value)


def joint_decision(
    protocol: JointProtocol, evidence: PairedEvidence, *, alpha: float, stage: str
) -> dict:
    """Compute paired seed means, preserve prompt weights, then bound cluster sums."""
    reasons = list(protocol.validate())
    if stage not in {"confirmation", "final"} or not _number(alpha) or not 0 < alpha <= 0.05:
        reasons.append("invalid-stage-risk")
    elif (stage == "confirmation" and alpha not in protocol.confirmation_alphas) or (
        stage == "final" and alpha != 0.05
    ):
        reasons.append("risk-not-frozen")
    parts = {part.slice_id: part for part in protocol.slices}
    if evidence.slice_id not in parts or parts[evidence.slice_id].purpose != stage:
        reasons.append("slice-purpose-mismatch")
    if evidence.q0_digest != protocol.q0_digest or evidence.scorer_digest != protocol.scorer_digest:
        reasons.append("evidence-frozen-identity-mismatch")
    if (
        not evidence.candidate_id
        or any(
            not _hash(value)
            for value in (evidence.candidate_digest, evidence.current_digest, evidence.q0_digest)
        )
        or type(evidence.center_version) is not int
        or evidence.center_version < 0
    ):
        reasons.append("invalid-candidate-identity")
    if reasons:
        return {"status": "invalid-request", "reasons": sorted(set(reasons)), "formal": False}
    part = next(item for item in protocol.slices if item.slice_id == evidence.slice_id)
    identities = {item.sample_id: item for item in part.samples}
    row_ids = [row.sample_id for row in evidence.rows]
    if len(set(row_ids)) != len(row_ids) or set(row_ids) != set(identities):
        reasons.append("incomplete-duplicate-or-mismatched-paired-samples")
    for row in evidence.rows:
        identity = identities.get(row.sample_id)
        if identity is None:
            continue
        for reference in (
            ("candidate", "current", "q0") if stage == "confirmation" else ("candidate", "q0")
        ):
            scores = getattr(row, reference)
            if scores is None or set(scores) != set(AXES):
                reasons.append("missing-or-invalid-score")
                continue
            for axis in AXES:
                value = scores[axis]
                if axis == "ocr" and identity.prompt_id not in part.ocr_prompt_ids:
                    if value is not None:
                        reasons.append("ocr-not-applicable-must-be-none")
                elif not _number(value):
                    reasons.append("missing-or-invalid-score")
                elif not protocol.score_bounds[axis][0] <= value <= protocol.score_bounds[axis][1]:
                    reasons.append("score-outside-frozen-support")
    if reasons:
        return {"status": "invalid-request", "reasons": sorted(set(reasons)), "formal": False}
    if len({item.source_group for item in part.samples}) < protocol.min_source_clusters:
        reasons.append("too-few-source-clusters")
    if (
        len({item.source_group for item in part.samples if item.prompt_id in part.ocr_prompt_ids})
        < protocol.min_ocr_source_clusters
    ):
        reasons.append("too-few-ocr-source-clusters")
    if reasons:
        return {"status": "insufficient-evidence", "reasons": sorted(set(reasons)), "formal": False}
    prompt_rows = defaultdict(list)
    for row in evidence.rows:
        prompt_rows[identities[row.sample_id].prompt_id].append(row)
    groups = {item.prompt_id: item.source_group for item in part.samples}
    comparisons = ("current", "q0") if stage == "confirmation" else ("q0",)
    per_check_alpha = alpha / (7 if stage == "confirmation" else 4)
    checks = {}
    all_contributions = {}
    all_widths = {}
    margins = {
        "clip": 0.05 * protocol.scales["clip"],
        "hps": 0.05 * protocol.scales["hps"],
        "ocr": 0.01,
    }
    for reference in comparisons:
        for axis in AXES:
            prompts = sorted(
                prompt for prompt in prompt_rows if axis != "ocr" or prompt in part.ocr_prompt_ids
            )
            contributions = defaultdict(list)
            counts = defaultdict(int)
            for prompt in prompts:
                rows = sorted(prompt_rows[prompt], key=lambda row: row.sample_id)
                mean = math.fsum(
                    row.candidate[axis] - getattr(row, reference)[axis] for row in rows
                ) / len(rows)
                contributions[groups[prompt]].append(mean / len(prompts))
                counts[groups[prompt]] += 1
            low, high = protocol.score_bounds[axis]
            widths = {
                group: 2 * (high - low) * count / len(prompts) for group, count in counts.items()
            }
            values = {group: math.fsum(items) for group, items in contributions.items()}
            key = f"{reference}.{axis}"
            checks[key] = _lower_bound(values, widths, per_check_alpha, -margins[axis])
            all_contributions[key], all_widths[key] = values, widths
    composite_reference = "current" if stage == "confirmation" else "q0"
    composite_values, composite_widths = defaultdict(list), defaultdict(list)
    for axis in AXES:
        weight = WEIGHTS[axis] / protocol.scales[axis]
        key = f"{composite_reference}.{axis}"
        for group, value in all_contributions[key].items():
            composite_values[group].append(weight * value)
            composite_widths[group].append(weight * all_widths[key][group])
    composite = _lower_bound(
        {group: math.fsum(items) for group, items in composite_values.items()},
        {group: math.fsum(items) for group, items in composite_widths.items()},
        per_check_alpha,
        0,
    )
    composite["passed"] = composite["lower"] > 0
    checks["composite"] = composite
    axes_passed = all(value["passed"] for key, value in checks.items() if key != "composite")
    status = (
        "accepted"
        if axes_passed and composite["passed"]
        else ("noninferiority-failed" if not axes_passed else "insufficient-evidence")
    )
    return {
        "status": status,
        "method": METHOD,
        "protocol_digest": protocol.digest,
        "evidence_digest": evidence.digest,
        "stage": stage,
        "alpha": alpha,
        "formal": protocol.execution_kind == "formal",
        "checks": checks,
        "reason": "joint-lower-bound-gates; failure-is-not-proof-of-inferiority",
    }


def _lower_bound(values, widths, alpha, threshold):
    estimate = math.fsum(values[group] for group in sorted(values))
    radius = math.sqrt(-math.log(alpha) * math.fsum(width * width for width in widths.values()) / 2)
    lower = estimate - radius
    return {
        "estimate": estimate,
        "radius": radius,
        "lower": lower,
        "alpha": alpha,
        "threshold": threshold,
        "passed": lower >= threshold,
        "source_clusters": len(values),
    }


def _number(value):
    return type(value) in {int, float} and math.isfinite(value)


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


class EvidenceError(ValueError):
    """A frozen evidence attempt cannot be started, changed or repeated."""


class EvidenceLedger:
    """Persist pre-exposure claims and irreversible risk/slice consumption."""

    def __init__(self, path: str | Path, protocol: JointProtocol, *, clock=time.time):
        errors = protocol.validate()
        if errors:
            raise EvidenceError("invalid-protocol:" + ",".join(errors))
        self.path, self.protocol, self.clock = Path(path), protocol, clock
        self._protocol_digest = protocol.digest
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS attempts (attempt_id TEXT PRIMARY KEY, "
                "slice_id TEXT UNIQUE NOT NULL, stage TEXT NOT NULL, candidate_digest TEXT NOT NULL, "
                "claim TEXT NOT NULL, visible_at REAL, visibility_digest TEXT, "
                "receipt TEXT, evidence_digest TEXT, retired_at REAL, "
                "UNIQUE(stage,candidate_digest))"
            )
            row = db.execute("SELECT value FROM state WHERE key='protocol'").fetchone()
            if row and row[0] != protocol.digest:
                raise EvidenceError("protocol-mismatch-on-reopen")
            db.execute("INSERT OR IGNORE INTO state VALUES ('protocol', ?)", (protocol.digest,))
            initial = {
                "digest": protocol.q0_digest,
                "version": 0,
                "candidate_id": None,
                "confirmed": False,
                "formal": protocol.execution_kind == "formal",
            }
            db.execute(
                "INSERT OR IGNORE INTO state VALUES ('center', ?)",
                (json.dumps(initial, sort_keys=True),),
            )

    @contextmanager
    def _transaction(self):
        if self.protocol.digest != self._protocol_digest:
            raise EvidenceError("frozen-protocol-mutated")
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def claim_confirmation(
        self,
        attempt_id: str,
        slice_id: str,
        candidate_id: str,
        candidate_digest: str,
        *,
        current_digest: str,
        center_version: int,
        selection_digest: str,
    ) -> dict:
        """Freeze the screened winner before exposing one fresh confirmation slice."""
        if not _hash(selection_digest):
            raise EvidenceError("missing-frozen-winner-selection")
        claim = {
            "attempt_id": attempt_id,
            "stage": "confirmation",
            "slice_id": slice_id,
            "candidate_id": candidate_id,
            "candidate_digest": candidate_digest,
            "current_digest": current_digest,
            "center_version": center_version,
            "q0_digest": self.protocol.q0_digest,
            "selection_digest": selection_digest,
        }
        with self._transaction() as db:
            return self._claim(db, claim)

    def claim_final(
        self, attempt_id: str, slice_id: str, candidate_id: str, candidate_digest: str
    ) -> dict:
        """Freeze the sole final model; no replacement, second look or expansion."""
        with self._transaction() as db:
            center = json.loads(
                db.execute("SELECT value FROM state WHERE key='center'").fetchone()[0]
            )
            if not center["confirmed"]:
                raise EvidenceError("no-confirmed-center")
            if (candidate_id, candidate_digest) != (center["candidate_id"], center["digest"]):
                raise EvidenceError("final-model-not-latest-confirmed-center")
            return self._claim(
                db,
                {
                    "attempt_id": attempt_id,
                    "stage": "final",
                    "slice_id": slice_id,
                    "candidate_id": candidate_id,
                    "candidate_digest": candidate_digest,
                    "current_digest": center["digest"],
                    "center_version": center["version"],
                    "q0_digest": self.protocol.q0_digest,
                    "selection_digest": None,
                },
            )

    def _claim(self, db, claim):
        if (
            not claim["attempt_id"]
            or not claim["candidate_id"]
            or not _hash(claim["candidate_digest"])
        ):
            raise EvidenceError("invalid-claim-identity")
        previous = db.execute(
            "SELECT claim,retired_at FROM attempts WHERE attempt_id=?", (claim["attempt_id"],)
        ).fetchone()
        if previous:
            old = json.loads(previous[0])
            if any(old[key] != value for key, value in claim.items()):
                raise EvidenceError("claim-replay-mismatch")
            if previous["retired_at"] is not None:
                raise EvidenceError("retired-attempt")
            return old
        if db.execute("SELECT 1 FROM attempts WHERE stage='final'").fetchone():
            raise EvidenceError("final-already-opened")
        part = next(
            (part for part in self.protocol.slices if part.slice_id == claim["slice_id"]), None
        )
        if part is None or part.purpose != claim["stage"]:
            raise EvidenceError("slice-purpose-mismatch")
        if db.execute(
            "SELECT 1 FROM attempts WHERE stage=? AND candidate_digest=?",
            (claim["stage"], claim["candidate_digest"]),
        ).fetchone():
            raise EvidenceError("candidate-already-attempted")
        if db.execute("SELECT 1 FROM attempts WHERE slice_id=?", (claim["slice_id"],)).fetchone():
            raise EvidenceError("slice-already-consumed")
        center = json.loads(db.execute("SELECT value FROM state WHERE key='center'").fetchone()[0])
        if (claim["current_digest"], claim["center_version"]) != (
            center["digest"],
            center["version"],
        ):
            raise EvidenceError("stale-center-claim")
        if db.execute("SELECT 1 FROM attempts WHERE retired_at IS NULL").fetchone():
            raise EvidenceError("evidence-attempt-in-flight")
        if claim["stage"] == "confirmation":
            count = db.execute(
                "SELECT COUNT(*) FROM attempts WHERE stage='confirmation'"
            ).fetchone()[0]
            if count >= len(self.protocol.confirmation_alphas):
                raise EvidenceError("confirmation-risk-exhausted")
            alpha = self.protocol.confirmation_alphas[count]
        else:
            alpha = 0.05
        claim = {
            **claim,
            "alpha": alpha,
            "claimed_at": float(self.clock()),
            "protocol_digest": self.protocol.digest,
        }
        db.execute(
            "INSERT INTO attempts (attempt_id,slice_id,stage,candidate_digest,claim) VALUES (?,?,?,?,?)",
            (
                claim["attempt_id"],
                claim["slice_id"],
                claim["stage"],
                claim["candidate_digest"],
                json.dumps(claim, sort_keys=True),
            ),
        )
        return claim

    def mark_visible(self, attempt_id: str, visibility_digest: str) -> None:
        if not _hash(visibility_digest):
            raise EvidenceError("invalid-visibility-identity")
        with self._transaction() as db:
            row = db.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row is None:
                raise EvidenceError("unclaimed-attempt")
            if row["visible_at"] is None:
                if row["retired_at"] is not None:
                    raise EvidenceError("retired-attempt")
                db.execute(
                    "UPDATE attempts SET visible_at=?,visibility_digest=? WHERE attempt_id=?",
                    (float(self.clock()), visibility_digest, attempt_id),
                )

    def abort(self, attempt_id: str, *, reason: str) -> dict:
        """Even an unseen aborted claim burns its allocation; there is no refund API."""
        if not reason:
            raise EvidenceError("missing-abort-reason")
        with self._transaction() as db:
            row = db.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row is None:
                raise EvidenceError("unclaimed-attempt")
            if row["receipt"] is not None:
                return json.loads(row["receipt"])
            result = {
                "status": "insufficient-evidence",
                "reasons": [reason],
                "formal": False,
                "attempt_id": attempt_id,
                "retired": True,
                "claim": json.loads(row["claim"]),
            }
            db.execute(
                "UPDATE attempts SET receipt=?,retired_at=? WHERE attempt_id=?",
                (json.dumps(result, sort_keys=True), float(self.clock()), attempt_id),
            )
            return result

    def submit(self, attempt_id: str, evidence: PairedEvidence) -> dict:
        """Persist exposure before computing; repeated offline evidence is idempotent."""
        try:
            fingerprint = evidence.digest
        except (TypeError, ValueError):
            fingerprint = digest_json({"invalid_evidence": repr(evidence)})
        with self._transaction() as db:
            row = db.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row is None:
                raise EvidenceError("unclaimed-attempt")
            if row["evidence_digest"] is not None and row["evidence_digest"] != fingerprint:
                raise EvidenceError("visible-evidence-cannot-be-replaced")
            if row["receipt"] is not None:
                if row["evidence_digest"] is None:
                    raise EvidenceError("retired-attempt")
                return json.loads(row["receipt"])
            claim = json.loads(row["claim"])
            db.execute(
                "UPDATE attempts SET visible_at=COALESCE(visible_at,?), "
                "visibility_digest=COALESCE(visibility_digest,?), evidence_digest=? WHERE attempt_id=?",
                (float(self.clock()), fingerprint, fingerprint, attempt_id),
            )
        matches = all(
            getattr(evidence, key) == claim[key]
            for key in (
                "candidate_id",
                "candidate_digest",
                "current_digest",
                "center_version",
                "q0_digest",
                "slice_id",
            )
        )
        if not matches:
            result = {
                "status": "invalid-request",
                "reasons": ["evidence-does-not-match-frozen-claim"],
                "formal": False,
            }
        else:
            result = joint_decision(
                self.protocol, evidence, alpha=claim["alpha"], stage=claim["stage"]
            )
        result = {**result, "attempt_id": attempt_id, "retired": True, "claim": claim}
        with self._transaction() as db:
            row = db.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row["receipt"] is not None:
                return json.loads(row["receipt"])
            if claim["stage"] == "confirmation" and result["status"] == "accepted":
                center = {
                    "digest": claim["candidate_digest"],
                    "version": claim["center_version"] + 1,
                    "candidate_id": claim["candidate_id"],
                    "confirmed": True,
                    "formal": self.protocol.execution_kind == "formal",
                }
                db.execute(
                    "UPDATE state SET value=? WHERE key='center'",
                    (json.dumps(center, sort_keys=True),),
                )
            db.execute(
                "UPDATE attempts SET receipt=?,retired_at=? WHERE attempt_id=?",
                (
                    json.dumps(result, sort_keys=True, allow_nan=False),
                    float(self.clock()),
                    attempt_id,
                ),
            )
            return result

    def snapshot(self) -> dict:
        with self._transaction() as db:
            attempts = []
            for row in db.execute("SELECT * FROM attempts ORDER BY rowid"):
                item = dict(row)
                item["claim"] = json.loads(item["claim"])
                item["receipt"] = json.loads(item["receipt"]) if item["receipt"] else None
                attempts.append(item)
            return {
                "protocol_digest": self.protocol.digest,
                "latest_center": json.loads(
                    db.execute("SELECT value FROM state WHERE key='center'").fetchone()[0]
                ),
                "confirmation_alpha_committed": math.fsum(
                    item["claim"]["alpha"] for item in attempts if item["stage"] == "confirmation"
                ),
                "final_alpha_committed": math.fsum(
                    item["claim"]["alpha"] for item in attempts if item["stage"] == "final"
                ),
                "attempts": attempts,
            }


def coverage_validation(*, trials: int, source_clusters: int, seed: int) -> dict:
    """Synthetic applicability diagnostics; run substantial simulations on GB10 CPU.

    This validates implementation under the declared independent-cluster model.
    It cannot establish independence, valid bounds or practical power for real data.
    """
    if (
        type(trials) is not int
        or trials < 1
        or type(source_clusters) is not int
        or source_clusters < 8
    ):
        raise EvidenceError("invalid-simulation-size")
    rng = random.Random(seed)
    alpha, mc_beta = 0.025, 0.01
    scenarios = (
        ("null-correlated", (0.0, 0.0, 0.0), False, False),
        ("null-unequal-small-ocr", (0.0, 0.0, 0.0), True, True),
        ("noninferiority-boundary", (-0.05, -0.05, -0.01), False, False),
        ("strong-gain", (0.4, 0.4, 0.4), False, False),
    )
    mc_radius = math.sqrt(math.log(2 * 3 * len(scenarios) / mc_beta) / (2 * trials))
    started = time.process_time()
    results = []
    for name, deltas, unequal, small_ocr in scenarios:
        samples = tuple(
            RequestedSample(f"p-{group}-{member}", digest_json([group, member]), f"g-{group}", 1)
            for group in range(source_clusters)
            for member in range(1 + group % 3 if unequal else 1)
        )
        ocr_groups = max(2, source_clusters // 8) if small_ocr else source_clusters
        ocr_prompts = tuple(
            sample.prompt_id for sample in samples if int(sample.source_group[2:]) < ocr_groups
        )
        part = SliceSpec(
            "simulation-slice", "confirmation", samples, ocr_prompts, digest_json(name)
        )
        protocol = JointProtocol(
            "coverage-synthetic/v1",
            "a" * 64,
            dict.fromkeys(AXES, 1.0),
            dict.fromkeys(AXES, (0.0, 1.0)),
            (alpha, alpha),
            2,
            2,
            digest_json("independent-generated-clusters"),
            digest_json("explicit-synthetic-zero-one-support"),
            (part,),
            scorer_digest="e" * 64,
        )
        missed, accepted = 0, 0
        for _ in range(trials):
            signs = {f"g-{group}": rng.choice((-1, 1)) for group in range(source_clusters)}
            rows = []
            for sample in samples:
                candidate, reference = {}, {}
                for axis, delta in zip(AXES, deltas, strict=True):
                    if axis == "ocr" and sample.prompt_id not in ocr_prompts:
                        candidate[axis] = reference[axis] = None
                    else:
                        sign = signs[sample.source_group] * (-1 if unequal and axis == "hps" else 1)
                        candidate[axis] = 0.5 + delta / 2 + 0.1 * sign
                        reference[axis] = 0.5 - delta / 2 - 0.1 * sign
                rows.append(PairedRow(sample.sample_id, candidate, reference, reference))
            evidence = PairedEvidence(
                "candidate-simulation",
                "d" * 64,
                "a" * 64,
                "a" * 64,
                0,
                part.slice_id,
                "e" * 64,
                tuple(rows),
            )
            decision = joint_decision(protocol, evidence, alpha=alpha, stage="confirmation")
            truth = dict(zip(AXES, deltas, strict=True))
            truth["composite"] = math.fsum(WEIGHTS[axis] * truth[axis] for axis in AXES)
            missed += any(
                check["lower"] > truth[key.split(".")[-1]]
                for key, check in decision["checks"].items()
            )
            accepted += decision["status"] == "accepted"
        rate, accept_rate = missed / trials, accepted / trials
        results.append(
            {
                "scenario": name,
                "trials": trials,
                "source_clusters": source_clusters,
                "prompts": len({sample.prompt_id for sample in samples}),
                "ocr_source_clusters": ocr_groups,
                "true_deltas": dict(zip(AXES, deltas, strict=True)),
                "joint_noncoverage_count": missed,
                "joint_noncoverage_rate": rate,
                "noncoverage_mc_upper": min(1.0, rate + mc_radius),
                "acceptance_rate": accept_rate,
                "acceptance_mc_interval": [
                    max(0.0, accept_rate - mc_radius),
                    min(1.0, accept_rate + mc_radius),
                ],
                "coverage_diagnostic_passed": rate + mc_radius <= alpha,
            }
        )
    return {
        "method": METHOD,
        "execution_kind": "synthetic-cpu-validation",
        "seed": seed,
        "attempt_alpha": alpha,
        "monte_carlo_family_risk": mc_beta,
        "monte_carlo_radius": mc_radius,
        "cpu_seconds": time.process_time() - started,
        "results": results,
        "status": "synthetic-coverage-supported"
        if all(item["coverage_diagnostic_passed"] for item in results)
        else "synthetic-coverage-inconclusive-or-failed",
        "formal_applicability": "blocked-until-real-support-independence-and-cost-audits",
        "limitation": "Independent synthetic clusters only; no universal or real-workload power guarantee.",
    }


def _main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GB10 CPU synthetic joint-bound coverage diagnostics."
    )
    parser.add_argument("--coverage-json", type=Path, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--source-clusters", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    result = coverage_validation(
        trials=args.trials, source_clusters=args.source_clusters, seed=args.seed
    )
    args.coverage_json.parent.mkdir(parents=True, exist_ok=True)
    args.coverage_json.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    _main()
