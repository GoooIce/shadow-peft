"""Frozen development requests for teacher-free quantized candidate execution.

This module validates declarations and content identities. Native execution is
proved separately by the installer/runner; an honest receipt states that limit.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .budget import BudgetError

PROTOCOL = "mage-direct-w4a4/v1"
DEPENDENCY_ROLES = frozenset(
    {
        "quantized_packed",
        "weight_scales",
        "quantized_remainder",
        "layer_manifest",
        "text_encoder",
        "vae",
        "clip_scorer",
        "hps_scorer",
        "ocr_scorer",
    }
)
DEVELOPMENT_USES = frozenset({"ocr_selection", "scale_calibration", "candidate_development"})
GENERATION_ROLES = DEPENDENCY_ROLES - {"clip_scorer", "hps_scorer", "ocr_scorer"}


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ArtifactIdentity:
    role: str
    sha256: str
    size_bytes: int
    path: str | None = None


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    prompt: str
    source_group: str
    pool: str
    development_uses: tuple[str, ...] = ()
    target_text: tuple[str, ...] = ()
    complete_target_inventory: bool = False
    retired: bool = False


@dataclass(frozen=True)
class DataManifest:
    records: tuple[PromptRecord, ...]
    exposure_log: tuple[dict, ...] = ()

    @classmethod
    def from_dict(cls, value: dict) -> DataManifest:
        payload = dict(value)
        payload["records"] = tuple(
            PromptRecord(
                **{
                    **record,
                    "development_uses": tuple(record.get("development_uses", ())),
                    "target_text": tuple(record.get("target_text", ())),
                }
            )
            for record in payload["records"]
        )
        payload["exposure_log"] = tuple(payload.get("exposure_log", ()))
        return cls(**payload)

    @property
    def digest(self) -> str:
        return digest_json(asdict(self))

    def validate(self) -> tuple[str, ...]:
        reasons = []
        by_id = {}
        groups = {}
        contents = {}
        for record in self.records:
            if not record.prompt_id or not record.prompt or not record.source_group:
                reasons.append("missing-prompt-identity")
            if record.prompt_id in by_id:
                reasons.append(f"duplicate-prompt-id:{record.prompt_id}")
            by_id[record.prompt_id] = record
            if record.pool not in {"development", "confirmation", "final"}:
                reasons.append(f"invalid-pool:{record.prompt_id}")
            uses = frozenset(record.development_uses)
            if record.pool == "development":
                if len(uses) != 1 or not uses <= DEVELOPMENT_USES:
                    reasons.append(f"invalid-development-use:{record.prompt_id}")
            elif uses:
                reasons.append(f"sealed-prompt-has-development-use:{record.prompt_id}")
            identity = (record.pool, uses)
            if record.source_group in groups:
                previous = groups[record.source_group]
                if previous[0] != record.pool:
                    reasons.append(f"source-group-crosses-pools:{record.source_group}")
                elif previous != identity:
                    reasons.append(f"source-group-crosses-development-uses:{record.source_group}")
            groups[record.source_group] = identity
            content = " ".join(unicodedata.normalize("NFC", record.prompt).split())
            if content in contents and contents[content] != record.source_group:
                reasons.append(f"duplicate-prompt-content-crosses-source-groups:{record.prompt_id}")
            contents[content] = record.source_group
            if record.complete_target_inventory and (
                not record.target_text or any(not text.strip() for text in record.target_text)
            ):
                reasons.append(f"empty-complete-target-inventory:{record.prompt_id}")
            if record.target_text and not record.complete_target_inventory:
                reasons.append(f"partial-target-inventory:{record.prompt_id}")
        exposure_ids = set()
        retired_at = {}
        for event in self.exposure_log:
            if not isinstance(event, dict) or not all(
                event.get(key) is not None
                for key in ("exposure_id", "prompt_id", "purpose", "kind", "run_id", "at")
            ):
                reasons.append("invalid-exposure-record")
                continue
            if (
                not event["exposure_id"]
                or not event["run_id"]
                or event["exposure_id"] in exposure_ids
            ):
                reasons.append("invalid-or-duplicate-exposure-id")
            exposure_ids.add(event["exposure_id"])
            if type(event["at"]) not in {int, float} or not math.isfinite(event["at"]):
                reasons.append("invalid-exposure-time")
                continue
            record = by_id.get(event["prompt_id"])
            if record is None:
                reasons.append(f"exposure-unknown-prompt:{event['prompt_id']}")
                continue
            allowed = (
                set(record.development_uses) if record.pool == "development" else {record.pool}
            )
            if event["kind"] == "metadata":
                allowed.add(record.pool)
            if event["purpose"] not in allowed:
                reasons.append(f"exposure-purpose-mismatch:{record.prompt_id}")
            if event["kind"] not in {"metadata", "generation", "score", "retirement"}:
                reasons.append(f"invalid-exposure-kind:{record.prompt_id}")
            if record.pool != "development" and event["kind"] in {"generation", "score"}:
                reasons.append(f"sealed-content-exposure:{record.prompt_id}")
            if event["kind"] == "retirement":
                retired_at[record.prompt_id] = min(
                    retired_at.get(record.prompt_id, math.inf), event["at"]
                )
                if not record.retired:
                    reasons.append(f"retired-prompt-reactivated:{record.prompt_id}")
        for event in self.exposure_log:
            if (
                isinstance(event, dict)
                and event.get("prompt_id") in retired_at
                and event.get("kind") in {"generation", "score"}
            ):
                at = event.get("at")
                if type(at) in {int, float} and at >= retired_at[event["prompt_id"]]:
                    reasons.append(f"content-exposure-after-retirement:{event['prompt_id']}")
        return tuple(dict.fromkeys(reasons))


@dataclass(frozen=True)
class RequestedSample:
    prompt_id: str
    prompt_digest: str
    source_group: str
    seed: int

    @property
    def sample_id(self) -> str:
        return digest_json(asdict(self))


@dataclass(frozen=True)
class CandidateRequest:
    request_id: str
    run_id: str
    ticket_id: str
    center_version: int
    center_digest: str
    q0_digest: str
    candidate_id: str
    candidate_digest: str
    group_id: int
    mutation_seed: int
    effective_edits: int
    samples: tuple[RequestedSample, ...]
    purpose: str
    device_id: str
    dependencies: tuple[ArtifactIdentity, ...]
    runtime_identity: dict
    manifest_digest: str
    protocol: str = PROTOCOL
    lease_id: str | None = None
    evaluation_mode: str = "complete"

    @classmethod
    def from_dict(cls, value: dict) -> CandidateRequest:
        payload = dict(value)
        payload["dependencies"] = tuple(
            ArtifactIdentity(**item) for item in payload["dependencies"]
        )
        payload["samples"] = tuple(RequestedSample(**item) for item in payload["samples"])
        return cls(**payload)

    @property
    def request_digest(self) -> str:
        value = asdict(self)
        value.pop("lease_id")
        return digest_json(value)

    @property
    def cache_key(self) -> str:
        """Content identity excludes routing IDs; includes every numerical dependency."""
        value = asdict(self)
        for key in ("request_id", "run_id", "ticket_id", "candidate_id", "lease_id"):
            value.pop(key)
        for artifact in value["dependencies"]:
            artifact.pop("path")
        return digest_json(value)


@dataclass(frozen=True)
class PreflightReceipt:
    status: str
    reasons: tuple[str, ...]
    request_digest: str | None
    cache_key: str | None
    lease: dict | None = None
    artifact_verification: str = "declared-identities-only"
    evaluation_mode: str = "complete"

    def to_dict(self) -> dict:
        return asdict(self)


def preflight(
    request: CandidateRequest | dict,
    manifest: DataManifest | dict,
    *,
    expected_center_version: int,
    ledger=None,
    reserve_seconds: float | None = None,
) -> PreflightReceipt:
    """Return a rejection before spending budget; sealed pools are metadata-only."""
    try:
        if isinstance(request, dict):
            request = CandidateRequest.from_dict(request)
        if isinstance(manifest, dict):
            manifest = DataManifest.from_dict(manifest)
        reasons = list(manifest.validate())
    except (KeyError, TypeError, AttributeError, ValueError):
        return PreflightReceipt("rejected", ("malformed-request-or-manifest",), None, None)
    try:
        request_digest, cache_key = request.request_digest, request.cache_key
    except (TypeError, ValueError):
        return PreflightReceipt("rejected", ("noncanonical-request",), None, None)
    for name in ("request_id", "run_id", "ticket_id", "candidate_id", "device_id"):
        if not isinstance(getattr(request, name), str) or not getattr(request, name).strip():
            reasons.append(f"missing-identity:{name}")
    if request.device_id not in {"gooo", "goooinext"}:
        reasons.append("unsupported-pilot-device")
    for name in ("center_digest", "q0_digest", "candidate_digest", "manifest_digest"):
        if not _is_digest(getattr(request, name)):
            reasons.append(f"invalid-digest:{name}")
    if request.protocol != PROTOCOL:
        reasons.append("unsupported-protocol")
    if request.purpose not in DEVELOPMENT_USES:
        reasons.append("sealed-or-invalid-purpose")
    if request.manifest_digest != manifest.digest:
        reasons.append("manifest-identity-mismatch")
    if type(request.group_id) is not int or not 0 <= request.group_id < 13:
        reasons.append("invalid-layer-group")
    for name in ("center_version", "mutation_seed", "effective_edits"):
        if type(getattr(request, name)) is not int or getattr(request, name) < 0:
            reasons.append(f"invalid-nonnegative-integer:{name}")
    if not isinstance(request.runtime_identity, dict):
        reasons.append("missing-runtime-identity")
    else:
        for name in ("software", "kernel", "sampler", "generation", "device"):
            if not request.runtime_identity.get(name):
                reasons.append(f"missing-runtime-identity:{name}")
        reasons.extend(_forbidden_runtime(request.runtime_identity))
    roles = [dependency.role for dependency in request.dependencies]
    reasons.extend(f"forbidden-dependency:{role}" for role in roles if role not in DEPENDENCY_ROLES)
    if request.evaluation_mode not in {"generation", "replay", "complete"}:
        reasons.append("invalid-evaluation-mode")
    required = DEPENDENCY_ROLES if request.evaluation_mode == "complete" else GENERATION_ROLES
    reasons.extend(f"missing-dependency:{role}" for role in sorted(required - set(roles)))
    if len(roles) != len(set(roles)):
        reasons.append("duplicate-dependency-role")
    verified_paths = 0
    for dependency in request.dependencies:
        if (
            not _is_digest(dependency.sha256)
            or type(dependency.size_bytes) is not int
            or dependency.size_bytes < 0
        ):
            reasons.append(f"invalid-artifact-identity:{dependency.role}")
        if dependency.path is not None:
            try:
                path = Path(dependency.path)
                before = path.stat()
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                after = path.stat()
                changed = any(
                    getattr(before, key) != getattr(after, key)
                    for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
                )
                if (
                    changed
                    or after.st_size != dependency.size_bytes
                    or digest.hexdigest() != dependency.sha256
                ):
                    reasons.append(f"artifact-content-mismatch:{dependency.role}")
                else:
                    verified_paths += 1
            except OSError:
                reasons.append(f"artifact-unreadable:{dependency.role}")
    if request.center_version != expected_center_version:
        reasons.append("stale-center-version")
    by_id = {record.prompt_id: record for record in manifest.records}
    sample_ids = set()
    content_samples = set()
    has_text = False
    if not request.samples:
        reasons.append("empty-sample-request")
    for sample in request.samples:
        if sample.sample_id in sample_ids:
            reasons.append("duplicate-sample")
        sample_ids.add(sample.sample_id)
        content_identity = (sample.prompt_digest, sample.seed)
        if content_identity in content_samples:
            reasons.append("duplicate-sample")
        content_samples.add(content_identity)
        if type(sample.seed) is not int or not 0 <= sample.seed < 2**32:
            reasons.append(f"invalid-generation-seed:{sample.prompt_id}")
        record = by_id.get(sample.prompt_id)
        if record is None:
            reasons.append(f"unknown-prompt:{sample.prompt_id}")
            continue
        if sample.prompt_digest != digest_json(record.prompt):
            reasons.append(f"sample-prompt-mismatch:{sample.prompt_id}")
        if sample.source_group != record.source_group:
            reasons.append(f"sample-source-group-mismatch:{sample.prompt_id}")
        if record.pool != "development" or request.purpose not in record.development_uses:
            reasons.append(f"sample-purpose-forbidden:{sample.prompt_id}")
        if record.retired:
            reasons.append(f"sample-retired:{sample.prompt_id}")
        has_text |= record.complete_target_inventory and bool(record.target_text)
    if request.evaluation_mode == "complete" and not has_text:
        reasons.append("complete-evaluation-missing-ocr-subset")
    lease = None
    if not reasons:
        try:
            if ledger is None and (request.lease_id is not None or reserve_seconds is not None):
                raise BudgetError("missing-budget-ledger")
            if ledger is not None:
                if request.lease_id is not None:
                    lease = ledger.assert_active(request.lease_id)
                    if (lease.ticket_id, lease.device_id, lease.run_id) != (
                        request.ticket_id,
                        request.device_id,
                        request.run_id,
                    ):
                        raise BudgetError("lease-request-owner-mismatch")
                elif reserve_seconds is not None:
                    lease = ledger.reserve(
                        request.request_id,
                        request.ticket_id,
                        request.device_id,
                        reserve_seconds,
                        run_id=request.run_id,
                    )
                    ledger.assert_active(lease.lease_id)
                if lease is not None:
                    ledger.bind_request(lease.lease_id, request.request_id, request_digest)
        except BudgetError as exc:
            reasons.append(str(exc))
    verification = "local-paths-verified" if verified_paths else "declared-identities-only"
    return PreflightReceipt(
        "rejected" if reasons else ("authorized" if lease is not None else "ready_for_reservation"),
        tuple(dict.fromkeys(reasons)),
        request_digest,
        cache_key,
        lease=lease.to_dict() if lease is not None and not reasons else None,
        artifact_verification=verification,
        evaluation_mode=request.evaluation_mode,
    )


def _is_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _forbidden_runtime(value):
    reasons = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized.startswith(("teacher", "master", "healedmaster")) or normalized in {
                "tloss",
                "clip",
                "clips",
                "clipvalues",
                "clipparams",
                "clipparameters",
                "clipcoefficients",
                "clipcoefficient",
                "clipcoef",
                "clippingcoefficient",
            }:
                reasons.append(f"forbidden-runtime-key:{key}")
            reasons.extend(_forbidden_runtime(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            reasons.extend(_forbidden_runtime(child))
    return reasons
