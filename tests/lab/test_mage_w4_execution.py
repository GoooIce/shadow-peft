from dataclasses import asdict

import pytest

from shadow_lab.mage_w4_search.budget import DeviceLease
from shadow_lab.mage_w4_search.contracts import (
    GENERATION_ROLES,
    ArtifactIdentity,
    CandidateRequest,
    DataManifest,
    PromptRecord,
    RequestedSample,
    digest_json,
)
from shadow_lab.mage_w4_search.execution import validate_worker_grant


def test_worker_consumes_authenticated_grant_without_second_ledger():
    manifest = DataManifest(
        (PromptRecord("p", "A cup", "source", "development", ("candidate_development",)),)
    )
    request = CandidateRequest(
        "req",
        "run",
        "02",
        0,
        "a" * 64,
        "a" * 64,
        "q0",
        "a" * 64,
        0,
        0,
        0,
        (RequestedSample("p", digest_json("A cup"), "source", 3),),
        "candidate_development",
        "gooo",
        tuple(ArtifactIdentity(role, "b" * 64, 1) for role in sorted(GENERATION_ROLES)),
        {key: "frozen" for key in ("software", "kernel", "sampler", "generation", "device")},
        manifest.digest,
        lease_id="lease",
        evaluation_mode="replay",
    )
    lease = DeviceLease("lease", "occupancy", "02", "gooo", 100, 200, 100, "run")
    grant = {
        "status": "authorized",
        "request": asdict(request),
        "request_digest": request.request_digest,
    }
    assert validate_worker_grant(grant, lease, manifest, device_id="gooo", now=110) == request
    with pytest.raises(ValueError, match="grant/lease mismatch"):
        validate_worker_grant(grant, lease, manifest, device_id="goooinext", now=110)


def test_json_manifest_defaults_match_shared_contract_and_preserve_digest():
    import json

    from test_mage_w4_rewards import screening_batch

    from shadow_lab.mage_w4_search.execution import read_manifest, read_request

    batch = screening_batch("q0", (0, 0, 0.5))
    raw_manifest = json.loads(json.dumps(asdict(batch.manifest)))
    raw_manifest.pop("exposure_log")
    # These optional defaults are absent in older delivered metadata.
    record = raw_manifest["records"][0]
    record.pop("retired")
    decoded = read_manifest(raw_manifest)
    assert decoded == batch.manifest
    assert decoded.digest == batch.manifest.digest
    assert decoded.exposure_log == ()
    raw_request = json.loads(json.dumps(asdict(batch.request)))
    assert read_request(raw_request) == batch.request
    assert read_request(raw_request).request_digest == batch.request.request_digest
