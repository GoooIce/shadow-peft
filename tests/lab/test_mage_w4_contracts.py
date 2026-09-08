"""A frozen development request passes; forbidden stages never reserve hardware."""

import hashlib
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from shadow_lab.mage_w4_search.budget import BudgetLedger
from shadow_lab.mage_w4_search.contracts import (
    ArtifactIdentity,
    CandidateRequest,
    DataManifest,
    PromptRecord,
    RequestedSample,
    digest_json,
    preflight,
)


def legal_request():
    record = PromptRecord(
        "text-1",
        'A sign saying only "OPEN"',
        "source-1",
        "development",
        ("candidate_development",),
        ("OPEN",),
        True,
    )
    manifest = DataManifest((record,))
    request = CandidateRequest(
        request_id="request-1",
        run_id="run-1",
        ticket_id="01",
        center_version=0,
        center_digest="a" * 64,
        q0_digest="a" * 64,
        candidate_id="candidate-1",
        candidate_digest="b" * 64,
        group_id=0,
        mutation_seed=7,
        effective_edits=1,
        samples=(RequestedSample("text-1", digest_json(record.prompt), "source-1", 42),),
        purpose="candidate_development",
        device_id="gooo",
        dependencies=tuple(
            ArtifactIdentity(role, "c" * 64, 12)
            for role in (
                "quantized_packed",
                "weight_scales",
                "quantized_remainder",
                "layer_manifest",
                "text_encoder",
                "vae",
                "clip_scorer",
                "hps_scorer",
                "ocr_scorer",
            )
        ),
        runtime_identity={
            "software": "fixture-versions",
            "kernel": "native-fixture",
            "sampler": "fixture-sampler",
            "generation": {"steps": 4},
            "device": "gooo",
        },
        manifest_digest=manifest.digest,
    )
    return request, manifest


def test_preflight_distinguishes_allowed_clip_scorer_from_forbidden_clip_parameters():
    request, manifest = legal_request()
    receipt = preflight(request, manifest, expected_center_version=0)
    assert receipt.status == "ready_for_reservation"
    assert receipt.reasons == ()

    forbidden = replace(
        request,
        dependencies=request.dependencies + (ArtifactIdentity("clip_parameters", "d" * 64, 1),),
    )
    rejected = preflight(forbidden, manifest, expected_center_version=0)
    assert rejected.status == "rejected"
    assert "forbidden-dependency:clip_parameters" in rejected.reasons


def test_generation_component_does_not_require_unfrozen_scorers_or_claim_complete_evaluation():
    request, manifest = legal_request()
    request = replace(
        request,
        evaluation_mode="generation",
        dependencies=tuple(d for d in request.dependencies if not d.role.endswith("_scorer")),
    )
    receipt = preflight(request, manifest, expected_center_version=0)
    assert receipt.status == "ready_for_reservation"
    assert receipt.evaluation_mode == "generation"
    complete = preflight(
        replace(request, evaluation_mode="complete"), manifest, expected_center_version=0
    )
    assert complete.status == "rejected"
    assert "missing-dependency:ocr_scorer" in complete.reasons


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"run_id": ""}, "missing-identity:run_id"),
        ({"center_digest": "same-name.pth"}, "invalid-digest:center_digest"),
        ({"protocol": "legacy-clip-es"}, "unsupported-protocol"),
        ({"purpose": "confirmation"}, "sealed-or-invalid-purpose"),
        ({"manifest_digest": "f" * 64}, "manifest-identity-mismatch"),
        (
            {"runtime_identity": {"teacher_path": "hidden.pth"}},
            "forbidden-runtime-key:teacher_path",
        ),
    ],
)
def test_invalid_identity_or_forbidden_runtime_metadata_is_rejected(change, reason):
    request, manifest = legal_request()
    receipt = preflight(replace(request, **change), manifest, expected_center_version=0)
    assert receipt.status == "rejected"
    assert reason in receipt.reasons


def test_changed_prompt_duplicate_sample_and_sealed_source_group_never_form_complete_request():
    request, manifest = legal_request()
    wrong = replace(request.samples[0], prompt_digest="e" * 64)
    rejected = preflight(
        replace(request, samples=(wrong, wrong)), manifest, expected_center_version=0
    )
    assert "sample-prompt-mismatch:text-1" in rejected.reasons
    assert "duplicate-sample" in rejected.reasons

    sealed = replace(
        manifest.records[0],
        prompt_id="sealed-1",
        prompt="Different prompt",
        pool="final",
        development_uses=(),
    )
    leaky = DataManifest(manifest.records + (sealed,))
    rejected = preflight(
        replace(request, manifest_digest=leaky.digest), leaky, expected_center_version=0
    )
    assert "source-group-crosses-pools:source-1" in rejected.reasons


def test_complete_evaluation_cannot_silently_omit_applicable_text_subset():
    request, manifest = legal_request()
    record = replace(manifest.records[0], target_text=(), complete_target_inventory=False)
    manifest = DataManifest((record,))
    request = replace(request, manifest_digest=manifest.digest)
    receipt = preflight(request, manifest, expected_center_version=0)
    assert "complete-evaluation-missing-ocr-subset" in receipt.reasons


def test_content_verified_request_reserves_once_and_changed_same_name_file_is_rejected(tmp_path):
    request, manifest = legal_request()
    payload = tmp_path / "model.packed"
    payload.write_bytes(b"authentic")
    dependency = replace(
        request.dependencies[0],
        path=str(payload),
        sha256=hashlib.sha256(b"authentic").hexdigest(),
        size_bytes=9,
    )
    request = replace(request, dependencies=(dependency,) + request.dependencies[1:])
    ledger = BudgetLedger(tmp_path / "budget.sqlite", clock=lambda: 1_000)
    receipt = preflight(
        request, manifest, expected_center_version=0, ledger=ledger, reserve_seconds=20
    )
    assert receipt.status == "authorized"
    assert receipt.artifact_verification == "local-paths-verified"
    replay = preflight(
        request, manifest, expected_center_version=0, ledger=ledger, reserve_seconds=20
    )
    assert replay.lease == receipt.lease
    assert ledger.snapshot()["committed_seconds"] == 20

    payload.write_bytes(b"imposture")
    rejected = preflight(
        request, manifest, expected_center_version=0, ledger=ledger, reserve_seconds=20
    )
    assert "artifact-content-mismatch:quantized_packed" in rejected.reasons
    assert rejected.status == "rejected"
    assert ledger.snapshot()["committed_seconds"] == 20


def test_content_verification_ignores_read_only_atime_changes(tmp_path, monkeypatch):
    request, manifest = legal_request()
    payload = tmp_path / "packed"
    payload.write_bytes(b"authentic")
    dependency = replace(
        request.dependencies[0],
        path=str(payload),
        sha256=hashlib.sha256(b"authentic").hexdigest(),
        size_bytes=9,
    )
    request = replace(request, dependencies=(dependency,) + request.dependencies[1:])
    original_stat = Path.stat
    reads = []

    def access_time_changes(path, **kwargs):
        actual = original_stat(path, **kwargs)
        if path != payload:
            return actual
        fields = {name: getattr(actual, name) for name in dir(actual) if name.startswith("st_")}
        reads.append(actual.st_atime_ns + len(reads) + 1)
        fields.update(st_atime_ns=reads[-1], st_atime=reads[-1] / 1e9)
        return SimpleNamespace(**fields)

    monkeypatch.setattr(Path, "stat", access_time_changes)
    result = preflight(request, manifest, expected_center_version=0)
    assert len(reads) >= 2 and reads[0] != reads[-1]
    assert result.status == "ready_for_reservation"
    assert result.artifact_verification == "local-paths-verified"


def test_prompt_seeds_must_fit_native_generation_uint32_contract():
    request, manifest = legal_request()
    request = replace(request, samples=(replace(request.samples[0], seed=2**32),))
    result = preflight(request, manifest, expected_center_version=0)
    assert result.status == "rejected"
    assert "invalid-generation-seed:text-1" in result.reasons


def test_existing_occupancy_cannot_be_borrowed_by_another_run(tmp_path):
    request, manifest = legal_request()
    ledger = BudgetLedger(tmp_path / "budget.sqlite", clock=lambda: 1_000)
    lease = ledger.reserve("occupancy-1", "01", "gooo", 100, run_id="run-1")
    request = replace(request, lease_id=lease.lease_id)
    assert (
        preflight(request, manifest, expected_center_version=0, ledger=ledger).status
        == "authorized"
    )
    wrong = preflight(
        replace(request, run_id="run-2"), manifest, expected_center_version=0, ledger=ledger
    )
    assert "lease-request-owner-mismatch" in wrong.reasons


def test_exposure_history_allows_sealed_metadata_but_rejects_generation_and_reassignment():
    _, development = legal_request()
    sealed = PromptRecord("sealed-1", "A future prompt", "future-source", "confirmation")
    exposure = {
        "exposure_id": "e-1",
        "prompt_id": "sealed-1",
        "purpose": "confirmation",
        "kind": "metadata",
        "run_id": "freeze-1",
        "at": 1000,
    }
    manifest = DataManifest(development.records + (sealed,), (exposure,))
    assert manifest.validate() == ()
    consumed = replace(manifest, exposure_log=({**exposure, "kind": "score"},))
    assert "sealed-content-exposure:sealed-1" in consumed.validate()
    unknown = replace(manifest, exposure_log=({**exposure, "prompt_id": "unknown"},))
    assert "exposure-unknown-prompt:unknown" in unknown.validate()
    reclassified = replace(
        manifest,
        records=development.records
        + (replace(sealed, pool="development", development_uses=("candidate_development",)),),
    )
    assert "exposure-purpose-mismatch:sealed-1" in reclassified.validate()


def test_json_boundary_returns_a_structured_rejection_for_missing_required_field():
    request, manifest = legal_request()
    raw = asdict(request)
    assert (
        preflight(raw, asdict(manifest), expected_center_version=0).status
        == "ready_for_reservation"
    )
    raw.pop("run_id")
    result = preflight(raw, asdict(manifest), expected_center_version=0)
    assert result.status == "rejected"
    assert result.reasons == ("malformed-request-or-manifest",)
