import pytest

from shadow_lab.mage_w4_search.contracts import DataManifest, PromptRecord
from shadow_lab.mage_w4_search.pilot import formal_readiness, register_group_pilot


def test_report_does_not_turn_component_success_into_formal_readiness():
    evidence = {
        "native_two_host": {"qualified": True, "sha256": "a" * 64},
        "ocr_lock": None,
        "scale_receipt": None,
        "groups": {str(i): {"actual_changes": 1} for i in range(13)},
    }
    report = formal_readiness(
        evidence, budget_used_seconds=300, per_device_seconds={"gooo": 0, "goooinext": 300}
    )
    assert report["status"] == "insufficient-evidence"
    assert "ocr_lock" in report["missing"]
    assert "measured_group_calibrations" in report["missing"]
    assert report["formal_execution_authorized"] is False
    assert report["pilot_used_seconds"] == 300


def test_registered_pilot_is_finite_and_cannot_open_sealed_prompts():
    from test_mage_w4_rewards import frozen_scales

    manifest = DataManifest(
        (
            PromptRecord(
                "p",
                "A sign reading OPEN and no other text",
                "dev",
                "development",
                ("candidate_development",),
                ("OPEN",),
                True,
            ),
            PromptRecord("sealed", "A mountain", "new", "final"),
        )
    )
    plan = register_group_pilot(
        manifest,
        q0_digest="a" * 64,
        prompt_ids=("p",),
        seeds=(9,),
        edit_counts=(1, 16, 256),
        proposals_per_count=2,
        max_attempts=8,
        random_seed=10,
        registered_at_utc="2026-09-09T08:00:00+00:00",
        scales=frozen_scales(),
        execution_kind="fixture",
    )
    schedule = plan.candidates()
    assert len(schedule) == 78
    assert [row["group_id"] for row in schedule[:7]] == [0, 0, 0, 0, 0, 0, 1]
    assert len({row["candidate_id"] for row in schedule}) == 78
    assert plan.candidates() == schedule
    with pytest.raises(ValueError, match="candidate development"):
        register_group_pilot(
            manifest,
            q0_digest="a" * 64,
            prompt_ids=("sealed",),
            seeds=(9,),
            edit_counts=(1,),
            proposals_per_count=1,
            max_attempts=8,
            random_seed=10,
            registered_at_utc="2026-09-09T08:00:00+00:00",
            scales=frozen_scales(),
            execution_kind="fixture",
        )


def test_real_pilot_refuses_fixture_scales_and_freezes_cfg_identity():
    from test_mage_w4_rewards import frozen_scales

    from shadow_lab.mage_w4_search.pilot import register_group_pilot

    manifest = DataManifest(
        (
            PromptRecord(
                "p", "OPEN", "dev", "development", ("candidate_development",), ("OPEN",), True
            ),
        )
    )
    args = dict(
        q0_digest="a" * 64,
        prompt_ids=("p",),
        seeds=(1,),
        edit_counts=(1,),
        proposals_per_count=1,
        max_attempts=8,
        random_seed=10,
        registered_at_utc="2026-09-09T08:00:00+00:00",
        scales=frozen_scales(),
    )
    with pytest.raises(ValueError, match="measured"):
        register_group_pilot(manifest, execution_kind="gb10_measured", **args)
    plan = register_group_pilot(manifest, execution_kind="fixture", **args)
    assert plan.fixed_generation_sha256 == frozen_scales().fixed_generation_sha256
    assert plan.scales_sha256 == frozen_scales().sha256


def test_file_hash_and_qualified_labels_alone_cannot_freeze_formal_values(tmp_path):
    import hashlib
    import json

    from shadow_lab.mage_w4_search.pilot import FORMAL_FIELDS

    path = tmp_path / "fake.json"
    path.write_text(json.dumps({"qualified": True, "value": 1}))
    artifact = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }
    evidence = {key: {"value": 1, "source": artifact, "pointer": "/value"} for key in FORMAL_FIELDS}
    report = formal_readiness(
        evidence, budget_used_seconds=0, per_device_seconds={"gooo": 0, "goooinext": 0}
    )
    assert report["status"] == "insufficient-evidence"
    assert report["formal_budget_includes_final"] is False
    assert report["missing"]


def test_unique_occupancy_recomputes_cost_and_keeps_nested_timings_descriptive():
    from shadow_lab.mage_w4_search.pilot import audit_occupancy

    lease = dict(
        lease_id="a",
        request_id="request",
        ticket_id="02",
        device_id="gooo",
        started_at=100,
        stopped_at=120,
        hard_deadline=130,
        run_id="run",
    )
    snapshot = dict(
        campaign_id="pilot",
        at=140,
        leases=[lease, dict(lease)],
        used_seconds=20,
        devices={"gooo": {"used_seconds": 20}, "goooinext": {"used_seconds": 0}},
        timing_events=[
            dict(
                event_id="parent",
                lease_id="a",
                name="worker",
                kind="device_component",
                started_at=100,
                ended_at=120,
                parent_event_id=None,
            ),
            dict(
                event_id="child",
                lease_id="a",
                name="scoring",
                kind="device_component",
                started_at=105,
                ended_at=110,
                parent_event_id="parent",
            ),
        ],
    )
    audit = audit_occupancy(snapshot)
    assert audit["used_seconds"] == 20
    assert audit["wall_seconds"] == 20
    assert audit["duplicate_lease_ids"] == ["a"]
    assert audit["component_intervals_are_additional_charges"] is False
    assert audit["remaining_seconds"] == 57600 - 20
    changed = {
        **snapshot,
        "leases": [lease, {**lease, "lease_id": "b", "started_at": 115, "stopped_at": 125}],
    }
    with pytest.raises(ValueError, match="overlapping"):
        audit_occupancy(changed)
    with pytest.raises(ValueError, match="total"):
        audit_occupancy({**snapshot, "used_seconds": 21})


def test_component_closeout_reads_only_explicit_metadata_and_records_missing_values(tmp_path):
    import hashlib
    import json

    from shadow_lab.mage_w4_search.pilot import closeout_report

    native = tmp_path / "native.json"
    native.write_text(
        json.dumps(
            {
                "schema": "mage-native-probe/v1",
                "qualified": True,
                "scope": "development-replay-only",
                "lease_id": "a",
            }
        )
    )
    source = dict(
        path=str(native),
        sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        size_bytes=native.stat().st_size,
    )
    snapshot = dict(
        campaign_id="pilot",
        at=120,
        leases=[
            dict(
                lease_id="a",
                request_id="request",
                ticket_id="02",
                device_id="gooo",
                started_at=100,
                stopped_at=120,
                hard_deadline=130,
                run_id="run",
            )
        ],
        used_seconds=20,
        devices={"gooo": {"used_seconds": 20}, "goooinext": {"used_seconds": 0}},
        timing_events=[],
    )
    report = closeout_report(
        snapshot, components=[dict(ticket_id="02", device_id="gooo", artifact=source)], evidence={}
    )
    assert report["readiness"]["status"] == "insufficient-evidence"
    assert report["groups"]["0"]["selected_edit_count"] is None
    assert report["components"][0]["file_verified"] is True
    assert report["components"][0]["claimed_qualified"] is True
    assert report["formal_execution_authorized"] is False
    native.write_text("{}")
    with pytest.raises(ValueError, match="artifact"):
        closeout_report(
            snapshot,
            components=[dict(ticket_id="02", device_id="gooo", artifact=source)],
            evidence={},
        )


def _group_fixture(*, counts=(1, 16), reps=2):
    from dataclasses import replace

    from test_mage_w4_rewards import frozen_scales, screening_batch

    from shadow_lab.mage_w4_search.pilot import ProposalOutcome

    scales = frozen_scales()
    baseline = screening_batch("q0", (0.0, 0.0, 0.5))
    plan = register_group_pilot(
        baseline.manifest,
        q0_digest="a" * 64,
        prompt_ids=("s1", "s2", "s3"),
        seeds=(1,),
        edit_counts=counts,
        proposals_per_count=reps,
        max_attempts=8,
        random_seed=10,
        registered_at_utc="2026-01-01T01:00:00+00:00",
        scales=scales,
        execution_kind="fixture",
    )
    outcomes, batches = [], []
    for index, proposal in enumerate(plan.candidates()[: len(counts) * reps]):
        batch = screening_batch(
            proposal["candidate_id"], (0.1, 1.0, 0.5), candidate_digest=f"{index + 1:064x}"
        )
        batches.append(
            replace(
                batch,
                request=replace(
                    batch.request,
                    mutation_seed=proposal["mutation_seed"],
                    effective_edits=proposal["mutation_count"],
                ),
            )
        )
        outcomes.append(
            ProposalOutcome(
                proposal["candidate_id"],
                "scored",
                1,
                proposal["mutation_count"],
                batch.request.candidate_digest,
                5.0,
            )
        )
    refs = {row.request.candidate_id: baseline for row in batches}
    return plan, scales, tuple(outcomes), tuple(batches), refs


def test_density_selects_positive_feasible_then_fewer_edits_then_frozen_order():
    from dataclasses import replace

    from shadow_lab.mage_w4_search.pilot import calibrate_group

    plan, scales, outcomes, batches, refs = _group_fixture()
    receipt = calibrate_group(
        plan,
        scales,
        group_id=0,
        outcomes=outcomes,
        candidates=tuple(reversed(batches)),
        current_references=refs,
        q0_references=refs,
    )
    assert receipt["selected_edit_count"] == 1
    assert receipt["selected_candidate_id"] == "g00-m1-r0"
    assert receipt["status"] == "fixture-selection-only"
    assert receipt["formal_density_frozen"] is False
    assert receipt["counts"] == {"scored": 4, "failed": 0, "duplicate": 0, "positive_feasible": 4}
    assert receipt["component_seconds"] == 20
    bad = tuple(
        replace(batch, samples=tuple(replace(row, clip=-1, hps=100) for row in batch.samples))
        for batch in batches
    )
    failed = calibrate_group(
        plan,
        scales,
        group_id=0,
        outcomes=outcomes,
        candidates=bad,
        current_references=refs,
        q0_references=refs,
    )
    assert failed["selected_edit_count"] is None
    assert failed["status"] == "no-positive-feasible-candidate"


def test_density_calibration_refuses_partial_grid_or_different_plan():
    from dataclasses import replace

    from shadow_lab.mage_w4_search.pilot import calibrate_group

    plan, scales, outcomes, batches, refs = _group_fixture()
    with pytest.raises(ValueError, match="complete finite grid"):
        calibrate_group(
            plan,
            scales,
            group_id=0,
            outcomes=outcomes[:-1],
            candidates=batches,
            current_references=refs,
            q0_references=refs,
        )
    with pytest.raises(ValueError, match="registered proposal"):
        wrong = (
            replace(batches[0], request=replace(batches[0].request, mutation_seed=999)),
        ) + batches[1:]
        calibrate_group(
            plan,
            scales,
            group_id=0,
            outcomes=outcomes,
            candidates=wrong,
            current_references=refs,
            q0_references=refs,
        )
    with pytest.raises(ValueError, match="fixed scales"):
        calibrate_group(
            replace(plan, scales_sha256="e" * 64),
            scales,
            group_id=0,
            outcomes=outcomes,
            candidates=batches,
            current_references=refs,
            q0_references=refs,
        )


def test_all_failed_grid_is_complete_evidence_without_an_invented_density():
    from dataclasses import replace

    from shadow_lab.mage_w4_search.pilot import calibrate_group

    plan, scales, outcomes, _, _ = _group_fixture()
    failed = tuple(
        replace(
            row,
            status="failed",
            candidate_digest=None,
            actual_changes=0,
            attempts=8,
            reason="no-effective-mutation",
        )
        for row in outcomes
    )
    result = calibrate_group(
        plan,
        scales,
        group_id=0,
        outcomes=failed,
        candidates=(),
        current_references={},
        q0_references={},
    )
    assert result["selected_edit_count"] is None
    assert result["counts"]["failed"] == 4
    assert result["screening"] is None
    assert result["status"] == "no-positive-feasible-candidate"


def test_relabelling_fixtures_does_not_make_measured_calibration_evidence():
    from dataclasses import replace

    from shadow_lab.mage_w4_search.pilot import calibrate_group

    plan, scales, outcomes, batches, refs = _group_fixture()
    scales = replace(scales, execution_kind="gb10_measured")
    plan = replace(plan, execution_kind="gb10_measured", scales_sha256=scales.sha256)
    batches = tuple(replace(row, execution_kind="gb10_measured") for row in batches)
    refs = {key: replace(row, execution_kind="gb10_measured") for key, row in refs.items()}
    with pytest.raises(ValueError, match="verified native"):
        calibrate_group(
            plan,
            scales,
            group_id=0,
            outcomes=outcomes,
            candidates=batches,
            current_references=refs,
            q0_references=refs,
        )


def test_every_qualified_label_with_13_group_names_still_cannot_freeze():
    from shadow_lab.mage_w4_search.pilot import FORMAL_FIELDS

    values = {key: {"qualified": True, "sha256": "a" * 64} for key in FORMAL_FIELDS}
    values["measured_group_calibrations"]["groups"] = {str(i): {} for i in range(13)}
    report = formal_readiness(
        values, budget_used_seconds=0, per_device_seconds={"gooo": 0, "goooinext": 0}
    )
    assert report["status"] == "insufficient-evidence"
    assert set(report["missing"]) == set(FORMAL_FIELDS)


def test_budget_exhaustion_reports_existing_evidence_without_extra_allowance():
    from shadow_lab.mage_w4_search.pilot import closeout_report

    snapshot = dict(
        campaign_id="pilot",
        at=30000,
        leases=[
            dict(
                lease_id="a",
                request_id="request",
                ticket_id="08",
                device_id="gooo",
                started_at=0,
                stopped_at=28800,
                hard_deadline=28800,
                run_id="run",
            )
        ],
        used_seconds=28800,
        devices={"gooo": {"used_seconds": 28800}, "goooinext": {"used_seconds": 0}},
        timing_events=[],
    )
    report = closeout_report(snapshot, components=[], evidence={})
    assert report["stop_reason"] == "budget-exhausted"
    assert report["accounting"]["remaining_per_device_seconds"]["gooo"] == 0
    assert report["readiness"]["status"] == "insufficient-evidence"
    assert report["formal_budget_seconds"] is None


def test_ocr_public_identity_archive_roundtrip_keeps_fixture_unqualified(tmp_path):
    import json
    from dataclasses import asdict, replace

    from test_mage_w4_ocr import config

    from shadow_lab.mage_w4_search.contracts import digest_json
    from shadow_lab.mage_w4_search.ocr import (
        CalibrationPlan,
        HumanTranscription,
        RawSegment,
        compare_ocr,
        freeze_scorer,
        recognize,
        scorer_lock_identity,
    )
    from shadow_lab.mage_w4_search.pilot import artifact_reference, validate_ocr_archive

    def archive(name, value):
        path = tmp_path / name
        # Deliberately use a third whitespace style: raw-file, contracts and
        # OCR object's hashes have distinct contracts and must not be conflated.
        path.write_text(json.dumps(value, ensure_ascii=False, indent=4) + "\n")
        return artifact_reference(path)

    configs = (config(), config("EasyOCR"))
    observations = tuple(
        recognize(
            b"OCR identity fixture",
            engine,
            lambda image, frozen: (RawSegment("OPFN", (0, 0, 40, 10)),),
        )
        for engine in configs
    )
    truth_data = dict(
        image_sha256=observations[0].image_sha256,
        actual_segments=["OPFN"],
        source_kind="human_image_transcription",
        annotator_id="synthetic-unit-test-fixture",
        complete=True,
    )
    truth = archive("synthetic-transcription.json", truth_data)
    plan = CalibrationPlan(
        (observations[0].image_sha256,), configs, ("b" * 64,), "2026-01-01T00:00:00+00:00", 0.5
    )
    report = compare_ocr(
        plan,
        (
            HumanTranscription(
                **{**truth_data, "actual_segments": ("OPFN",), "artifact_sha256": truth["sha256"]}
            ),
        ),
        observations,
        execution_kind="fixture",
        occupied_seconds_by_config=((configs[0].sha256, 2), (configs[1].sha256, 3)),
    )
    identity = scorer_lock_identity(report)
    assert identity["config_sha256"] != digest_json(asdict(configs[0]))
    value = dict(
        lock=archive("identity-only.json", identity),
        calibration=archive("fixture-report.json", asdict(report)),
        human_transcriptions=[truth],
        observations=[
            archive(f"recognition-{i}.json", asdict(row)) for i, row in enumerate(observations)
        ],
        device_evidence=None,
    )
    restored = validate_ocr_archive(value)
    assert restored.sha256 == report.sha256
    assert restored.execution_kind == "fixture"
    with pytest.raises(ValueError, match="GB10"):
        freeze_scorer(restored)
    field = dict(
        schema="mage-formal-field/v1",
        field="ocr_lock",
        value=value,
        rationale="fixture only; no measured freeze claim",
        basis=[value["calibration"]],
    )
    result = formal_readiness(
        {"ocr_lock": archive("fixture-field.json", field)},
        budget_used_seconds=0,
        per_device_seconds={"gooo": 0, "goooinext": 0},
    )
    assert "ocr_lock" in result["missing"]
    changed_report = compare_ocr(
        plan,
        (replace(report.transcriptions[0], annotator_id="different-fixture-author"),),
        observations,
        execution_kind="fixture",
        occupied_seconds_by_config=((configs[0].sha256, 2), (configs[1].sha256, 3)),
    )
    changed_value = {
        **value,
        "lock": archive("different-author-identity.json", scorer_lock_identity(changed_report)),
        "calibration": archive("different-author-report.json", asdict(changed_report)),
    }
    with pytest.raises(ValueError, match="actual human image transcription"):
        validate_ocr_archive(changed_value)


def complete_human_archive_fixture(tmp_path):
    import hashlib
    import json
    from dataclasses import asdict

    from test_mage_w4_ocr import config

    from shadow_lab.mage_w4_search.human_transcription import import_human_work
    from shadow_lab.mage_w4_search.ocr import (
        CalibrationPlan,
        RawSegment,
        compare_ocr,
        recognize,
        scorer_lock_identity,
    )
    from shadow_lab.mage_w4_search.pilot import artifact_reference

    def archive(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value, ensure_ascii=False, indent=4) + "\n")
        return artifact_reference(path)

    images = tuple(f"whole-human-work-fixture-{i}".encode() for i in range(5))
    image_ids = tuple(hashlib.sha256(image).hexdigest() for image in images)
    inventory = archive(
        "all-images.json",
        {"purpose": "ocr_selection", "rows": [{"image_sha256": sha} for sha in image_ids]},
    )
    human = archive(
        "whole-human.json",
        {
            "schema": "mage-ocr-human-transcription-entry/v1",
            "purpose": "ocr_selection",
            "transcribed_by": "synthetic-unit-test-author",
            "transcribed_at_utc": "2026-01-01T00:00:00Z",
            "rows": [
                {
                    "index": i,
                    "image_sha256": sha,
                    "complete": i != 3,
                    "actual_segments": [f"Ab{i}!"],
                }
                for i, sha in enumerate(image_ids)
            ],
        },
    )
    selected = (0, 1, 2, 4)
    configs = (config(), config("EasyOCR"))
    plan = CalibrationPlan(
        tuple(image_ids[i] for i in selected),
        configs,
        ("b" * 64,) * len(selected),
        "2026-01-01T00:01:00Z",
        0.25,
    )
    work = import_human_work(human["path"], inventory["path"])
    observations = tuple(
        recognize(
            images[i],
            engine,
            lambda image, frozen, i=i: (RawSegment(f"Ab{i}!", (0, 0, 40, 10)),),
        )
        for engine in configs
        for i in selected
    )
    report = compare_ocr(plan, work.for_plan(plan), observations, execution_kind="fixture")
    value = {
        "schema": "mage-w4-ocr-archive/v2",
        "human_source": {
            "format": "complete-human-work/v1",
            "artifact": human,
            "inventory": inventory,
        },
        "lock": archive("lock-identity.json", scorer_lock_identity(report)),
        "calibration": archive("calibration.json", asdict(report)),
        "observations": [
            archive(f"observations-{i}.json", asdict(row)) for i, row in enumerate(observations)
        ],
        "device_evidence": None,
    }
    return value, report, archive


def test_complete_human_archive_replays_original_file_identity_without_qualifying_fixture(tmp_path):
    from shadow_lab.mage_w4_search.pilot import validate_ocr_archive

    value, report, archive = complete_human_archive_fixture(tmp_path)
    restored = validate_ocr_archive(value)
    assert restored.sha256 == report.sha256
    assert len(restored.transcriptions) == 4
    assert {truth.artifact_sha256 for truth in restored.transcriptions} == {
        value["human_source"]["artifact"]["sha256"]
    }
    assert restored.transcriptions[0].actual_segments == ("Ab0!",)
    field = archive(
        "formal-field.json",
        {
            "schema": "mage-formal-field/v1",
            "field": "ocr_lock",
            "value": value,
            "basis": [value["calibration"]],
            "rationale": "Whole-file public identity fixture; no measured OCR claim.",
        },
    )
    readiness = formal_readiness(
        {"ocr_lock": field}, budget_used_seconds=0, per_device_seconds={"gooo": 0, "goooinext": 0}
    )
    assert "ocr_lock" in readiness["missing"]
    assert readiness["formal_execution_authorized"] is False


@pytest.mark.parametrize(
    "change",
    [
        "no_explicit_schema",
        "unknown_source_format",
        "mixed_source_formats",
        "partial_inventory",
        "reordered_inventory",
        "incomplete_selected_image",
        "human_after_registration",
        "changed_author_text",
        "same_text_different_raw_file",
    ],
)
def test_complete_human_archive_rejects_incomplete_or_reidentified_originals(tmp_path, change):
    import json
    from pathlib import Path

    from shadow_lab.mage_w4_search.pilot import artifact_reference, validate_ocr_archive

    value, _, archive = complete_human_archive_fixture(tmp_path)
    source = value["human_source"]
    if change == "no_explicit_schema":
        value.pop("schema")
    elif change == "unknown_source_format":
        source["format"] = "guess-from-fields"
    elif change == "mixed_source_formats":
        value["human_transcriptions"] = []
    elif change in {"partial_inventory", "reordered_inventory"}:
        inventory = json.loads(Path(source["inventory"]["path"]).read_text())
        if change == "partial_inventory":
            inventory["rows"].pop(3)  # Even an excluded row belongs to the complete inventory.
        else:
            inventory["rows"].reverse()
        source["inventory"] = archive("changed-inventory.json", inventory)
    else:
        human = json.loads(Path(source["artifact"]["path"]).read_text())
        if change == "incomplete_selected_image":
            human["rows"][0]["complete"] = False
        elif change == "human_after_registration":
            human["transcribed_at_utc"] = "2026-01-01T00:02:00Z"
        elif change == "changed_author_text":
            human["rows"][0]["actual_segments"] = ["AB0!"]
        if change == "same_text_different_raw_file":
            path = tmp_path / "reformatted-human.json"
            path.write_text(json.dumps(human, separators=(",", ":")))
            source["artifact"] = artifact_reference(path)
        else:
            source["artifact"] = archive("changed-human.json", human)
    with pytest.raises(ValueError):
        validate_ocr_archive(value)


def test_native_generate_sample_hash_is_not_complete_request_hash(tmp_path):
    from dataclasses import replace

    from test_mage_w4_native import _runtime
    from test_mage_w4_rewards import screening_batch

    from shadow_lab.mage_qad_checkpoint import content_hash
    from shadow_lab.mage_w4_search.pilot import validate_native_sample_binding

    runtime = _runtime(tmp_path)
    batch = screening_batch("q0", (0, 0, 0.5))
    prompts = {row.prompt_id: row.prompt for row in batch.manifest.records}
    samples = [
        {"sample_id": row.sample_id, "prompt": prompts[row.prompt_id], "seed": row.seed}
        for row in batch.request.samples
    ]
    report = runtime.generate(samples, tmp_path / "native-fixture")
    assert report["native_compute"] is False
    assert report["qualified"] is False
    assert report["request_sha256"] == content_hash(samples)
    assert report["request_sha256"] != batch.request.request_digest
    validate_native_sample_binding(batch.request, batch.manifest, report)
    changed = replace(
        batch.request,
        samples=(replace(batch.request.samples[0], seed=99),) + batch.request.samples[1:],
    )
    with pytest.raises(ValueError, match="sample"):
        validate_native_sample_binding(changed, batch.manifest, report)
    with pytest.raises(ValueError, match="sample"):
        validate_native_sample_binding(
            batch.request,
            batch.manifest,
            {**report, "request_sha256": batch.request.request_digest},
        )


def test_component_html_explains_missing_evidence_without_claiming_campaign_end():
    from shadow_lab.mage_w4_search.pilot import closeout_report, render_closeout_html

    snapshot = dict(
        campaign_id="pilot",
        at=0,
        leases=[],
        used_seconds=0,
        devices={"gooo": {"used_seconds": 0}, "goooinext": {"used_seconds": 0}},
        timing_events=[],
    )
    report = closeout_report(snapshot, components=[], evidence={})
    page = render_closeout_html(report)
    assert "组件阶段证据快照" in page
    assert "OCR 评分器冻结" in page
    assert "尚无已验证的依据文件" in page
    assert "05–08 真实闭环尚未验收" in page
    assert "此快照不表示整个先导已完成或 campaign 已停止" in page
    # Raw technical reasons remain available in the collapsed JSON for audit.
    assert "artifact requires a path" in page[page.index("<details>") :]
    assert "artifact requires a path" not in page[: page.index("<details>")]
