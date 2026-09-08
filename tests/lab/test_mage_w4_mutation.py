"""Small packed fixtures for the direct adjacent-code proposal boundary."""

# ruff: noqa: E402 -- optional torch must be checked before torch-dependent imports.
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from shadow_lab.mage_qad_checkpoint import content_hash as _hash
from shadow_lab.mage_w4_search.mutation import (
    OUTSIDE_LAYERS,
    build_layer_groups,
    propose_candidate,
    replay_candidate,
)
from shadow_lab.mage_w4_search.native import NativeContractError, PackedLayer, QuantizedAssets


def _assets(*, code=0x88, zero_scale=False):
    names = [f"transformer_blocks.{b}.layer_{i}" for b in range(12) for i in range(14)]
    names += list(OUTSIDE_LAYERS)
    layers = {
        name: PackedLayer(
            torch.full((16, 8), code, dtype=torch.uint8),
            torch.full((16, 1), 0 if zero_scale else 1).to(torch.float8_e4m3fn),
            torch.tensor(0.25),
            (16, 16),
        )
        for name in names
    }
    record = {"packed": {n: v.record() for n, v in layers.items()}, "remainder": {}}
    fixed = {
        "scales": {n: [v.block_scale, v.tensor_scale] for n, v in layers.items()},
        "remainder": {},
    }
    return QuantizedAssets(layers, {}, {}, _hash(record), _hash(fixed))


def test_groups_cover_174_layers_once_with_explicit_outside_group():
    assets = _assets()
    groups = build_layer_groups(assets)
    assert len(groups) == 13
    assert sorted(n for g in groups for n in g.layers) == sorted(assets.layers)
    assert [len(g.layers) for g in groups] == [14] * 12 + [6]
    assert groups[-1].weight_count == 6 * 256


def test_one_selected_nibble_changes_and_candidate_replays_exactly():
    assets = _assets()
    groups = build_layer_groups(assets)
    proposal = propose_candidate(
        assets, groups, group_id=groups[0].group_id, mutation_count=1, seed=10, candidate_index=0
    )
    assert proposal.receipt["status"] == "candidate"
    edit = proposal.receipt["edits"][0]
    assert edit["old_code"] == 8
    assert edit["new_code"] in {1, 9}
    rebuilt = replay_candidate(assets, proposal.receipt)
    assert rebuilt.sha256 == proposal.assets.sha256
    assert rebuilt.fixed_sha256 == assets.fixed_sha256
    changed_bytes = sum(
        int((assets.layers[n].qdata != rebuilt.layers[n].qdata).sum()) for n in assets.layers
    )
    assert changed_bytes == 1
    layer = rebuilt.layers[edit["layer"]]
    byte = int(layer.qdata.view(-1)[edit["flat_index"] // 2])
    untouched = (byte >> 4) if edit["flat_index"] % 2 == 0 else (byte & 15)
    assert untouched == 8


def test_zero_scales_and_zero_request_are_explicit_non_candidates():
    assets = _assets(zero_scale=True)
    groups = build_layer_groups(assets)
    zero = propose_candidate(
        assets, groups, group_id=groups[0].group_id, mutation_count=1, seed=1, candidate_index=0
    )
    noop = propose_candidate(
        assets, groups, group_id=groups[0].group_id, mutation_count=0, seed=1, candidate_index=1
    )
    assert zero.assets is None and zero.receipt["status"] == "no_movable_coordinates"
    assert noop.assets is None and noop.receipt["status"] == "no_op"


@pytest.mark.parametrize("packed,expected", [(0xFF, 14), (0x77, 6)])
def test_numerical_endpoints_only_move_inward(packed, expected):
    assets = _assets(code=packed)
    groups = build_layer_groups(assets)
    result = propose_candidate(
        assets, groups, group_id="outside", mutation_count=20, seed=3, candidate_index=4
    )
    assert {e["new_code"] for e in result.receipt["edits"]} == {expected}
    coordinates = [(e["layer"], e["flat_index"]) for e in result.receipt["edits"]]
    assert len(set(coordinates)) == 20


def test_duplicate_retry_is_bounded_and_does_not_count_as_a_candidate():
    assets = _assets()
    groups = build_layer_groups(assets)
    kwargs = dict(group_id="outside", mutation_count=1, seed=1, candidate_index=2, max_attempts=1)
    first = propose_candidate(assets, groups, **kwargs)
    duplicate = propose_candidate(
        assets, groups, **kwargs, seen_candidate_sha256=[first.assets.sha256]
    )
    assert duplicate.assets is None
    assert duplicate.receipt["status"] == "duplicate_exhausted"
    assert duplicate.receipt["effective_candidates"] == 0
    assert len(duplicate.receipt["attempts"]) == 1
    assert duplicate.receipt["next_candidate_index"] == 3
    repeated = propose_candidate(assets, groups, **kwargs)
    assert first.receipt == repeated.receipt


def test_replay_rejects_hidden_extra_changes_or_wrong_parent():
    import copy

    assets = _assets()
    proposal = propose_candidate(
        assets,
        build_layer_groups(assets),
        group_id="outside",
        mutation_count=1,
        seed=10,
        candidate_index=0,
    )
    tampered = copy.deepcopy(proposal.receipt)
    tampered["edits"].append(tampered["edits"][0].copy())
    tampered["actual_changes"] = 2
    with pytest.raises(NativeContractError, match="duplicate coordinate"):
        replay_candidate(assets, tampered)
    with pytest.raises(NativeContractError, match="center"):
        replay_candidate(proposal.assets, proposal.receipt)


def test_selected_coordinates_ignore_zero_scale_blocks_and_canonicalize_new_zero():
    from dataclasses import replace

    assets = _assets(code=0x99, zero_scale=True)
    name = OUTSIDE_LAYERS[0]
    assets.layers[name].block_scale[0, 0] = 1
    fixed = {
        "scales": {n: [v.block_scale, v.tensor_scale] for n, v in assets.layers.items()},
        "remainder": {},
    }
    assets = replace(assets, sha256=_hash(assets.record()), fixed_sha256=_hash(fixed))
    proposal = propose_candidate(
        assets,
        build_layer_groups(assets),
        group_id="outside",
        mutation_count=16,
        seed=99,
        candidate_index=0,
    )
    assert proposal.receipt["movable_coordinates"] == 16
    assert {(e["layer"], e["flat_index"]) for e in proposal.receipt["edits"]} == {
        (name, i) for i in range(16)
    }
    assert {e["new_code"] for e in proposal.receipt["edits"]} == {0, 10}


def test_effective_scale_underflow_is_not_a_movable_weight():
    from dataclasses import replace

    assets = _assets()
    for layer in assets.layers.values():
        layer.block_scale.fill_(1 / 512)
        layer.tensor_scale.fill_(torch.finfo(torch.float32).tiny / 2**23)
    fixed = {
        "scales": {n: [v.block_scale, v.tensor_scale] for n, v in assets.layers.items()},
        "remainder": {},
    }
    assets = replace(assets, sha256=_hash(assets.record()), fixed_sha256=_hash(fixed))
    proposal = propose_candidate(
        assets,
        build_layer_groups(assets),
        group_id="outside",
        mutation_count=1,
        seed=99,
        candidate_index=0,
    )
    assert proposal.receipt["status"] == "no_movable_coordinates"


def test_full_packed_difference_independently_matches_the_sparse_receipt():
    from shadow_lab.mage_w4_search.mutation import audit_payload_difference

    assets = _assets()
    proposal = propose_candidate(
        assets,
        build_layer_groups(assets),
        group_id="outside",
        mutation_count=3,
        seed=6,
        candidate_index=0,
    )
    audit = audit_payload_difference(assets, proposal.assets)
    assert audit["edits"] == proposal.receipt["edits"]
    assert audit["code_changes"] == audit["numerical_changes"] == 3
    assert audit["fixed_state_equal"] is True
    assert audit["layers_compared"] == 174
