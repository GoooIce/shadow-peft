"""Reproducible single-group, sparse adjacent E2M1 proposals on packed bytes."""

from __future__ import annotations

import bisect
import hashlib
import json
import random
import re
from dataclasses import dataclass, replace

from .native import NativeContractError, QuantizedAssets

OUTSIDE_LAYERS = (
    "img_in",
    "txt_in",
    "time_text_embed.timestep_embedder.linear_1",
    "time_text_embed.timestep_embedder.linear_2",
    "norm_out.linear",
    "proj_out",
)
# Signed E2M1 codes in numerical order. Negative zero 8 shares zero's index.
ORDERED_CODES = (15, 14, 13, 12, 11, 10, 9, 0, 1, 2, 3, 4, 5, 6, 7)
CODE_RANK = {code: rank for rank, code in enumerate(ORDERED_CODES)} | {8: 7}
PROTOCOL = "mage-adjacent-packed/v1"


def _hash(value):
    from ..mage_qad_checkpoint import content_hash

    return content_hash(value)


@dataclass(frozen=True)
class LayerGroup:
    group_id: str
    layers: tuple[str, ...]
    weight_count: int


@dataclass(frozen=True)
class CandidateProposal:
    assets: QuantizedAssets | None
    receipt: dict


def build_layer_groups(assets: QuantizedAssets) -> tuple[LayerGroup, ...]:
    """Twelve complete 14-Linear blocks, followed by all six outside layers."""
    assets.verify()
    grouped = [[] for _ in range(13)]
    for name in assets.layers:
        match = re.fullmatch(r"transformer_blocks\.(\d+)\.(.+)", name)
        if match and 0 <= int(match[1]) < 12:
            grouped[int(match[1])].append(name)
        elif name in OUTSIDE_LAYERS:
            grouped[12].append(name)
        else:
            raise NativeContractError(f"layer-group: unexpected layer {name}")
    if [len(g) for g in grouped] != [14] * 12 + [6]:
        raise NativeContractError("layer-group: expected 12x14 + 6 complete coverage")
    return tuple(
        LayerGroup(
            f"block_{i:02d}" if i < 12 else "outside",
            tuple(sorted(names)),
            sum(assets.layers[n].shape[0] * assets.layers[n].shape[1] for n in names),
        )
        for i, names in enumerate(grouped)
    )


def _valid_blocks(layer, start, stop):
    import torch

    effective = layer.block_scale.reshape(-1)[start:stop].float() * layer.tensor_scale.float()
    if not torch.isfinite(effective).all():
        raise NativeContractError("effective-scale: nonfinite scale product")
    return effective > 0


def _movable_chunks(assets, group):
    """Index scale chunks, not all individual weights; 16 coordinates per block."""
    chunks, cumulative = [], 0
    for name in group.layers:
        scales = assets.layers[name].block_scale.reshape(-1)
        for start in range(0, scales.numel(), 65536):
            valid = _valid_blocks(assets.layers[name], start, start + 65536)
            count = int(valid.sum()) * 16
            if count:
                cumulative += count
                chunks.append((cumulative, name, start))
    return chunks, cumulative


def _coordinates(assets, chunks, ranks):
    ends = [chunk[0] for chunk in chunks]
    cache = {}
    result = []
    for rank in ranks:
        chunk_index = bisect.bisect_right(ends, rank)
        _, name, block_start = chunks[chunk_index]
        local = rank - (ends[chunk_index - 1] if chunk_index else 0)
        if chunk_index not in cache:
            valid = _valid_blocks(assets.layers[name], block_start, block_start + 65536)
            cache[chunk_index] = valid.nonzero().reshape(-1).tolist()
        block = block_start + cache[chunk_index][local // 16]
        result.append((name, block * 16 + local % 16))
    return result


def _new_code(code, rng):
    rank = CODE_RANK[code]
    if rank == 0:
        rank = 1
    elif rank == len(ORDERED_CODES) - 1:
        rank -= 1
    else:
        rank += -1 if rng.getrandbits(1) == 0 else 1
    return ORDERED_CODES[rank]


def _apply_edits(center, edits, group):
    layers = dict(center.layers)
    changed = {}
    seen = set()
    for edit in edits:
        name, coordinate = edit["layer"], edit["flat_index"]
        if name not in group.layers or not isinstance(coordinate, int):
            raise NativeContractError("candidate-edit: invalid layer or coordinate")
        layer = center.layers[name]
        if not 0 <= coordinate < layer.shape[0] * layer.shape[1] or (name, coordinate) in seen:
            raise NativeContractError("candidate-edit: out-of-range or duplicate coordinate")
        seen.add((name, coordinate))
        if not _valid_blocks(layer, coordinate // 16, coordinate // 16 + 1).item():
            raise NativeContractError("candidate-edit: zero effective scale")
        old, new = edit["old_code"], edit["new_code"]
        if (
            old not in CODE_RANK
            or new not in CODE_RANK
            or new == 8
            or abs(CODE_RANK[old] - CODE_RANK[new]) != 1
        ):
            raise NativeContractError("candidate-edit: not a canonical adjacent numerical move")
        original_byte = int(layer.qdata.reshape(-1)[coordinate // 2])
        shift = 4 * (coordinate % 2)
        if (original_byte >> shift) & 15 != old:
            raise NativeContractError("candidate-edit: old code differs from frozen center")
        if name not in changed:
            changed[name] = layer.qdata.clone()
        flat = changed[name].reshape(-1)
        value = int(flat[coordinate // 2])
        flat[coordinate // 2] = (value & ~(15 << shift)) | (new << shift)
    for name, qdata in changed.items():
        layers[name] = replace(center.layers[name], qdata=qdata)
    record = {
        "packed": {n: v.record() for n, v in layers.items()},
        "remainder": dict(center.remainder),
    }
    result = replace(center, layers=layers, sha256=_hash(record))
    result.verify()
    return result


def propose_candidate(
    center: QuantizedAssets,
    groups,
    *,
    group_id: str,
    mutation_count: int,
    seed: int,
    candidate_index: int,
    max_attempts: int = 8,
    seen_candidate_sha256=(),
) -> CandidateProposal:
    """Uniform legal coordinates without replacement; bounded duplicate retries.

    ``mutation_count`` is an explicit count, never inferred from a density.
    A tiny engineering smoke count is not a calibrated scientific search size.
    Explicit edits are the authoritative replay; seed and index allow repeatable
    proposal generation in the frozen Python environment. Failed proposals do
    not count as effective candidates and still advance the returned index.
    """
    expected = build_layer_groups(center)
    if tuple(groups) != expected:
        raise NativeContractError("layer-group: groups differ from complete center census")
    if (
        not isinstance(mutation_count, int)
        or mutation_count < 0
        or not isinstance(seed, int)
        or not isinstance(candidate_index, int)
        or candidate_index < 0
        or not isinstance(max_attempts, int)
        or max_attempts < 1
    ):
        raise NativeContractError("candidate-request: invalid count, seed, index or retry bound")
    matching = [g for g in groups if g.group_id == group_id]
    if len(matching) != 1:
        raise NativeContractError("candidate-group: unknown group")
    group = matching[0]
    receipt = {
        "protocol": PROTOCOL,
        "base_sha256": center.sha256,
        "fixed_sha256": center.fixed_sha256,
        "group_id": group_id,
        "requested_changes": mutation_count,
        "seed": seed,
        "candidate_index": candidate_index,
        "next_candidate_index": candidate_index + 1,
        "max_attempts": max_attempts,
        "group_weight_count": group.weight_count,
        "attempts": [],
        "effective_candidates": 0,
    }
    if mutation_count == 0:
        return CandidateProposal(None, {**receipt, "status": "no_op", "actual_changes": 0})
    chunks, movable = _movable_chunks(center, group)
    receipt.update(movable_coordinates=movable, zero_scale_coordinates=group.weight_count - movable)
    if not movable or mutation_count > movable:
        status = "no_movable_coordinates" if not movable else "insufficient_movable_coordinates"
        return CandidateProposal(None, {**receipt, "status": status, "actual_changes": 0})
    seen = set(seen_candidate_sha256) | {center.sha256}
    for attempt in range(max_attempts):
        seed_material = json.dumps(
            [PROTOCOL, seed, candidate_index, group_id, attempt], separators=(",", ":")
        )
        rng = random.Random(int.from_bytes(hashlib.sha256(seed_material.encode()).digest(), "big"))
        coordinates = _coordinates(center, chunks, rng.sample(range(movable), mutation_count))
        edits = []
        for name, coordinate in coordinates:
            byte = int(center.layers[name].qdata.reshape(-1)[coordinate // 2])
            old = (byte >> (4 * (coordinate % 2))) & 15
            edits.append(
                {
                    "layer": name,
                    "flat_index": coordinate,
                    "old_code": old,
                    "new_code": _new_code(old, rng),
                }
            )
        edits.sort(key=lambda e: (e["layer"], e["flat_index"]))
        candidate = _apply_edits(center, edits, group)
        duplicate = candidate.sha256 in seen
        receipt["attempts"].append(
            {
                "attempt": attempt,
                "candidate_sha256": candidate.sha256,
                "status": "duplicate" if duplicate else "candidate",
            }
        )
        if duplicate:
            continue
        return CandidateProposal(
            candidate,
            {
                **receipt,
                "status": "candidate",
                "edits": edits,
                "candidate_sha256": candidate.sha256,
                "actual_changes": len(edits),
                "effective_candidates": 1,
            },
        )
    return CandidateProposal(
        None, {**receipt, "status": "duplicate_exhausted", "actual_changes": 0}
    )


def replay_candidate(center: QuantizedAssets, receipt: dict) -> QuantizedAssets:
    """Apply audited edits to exactly their frozen center, without resampling."""
    groups = build_layer_groups(center)
    if (
        receipt.get("protocol") != PROTOCOL
        or receipt.get("status") != "candidate"
        or receipt.get("base_sha256") != center.sha256
        or receipt.get("fixed_sha256") != center.fixed_sha256
    ):
        raise NativeContractError("candidate-replay: incompatible receipt or center")
    matching = [g for g in groups if g.group_id == receipt.get("group_id")]
    if len(matching) != 1 or not receipt.get("edits"):
        raise NativeContractError("candidate-replay: missing group or edits")
    if receipt.get("actual_changes") != len(receipt["edits"]):
        raise NativeContractError("candidate-replay: change count differs")
    result = _apply_edits(center, receipt["edits"], matching[0])
    if result.sha256 != receipt.get("candidate_sha256"):
        raise NativeContractError("candidate-replay: resulting identity differs")
    return result


def audit_movable_coordinates(assets: QuantizedAssets) -> dict:
    """CPU-only real-payload census; does not load models or read prompt data."""
    rows = []
    for group in build_layer_groups(assets):
        _, movable = _movable_chunks(assets, group)
        rows.append(
            {
                "group_id": group.group_id,
                "layers": list(group.layers),
                "weight_count": group.weight_count,
                "movable_coordinates": movable,
                "zero_scale_coordinates": group.weight_count - movable,
            }
        )
    return {"center_sha256": assets.sha256, "fixed_sha256": assets.fixed_sha256, "groups": rows}


def audit_payload_difference(
    center: QuantizedAssets, candidate: QuantizedAssets, *, max_edits=10000
) -> dict:
    """Compare every packed byte independently of the proposal's declared edits.

    Intended for small engineering artifacts loaded from saved checkpoints. The
    bounded record limit fails explicitly; it never silently truncates proof.
    No full matrix of decoded weights is created.
    """
    import torch

    center.verify()
    candidate.verify()
    if center.fixed_sha256 != candidate.fixed_sha256 or set(center.layers) != set(candidate.layers):
        raise NativeContractError("packed-diff: fixed state or layer census changed")
    edits, changed_bytes, numerical_changes = [], 0, 0
    changed_layers = []
    for name in sorted(center.layers):
        a, b = center.layers[name], candidate.layers[name]
        if a.shape != b.shape or a.qdata.shape != b.qdata.shape or a.qdata.dtype != b.qdata.dtype:
            raise NativeContractError(f"packed-diff-layout: {name}")
        if torch.equal(a.qdata, b.qdata):
            continue
        changed_layers.append(name)
        old, new = a.qdata.reshape(-1), b.qdata.reshape(-1)
        for start in range(0, old.numel(), 65536):
            indices = (
                (old[start : start + 65536] != new[start : start + 65536])
                .nonzero()
                .reshape(-1)
                .tolist()
            )
            changed_bytes += len(indices)
            for local in indices:
                position = start + local
                old_byte, new_byte = int(old[position]), int(new[position])
                for nibble in (0, 1):
                    old_code, new_code = (
                        (old_byte >> (nibble * 4)) & 15,
                        (new_byte >> (nibble * 4)) & 15,
                    )
                    if old_code == new_code:
                        continue
                    coordinate = position * 2 + nibble
                    edits.append(
                        {
                            "layer": name,
                            "flat_index": coordinate,
                            "old_code": old_code,
                            "new_code": new_code,
                        }
                    )
                    if len(edits) > max_edits:
                        raise NativeContractError("packed-diff: edit record limit exceeded")
                    if (
                        CODE_RANK[old_code] != CODE_RANK[new_code]
                        and _valid_blocks(a, coordinate // 16, coordinate // 16 + 1).item()
                    ):
                        numerical_changes += 1
    return {
        "base_sha256": center.sha256,
        "candidate_sha256": candidate.sha256,
        "fixed_state_equal": True,
        "fixed_sha256": center.fixed_sha256,
        "layers_compared": len(center.layers),
        "changed_layers": changed_layers,
        "changed_packed_bytes": changed_bytes,
        "code_changes": len(edits),
        "numerical_changes": numerical_changes,
        "edits": edits,
    }
