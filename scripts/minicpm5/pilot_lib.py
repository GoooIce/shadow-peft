#!/usr/bin/env python3
"""CPU-only logic for the ticket-14 pilot loop (MiniCPM5-2B NVFP4 W4A4).

Ticket 09 froze the first updater:

* perturbations live in **rank coordinates** (rank ``+/-1`` sparse moves on the
  E2M1 grid), one layer group per generation, 13 groups in rotation;
* aggregation happens in the **realised displacement** space, with the rank
  shape reward ``1 - 2*rank/(M-1)`` and mirror pairs dropped as a whole when
  either side fails the non-inferiority gate;
* the residual is numeric, ``e = u - delta_realised``, clamped at +/-1.5, kept
  at ``gamma = 1.0`` (full residual, FP32) in the first round;
* ``Theta = D(q, scales) + e``.

This module owns the arithmetic and the bookkeeping, so the GPU runner stays a
thin driver.  **Code coordinates are expressed in the checkpoint layout**
(the 294 quantised ``Linear`` layers), because that is the layout of the start
payload, of the packet-byte census and of the offline kernel export; the
fused-engine layout is reached through :func:`engine_layout`.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROTOCOL = "minicpm5-rank-mirror/v1"
GROUP_PROTOCOL = "minicpm5-13-groups/v1"
RESIDUAL_COORDS = "code-value/e2m1/fp32"
CLAMP = 1.5
GAMMA = 1.0
BLOCK = 16

# --------------------------------------------------------------------------- #
# E2M1 grid
# --------------------------------------------------------------------------- #

E2M1_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
#: signed codes in numerical order; negative zero (8) shares zero's index
ORDERED_CODES = (15, 14, 13, 12, 11, 10, 9, 0, 1, 2, 3, 4, 5, 6, 7)
CODE_RANK = {code: rank for rank, code in enumerate(ORDERED_CODES)} | {8: 7}
VALUE = np.asarray(E2M1_VALUES, dtype=np.float64)
MAX_VALUE = 6.0


def canonical_code(code: int) -> int:
    """Fold negative zero onto code 0 (ticket 08: ``e2m1_encode`` can emit 8)."""
    return 0 if int(code) == 8 else int(code)


def rank_of(code: int) -> int:
    return CODE_RANK[canonical_code(code)]


def code_at_rank(rank: int) -> int:
    return ORDERED_CODES[rank]


def step_code(code: int, sign: int) -> int:
    """Rank move ``+/-1`` with endpoint clamping (never emits code 8)."""
    rank = rank_of(code)
    if rank == 0:
        rank = 1
    elif rank == len(ORDERED_CODES) - 1:
        rank -= 1
    else:
        rank += -1 if sign < 0 else 1
    return ORDERED_CODES[rank]


def gap(code_a: int, code_b: int) -> float:
    return float(VALUE[canonical_code(code_b)] - VALUE[canonical_code(code_a)])


def project(value: float) -> tuple[int, bool]:
    """Nearest canonical grid code.  Returns ``(code, in_range)``.

    ``in_range`` is False when the value falls beyond the grid endpoints, which
    ticket 08's source audit turns into a whole-step rejection.
    """
    if value > MAX_VALUE or value < -MAX_VALUE:
        return 0, False
    mag = abs(value)
    best_code, best_dist = 0, None
    for code in range(16):
        if code == 8:
            continue
        dist = abs(VALUE[code] - mag)
        if best_dist is None or dist < best_dist - 1e-12:
            best_code, best_dist = code, dist
    return canonical_code(best_code if value >= 0 else best_code + 8), True


# --------------------------------------------------------------------------- #
# layer groups
# --------------------------------------------------------------------------- #

LINEAR_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def block_of(layer: str) -> int | None:
    match = re.search(r"layers\.(\d+)\.", layer)
    return int(match.group(1)) if match else None


def leaf_of(layer: str) -> str:
    return layer.rsplit(".", 1)[-1]


@dataclass(frozen=True)
class LayerGroup:
    group_id: str
    layers: tuple[str, ...]
    coordinate_count: int

    def as_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "layers": list(self.layers),
            "coordinate_count": self.coordinate_count,
        }


def build_groups(layers: list[str], blocks_per_group: int = 3) -> tuple[LayerGroup, ...]:
    """13 complete-coverage groups: 12 uniform bands plus a remainder group.

    The Mage implementation queue used ``12 uniform blocks + 1 outside`` = 13
    groups (see ``mage_w4_search.mutation.build_layer_groups``); MiniCPM5-2B has
    42 decoder blocks and 7 quantised linears per block, so the same rule gives
    12 bands of 3 blocks plus a tail band of the remaining 6 blocks.
    """
    blocks = sorted({block_of(layer) for layer in layers})
    if any(block is None for block in blocks):
        raise ValueError("layer names do not carry a decoder-block index")
    groups: list[LayerGroup] = []
    cursor = 0
    # 12 uniform bands plus one remainder band == the 13-group rotation of the
    # Mage implementation queue (12 blocks + 1 outside group).
    while cursor < len(blocks) and len(groups) < 12:
        band = set(blocks[cursor : cursor + blocks_per_group])
        cursor += len(band)
        members = tuple(
            layer
            for layer in sorted(layers, key=lambda name: (block_of(name), leaf_of(name)))
            if block_of(layer) in band
        )
        groups.append(LayerGroup(f"band_{len(groups):02d}", members, 0))
    remainder = set(blocks[cursor:])
    if remainder:
        members = tuple(
            layer
            for layer in sorted(layers, key=lambda name: (block_of(name), leaf_of(name)))
            if block_of(layer) in remainder
        )
        groups.append(LayerGroup(f"band_{len(groups):02d}", members, 0))
    if len(groups) != 13:
        raise ValueError(f"expected 13 groups, derived {len(groups)}")
    return tuple(groups)


def coordinate_count(layers: list[str], shapes: dict[str, tuple[int, int]]) -> int:
    return int(sum(shapes[layer][0] * shapes[layer][1] for layer in layers))


# --------------------------------------------------------------------------- #
# payload access (checkpoint layout -> engine layout)
# --------------------------------------------------------------------------- #

FUSED_ORDER = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}


def engine_layout(
    layers: list[str], shapes: dict[str, tuple[int, int]] | None = None
) -> dict[str, tuple[str, int]]:
    """Map ``layer -> (engine module, **row** offset)`` for the fused kernel layout.

    The engine concatenates the shards **row-wise** (``build_kernel_payload``
    does ``torch.cat(parts, dim=0)``, exactly as
    ``process_weights_after_loading`` does), so a shard's offset is the number
    of rows of the shards before it: q/k/v -> 0 / 2048 / 2304 and gate/up ->
    0 / 6144 on MiniCPM5-2B.

    Ticket 18 (2026-09-10) found that this function used to return a
    *layer-count* offset (0/1/2) while every caller treated it as a row offset.
    That silently wrote each ``k_proj`` / ``v_proj`` / ``up_proj`` edit into the
    **first** shard's rows of the fused engine weight (and made the byte-level
    hard gate fail on the fused modules for reasons no readback could see,
    because the readback used the same formula).  ``shapes`` is therefore
    required for fused parents: a caller that cannot supply it gets a loud
    failure instead of a silently wrong mapping.
    """
    mapping: dict[str, tuple[str, int]] = {}
    by_parent: dict[str, list[str]] = {}
    for layer in layers:
        parts = layer.split(".")
        by_parent.setdefault(".".join(parts[:-1]), []).append(layer)
    for parent, members in by_parent.items():
        leaves: dict[str, list[str]] = {}
        for layer in members:
            leaves.setdefault(leaf_of(layer), []).append(layer)
        for engine_leaf, order in FUSED_ORDER.items():
            present = [leaf for leaf in order if leaf in leaves]
            if not present:
                continue
            if len(present) > 1 and shapes is None:
                raise ValueError(
                    f"{parent}.{engine_leaf} is a fused module and needs shapes "
                    "to compute row offsets"
                )
            offset = 0
            for leaf in present:
                for layer in leaves[leaf]:
                    mapping[layer] = (f"{parent}.{engine_leaf}", offset)
                    offset += int(shapes[layer][0]) if shapes is not None else 1
        for leaf in ("o_proj", "down_proj"):
            for layer in leaves.get(leaf, []):
                mapping[layer] = (layer, 0)
    return mapping


class PayloadReader:
    """Lazy nibble/block-scale reader for one compressed-tensors payload."""

    def __init__(self, payload: Path, fused_names: dict[str, str] | None = None):
        from safetensors import safe_open

        self.payload = Path(payload)
        self._handle = safe_open(self.payload / "model.safetensors", framework="pt", device="cpu")
        self._keys = set(self._handle.keys())
        self.fused_names = fused_names or {}
        self._codes: dict[str, np.ndarray] = {}
        self._scales: dict[str, np.ndarray] = {}
        self._raw: dict[str, np.ndarray] = {}

    def close(self) -> None:
        self._handle.__exit__(None, None, None)

    def release_codes(self) -> None:
        """Drop the decoded-code cache (the engine core is forked from us)."""
        self._codes.clear()
        self._raw.clear()

    def raw_code(self, layer: str, flat: int) -> int:
        """Un-canonicalised nibble straight from the payload (keeps code 8).

        Reverting an edited coordinate to the centre has to restore the *raw*
        byte, otherwise code 8 (negative zero, numerically equal to code 0)
        gets rewritten to 0 and the byte-level hard gate reports a mismatch
        that is numerically empty.
        """
        if layer not in self._raw:
            self._raw[layer] = self._handle.get_tensor(f"{layer}.weight_packed").numpy().reshape(-1)
        byte = int(self._raw[layer][int(flat) // 2])
        return (byte >> (4 * (int(flat) % 2))) & 0x0F

    def has(self, name: str) -> bool:
        return f"{name}.weight_packed" in self._keys

    def tensors(self) -> dict[str, np.ndarray]:
        """All non-quantised tensors (config-independent BF16 side)."""
        out = {}
        for key in sorted(self._keys):
            if key.endswith((".weight_packed", ".weight_scale", ".weight_global_scale", ".input_global_scale")):
                continue
            out[key] = self._handle.get_tensor(key).numpy()
        return out

    def codes(self, layer: str) -> np.ndarray:
        """Canonical E2M1 codes, shape ``[out, in]`` (uint8)."""
        if layer not in self._codes:
            packed = self._handle.get_tensor(f"{layer}.weight_packed").numpy()
            low = packed & 0x0F
            high = packed >> 4
            codes = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
            codes[:, 0::2] = low
            codes[:, 1::2] = high
            codes[codes == 8] = 0
            self._codes[layer] = codes
        return self._codes[layer]

    def block_scales(self, layer: str) -> np.ndarray:
        """E4M3 block scales as float32, shape ``[out, in//16]``."""
        if layer not in self._scales:
            scale = self._handle.get_tensor(f"{layer}.weight_scale").float().numpy()
            self._scales[layer] = scale
        return self._scales[layer]

    def global_scale(self, layer: str) -> float:
        stored = float(self._handle.get_tensor(f"{layer}.weight_global_scale").float().reshape(-1)[0])
        return 1.0 / stored if stored else float("nan")

    def shape(self, layer: str) -> tuple[int, int]:
        packed = self._handle.get_tensor(f"{layer}.weight_packed")
        return int(packed.shape[0]), int(packed.shape[1] * 2)


# --------------------------------------------------------------------------- #
# centre state
# --------------------------------------------------------------------------- #


@dataclass
class CentreState:
    """Sparse code delta against q0 plus the FP32 residual state."""

    generation: int = 0
    codes: dict[str, dict[int, int]] = field(default_factory=dict)   # layer -> {flat: code}
    residual: dict[str, dict[int, float]] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def code_at(self, layer: str, flat: int, base: int) -> int:
        return self.codes.get(layer, {}).get(flat, base)

    def set_code(self, layer: str, flat: int, code: int, base: int) -> None:
        canonical = canonical_code(code)
        bucket = self.codes.setdefault(layer, {})
        if canonical == canonical_code(base):
            bucket.pop(flat, None)
        else:
            bucket[flat] = canonical

    def residual_at(self, layer: str, flat: int) -> float:
        return self.residual.get(layer, {}).get(flat, 0.0)

    def set_residual(self, layer: str, flat: int, value: float) -> None:
        bucket = self.residual.setdefault(layer, {})
        if abs(value) < 1e-12:
            bucket.pop(flat, None)
        else:
            bucket[flat] = float(value)

    def delta_map(self) -> dict[str, dict[str, int]]:
        return {layer: {str(flat): code for flat, code in sorted(bucket.items())}
                for layer, bucket in sorted(self.codes.items()) if bucket}

    def residual_map(self) -> dict[str, dict[str, float]]:
        return {layer: {str(flat): round(value, 6) for flat, value in sorted(bucket.items())}
                for layer, bucket in sorted(self.residual.items()) if bucket}

    def sha256(self, base_hash_of) -> str:
        """Identity of the centre: per-layer base hash + sparse delta.

        ``base_hash_of`` is a ``layer -> short hash`` callable (``dict`` also
        works) so the runner can keep the payload code cache lazy.
        """
        lookup = base_hash_of if callable(base_hash_of) else base_hash_of.__getitem__
        digest = hashlib.sha256()
        digest.update(PROTOCOL.encode())
        for layer in sorted(self.codes):
            digest.update(layer.encode())
            digest.update(str(lookup(layer)).encode())
            bucket = self.codes[layer]
            for flat in sorted(bucket):
                digest.update(f"{flat}:{bucket[flat]}".encode())
        return digest.hexdigest()

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "protocol": PROTOCOL,
            "residual_coordinates": RESIDUAL_COORDS,
            "clamp": CLAMP,
            "gamma": GAMMA,
            "codes": self.delta_map(),
            "residual": self.residual_map(),
        }

    @classmethod
    def from_dict(cls, blob: dict) -> "CentreState":
        state = cls(generation=int(blob.get("generation", 0)))
        for layer, bucket in (blob.get("codes") or {}).items():
            state.codes[layer] = {int(flat): canonical_code(code) for flat, code in bucket.items()}
        for layer, bucket in (blob.get("residual") or {}).items():
            state.residual[layer] = {int(flat): float(value) for flat, value in bucket.items()}
        return state


# --------------------------------------------------------------------------- #
# candidate generation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Candidate:
    index: int
    pair: int
    side: int
    group_id: str
    sign: int
    coordinates: tuple[tuple[str, int], ...]

    def edits(self, center: CentreState, reader: PayloadReader) -> list[dict]:
        edits = []
        for layer, flat in self.coordinates:
            old = center.code_at(layer, flat, int(reader.codes(layer).reshape(-1)[flat]))
            new = step_code(old, self.sign)
            edits.append(
                {
                    "layer": layer,
                    "flat": int(flat),
                    "old_code": int(old),
                    "new_code": int(new),
                    "gap": gap(old, new),
                }
            )
        return edits


def group_census(
    groups: tuple[LayerGroup, ...], reader: PayloadReader, block_min: float = 0.0
) -> tuple[dict[str, np.ndarray], dict]:
    """Per-layer movable-block masks plus a group-level census."""
    masks: dict[str, np.ndarray] = {}
    report = {"groups": [], "totals": {"coordinates": 0, "movable": 0, "zero_scale": 0}}
    for group in groups:
        movable = zero = 0
        for layer in group.layers:
            mask = reader.block_scales(layer) > block_min
            masks[layer] = mask
            movable += int(mask.sum()) * BLOCK
            zero += int((~mask).sum()) * BLOCK
        report["groups"].append(
            {
                "group_id": group.group_id,
                "layers": len(group.layers),
                "coordinates": int(
                    sum(reader.shape(layer)[0] * reader.shape(layer)[1] for layer in group.layers)
                ),
                "movable": movable,
                "zero_scale": zero,
            }
        )
        report["totals"]["coordinates"] += report["groups"][-1]["coordinates"]
        report["totals"]["movable"] += movable
        report["totals"]["zero_scale"] += zero
    return masks, report


def _uniform_coordinates(
    group: LayerGroup,
    masks: dict[str, np.ndarray],
    rng: random.Random,
    count: int,
) -> list[tuple[str, int]]:
    """Uniform sample (without replacement) over the group's movable coordinates."""
    per_layer = []
    total = 0
    for layer in group.layers:
        movable = int(masks[layer].sum())
        if not movable:
            continue
        per_layer.append((layer, np.flatnonzero(masks[layer].reshape(-1)), movable))
        total += movable * BLOCK
    if count > total:
        raise ValueError(f"group {group.group_id}: requested {count} > movable {total}")
    chosen = sorted(rng.sample(range(total), count))
    ends = []
    running = 0
    for layer, _indices, movable in per_layer:
        running += movable * BLOCK
        ends.append(running)
    result: list[tuple[str, int]] = []
    for position in chosen:
        for (layer, indices, _movable), end in zip(per_layer, ends):
            start = end - len(indices) * BLOCK
            if start <= position < end:
                local = position - start
                block = int(indices[local // BLOCK])
                result.append((layer, block * BLOCK + local % BLOCK))
                break
    return result


def propose_generation(
    groups: tuple[LayerGroup, ...],
    masks: dict[str, np.ndarray],
    generation: int,
    group_index: int,
    pairs: int,
    edits_per_candidate: int,
    seed: int,
    base_coordinate: int = 0,
) -> list[Candidate]:
    """Deterministic mirror-pair population for one generation."""
    group = groups[group_index % len(groups)]
    population: list[Candidate] = []
    for pair in range(pairs):
        material = json.dumps(
            [PROTOCOL, seed, generation, group.group_id, pair], separators=(",", ":")
        )
        rng = random.Random(int.from_bytes(hashlib.sha256(material.encode()).digest(), "big"))
        coordinates = tuple(_uniform_coordinates(group, masks, rng, edits_per_candidate))
        for side, sign in ((0, 1), (1, -1)):
            population.append(
                Candidate(
                    index=base_coordinate + len(population),
                    pair=pair,
                    side=side,
                    group_id=group.group_id,
                    sign=sign,
                    coordinates=coordinates,
                )
            )
    return population


# --------------------------------------------------------------------------- #
# aggregation -> centre step -> residual
# --------------------------------------------------------------------------- #


def rank_shape_reward(values: list[float]) -> list[float]:
    """``1 - 2*rank/(M-1)`` over the descending ordering (ties by index)."""
    m = len(values)
    if m == 1:
        return [0.0]
    order = sorted(range(m), key=lambda i: (-values[i], i))
    reward = [0.0] * m
    for rank, index in enumerate(order):
        reward[index] = 1.0 - 2.0 * rank / (m - 1)
    return reward


def aggregate(
    pairs: list[dict],
    realized: list[dict[tuple[str, int], float]],
    tau: float,
) -> dict:
    """Mirror-pair aggregation in the realised-displacement space.

    ``pairs`` carry ``fitness`` for both sides plus the paired difference against
    the centre; a pair is dropped as a whole when either side breaches the
    non-inferiority gate (``delta_center >= -tau``).
    """
    qualified = []
    dropped = []
    for pair in pairs:
        plus, minus = pair["plus"], pair["minus"]
        ok = plus["delta_center"] >= -tau and minus["delta_center"] >= -tau
        (qualified if ok else dropped).append(pair)
    diffs = [pair["plus"]["fitness"] - pair["minus"]["fitness"] for pair in qualified]
    rewards = rank_shape_reward(diffs)
    gradient: dict[tuple[str, int], float] = {}
    weight_total = 0.0
    for pair, reward in zip(qualified, rewards):
        weight_total += 2.0 * abs(reward)
        for side, coefficient in (("plus", reward), ("minus", -reward)):
            for coordinate, value in realized[pair[side]["index"]].items():
                gradient[coordinate] = gradient.get(coordinate, 0.0) + coefficient * value
    if weight_total:
        gradient = {key: value / weight_total for key, value in gradient.items()}
    return {
        "gamma": GAMMA,
        "qualified_pairs": len(qualified),
        "dropped_pairs": [pair["pair"] for pair in dropped],
        "pair_diffs": diffs,
        "pair_rewards": rewards,
        "gradient": gradient,
        "weight_total": weight_total,
    }


def step_coordinates(center: CentreState, gradient: dict) -> set:
    """Exactly the coordinate set :func:`apply_step` will visit this generation.

    ``touched = support(gradient) | support(residual)``.  The retention snapshot
    has to cover this set *before* the step, so the set is computed here rather
    than duplicated at the call site; the unit test pins it to ``apply_step``'s
    own ``touched`` count.
    """
    return set(gradient) | {
        (layer, flat) for layer, bucket in center.residual.items() for flat in bucket
    }


def apply_step(
    center: CentreState,
    reader: PayloadReader,
    gradient: dict[tuple[str, int], float],
    alpha: float,
    gamma: float = GAMMA,
    clamp: float = CLAMP,
) -> dict:
    """Move the centre by ``u = alpha*g + gamma*e`` and book-keep the residual.

    Returns a per-coordinate ledger: rejection (beyond the grid endpoints),
    residual clamp hits and the realised displacement in code-value units.
    """
    touched = step_coordinates(center, gradient)
    ledger = {
        "alpha": alpha,
        "gamma": gamma,
        "touched": len(touched),
        "moved": 0,
        "rejected_beyond_grid": 0,
        "residual_clamped": 0,
        "unmoved_below_projection": 0,
        "realised_gaps": [],
        "moved_coordinates": [],
        "u_abs_max": 0.0,
        "residual_abs_max": 0.0,
    }
    for layer, flat in sorted(touched):
        base = int(reader.codes(layer).reshape(-1)[flat])
        old = center.code_at(layer, flat, base)
        residual = center.residual_at(layer, flat)
        u = alpha * gradient.get((layer, flat), 0.0) + gamma * residual
        ledger["u_abs_max"] = max(ledger["u_abs_max"], abs(u))
        theta = float(VALUE[old]) + u
        new_code, in_range = project(theta)
        if not in_range:
            # source-semantics: a step rounded beyond the endpoint is rejected
            # whole, and the float step stays in the residual
            ledger["rejected_beyond_grid"] += 1
            delta_realised = 0.0
            new_code = old
        else:
            delta_realised = float(VALUE[new_code] - VALUE[old])
            if new_code == old:
                ledger["unmoved_below_projection"] += 1
            else:
                ledger["moved"] += 1
                ledger["realised_gaps"].append(abs(delta_realised))
                ledger["moved_coordinates"].append(
                    {"layer": layer, "flat": int(flat), "old": int(old), "new": int(new_code)}
                )
        residual_next = u - delta_realised
        if abs(residual_next) > clamp:
            residual_next = clamp if residual_next > 0 else -clamp
            ledger["residual_clamped"] += 1
        center.set_code(layer, flat, new_code, base)
        center.set_residual(layer, flat, residual_next)
        ledger["residual_abs_max"] = max(ledger["residual_abs_max"], abs(residual_next))
    if ledger["realised_gaps"]:
        gaps = np.asarray(ledger["realised_gaps"], dtype=np.float64)
        ledger["realised_gap_mean"] = float(gaps.mean())
        ledger["realised_gap_median"] = float(np.median(gaps))
    else:
        ledger["realised_gap_mean"] = None
        ledger["realised_gap_median"] = None
    return ledger


def realized_displacement(edits: list[dict]) -> dict[tuple[str, int], float]:
    return {(edit["layer"], edit["flat"]): float(edit["gap"]) for edit in edits}


# --------------------------------------------------------------------------- #
# step snapshot / rollback (round 2: per-generation retention)
# --------------------------------------------------------------------------- #


def snapshot_layers(center: CentreState, coordinates) -> dict:
    """Exact, invertible snapshot of ``center`` over the layers a step will touch.

    Round 2's retention rule (the mechanism HarnessDev's 0.57-correlation
    revision loop corresponds to) may have to *undo* a centre step after it has
    already been applied.  Ticket 09 §4 froze the residual semantics for a
    direction that is rejected: *"acceptance rejection must re-anchor the
    residual"*, so that the direction **not** taken cannot come back through the
    residual in the next generation.  Undoing :func:`apply_step` is exactly that
    case, and the only faithful way to do it is to restore the per-coordinate
    pair ``(code, residual)`` that existed **before** the step.

    Two details separate "looks reverted" from "is reverted":

    * the residual of every coordinate the step touched must go back to its
      pre-step value -- ``e = u - delta_realised`` must not stay behind, or the
      rolled-back float step re-applies next generation through ``gamma*e``;
    * ``CentreState.codes`` / ``residual`` are ``layer -> bucket`` dicts whose
      **presence** is part of the centre identity (``CentreState.sha256``
      iterates ``self.codes`` *including* empty buckets -- ticket 14 §5.1), so
      the restore reproduces bucket presence as well: a rolled-back centre must
      be byte-identical to the centre that was there before, not merely equal in
      value.

    The snapshot is a plain JSON-dumpable dict so the rollback evidence lands in
    the per-generation record.
    """
    layers = sorted({layer for layer, _flat in coordinates})
    return {
        "layers": layers,
        "codes": {layer: {int(flat): int(code)
                          for flat, code in center.codes.get(layer, {}).items()}
                  for layer in layers},
        "codes_present": {layer: layer in center.codes for layer in layers},
        "residual": {layer: {int(flat): float(value)
                             for flat, value in center.residual.get(layer, {}).items()}
                     for layer in layers},
        "residual_present": {layer: layer in center.residual for layer in layers},
    }


def restore_layers(center: CentreState, snapshot: dict) -> dict:
    """Inverse of :func:`snapshot_layers`; returns the rollback ledger.

    ``residual_mismatches`` must be zero for a faithful re-anchoring; the
    retention path asserts on it (and the unit test checks the centre's
    ``to_dict()`` is byte-identical before the step and after the rollback).
    """
    ledger = {
        "layers": len(snapshot["layers"]),
        "codes_restored": 0,
        "codes_removed": 0,
        "residual_restored": 0,
        "residual_cleared": 0,
        "residual_mismatches": 0,
        "empty_buckets_pruned": 0,
        "per_layer": {},
    }
    for layer in snapshot["layers"]:
        before_codes = snapshot["codes"][layer]
        current_codes = dict(center.codes.get(layer, {}))
        for flat in sorted(set(current_codes) | set(before_codes)):
            if flat in before_codes:
                if current_codes.get(flat) != before_codes[flat]:
                    ledger["codes_restored"] += 1
            elif flat in current_codes:
                ledger["codes_removed"] += 1
        if snapshot["codes_present"][layer]:
            center.codes[layer] = dict(before_codes)
        else:
            # a layer key that did not exist must not come back as an empty bucket
            center.codes.pop(layer, None)
            if before_codes:  # defensive: presence flag disagreed with the data
                center.codes[layer] = dict(before_codes)
        if not center.codes.get(layer) and not snapshot["codes_present"][layer]:
            center.codes.pop(layer, None)
            ledger["empty_buckets_pruned"] += 1

        before_residual = snapshot["residual"][layer]
        current_residual = dict(center.residual.get(layer, {}))
        for flat in sorted(set(current_residual) | set(before_residual)):
            if flat in before_residual:
                if current_residual.get(flat) != before_residual[flat]:
                    ledger["residual_restored"] += 1
            elif flat in current_residual:
                ledger["residual_cleared"] += 1
        if snapshot["residual_present"][layer]:
            center.residual[layer] = dict(before_residual)
        else:
            center.residual.pop(layer, None)
        if not center.residual.get(layer) and not snapshot["residual_present"][layer]:
            center.residual.pop(layer, None)

        after_residual = center.residual.get(layer, {})
        mismatches = sum(1 for flat, value in before_residual.items()
                         if after_residual.get(flat) != value)
        mismatches += sum(1 for flat in after_residual if flat not in before_residual)
        ledger["residual_mismatches"] += mismatches
        ledger["per_layer"][layer] = {
            "codes_before": len(before_codes),
            "codes_after": len(center.codes.get(layer, {})),
            "residual_before": len(before_residual),
            "residual_after": len(after_residual),
            "residual_mismatches": mismatches,
        }
    return ledger


def residual_reanchor(step_ledger: dict | None, coordinates) -> dict:
    """Ticket 09's residual rule, restated for a step that was **not** taken.

    A rolled-back step contributes *nothing* to the residual: the attempted
    displacement ``u`` (both its ``alpha*g`` term and the ``gamma*e`` term it
    re-applied) is re-anchored to the pre-step state.  Otherwise the rejected
    direction would come back through ``gamma*e`` on the next generation -- the
    failure mode ticket 09 §4 names for acceptance rejections.  The arithmetic
    itself lives in :func:`restore_layers`; this helper writes the rule and the
    magnitudes into the evidence file.
    """
    step_ledger = step_ledger or {}
    return {
        "rule": ("ticket 09 §4: a step that is not taken must re-anchor the residual -- "
                 "restore (code, residual) per touched coordinate to the pre-step values, "
                 "so the rejected direction cannot return through gamma*e"),
        "coordinates": len(set(coordinates)) if coordinates else int(step_ledger.get("touched", 0)),
        "touched": int(step_ledger.get("touched", 0)),
        "u_abs_max": step_ledger.get("u_abs_max"),
        "residual_abs_max_after_step": step_ledger.get("residual_abs_max"),
        "rejected_beyond_grid": int(step_ledger.get("rejected_beyond_grid", 0)),
        "residual_clamped": int(step_ledger.get("residual_clamped", 0)),
    }


# --------------------------------------------------------------------------- #
# fitness helpers
# --------------------------------------------------------------------------- #


def episode_scores(rows: list[dict], key: str = "tf_correct") -> dict[str, list[float]]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        buckets.setdefault(row["episode"], []).append(1.0 if row[key] else 0.0)
    return {episode: values for episode, values in sorted(buckets.items())}


def paired_stats(left: dict[str, list[float]], right: dict[str, list[float]]) -> dict:
    """Episode-level paired difference summary (17's statistics unit)."""
    episodes = sorted(set(left) & set(right))
    diffs = [
        float(np.mean(left[episode]) - np.mean(right[episode])) for episode in episodes
    ]
    array = np.asarray(diffs, dtype=np.float64)
    return {
        "episodes": len(episodes),
        "mean": float(array.mean()) if array.size else None,
        "sd": float(array.std(ddof=1)) if array.size > 1 else None,
        "mean_abs": float(np.abs(array).mean()) if array.size else None,
        "zero_fraction": float((array == 0).mean()) if array.size else None,
    }


def fitness_summary(rows: list[dict]) -> dict:
    hits = [1.0 if row["tf_correct"] else 0.0 for row in rows]
    logprobs = [row["tf_gold_logprob"] for row in rows if row.get("tf_gold_logprob") is not None]
    acc = float(np.mean(hits)) if hits else None
    gold_prob = float(np.exp(np.mean(logprobs))) if logprobs else None
    episodes = episode_scores(rows)
    episode_acc = [float(np.mean(values)) for values in episodes.values()]
    return {
        "items": len(rows),
        "episodes": len(episodes),
        "acc": acc,
        "episode_acc": float(np.mean(episode_acc)) if episode_acc else None,
        "gold_prob": gold_prob,
        "gold_logprob_mean": float(np.mean(logprobs)) if logprobs else None,
        "im_end_rate": float(np.mean([1.0 if row.get("tf_next_is_im_end") else 0.0 for row in rows]))
        if rows
        else None,
    }


def json_dump(path: Path, blob) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
