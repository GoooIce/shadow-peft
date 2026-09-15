#!/usr/bin/env python3
"""QES -> NVFP4 arithmetic reference prototype (wayfinder ticket 08).

Pure CPU, pure stdlib, no model, no GPU.  Runs with a bare ``python3``
(verified on CPython 3.9.6 and 3.12) and prints every number cited by
``docs/research/qes-arithmetic-reference-20260909.md``.

What this file is
-----------------
A *cheap, re-computable arithmetic reference* for the seam between QES
(code-space evolution strategies) and NVFP4 (E2M1 nibbles + E4M3 block
scales).  It is deliberately not an ES implementation and not a trainer.

It implements, on one fixed single-group quantized vector:

* two perturbation/projection coordinates -- ``rank`` and ``value`` --
  each with the full ``perturb -> project -> residual bookkeeping`` chain;
* three feedback variants on the *same* update sequence:
  (A) no feedback ``e == 0``, (B) full residual (FP32, gate accounting),
  (C) finite-K replay (zero start, current boundary, per-step clamp +-1.5).
  B is the arithmetic oracle for C;
* the two NVFP4 counterexamples (raw code ``8 - 1`` jump; code 0 <-> 8
  equivalent rewrite);
* the two QES counterexamples (gamma=0.9 residual converges but never
  crosses the grid; a rejected direction returns through the residual);
* record-only replay: A/B/C are pure functions of a serialized
  ``UpdateRecord`` list plus a frozen ``ReplayProtocol`` -- no runtime
  config object is consulted.

Provenance of the constants
---------------------------
``E2M1_BOUNDS`` / ``E2M1_VALUES`` are copied verbatim from the read-only
main workspace ``src/shadow_lab/quasar.py:34-52``; ``ORDERED_CODES`` /
``CODE_RANK`` from ``src/shadow_lab/mage_w4_search/mutation.py:23-24``.
``encode_scaled`` is a pure-Python mirror of ``quasar.py:91-106``
``e2m1_encode`` (searchsorted + ties-to-even on the mantissa LSB, code 8
reachable for small negative inputs).  If torch is importable the script
cross-checks the mirror against the real primitive and reports the result;
otherwise the check is skipped and the pure-Python numbers stand alone.

Usage
-----
    python3 scripts/prototype_qes_arithmetic_reference.py
    python3 scripts/prototype_qes_arithmetic_reference.py --emit-record /tmp/qes_case.json
"""

from __future__ import annotations

import bisect
import functools
import hashlib
import json
import math
import random
import struct
import sys
from collections import deque
from dataclasses import asdict, dataclass

# --------------------------------------------------------------------------
# 0. constants (copied from the read-only main workspace; see module docstring)
# --------------------------------------------------------------------------

# src/shadow_lab/quasar.py:34
E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# src/shadow_lab/quasar.py:35-52 (note the two zeros at index 0 and 8)
E2M1_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
# src/shadow_lab/quasar.py:101-103 (midpoint indices 1/3/5 round up)
ROUND_UP_MASK = (0, 1, 0, 1, 0, 1, 0)
# src/shadow_lab/mage_w4_search/mutation.py:23-24 (negative zero 8 shares rank 7)
ORDERED_CODES = (15, 14, 13, 12, 11, 10, 9, 0, 1, 2, 3, 4, 5, 6, 7)
CODE_RANK = {code: rank for rank, code in enumerate(ORDERED_CODES)}
CODE_RANK[8] = 7

# Prototype protocol constants.  One reconstruction group, one fixed effective
# scale ``weight_scale * tensor_scale`` (the ticket's "fixed codebook, fixed
# scale"): every coordinate shares the same 15-value grid.
S_EFF = 1.0
DIM = 128
POPULATION = 8
GENERATIONS = 48
SIGMA_RANK = 0.5          # rank units
SIGMA_VALUE = 0.5 * S_EFF  # value units (no unique cross-coordinate conversion)
ALPHA = 1.0
GAMMA = 0.9
WINDOW_K = 8
RESIDUAL_CLAMP = 1.5      # native units, mirrors the source's clamp(-1.5, 1.5)
BASE_SEED = 20260909
LAYER_ID = "group0"
DRIFT_RATE = 0.02

RANK_VALUES = tuple(E2M1_VALUES[code] * S_EFF for code in ORDERED_CODES)
RANK_GAPS = tuple(
    RANK_VALUES[index + 1] - RANK_VALUES[index] for index in range(len(RANK_VALUES) - 1)
)


# --------------------------------------------------------------------------
# 1. projections: the two primitive operations the seam must reuse
# --------------------------------------------------------------------------

def encode_scaled(pre_scaled: float) -> int:
    """Pure-Python mirror of ``quasar.e2m1_encode`` for one pre-scaled value.

    ``pre_scaled`` is already divided by the effective scale, exactly like the
    tensor argument of the real primitive.  Ties-to-even on the mantissa LSB
    is reproduced through ``ROUND_UP_MASK``.  Code 8 is emitted for
    ``-0.25 <= pre_scaled < 0`` because ``sign`` is taken from the sign bit
    while the ordinal is 0.
    """

    sign = 8 if pre_scaled < 0 else 0
    magnitude = abs(pre_scaled)
    ordinal = bisect.bisect_left(E2M1_BOUNDS, magnitude)
    if ordinal > len(E2M1_BOUNDS):
        # torch.searchsorted returns 7 for magnitude > 5.0; codes 7 and 15
        # saturate at +-6.0.  Ordinal 7 is not a bound index, so no tie.
        ordinal = len(E2M1_BOUNDS)
    round_up = (
        ordinal < len(ROUND_UP_MASK)
        and magnitude == E2M1_BOUNDS[ordinal]
        and ROUND_UP_MASK[ordinal]
    )
    return sign + ordinal + (1 if round_up else 0)


def decode_value(code: int) -> float:
    """``D(q, scales)`` for the fixed effective scale."""

    return E2M1_VALUES[code & 0x0F] * S_EFF


def project_value(value: float) -> tuple[int, float]:
    """Nearest grid point with the repo's ties-to-even rule."""

    code = encode_scaled(value / S_EFF)
    return code, decode_value(code)


def round_half_even(value: float) -> int:
    """Deterministic ties-to-even integer rounding (Python ``round`` is float-y)."""

    floor = math.floor(value)
    fraction = value - floor
    if fraction > 0.5:
        return floor + 1
    if fraction < 0.5:
        return floor
    return floor if floor % 2 == 0 else floor + 1


def local_gaps(rank: int) -> tuple[float, float]:
    """(gap towards -inf, gap towards +inf) at one rank, in value units."""

    left = RANK_GAPS[rank - 1] if rank > 0 else float("inf")
    right = RANK_GAPS[rank] if rank < len(RANK_GAPS) else float("inf")
    return left, right


def half_gap(code: int) -> float:
    """Conservative half step at a code (min of the two adjacent half gaps)."""

    rank = CODE_RANK[code]
    left, right = local_gaps(rank)
    return 0.5 * min(left, right)


# --------------------------------------------------------------------------
# 2. the two coordinates
# --------------------------------------------------------------------------

class RankCoordinate:
    """Code-space arithmetic done on the *ordered rank* (0..14).

    rank == the paper's code index: perturbation +-1 rank step, aggregation in
    rank units, residual in rank units.  Every candidate is exactly a grid
    point, so no projection is needed at candidate evaluation time.  Value
    displacements are derived only for reporting (``value_delta``).
    """

    name = "rank"

    def native(self, code: int) -> float:
        return float(CODE_RANK[code])

    def value(self, code: int) -> float:
        return decode_value(code)

    def value_delta(self, code: int, delta_native: float) -> float:
        """Value-unit displacement of a rank step (exact, on-grid)."""

        new_code = ORDERED_CODES[CODE_RANK[code] + int(delta_native)]
        return decode_value(new_code) - decode_value(code)

    def sample_delta(self, native: float, sigma: float, epsilon: float) -> float:
        step = round_half_even(sigma * epsilon)
        return float(max(-1, min(1, step)))

    def gate_perturbation(self, native: float, delta: float) -> tuple[float, bool]:
        target = native + delta
        if target < 0.0 or target > len(ORDERED_CODES) - 1:
            return 0.0, True
        return delta, False

    def realize(self, code: int, delta: float) -> tuple[int, float]:
        new_code = ORDERED_CODES[CODE_RANK[code] + int(delta)]
        return new_code, float(int(delta))

    def update(self, code: int, u: float) -> tuple[int, float, bool]:
        native = self.native(code)
        target = native + u
        top = len(ORDERED_CODES) - 1
        new_native = round_half_even(target)
        if new_native < 0 or new_native > top:
            # QES wfp.py:363-372 rejects the *whole* rounded step; it never
            # clips it to the endpoint.  The rejected float step stays in e.
            return code, 0.0, True
        new_code = ORDERED_CODES[new_native]
        return new_code, float(new_native) - native, False


class ValueCoordinate:
    """Dequantized-numerical arithmetic (value units).

    Perturbation is a real number, so the candidate must be projected back to
    the grid before it can be evaluated; the *realized* displacement is what
    aggregation must use, otherwise phantom components appear.
    """

    name = "value"

    def native(self, code: int) -> float:
        return decode_value(code)

    def value(self, code: int) -> float:
        return decode_value(code)

    def value_delta(self, code: int, delta_native: float) -> float:
        """Identity: the value coordinate's native unit is the value unit."""

        return delta_native

    def sample_delta(self, native: float, sigma: float, epsilon: float) -> float:
        return sigma * epsilon

    def gate_perturbation(self, native: float, delta: float) -> tuple[float, bool]:
        limit = 6.0 * S_EFF
        if native + delta < -limit or native + delta > limit:
            return 0.0, True
        return delta, False

    def realize(self, code: int, delta: float) -> tuple[int, float]:
        candidate = decode_value(code) + delta
        new_code = encode_scaled(candidate / S_EFF)
        return new_code, decode_value(new_code) - decode_value(code)

    def update(self, code: int, u: float) -> tuple[int, float, bool]:
        base = decode_value(code)
        target = base + u
        limit = 6.0 * S_EFF
        if target < -limit or target > limit:
            # Analog of the source's code-range gate: a step that wants to
            # leave the representable range is rejected whole, not clipped.
            return code, 0.0, True
        new_code = encode_scaled(target / S_EFF)
        return new_code, decode_value(new_code) - base, False


def make_coordinate(name: str):
    if name == "rank":
        return RankCoordinate()
    if name == "value":
        return ValueCoordinate()
    raise ValueError(f"unknown coordinate {name!r}")


# --------------------------------------------------------------------------
# 3. record types (the only replay inputs)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ReplayProtocol:
    coordinate: str
    dim: int
    s_eff: float
    gamma: float
    window_k: int
    residual_clamp: float
    aggregation: str = "realized"   # "realized" | "unprojected" (value only)
    residual_dtype: str = "fp32"    # "fp32" | "fp16" (source stores FP16)
    layer_id: str = LAYER_ID


@dataclass(frozen=True)
class UpdateRecord:
    generation: int
    seeds: tuple[int, ...]
    coeffs: tuple[float, ...]
    sigma: float
    alpha: float
    realized_deltas: tuple[float, ...]


@dataclass(frozen=True)
class ReplayCase:
    protocol: ReplayProtocol
    initial_codes: tuple[int, ...]
    initial_residual: tuple[float, ...]
    records: tuple[UpdateRecord, ...]


def case_to_json(case: ReplayCase) -> str:
    payload = {
        "protocol": asdict(case.protocol),
        "initial_codes": list(case.initial_codes),
        "initial_residual": list(case.initial_residual),
        "records": [asdict(record) for record in case.records],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def case_from_json(text: str) -> ReplayCase:
    payload = json.loads(text)
    protocol = ReplayProtocol(**payload["protocol"])
    records = tuple(
        UpdateRecord(
            generation=item["generation"],
            seeds=tuple(item["seeds"]),
            coeffs=tuple(item["coeffs"]),
            sigma=item["sigma"],
            alpha=item["alpha"],
            realized_deltas=tuple(item["realized_deltas"]),
        )
        for item in payload["records"]
    )
    return ReplayCase(
        protocol=protocol,
        initial_codes=tuple(payload["initial_codes"]),
        initial_residual=tuple(payload["initial_residual"]),
        records=records,
    )


# --------------------------------------------------------------------------
# 4. noise: derived once for the reference run, stored as seeds in the record
# --------------------------------------------------------------------------

def noise_seed(base_seed: int, generation: int, pair_index: int, layer_id: str) -> int:
    """Mirror of ``es.py:172-174`` plus the ``layer_id`` from seam note 2.6.

    The record stores the *derived* seed, so replay never depends on this
    function or on any runtime config.
    """

    payload = f"{base_seed}\0{generation}\0{pair_index}\0{layer_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


@functools.lru_cache(maxsize=None)
def noise_vector(seed: int, dim: int) -> tuple[float, ...]:
    """CPU, pure-function noise, mirroring ``es.py:194-197`` ``replay_noise``.

    ``random.Random.gauss`` was verified bit-identical on CPython 3.9.6 and
    3.12 for the same seed.  Memoized so the three variants and the window
    replay consume the *same* numbers.
    """

    generator = random.Random(seed)
    return tuple(generator.gauss(0.0, 1.0) for _ in range(dim))


def zscore_coeffs(rewards: list[float], clip: float = 3.0) -> list[float]:
    """QES default shaping (``int4_perturb.py:648-671``): clip((r-mean)/std, +-3)."""

    count = len(rewards)
    mean = sum(rewards) / count
    variance = sum((reward - mean) ** 2 for reward in rewards) / count
    std = math.sqrt(variance)
    if std == 0.0:
        return [0.0] * count
    return [max(-clip, min(clip, (reward - mean) / std)) for reward in rewards]


# --------------------------------------------------------------------------
# 5. the shared step primitives
# --------------------------------------------------------------------------

@dataclass
class Step:
    generation: int
    g: list[float]
    u: list[float]
    delta_value: list[float]
    codes_before: list[int]
    codes_after: list[int]
    residual_native: list[float]
    residual_value: list[float]
    clamp_hits: int
    perturb_gate_flags: list[bool]
    update_gate_flags: list[bool]
    phantom_components: int


def aggregate(
    coordinate,
    codes: list[int],
    coeffs: tuple[float, ...],
    sigma: float,
    seeds: tuple[int, ...],
    aggregation: str,
) -> tuple[list[float], list[bool], int]:
    """Gradient estimate from the *same* seeds/coeffs against a given boundary.

    ``aggregation="realized"`` accumulates the post-projection value
    displacement (the recommendation); ``aggregation="unprojected"``
    accumulates the raw perturbation, which is only well defined for the
    value coordinate and is kept as the phantom-component diagnostic.
    """

    dim = len(codes)
    estimate = [0.0] * dim
    gate_flags = [False] * dim
    phantom = 0
    for coeff, seed in zip(coeffs, seeds):
        epsilon = noise_vector(seed, dim)
        for index in range(dim):
            code = codes[index]
            native = coordinate.native(code)
            delta = coordinate.sample_delta(native, sigma, epsilon[index])
            delta, gated = coordinate.gate_perturbation(native, delta)
            if gated:
                gate_flags[index] = True
                continue
            if aggregation == "unprojected":
                estimate[index] += coeff * delta
                continue
            _, realized = coordinate.realize(code, delta)
            if realized == 0.0 and delta != 0.0:
                phantom += 1
            estimate[index] += coeff * realized
    count = len(coeffs)
    return [value / count for value in estimate], gate_flags, phantom


def float16_roundtrip(value: float) -> float:
    """Round a float through IEEE binary16, as the source stores residuals.

    ``wfp.py:380-384`` casts the residual to ``torch.float16`` before keeping
    it, so the *oracle* carries less precision than the FP32 replay path.
    """

    return struct.unpack("e", struct.pack("e", value))[0]


def apply_update(
    coordinate,
    codes: list[int],
    u: list[float],
    clamp_native: float | None,
    residual_enabled: bool,
    residual_dtype: str = "fp32",
) -> tuple[list[int], list[float], list[float], list[bool], int]:
    """Project the virtual update, write the code, book the residual.

    Returns ``(new_codes, delta_value, residual_native, update_gate_flags,
    clamp_hits)``.  The update gate rejects the *whole* rounded step when it
    would leave the representable range (QES ``wfp.py:363-372``); the clamp is
    the source's per-step ``clamp(+-1.5)`` on the residual.
    """

    new_codes: list[int] = []
    deltas: list[float] = []
    residuals: list[float] = []
    gate_flags: list[bool] = []
    clamp_hits = 0
    for index, code in enumerate(codes):
        new_code, delta_native, clipped = coordinate.update(code, u[index])
        new_codes.append(new_code)
        deltas.append(coordinate.value_delta(code, delta_native))
        gate_flags.append(clipped)
        if not residual_enabled:
            residuals.append(0.0)
            continue
        raw = u[index] - delta_native
        if clamp_native is not None and abs(raw) > clamp_native:
            clamp_hits += 1
            raw = math.copysign(clamp_native, raw)
        if residual_dtype == "fp16":
            raw = float16_roundtrip(raw)
        residuals.append(raw)
    return new_codes, deltas, residuals, gate_flags, clamp_hits


def residual_value_view(coordinate, codes: list[int], residuals: list[float]) -> list[float]:
    """Express a native-unit residual in value units for reporting.

    For the rank coordinate the conversion is the local linearization
    ``e_value = e_rank * gap``; for the value coordinate it is the identity.
    """

    if coordinate.name == "value":
        return list(residuals)
    view = []
    for code, residual in zip(codes, residuals):
        rank = CODE_RANK[code]
        left, right = local_gaps(rank)
        span = min(left, right)
        if not math.isfinite(span):
            span = max(left, right)
        view.append(residual * span)
    return view


# --------------------------------------------------------------------------
# 6. the three feedback variants, all driven by the same record list
# --------------------------------------------------------------------------

def replay(
    case: ReplayCase,
    variant: str,
    *,
    window_k: int | None = None,
    clamp_native: float | None = None,
    aggregation: str | None = None,
    residual_dtype: str | None = None,
) -> list[Step]:
    """Replay A (no feedback), B (full residual) or C (finite-K) from the record.

    * ``variant="A"``: ``e == 0`` throughout, no residual bookkeeping.
    * ``variant="B"``: persistent residual, gate accounting, per-step clamp.
    * ``variant="C"``: at each generation, re-simulate the previous K records
      from a zero residual with the boundary fixed at the current center (the
      QES seed-replay worker), then use that residual once for this step.
    """

    protocol = case.protocol
    coordinate = make_coordinate(protocol.coordinate)
    aggregation = aggregation or protocol.aggregation
    window = window_k if window_k is not None else protocol.window_k
    clamp = protocol.residual_clamp if clamp_native is None else clamp_native
    dtype = residual_dtype or protocol.residual_dtype

    codes = list(case.initial_codes)
    residuals = list(case.initial_residual)
    history: deque[UpdateRecord] = deque(maxlen=window)
    trace: list[Step] = []

    for record in case.records:
        codes_before = list(codes)
        if variant == "A":
            estimate, gate_flags, phantom = aggregate(
                coordinate, codes, record.coeffs, record.sigma, record.seeds, aggregation
            )
            u = [record.alpha * value for value in estimate]
            codes, deltas, residuals, update_gate, clamp_hits = apply_update(
                coordinate, codes, u, clamp, residual_enabled=False
            )
        elif variant == "B":
            estimate, gate_flags, phantom = aggregate(
                coordinate, codes, record.coeffs, record.sigma, record.seeds, aggregation
            )
            u = [
                record.alpha * value + protocol.gamma * residuals[index]
                for index, value in enumerate(estimate)
            ]
            codes, deltas, residuals, update_gate, clamp_hits = apply_update(
                coordinate, codes, u, clamp, residual_enabled=True, residual_dtype=dtype
            )
        elif variant == "C":
            # Residual reconstruction, exactly the QES seed-replay worker
            # (wsr.py:_reconstruct_layer_residual): replay the *previous* K
            # records from a zero residual while the boundary stays fixed at
            # the current center; only the residual recurses, the weight never
            # moves inside the replay.  The reconstructed residual is then used
            # once for the current step and discarded.
            reconstructed = [0.0] * protocol.dim
            phantom = 0
            for window_record in list(history):
                estimate_w, _, phantom_w = aggregate(
                    coordinate,
                    codes,
                    window_record.coeffs,
                    window_record.sigma,
                    window_record.seeds,
                    aggregation,
                )
                u_w = [
                    window_record.alpha * value
                    + protocol.gamma * reconstructed[index]
                    for index, value in enumerate(estimate_w)
                ]
                _, _, reconstructed, _, _ = apply_update(
                    coordinate, codes, u_w, clamp, residual_enabled=True,
                    residual_dtype=dtype,
                )
                phantom += phantom_w
            estimate, gate_flags, current_phantom = aggregate(
                coordinate, codes, record.coeffs, record.sigma, record.seeds, aggregation
            )
            u = [
                record.alpha * value + protocol.gamma * reconstructed[index]
                for index, value in enumerate(estimate)
            ]
            codes, deltas, residuals, update_gate, clamp_hits = apply_update(
                coordinate, codes, u, clamp, residual_enabled=True, residual_dtype=dtype
            )
            phantom += current_phantom
        elif variant == "C_exact":
            # Engine self-check: reconstruct the residual by replaying *all*
            # previous records with a re-simulated center (no truncation, no
            # current-boundary approximation).  This must reproduce B exactly.
            simulated = list(case.initial_codes)
            reconstructed = [0.0] * protocol.dim
            for window_record in case.records[: record.generation]:
                estimate_w, _, _ = aggregate(
                    coordinate,
                    simulated,
                    window_record.coeffs,
                    window_record.sigma,
                    window_record.seeds,
                    aggregation,
                )
                u_w = [
                    window_record.alpha * value + protocol.gamma * reconstructed[index]
                    for index, value in enumerate(estimate_w)
                ]
                simulated, _, reconstructed, _, _ = apply_update(
                    coordinate, simulated, u_w, clamp, residual_enabled=True,
                    residual_dtype=dtype,
                )
            estimate, gate_flags, phantom = aggregate(
                coordinate, simulated, record.coeffs, record.sigma, record.seeds, aggregation
            )
            u = [
                record.alpha * value + protocol.gamma * reconstructed[index]
                for index, value in enumerate(estimate)
            ]
            codes, deltas, residuals, update_gate, clamp_hits = apply_update(
                coordinate, simulated, u, clamp, residual_enabled=True, residual_dtype=dtype
            )
        else:
            raise ValueError(f"unknown variant {variant!r}")

        trace.append(
            Step(
                generation=record.generation,
                g=estimate,
                u=u,
                delta_value=deltas,
                codes_before=codes_before,
                codes_after=list(codes),
                residual_native=list(residuals),
                residual_value=residual_value_view(coordinate, codes, residuals),
                clamp_hits=clamp_hits,
                perturb_gate_flags=list(gate_flags),
                update_gate_flags=list(update_gate),
                phantom_components=phantom,
            )
        )
        history.append(record)
    return trace


# --------------------------------------------------------------------------
# 7. reference run: builds the record (this is the (B) trajectory)
# --------------------------------------------------------------------------

def build_reference_case(
    coordinate_name: str,
    *,
    dim: int = DIM,
    population: int = POPULATION,
    generations: int = GENERATIONS,
    sigma: float,
    alpha: float,
    gamma: float,
    window_k: int,
    clamp: float,
    seed: int = BASE_SEED,
    drift_rate: float = DRIFT_RATE,
    aggregation: str = "realized",
) -> tuple[ReplayCase, list[Step]]:
    """Run (B) once, logging exactly the fields replay needs.

    The synthetic objective is a fixed target vector that drifts slowly; the
    reward is ``-||candidate - target||^2`` and the coefficients are QES's
    z-score shaping.  Neither the target nor the reward is needed afterwards:
    only ``seeds``/``coeffs``/``sigma``/``alpha`` reach the record.
    """

    coordinate = make_coordinate(coordinate_name)
    protocol = ReplayProtocol(
        coordinate=coordinate_name,
        dim=dim,
        s_eff=S_EFF,
        gamma=gamma,
        window_k=window_k,
        residual_clamp=clamp,
        aggregation=aggregation,
    )
    generator = random.Random(seed)
    initial_codes = [ORDERED_CODES[generator.randrange(len(ORDERED_CODES))] for _ in range(dim)]
    initial_values = [coordinate.value(code) for code in initial_codes]
    target = [value + generator.gauss(0.0, 2.0) for value in initial_values]
    direction = [generator.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(component * component for component in direction)) or 1.0
    direction = [component / norm for component in direction]

    codes = list(initial_codes)
    residuals = [0.0] * dim
    records: list[UpdateRecord] = []
    trace: list[Step] = []

    for generation in range(generations):
        seeds = tuple(
            noise_seed(seed, generation, index, LAYER_ID) for index in range(population)
        )
        target_t = [
            target[index] + drift_rate * generation * direction[index] for index in range(dim)
        ]
        rewards: list[float] = []
        for candidate_index in range(population):
            epsilon = noise_vector(seeds[candidate_index], dim)
            squared_error = 0.0
            for index in range(dim):
                code = codes[index]
                native = coordinate.native(code)
                delta = coordinate.sample_delta(native, sigma, epsilon[index])
                delta, gated = coordinate.gate_perturbation(native, delta)
                if gated:
                    candidate_value = coordinate.value(code)
                else:
                    candidate_code, _ = coordinate.realize(code, delta)
                    candidate_value = coordinate.value(candidate_code)
                squared_error += (candidate_value - target_t[index]) ** 2
            rewards.append(-squared_error)
        coeffs = tuple(zscore_coeffs(rewards))

        estimate, gate_flags, phantom = aggregate(
            coordinate, codes, coeffs, sigma, seeds, aggregation
        )
        u = [
            alpha * value + gamma * residuals[index] for index, value in enumerate(estimate)
        ]
        codes_before = list(codes)
        codes, deltas, residuals, update_gate, clamp_hits = apply_update(
            coordinate, codes, u, clamp, residual_enabled=True, residual_dtype="fp32"
        )
        records.append(
            UpdateRecord(
                generation=generation,
                seeds=seeds,
                coeffs=coeffs,
                sigma=sigma,
                alpha=alpha,
                realized_deltas=tuple(deltas),
            )
        )
        trace.append(
            Step(
                generation=generation,
                g=estimate,
                u=u,
                delta_value=list(deltas),
                codes_before=codes_before,
                codes_after=list(codes),
                residual_native=list(residuals),
                residual_value=residual_value_view(coordinate, codes, residuals),
                clamp_hits=clamp_hits,
                perturb_gate_flags=list(gate_flags),
                update_gate_flags=list(update_gate),
                phantom_components=phantom,
            )
        )

    case = ReplayCase(
        protocol=protocol,
        initial_codes=tuple(initial_codes),
        initial_residual=tuple([0.0] * dim),
        records=tuple(records),
    )
    return case, trace


# --------------------------------------------------------------------------
# 8. divergence metrics for (B) - (C)
# --------------------------------------------------------------------------

@dataclass
class Divergence:
    generation: int
    delta_mismatch: int
    changed_b: int
    changed_c: int
    changed_total: int
    code_mismatch: int
    equivalent_code_mismatch: int
    residual_inf: float
    residual_rms: float
    clamp_b: int
    clamp_c: int
    perturb_gate_flip: int
    update_gate_flip: int
    gate_denominator: int


def divergence(trace_b: list[Step], trace_c: list[Step]) -> list[Divergence]:
    rows = []
    for step_b, step_c in zip(trace_b, trace_c):
        mismatch = sum(
            1 for a, b in zip(step_b.delta_value, step_c.delta_value) if a != b
        )
        changed_b = sum(1 for value in step_b.delta_value if value != 0.0)
        changed_c = sum(1 for value in step_c.delta_value if value != 0.0)
        changed_total = sum(
            1
            for a, b in zip(step_b.delta_value, step_c.delta_value)
            if a != 0.0 or b != 0.0
        )
        code_mismatch = sum(
            1 for a, b in zip(step_b.codes_after, step_c.codes_after) if a != b
        )
        equivalent_code_mismatch = sum(
            1
            for a, b in zip(step_b.codes_after, step_c.codes_after)
            if a != b and decode_value(a) == decode_value(b)
        )
        difference = [
            a - b for a, b in zip(step_b.residual_value, step_c.residual_value)
        ]
        residual_inf = max((abs(value) for value in difference), default=0.0)
        residual_rms = math.sqrt(
            sum(value * value for value in difference) / len(difference)
        )
        perturb_flip = sum(
            1
            for a, b in zip(step_b.perturb_gate_flags, step_c.perturb_gate_flags)
            if a != b
        )
        update_flip = sum(
            1
            for a, b in zip(step_b.update_gate_flags, step_c.update_gate_flags)
            if a != b
        )
        denominator = sum(
            1
            for index in range(len(step_b.perturb_gate_flags))
            if step_b.perturb_gate_flags[index]
            or step_c.perturb_gate_flags[index]
            or step_b.update_gate_flags[index]
            or step_c.update_gate_flags[index]
        )
        rows.append(
            Divergence(
                generation=step_b.generation,
                delta_mismatch=mismatch,
                changed_b=changed_b,
                changed_c=changed_c,
                changed_total=changed_total,
                code_mismatch=code_mismatch,
                equivalent_code_mismatch=equivalent_code_mismatch,
                residual_inf=residual_inf,
                residual_rms=residual_rms,
                clamp_b=step_b.clamp_hits,
                clamp_c=step_c.clamp_hits,
                perturb_gate_flip=perturb_flip,
                update_gate_flip=update_flip,
                gate_denominator=denominator,
            )
        )
    return rows


def summarise(rows: list[Divergence]) -> dict:
    total_changed = sum(row.changed_total for row in rows)
    total_mismatch = sum(row.delta_mismatch for row in rows)
    first = next((row.generation for row in rows if row.delta_mismatch), None)
    return {
        "generations": len(rows),
        "first_divergent_generation": first,
        "total_changed_coordinates": total_changed,
        "total_delta_mismatch": total_mismatch,
        "mismatch_ratio": total_mismatch / total_changed if total_changed else 0.0,
        "max_residual_inf": max((row.residual_inf for row in rows), default=0.0),
        "max_residual_rms": max((row.residual_rms for row in rows), default=0.0),
        "total_code_mismatch": sum(row.code_mismatch for row in rows),
        "total_equivalent_code_mismatch": sum(
            row.equivalent_code_mismatch for row in rows
        ),
        "clamp_hits_b": sum(row.clamp_b for row in rows),
        "clamp_hits_c": sum(row.clamp_c for row in rows),
        "perturb_gate_flips": sum(row.perturb_gate_flip for row in rows),
        "update_gate_flips": sum(row.update_gate_flip for row in rows),
        "gate_denominator": sum(row.gate_denominator for row in rows),
    }


def delta_divergence(trace_a: list[Step], trace_b: list[Step]) -> dict:
    """(A)-(B): the residual effect, not the replay effect."""

    total_changed = sum(
        1
        for step_a, step_b in zip(trace_a, trace_b)
        for a, b in zip(step_a.delta_value, step_b.delta_value)
        if a != 0.0 or b != 0.0
    )
    total_mismatch = sum(
        1
        for step_a, step_b in zip(trace_a, trace_b)
        for a, b in zip(step_a.delta_value, step_b.delta_value)
        if a != b
    )
    return {
        "total_changed_coordinates": total_changed,
        "total_delta_mismatch": total_mismatch,
        "mismatch_ratio": total_mismatch / total_changed if total_changed else 0.0,
    }


def step_in_half_steps(coordinate_name: str, trace: list[Step]) -> tuple[float, float]:
    """Mean |u| and mean |e| expressed in units of the local half step.

    This is the regime coordinate that decides whether the replay can be
    faithful: when |u| stays well below half a step, projections almost never
    flip and (B)-(C) collapses; when |u| is comparable to the half step, every
    replay error flips a rounding decision.
    """

    u_ratios = []
    e_ratios = []
    for step in trace:
        for index, code in enumerate(step.codes_before):
            if coordinate_name == "rank":
                rank = CODE_RANK[code]
                left, right = local_gaps(rank)
                gap = min(left, right)
                if not math.isfinite(gap):
                    gap = max(left, right)
                half = 0.5 * gap
                u_ratio = abs(step.u[index]) * gap / half
                e_ratio = abs(step.residual_native[index]) * gap / half
            else:
                half = half_gap(code)
                u_ratio = abs(step.u[index]) / half
                e_ratio = abs(step.residual_value[index]) / half
            u_ratios.append(u_ratio)
            e_ratios.append(e_ratio)
    return (
        sum(u_ratios) / len(u_ratios),
        sum(e_ratios) / len(e_ratios),
    )


def regime_sweep(
    coordinate_name: str,
    alphas: list[float],
    *,
    sigma: float,
    generations: int = GENERATIONS,
    window_k: int = WINDOW_K,
    clamp: float,
) -> list[dict]:
    """How (B)-(C) depends on the update-to-grid-step ratio."""

    rows = []
    for alpha in alphas:
        case, _ = build_reference_case(
            coordinate_name,
            sigma=sigma,
            alpha=alpha,
            gamma=GAMMA,
            window_k=window_k,
            clamp=clamp,
            generations=generations,
        )
        loaded = case_from_json(case_to_json(case))
        trace_b = replay(loaded, "B")
        trace_c = replay(loaded, "C")
        summary = summarise(divergence(trace_b, trace_c))
        mean_u, mean_e = step_in_half_steps(coordinate_name, trace_b)
        rows.append(
            {
                "alpha": alpha,
                "mean_u_half_steps": mean_u,
                "mean_e_half_steps": mean_e,
                "mismatch_ratio": summary["mismatch_ratio"],
                "delta_mismatch": summary["total_delta_mismatch"],
                "changed": summary["total_changed_coordinates"],
                "code_mismatch": summary["total_code_mismatch"],
                "equivalent_code_mismatch": summary["total_equivalent_code_mismatch"],
                "clamp_b": summary["clamp_hits_b"],
                "clamp_c": summary["clamp_hits_c"],
                "residual_inf": summary["max_residual_inf"],
            }
        )
    return rows


# --------------------------------------------------------------------------
# 9. counterexamples
# --------------------------------------------------------------------------

def nvfp4_raw_code_jump() -> dict:
    """Raw code-number arithmetic is not a numerical coordinate."""

    rows = []
    for code in range(16):
        down = code - 1
        down_value = E2M1_VALUES[down] if 0 <= down < 16 else None
        up = code + 1
        up_value = E2M1_VALUES[up] if 0 <= up < 16 else None
        rows.append(
            {
                "code": code,
                "value": E2M1_VALUES[code],
                "rank": CODE_RANK[code],
                "value_at_code_minus_1": down_value,
                "jump_down": None if down_value is None else down_value - E2M1_VALUES[code],
                "value_at_code_plus_1": up_value,
                "jump_up": None if up_value is None else up_value - E2M1_VALUES[code],
            }
        )
    rank_rows = []
    for rank in range(len(ORDERED_CODES)):
        code = ORDERED_CODES[rank]
        step = RANK_VALUES[rank] - RANK_VALUES[rank - 1] if rank > 0 else None
        rank_rows.append({"rank": rank, "code": code, "value": RANK_VALUES[rank], "step": step})
    return {"code_table": rows, "rank_table": rank_rows}


def nvfp4_double_zero() -> dict:
    """Code 0 <-> code 8: identical value, different payload."""

    events = []
    for label, start, delta in (
        ("tiny negative from code 0", 0, -1e-9),
        ("half-gap negative from code 0", 0, -0.2),
        ("at the bound from code 0", 0, -0.25),
        ("tiny positive from code 0", 0, 1e-9),
        ("tiny positive from code 8", 8, 1e-9),
        ("tiny negative from code 8", 8, -1e-9),
    ):
        base_value = decode_value(start)
        code, value = project_value(base_value + delta)
        events.append(
            {
                "label": label,
                "start_code": start,
                "delta": delta,
                "code": code,
                "value": value,
                "value_change": value - base_value,
                "payload_changed": code != start,
                "numerical_change": CODE_RANK[code] != CODE_RANK[start],
            }
        )
    return {
        "events": events,
        "round_trip_8_to_0": project_value(0.0)[0],
        "packed_byte_code0": 0x0,
        "packed_byte_code8": 0x8,
    }


def qes_frozen_residual() -> dict:
    """gamma=0.9, fixed update magnitude: residual converges, never crosses."""

    def converge(alpha_g: float, gamma: float, step: float, steps: int = 400):
        """One coordinate on a uniform grid of ``step``; returns the crossing stats."""

        residual = 0.0
        centre = 0.0
        first_crossing = None
        trace = []
        for iteration in range(steps):
            u = alpha_g + gamma * residual
            applied = round_half_even(u / step) * step
            residual = u - applied
            if applied != 0.0:
                centre += applied
                if first_crossing is None:
                    first_crossing = iteration
            if iteration < 30:
                trace.append(residual)
        return {
            "alpha_g": alpha_g,
            "gamma": gamma,
            "step": step,
            "limit": alpha_g / (1.0 - gamma),
            "first_crossing_generation": first_crossing,
            "moves": centre,
            "residual_tail": trace[-1],
            "trace": trace,
        }

    frozen = converge(0.01, 0.9, 1.0)
    critical_gamma = 1.0 - 2.0 * 0.01
    gamma_sweep = [converge(0.01, gamma, 1.0) for gamma in (0.90, 0.95, 0.98, 0.99)]

    # Non-uniform grid: one (alpha_g, gamma) pair, three local grid steps.
    non_uniform = converge(0.03, 0.9, 1.0)
    gap_classes = {}
    for rank in range(len(RANK_VALUES) - 1):
        gap = RANK_GAPS[rank]
        gap_classes.setdefault(gap, 0)
        gap_classes[gap] += 1
    crossing_by_gap = {}
    for gap in sorted(gap_classes):
        crossing_by_gap[gap] = converge(0.03, 0.9, gap)["first_crossing_generation"]
    return {
        "frozen_case": frozen,
        "critical_gamma": critical_gamma,
        "gamma_sweep": gamma_sweep,
        "non_uniform_limit": non_uniform["limit"],
        "non_uniform_first_crossing": non_uniform["first_crossing_generation"],
        "gap_class_counts": gap_classes,
        "crossing_by_gap": crossing_by_gap,
    }


def qes_rejected_direction_returns() -> dict:
    """Endpoint rejection keeps the displacement; it comes back as a lagging bias."""

    gamma = 0.9
    clamp = RESIDUAL_CLAMP

    def run(desired: list[float], use_residual: bool) -> dict:
        residual = 0.0
        value = 6.0  # upper endpoint
        moves = []
        trace = []
        for step, magnitude in enumerate(desired):
            u = magnitude + (gamma * residual if use_residual else 0.0)
            target = value + u
            clipped = target > 6.0
            target = min(6.0, target)
            code = encode_scaled(target / S_EFF)
            new_value = decode_value(code)
            applied = new_value - value
            if use_residual:
                raw = u - applied
                if abs(raw) > clamp:
                    raw = math.copysign(clamp, raw)
                residual = raw
            else:
                residual = 0.0
            value = new_value
            moves.append(applied)
            trace.append(
                {
                    "generation": step,
                    "desired": magnitude,
                    "u": u,
                    "target": target,
                    "clipped": clipped,
                    "applied": applied,
                    "value": value,
                    "residual": residual,
                }
            )
        return {
            "first_move_generation": next(
                (step for step, applied in enumerate(moves) if applied != 0.0), None
            ),
            "moves": moves,
            "trace": trace,
        }

    desired = [0.6] + [-1.2] * 12
    with_feedback = run(desired, True)
    without_feedback = run(desired, False)

    # Closed form for the first inward crossing with a constant reversal.
    e0, alpha_g, gamma_rev = 0.6, -1.2, gamma
    crossing = None
    for step in range(1, 200):
        residual = e0 * gamma_rev**step + alpha_g * (1.0 - gamma_rev**step) / (1.0 - gamma_rev)
        if residual < -1.0:
            crossing = step
            break
    return {
        "with_feedback": with_feedback,
        "without_feedback": without_feedback,
        "analytic_first_crossing": crossing,
    }


# --------------------------------------------------------------------------
# 10. reporting helpers
# --------------------------------------------------------------------------

def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------
# 11. main
# --------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    emit_record = None
    if "--emit-record" in argv:
        emit_record = argv[argv.index("--emit-record") + 1]

    print("QES -> NVFP4 arithmetic reference prototype (ticket 08)")
    print(f"python {sys.version.split()[0]}  |  pure stdlib, no model, no GPU")
    print(
        f"fixed group: dim={DIM}  s_eff={S_EFF}  grid={len(ORDERED_CODES)} values  "
        f"pop={POPULATION}  generations={GENERATIONS}  gamma={GAMMA}  K={WINDOW_K}"
    )
    print(
        "replay inputs: ReplayProtocol + initial_codes + initial_residual + "
        "UpdateRecord[] (generation, seeds, coeffs, sigma, alpha, realized_deltas); "
        "no runtime config object is read"
    )

    section("0. provenance and the two coordinate tables")
    print("E2M1_VALUES / E2M1_BOUNDS copied from src/shadow_lab/quasar.py:34-52")
    print("ORDERED_CODES / CODE_RANK copied from src/shadow_lab/mage_w4_search/mutation.py:23-24")
    print()
    print("rank | code |  value | gap to next rank | half gap")
    print("-----+------+--------+------------------+---------")
    for rank in range(len(ORDERED_CODES)):
        if rank < len(RANK_GAPS):
            gap_text = f"{RANK_GAPS[rank]:16.2f}"
            half_text = f"{RANK_GAPS[rank] / 2.0:8.2f}"
        else:
            gap_text, half_text = f"{'n/a':>16}", f"{'n/a':>8}"
        print(
            f"{rank:4d} | {ORDERED_CODES[rank]:4d} | {RANK_VALUES[rank]:6.1f} |"
            f" {gap_text} | {half_text}"
        )
    gap_counts: dict[float, int] = {}
    for gap in RANK_GAPS:
        gap_counts[gap] = gap_counts.get(gap, 0) + 1
    print(f"gap classes: {gap_counts}  (non-equidistant: 0.5 / 1.0 / 2.0)")
    print(f"code 8 and code 0 both decode to {E2M1_VALUES[8]:.1f}; "
          f"CODE_RANK[8]={CODE_RANK[8]} == CODE_RANK[0]={CODE_RANK[0]}")

    # optional cross-check of the pure-Python mirror against the real primitive
    mirror_check = "skipped (torch not importable)"
    try:
        import torch  # noqa: PLC0415

        sys.path.insert(0, str(__file__).rsplit("/scripts/", 1)[0] + "/src")
        from shadow_lab.quasar import e2m1_encode  # noqa: PLC0415

        probes = [index * 0.005 - 7.0 for index in range(2801)]
        probes += [-1e-9, 1e-9, -0.2, 0.2, -0.25, 0.25, -6.0, 6.0, -6.1, 6.1, 0.0]
        probes += list(E2M1_BOUNDS) + [-value for value in E2M1_BOUNDS]
        probe = torch.tensor(probes, dtype=torch.float32)
        reference = e2m1_encode(probe).tolist()
        mirrored = [encode_scaled(value) for value in probes]
        mismatches = [row for row in zip(reference, mirrored) if row[0] != row[1]]
        code8 = sum(1 for value in mirrored if value == 8)
        mirror_check = (
            f"{len(mismatches)} mismatches on {len(probes)} probes in [-7, 7]; "
            f"the real primitive emits code 8 for small negatives "
            f"({code8} code-8 outputs in this probe set)"
        )
    except Exception as error:  # pragma: no cover - depends on environment
        mirror_check = f"skipped ({type(error).__name__})"
    print(f"mirror check vs quasar.e2m1_encode: {mirror_check}")

    section("1. NVFP4 counterexample: raw code arithmetic (code 8 - 1)")
    jump = nvfp4_raw_code_jump()
    print("code | value | rank | value(code-1) | jump(code-1) | value(code+1) | jump(code+1)")
    print("-----+-------+------+---------------+--------------+---------------+--------------")
    for row in jump["code_table"]:
        down = "n/a" if row["value_at_code_minus_1"] is None else f"{row['value_at_code_minus_1']:6.1f}"
        down_jump = "n/a" if row["jump_down"] is None else f"{row['jump_down']:7.1f}"
        up = "n/a" if row["value_at_code_plus_1"] is None else f"{row['value_at_code_plus_1']:6.1f}"
        up_jump = "n/a" if row["jump_up"] is None else f"{row['jump_up']:7.1f}"
        print(
            f"{row['code']:4d} | {row['value']:5.1f} | {row['rank']:4d} | {down} | {down_jump} |"
            f" {up} | {up_jump}"
        )
    print()
    print("rank | code | value | step from previous rank")
    print("-----+------+-------+------------------------")
    for row in jump["rank_table"]:
        step = "n/a" if row["step"] is None else f"{row['step']:.2f}"
        print(f"{row['rank']:4d} | {row['code']:4d} | {row['value']:5.1f} | {step:>10}")
    code8 = E2M1_VALUES[8]
    print()
    print(
        f"code 8 decodes to {code8:.1f}; code 8 - 1 = code 7 decodes to "
        f"{E2M1_VALUES[7]:.1f}  ->  jump {E2M1_VALUES[7] - code8:+.1f} "
        f"for a nominal single-code step"
    )
    print(
        f"rank 7 - 1 = rank 6 decodes to {RANK_VALUES[6]:.1f}  ->  step "
        f"{RANK_VALUES[6] - RANK_VALUES[7]:+.1f} (the same nominal step, "
        f"{abs((RANK_VALUES[6] - RANK_VALUES[7]) / (E2M1_VALUES[7] - code8)):.3f} of the code jump)"
    )

    section("2. NVFP4 counterexample: code 0 <-> 8 equivalent rewrite")
    double_zero = nvfp4_double_zero()
    print("case                          | start |    delta | code | value | value change | payload changed | numerical change")
    print("------------------------------+-------+----------+------+-------+--------------+-----------------+-----------------")
    for row in double_zero["events"]:
        print(
            f"{row['label']:<29} | {row['start_code']:5d} | {row['delta']:+8.1e} | {row['code']:4d} |"
            f" {row['value']:5.1f} | {row['value_change']:+12.1e} | {str(row['payload_changed']):>15} |"
            f" {str(row['numerical_change']):>15}"
        )
    print()
    print(
        "mutation.py's criterion (CODE_RANK[old] != CODE_RANK[new]) counts the "
        "0<->8 rewrite as code_change=1, numerical_change=0."
    )

    section("3. QES counterexample: gamma=0.9, fixed step, residual never crosses")
    frozen = qes_frozen_residual()
    case = frozen["frozen_case"]
    print(
        f"alpha*g={case['alpha_g']}, gamma={case['gamma']}, grid step={case['step']}, "
        f"half step={case['step'] / 2.0}: limit = alpha*g/(1-gamma) = {case['limit']:.3f}"
    )
    print(
        f"first crossing generation over 400 steps: {case['first_crossing_generation']}  "
        f"(centre moved {case['moves']} times)"
    )
    print("residual trace e_0..e_29: " + ", ".join(f"{value:.4f}" for value in case["trace"]))
    print()
    print(f"critical gamma for alpha*g=0.01, half step 0.5: gamma >= {frozen['critical_gamma']:.2f}")
    print("gamma | limit | first crossing generation (400 steps)")
    print("------+-------+---------------------------------------")
    for row in frozen["gamma_sweep"]:
        print(
            f"{row['gamma']:5.2f} | {row['limit']:5.3f} | {row['first_crossing_generation']}"
        )
    print()
    print(
        f"non-uniform grid, same alpha*g=0.03 and gamma=0.9: limit={frozen['non_uniform_limit']:.3f}; "
        f"gap classes={frozen['gap_class_counts']}"
    )
    print("grid step | half step | first crossing generation (400 steps)")
    print("----------+-----------+---------------------------------------")
    for gap, crossing in sorted(frozen["crossing_by_gap"].items()):
        print(f"{gap:9.2f} | {gap / 2.0:9.2f} | {crossing}")
    print(
        "-> the same (alpha, gamma) moves the fine coordinates and freezes the coarse ones; "
        "the paper's single-Delta bound does not transfer."
    )

    section("4. QES counterexample: a rejected direction returns through the residual")
    rejected = qes_rejected_direction_returns()
    print("with feedback (B): desired +0.6 at the endpoint, then -1.2 for twelve generations")
    print("gen | desired |      u | target | clipped | applied |  value | residual")
    print("----+---------+--------+--------+---------+---------+--------+---------")
    for row in rejected["with_feedback"]["trace"]:
        print(
            f"{row['generation']:3d} | {row['desired']:+7.2f} | {row['u']:+6.3f} |"
            f" {row['target']:6.3f} | {str(row['clipped']):>7} | {row['applied']:+7.1f} |"
            f" {row['value']:6.1f} | {row['residual']:+7.3f}"
        )
    print(
        f"first inward move: with feedback generation "
        f"{rejected['with_feedback']['first_move_generation']}, without feedback generation "
        f"{rejected['without_feedback']['first_move_generation']}; "
        f"analytic first crossing e_n < -1.0 at generation {rejected['analytic_first_crossing']}"
    )

    section("5. A/B/C replay divergence, rank coordinate")
    rank_case, rank_trace_ref = build_reference_case(
        "rank", sigma=SIGMA_RANK, alpha=ALPHA, gamma=GAMMA, window_k=WINDOW_K,
        clamp=RESIDUAL_CLAMP,
    )
    rank_json = case_to_json(rank_case)
    print(f"record: {len(rank_json)} bytes json, sha256={hashlib.sha256(rank_json.encode()).hexdigest()[:16]}")
    rank_loaded = case_from_json(rank_json)
    rank_b = replay(rank_loaded, "B")
    rank_a = replay(rank_loaded, "A")
    rank_c = replay(rank_loaded, "C")
    rank_c_full = replay(rank_loaded, "C", window_k=len(rank_loaded.records))
    rank_c_exact = replay(rank_loaded, "C_exact")
    rank_b_noclamp = replay(rank_loaded, "B", clamp_native=float("inf"))
    rank_b_fp16 = replay(rank_loaded, "B", residual_dtype="fp16")
    rank_repro = max(
        abs(a - b)
        for step, record in zip(rank_b, rank_loaded.records)
        for a, b in zip(step.delta_value, record.realized_deltas)
    )
    print(f"record-only replay of B reproduces stored realized_deltas: max abs diff = {rank_repro}")
    print(
        f"engine check: C_exact (all history, re-simulated boundary) vs B: max |delta diff| = "
        f"{max(abs(a - b) for step_a, step_b in zip(rank_c_exact, rank_b) for a, b in zip(step_a.delta_value, step_b.delta_value))}"
    )
    print(
        f"C with window K={len(rank_loaded.records)} (no truncation) vs B: max |delta diff| = "
        f"{max(abs(a - b) for step_a, step_b in zip(rank_c_full, rank_b) for a, b in zip(step_a.delta_value, step_b.delta_value))}"
        f"  <- boundary approximation alone"
    )
    print(
        f"B without the +-1.5 clamp vs B: max |delta diff| = "
        f"{max(abs(a - b) for step_a, step_b in zip(rank_b_noclamp, rank_b) for a, b in zip(step_a.delta_value, step_b.delta_value))}, "
        f"changed coordinates = "
        f"{sum(1 for step_a, step_b in zip(rank_b_noclamp, rank_b) for a, b in zip(step_a.delta_value, step_b.delta_value) if a != b)}"
    )
    print(
        f"B with FP16 residual storage vs B (FP32): max |delta diff| = "
        f"{max(abs(a - b) for step_a, step_b in zip(rank_b_fp16, rank_b) for a, b in zip(step_a.delta_value, step_b.delta_value))}, "
        f"changed coordinates = "
        f"{sum(1 for step_a, step_b in zip(rank_b_fp16, rank_b) for a, b in zip(step_a.delta_value, step_b.delta_value) if a != b)}"
    )
    print()
    print("gen | dDdiff | dDB!=0 | dDC!=0 | changed | codeDiff | equivDiff |  |eB-eC|inf |   rms   | clampB | clampC | pFlip | uFlip")
    print("----+--------+--------+--------+---------+----------+-----------+-------------+---------+--------+--------+-------+------")
    rank_rows = divergence(rank_b, rank_c)
    for row in rank_rows:
        print(
            f"{row.generation:3d} | {row.delta_mismatch:6d} | {row.changed_b:6d} | {row.changed_c:6d} |"
            f" {row.changed_total:7d} | {row.code_mismatch:8d} | {row.equivalent_code_mismatch:9d} | {row.residual_inf:11.4f} |"
            f" {row.residual_rms:7.4f} | {row.clamp_b:6d} | {row.clamp_c:6d} |"
            f" {row.perturb_gate_flip:5d} | {row.update_gate_flip:4d}"
        )
    rank_summary = summarise(rank_rows)
    print()
    print(f"rank summary: {rank_summary}")
    print(f"rank (A)-(B) residual effect: {delta_divergence(rank_a, rank_b)}")
    rank_res_inf = max(max(abs(value) for value in step.residual_native) for step in rank_b)
    print(f"rank B residual |e|inf = {rank_res_inf:.4f} rank units (bound 0.5 + clamp 1.5)")

    section("6. A/B/C replay divergence, value coordinate")
    value_case, value_trace_ref = build_reference_case(
        "value", sigma=SIGMA_VALUE, alpha=ALPHA, gamma=GAMMA, window_k=WINDOW_K,
        clamp=RESIDUAL_CLAMP * S_EFF,
    )
    value_json = case_to_json(value_case)
    print(f"record: {len(value_json)} bytes json, sha256={hashlib.sha256(value_json.encode()).hexdigest()[:16]}")
    value_loaded = case_from_json(value_json)
    value_b = replay(value_loaded, "B")
    value_a = replay(value_loaded, "A")
    value_c = replay(value_loaded, "C")
    value_c_full = replay(value_loaded, "C", window_k=len(value_loaded.records))
    value_c_exact = replay(value_loaded, "C_exact")
    value_b_noclamp = replay(value_loaded, "B", clamp_native=float("inf"))
    value_b_fp16 = replay(value_loaded, "B", residual_dtype="fp16")
    value_unprojected = replay(value_loaded, "B", aggregation="unprojected")
    value_repro = max(
        abs(a - b)
        for step, record in zip(value_b, value_loaded.records)
        for a, b in zip(step.delta_value, record.realized_deltas)
    )
    print(f"record-only replay of B reproduces stored realized_deltas: max abs diff = {value_repro}")
    print(
        f"engine check: C_exact (all history, re-simulated boundary) vs B: max |delta diff| = "
        f"{max(abs(a - b) for step_a, step_b in zip(value_c_exact, value_b) for a, b in zip(step_a.delta_value, step_b.delta_value))}"
    )
    print(
        f"C with window K={len(value_loaded.records)} (no truncation) vs B: max |delta diff| = "
        f"{max(abs(a - b) for step_a, step_b in zip(value_c_full, value_b) for a, b in zip(step_a.delta_value, step_b.delta_value))}"
        f"  <- boundary approximation alone"
    )
    print(
        f"B without the +-1.5 clamp vs B: max |delta diff| = "
        f"{max(abs(a - b) for step_a, step_b in zip(value_b_noclamp, value_b) for a, b in zip(step_a.delta_value, step_b.delta_value))}, "
        f"changed coordinates = "
        f"{sum(1 for step_a, step_b in zip(value_b_noclamp, value_b) for a, b in zip(step_a.delta_value, step_b.delta_value) if a != b)}"
    )
    print(
        f"B with FP16 residual storage vs B (FP32): max |delta diff| = "
        f"{max(abs(a - b) for step_a, step_b in zip(value_b_fp16, value_b) for a, b in zip(step_a.delta_value, step_b.delta_value))}, "
        f"changed coordinates = "
        f"{sum(1 for step_a, step_b in zip(value_b_fp16, value_b) for a, b in zip(step_a.delta_value, step_b.delta_value) if a != b)}"
    )
    print()
    print("gen | dDdiff | dDB!=0 | dDC!=0 | changed | codeDiff | equivDiff |  |eB-eC|inf |   rms   | clampB | clampC | pFlip | uFlip")
    print("----+--------+--------+--------+---------+----------+-----------+-------------+---------+--------+--------+-------+------")
    value_rows = divergence(value_b, value_c)
    for row in value_rows:
        print(
            f"{row.generation:3d} | {row.delta_mismatch:6d} | {row.changed_b:6d} | {row.changed_c:6d} |"
            f" {row.changed_total:7d} | {row.code_mismatch:8d} | {row.equivalent_code_mismatch:9d} | {row.residual_inf:11.4f} |"
            f" {row.residual_rms:7.4f} | {row.clamp_b:6d} | {row.clamp_c:6d} |"
            f" {row.perturb_gate_flip:5d} | {row.update_gate_flip:4d}"
        )
    value_summary = summarise(value_rows)
    print()
    print(f"value summary: {value_summary}")
    print(f"value (A)-(B) residual effect: {delta_divergence(value_a, value_b)}")
    value_res_inf = max(
        max(abs(value) for value in step.residual_value) for step in value_b
    )
    print(
        f"value B residual |e|inf = {value_res_inf:.4f} value units; "
        f"max |e| / half-gap = "
        f"{max(abs(value) / half_gap(code) for step in value_b for code, value in zip(step.codes_after, step.residual_value)):.3f}"
    )
    print(
        f"value coordinate phantom components (nonzero perturbation, zero realized "
        f"displacement): {sum(step.phantom_components for step in value_b)} over "
        f"{len(value_b) * POPULATION * DIM} candidate-coordinate draws"
    )
    print(
        f"unprojected aggregation vs realized aggregation, final center codes differ at "
        f"{sum(1 for a, b in zip(value_b[-1].codes_after, value_unprojected[-1].codes_after) if a != b)}"
        f" of {DIM} coordinates"
    )
    equivalent_rewrites = sum(
        1
        for step in value_b
        for before, after, delta in zip(
            step.codes_before, step.codes_after, step.delta_value
        )
        if before != after and delta == 0.0
    )
    code8_generations = sum(
        1 for step in value_b for code in step.codes_after if code == 8
    )
    print(
        f"value coordinate equivalent rewrites in the B center (code changed, value "
        f"unchanged): {equivalent_rewrites}; coordinate-generations parked at code 8: "
        f"{code8_generations}"
    )

    section("7. regime sweep: (B)-(C) versus the update-to-grid-step ratio")
    print(
        "The replay can only be faithful while the desired step stays small "
        "relative to the local half step; QES's published regime is |u| ~ 1e-2 code units."
    )
    sweep_alphas = [0.01, 0.03, 0.1, 0.3, 1.0]
    sweep_rows = {}
    for coordinate_name, sigma in (("rank", SIGMA_RANK), ("value", SIGMA_VALUE)):
        clamp = RESIDUAL_CLAMP if coordinate_name == "rank" else RESIDUAL_CLAMP * S_EFF
        sweep_rows[coordinate_name] = regime_sweep(
            coordinate_name, sweep_alphas, sigma=sigma, clamp=clamp
        )
    print()
    print("coordinate | alpha | mean|u| (half steps) | mean|e| (half steps) | (B)-(C) mismatch | code diff | of which 0<->8 | clamp B/C | max |eB-eC|inf")
    print("-----------+-------+----------------------+----------------------+------------------+-----------+--------------+-----------+---------------")
    for coordinate_name, rows in sweep_rows.items():
        for row in rows:
            print(
                f"{coordinate_name:>10} | {row['alpha']:5.2f} | {row['mean_u_half_steps']:20.3f} |"
                f" {row['mean_e_half_steps']:20.3f} | {row['mismatch_ratio']:16.4f} |"
                f" {row['code_mismatch']:9d} | {row['equivalent_code_mismatch']:12d} |"
                f" {str(row['clamp_b']) + '/' + str(row['clamp_c']):>9} | {row['residual_inf']:13.4f}"
            )

    section("8. summary table for ticket 09")
    print(f"{'metric':<52} | {'rank':>16} | {'value':>16}")
    print("-" * 92)
    print(
        f"{'perturbation unit':<52} | {'rank step (-1,0,1)':>16} | {'real number':>16}"
    )
    print(
        f"{'projection at candidate time':<52} | {'none (on grid)':>16} | {'ties-to-even encode':>16}"
    )
    print(
        f"{'residual unit':<52} | {'rank':>16} | {'value':>16}"
    )
    print(
        f"{'theoretical residual bound':<52} | {'0.5 rank':>16} | {'0.5*local gap':>16}"
    )
    print(
        f"{'empirical |e|inf':<52} | {rank_res_inf:16.4f} | {value_res_inf:16.4f}"
    )
    print(
        f"{'(B)-(C) mismatch ratio':<52} | {rank_summary['mismatch_ratio']:16.4f} | "
        f"{value_summary['mismatch_ratio']:16.4f}"
    )
    print(
        f"{'(B)-(C) first divergent generation':<52} | "
        f"{str(rank_summary['first_divergent_generation']):>16} | "
        f"{str(value_summary['first_divergent_generation']):>16}"
    )
    print(
        f"{'(B)-(C) max residual inf':<52} | {rank_summary['max_residual_inf']:16.4f} | "
        f"{value_summary['max_residual_inf']:16.4f}"
    )
    print(
        f"{'(B)-(C) max residual rms':<52} | {rank_summary['max_residual_rms']:16.4f} | "
        f"{value_summary['max_residual_rms']:16.4f}"
    )
    print(
        f"{'(B)-(C) clamp hits B / C':<52} | "
        f"{str(rank_summary['clamp_hits_b']) + ' / ' + str(rank_summary['clamp_hits_c']):>16} | "
        f"{str(value_summary['clamp_hits_b']) + ' / ' + str(value_summary['clamp_hits_c']):>16}"
    )
    print(
        f"{'(B)-(C) gate flips (perturb/update)':<52} | "
        f"{str(rank_summary['perturb_gate_flips']) + ' / ' + str(rank_summary['update_gate_flips']):>16} | "
        f"{str(value_summary['perturb_gate_flips']) + ' / ' + str(value_summary['update_gate_flips']):>16}"
    )
    print(
        f"{'(B)-(C) code mismatches':<52} | "
        f"{rank_summary['total_code_mismatch']:16d} | {value_summary['total_code_mismatch']:16d}"
    )
    print(
        f"{'(B)-(C) of which value-equal (0<->8 rewrite)':<52} | "
        f"{rank_summary['total_equivalent_code_mismatch']:16d} | "
        f"{value_summary['total_equivalent_code_mismatch']:16d}"
    )

    if emit_record is not None:
        for label, payload in (("rank", rank_json), ("value", value_json)):
            target = emit_record.replace(".json", f".{label}.json")
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
            print(f"wrote {label} record to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
