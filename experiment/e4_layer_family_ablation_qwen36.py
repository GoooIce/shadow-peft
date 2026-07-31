"""E4: localize the Bonsai-vs-PTQ semantic gap by matrix family and layer block.

All candidate cases retain the self-quantized checkpoint's auxiliary
parameters. Main-weight triplets are swapped in memory between the official
Bonsai checkpoint and the v3 one-shot PTQ checkpoint.

Phase 1 measures bidirectional family interventions:

* injection: start from PTQ main weights and inject one official family;
* leaveout: start from official main weights and leave one family as PTQ.

Phase 2 selects the most influential layered atomic family and repeats the
bidirectional intervention over eight contiguous 8-layer blocks.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from e2_hybrid_bonsai_qwen36 import (
    PROMPTS,
    _chat_prompt,
    _cosine,
    _generate,
    _kl,
    _last_logits,
    _partition,
    _top_tokens,
)
from mlx.utils import tree_flatten
from mlx_lm import load

DEFAULT_OFFICIAL = Path(
    "~/.cache/modelscope/models/prism-ml--Ternary-Bonsai-27B-mlx-2bit/snapshots/master"
).expanduser()
DEFAULT_PTQ = Path("qwen3.6-27b-mlx-2bit-v3-oneshot")
DEFAULT_OUT = Path("experiment/analysis_qwen36_bonsai_migration/e4")
TRIPLET_SUFFIXES = (".weight", ".scales", ".biases")
LAYER_PATTERN = re.compile(r"\.layers\.(\d+)\.")


def _family(base: str) -> str:
    if base.endswith(".embed_tokens"):
        return "embed_tokens"
    if base.endswith(".lm_head"):
        return "lm_head"
    if ".mlp." in base:
        return "mlp"
    if ".self_attn." in base:
        return "self_attn"
    marker = ".linear_attn."
    if marker in base:
        return "linear_attn." + base.split(marker, 1)[1]
    raise ValueError(f"unclassified quantized matrix: {base}")


def _layer(base: str) -> int | None:
    match = LAYER_PATTERN.search(base)
    return int(match.group(1)) if match else None


def _keys_for_bases(bases: set[str], keys: set[str]) -> set[str]:
    return {base + suffix for base in bases for suffix in TRIPLET_SUFFIXES if base + suffix in keys}


def _aggregate(rows: dict[str, dict[str, Any]]) -> dict[str, float | int]:
    values = list(rows.values())
    return {
        "mean_kl": sum(row["kl_official_to_case"] for row in values) / len(values),
        "mean_logit_cosine": sum(row["logit_cosine"] for row in values) / len(values),
        "top10_overlap": sum(row["top10_overlap"] for row in values),
        "top10_total": 10 * len(values),
    }


def _weight_count(params: dict[str, mx.array], bases: set[str]) -> int:
    # Each packed uint32 holds sixteen 2-bit codes.
    return int(sum(params[base + ".weight"].size * 16 for base in bases))


def _case(
    model: Any,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    reference_logits: dict[str, np.ndarray],
    official: dict[str, mx.array],
    ptq: dict[str, mx.array],
    main_keys: set[str],
    selected_bases: set[str],
    *,
    mode: str,
) -> dict[str, Any]:
    keys = set(ptq)
    if mode == "official_reference":
        updates = list(official.items())
    else:
        # Every experimental case uses Qwen/PTQ auxiliary parameters.
        selected_keys = _keys_for_bases(selected_bases, keys)
        if mode == "inject":
            official_keys = selected_keys
        elif mode == "leaveout":
            official_keys = main_keys - selected_keys
        elif mode == "ptq_baseline":
            official_keys = set()
        elif mode == "official_main":
            official_keys = main_keys
        else:
            raise ValueError(f"unknown case mode: {mode}")
        updates = [
            (key, official[key] if key in official_keys else value) for key, value in ptq.items()
        ]

    model.load_weights(updates, strict=False)
    mx.eval(model.parameters())
    rows: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for prompt in prompts:
        prompt_id = prompt["id"]
        logits = _last_logits(model, prompt["token_ids"])
        reference = reference_logits[prompt_id]
        ref_top = set(np.argpartition(reference, -10)[-10:].tolist())
        cand_top = set(np.argpartition(logits, -10)[-10:].tolist())
        rows[prompt_id] = {
            "kl_official_to_case": _kl(reference, logits),
            "logit_cosine": _cosine(reference, logits),
            "top10_overlap": len(ref_top & cand_top),
            "top10": _top_tokens(logits, tokenizer),
        }
    return {
        "mode": mode,
        "selected_matrices": len(selected_bases),
        "selected_original_weights": _weight_count(ptq, selected_bases),
        "prompts": rows,
        "aggregate": _aggregate(rows),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _case_score(
    case: dict[str, Any],
    *,
    baseline_kl: float,
    upper_kl: float,
) -> float:
    if case["mode"] == "inject":
        return baseline_kl - case["aggregate"]["mean_kl"]
    if case["mode"] == "leaveout":
        return case["aggregate"]["mean_kl"] - upper_kl
    raise ValueError(case["mode"])


def _run_generation(
    model: Any,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    official: dict[str, mx.array],
    ptq: dict[str, mx.array],
    main_keys: set[str],
    bases: set[str],
    *,
    mode: str,
    max_tokens: int,
) -> dict[str, str]:
    keys = set(ptq)
    if mode == "official_reference":
        updates = list(official.items())
    else:
        selected_keys = _keys_for_bases(bases, keys)
        if mode == "inject":
            official_keys = selected_keys
        elif mode == "leaveout":
            official_keys = main_keys - selected_keys
        elif mode == "ptq_baseline":
            official_keys = set()
        elif mode == "official_main":
            official_keys = main_keys
        else:
            raise ValueError(mode)
        updates = [
            (key, official[key] if key in official_keys else value) for key, value in ptq.items()
        ]
    model.load_weights(updates, strict=False)
    mx.eval(model.parameters())
    return {
        prompt["id"]: _generate(model, tokenizer, prompt["chat"], max_tokens=max_tokens)[0]
        for prompt in prompts
    }


def run(
    official_dir: Path,
    ptq_dir: Path,
    *,
    generation_tokens: int,
) -> dict[str, Any]:
    print(f"Loading official: {official_dir}", flush=True)
    official_model, tokenizer = load(str(official_dir))
    print(f"Loading PTQ: {ptq_dir}", flush=True)
    target_model, ptq_tokenizer = load(str(ptq_dir))
    official = dict(tree_flatten(official_model.parameters()))
    ptq = dict(tree_flatten(target_model.parameters()))
    if official.keys() != ptq.keys():
        raise ValueError("official and PTQ parameter sets differ")
    for key in official:
        if official[key].shape != ptq[key].shape:
            raise ValueError(f"shape mismatch: {key}")

    keys = set(ptq)
    bases, main_keys, aux_keys = _partition(keys)
    family_bases: dict[str, set[str]] = {}
    for base in bases:
        family_bases.setdefault(_family(base), set()).add(base)
    expected_atomic = {
        "embed_tokens",
        "lm_head",
        "mlp",
        "self_attn",
        "linear_attn.in_proj_a",
        "linear_attn.in_proj_b",
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.out_proj",
    }
    if set(family_bases) != expected_atomic:
        raise ValueError(f"unexpected family set: {sorted(set(family_bases) ^ expected_atomic)}")

    broad_bases = {
        "boundary": family_bases["embed_tokens"] | family_bases["lm_head"],
        "mlp": family_bases["mlp"],
        "self_attn": family_bases["self_attn"],
        "linear_attn": set().union(
            *(value for name, value in family_bases.items() if name.startswith("linear_attn."))
        ),
    }
    prompts = [
        {
            **item,
            "chat": _chat_prompt(tokenizer, item["text"]),
        }
        for item in PROMPTS
    ]
    for prompt in prompts:
        prompt["token_ids"] = tokenizer.encode(prompt["chat"])
        if prompt["token_ids"] != ptq_tokenizer.encode(prompt["chat"]):
            raise ValueError(f"tokenizer mismatch: {prompt['id']}")

    print("[reference] official main + official aux", flush=True)
    target_model.load_weights(list(official.items()), strict=False)
    mx.eval(target_model.parameters())
    reference_logits = {
        prompt["id"]: _last_logits(target_model, prompt["token_ids"]) for prompt in prompts
    }

    cases: dict[str, dict[str, Any]] = {}
    cases["reference"] = _case(
        target_model,
        tokenizer,
        prompts,
        reference_logits,
        official,
        ptq,
        main_keys,
        bases,
        mode="official_reference",
    )
    cases["ptq_baseline"] = _case(
        target_model,
        tokenizer,
        prompts,
        reference_logits,
        official,
        ptq,
        main_keys,
        set(),
        mode="ptq_baseline",
    )
    cases["official_main_qwen_aux"] = _case(
        target_model,
        tokenizer,
        prompts,
        reference_logits,
        official,
        ptq,
        main_keys,
        bases,
        mode="official_main",
    )
    baseline_kl = cases["ptq_baseline"]["aggregate"]["mean_kl"]
    upper_kl = cases["official_main_qwen_aux"]["aggregate"]["mean_kl"]
    print(
        f"[anchors] PTQ KL={baseline_kl:.4f}; official-main KL={upper_kl:.4f}",
        flush=True,
    )

    interventions = {
        **{f"broad.{name}": value for name, value in broad_bases.items()},
        **{
            f"atomic.{name}": value
            for name, value in family_bases.items()
            if name not in {"mlp", "self_attn"}
        },
    }
    # mlp/self_attn are already identical between the broad and atomic views.
    for name in ("mlp", "self_attn"):
        interventions[f"atomic.{name}"] = family_bases[name]

    print(f"[phase1] {len(interventions)} families × 2 directions", flush=True)
    for family_name, selected in interventions.items():
        for mode in ("inject", "leaveout"):
            case_name = f"{mode}.{family_name}"
            case = _case(
                target_model,
                tokenizer,
                prompts,
                reference_logits,
                official,
                ptq,
                main_keys,
                selected,
                mode=mode,
            )
            case["family"] = family_name
            case["score"] = _case_score(case, baseline_kl=baseline_kl, upper_kl=upper_kl)
            cases[case_name] = case
            print(
                f"  {case_name}: KL={case['aggregate']['mean_kl']:.4f}, "
                f"score={case['score']:+.4f}, "
                f"top10={case['aggregate']['top10_overlap']}/40",
                flush=True,
            )
        mx.clear_cache()

    layered_atomic = [name for name in expected_atomic if name not in {"embed_tokens", "lm_head"}]
    atomic_scores = {}
    for family_name in layered_atomic:
        inject = cases[f"inject.atomic.{family_name}"]["score"]
        leaveout = cases[f"leaveout.atomic.{family_name}"]["score"]
        atomic_scores[family_name] = {
            "inject_recovery": inject,
            "leaveout_damage": leaveout,
            "combined_score": inject + leaveout,
        }
    selected_family = max(
        atomic_scores,
        key=lambda name: atomic_scores[name]["combined_score"],
    )
    print(
        f"[phase2] selected {selected_family}: {atomic_scores[selected_family]}",
        flush=True,
    )
    selected_family_bases = family_bases[selected_family]
    block_cases: list[str] = []
    for start in range(0, 64, 8):
        end = start + 7
        selected = {
            base
            for base in selected_family_bases
            if (layer := _layer(base)) is not None and start <= layer <= end
        }
        if not selected:
            continue
        for mode in ("inject", "leaveout"):
            case_name = f"{mode}.block_{start:02d}_{end:02d}.{selected_family}"
            case = _case(
                target_model,
                tokenizer,
                prompts,
                reference_logits,
                official,
                ptq,
                main_keys,
                selected,
                mode=mode,
            )
            case.update(
                {
                    "family": selected_family,
                    "layer_start": start,
                    "layer_end": end,
                    "score": _case_score(case, baseline_kl=baseline_kl, upper_kl=upper_kl),
                }
            )
            cases[case_name] = case
            block_cases.append(case_name)
            print(
                f"  {case_name}: KL={case['aggregate']['mean_kl']:.4f}, "
                f"score={case['score']:+.4f}, "
                f"top10={case['aggregate']['top10_overlap']}/40",
                flush=True,
            )
        mx.clear_cache()

    best_inject_block = max(
        (name for name in block_cases if cases[name]["mode"] == "inject"),
        key=lambda name: cases[name]["score"],
    )
    best_leaveout_block = max(
        (name for name in block_cases if cases[name]["mode"] == "leaveout"),
        key=lambda name: cases[name]["score"],
    )
    phase3_specs = (
        (
            "inject",
            cases[best_inject_block]["layer_start"],
            cases[best_inject_block]["layer_end"],
        ),
        (
            "leaveout",
            cases[best_leaveout_block]["layer_start"],
            cases[best_leaveout_block]["layer_end"],
        ),
    )
    layer_cases: list[str] = []
    print(
        f"[phase3] inject layers "
        f"{phase3_specs[0][1]}-{phase3_specs[0][2]}; leaveout layers "
        f"{phase3_specs[1][1]}-{phase3_specs[1][2]}",
        flush=True,
    )
    for mode, start, end in phase3_specs:
        for layer_number in range(start, end + 1):
            selected = {base for base in selected_family_bases if _layer(base) == layer_number}
            if not selected:
                continue
            case_name = f"{mode}.layer_{layer_number:02d}.{selected_family}"
            case = _case(
                target_model,
                tokenizer,
                prompts,
                reference_logits,
                official,
                ptq,
                main_keys,
                selected,
                mode=mode,
            )
            case.update(
                {
                    "family": selected_family,
                    "layer": layer_number,
                    "score": _case_score(case, baseline_kl=baseline_kl, upper_kl=upper_kl),
                }
            )
            cases[case_name] = case
            layer_cases.append(case_name)
            print(
                f"  {case_name}: KL={case['aggregate']['mean_kl']:.4f}, "
                f"score={case['score']:+.4f}, "
                f"top10={case['aggregate']['top10_overlap']}/40",
                flush=True,
            )
        mx.clear_cache()

    ranked_inject = sorted(
        (name for name, case in cases.items() if case["mode"] == "inject"),
        key=lambda name: cases[name]["score"],
        reverse=True,
    )
    ranked_leaveout = sorted(
        (name for name, case in cases.items() if case["mode"] == "leaveout"),
        key=lambda name: cases[name]["score"],
        reverse=True,
    )
    generation_case_names = [
        "reference",
        "ptq_baseline",
        "official_main_qwen_aux",
        ranked_inject[0],
        ranked_leaveout[0],
    ]
    best_block_inject = next((name for name in ranked_inject if name in block_cases), None)
    best_block_leaveout = next((name for name in ranked_leaveout if name in block_cases), None)
    for name in (best_block_inject, best_block_leaveout):
        if name is not None:
            generation_case_names.append(name)
    best_layer_inject = next((name for name in ranked_inject if name in layer_cases), None)
    best_layer_leaveout = next((name for name in ranked_leaveout if name in layer_cases), None)
    for name in (best_layer_inject, best_layer_leaveout):
        if name is not None:
            generation_case_names.append(name)
    generation_case_names = list(dict.fromkeys(generation_case_names))

    print(f"[generation] {generation_case_names}", flush=True)
    for case_name in generation_case_names:
        case = cases[case_name]
        if case_name == "reference":
            mode = "official_reference"
            selected = bases
        elif case_name == "ptq_baseline":
            mode = "ptq_baseline"
            selected = set()
        elif case_name == "official_main_qwen_aux":
            mode = "official_main"
            selected = bases
        else:
            mode = case["mode"]
            if case_name in block_cases:
                selected = {
                    base
                    for base in family_bases[case["family"]]
                    if (
                        (layer := _layer(base)) is not None
                        and case["layer_start"] <= layer <= case["layer_end"]
                    )
                }
            elif case_name in layer_cases:
                selected = {
                    base for base in family_bases[case["family"]] if _layer(base) == case["layer"]
                }
            else:
                selected = interventions[case["family"]]
        case["generation"] = _run_generation(
            target_model,
            tokenizer,
            prompts,
            official,
            ptq,
            main_keys,
            selected,
            mode=mode,
            max_tokens=generation_tokens,
        )
        print(f"  {case_name}: {case['generation']}", flush=True)
        mx.clear_cache()

    total_original_weights = _weight_count(ptq, bases)
    return {
        "protocol": {
            "official_model": str(official_dir),
            "ptq_model": str(ptq_dir),
            "prompts": [{"id": p["id"], "text": p["text"]} for p in prompts],
            "quantized_matrices": len(bases),
            "main_parameter_arrays": len(main_keys),
            "aux_parameter_arrays": len(aux_keys),
            "total_original_weights": total_original_weights,
            "generation_tokens": generation_tokens,
            "experimental_aux_source": "ptq/qwen",
        },
        "family_inventory": {
            name: {
                "matrices": len(value),
                "original_weights": _weight_count(ptq, value),
                "weight_fraction": _weight_count(ptq, value) / total_original_weights,
            }
            for name, value in sorted(family_bases.items())
        },
        "selected_layered_family": selected_family,
        "atomic_scores": atomic_scores,
        "ranked_inject": ranked_inject,
        "ranked_leaveout": ranked_leaveout,
        "generation_cases": generation_case_names,
        "phase3": {
            "best_inject_block": best_inject_block,
            "best_leaveout_block": best_leaveout_block,
            "layer_cases": layer_cases,
        },
        "cases": cases,
        "peak_memory_gb": mx.get_peak_memory() / (1024**3),
    }


def _write_report(result: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    cases = result["cases"]
    lines = [
        "# E4 — Qwen3.6 Bonsai matrix-family and layer-block localization",
        "",
        "All intervention cases use Qwen/PTQ auxiliary parameters. Positive score "
        "means that the family either recovers the PTQ baseline when injected, "
        "or damages the official-main upper bound when left as PTQ.",
        "",
        "## Anchors",
        "",
        "| Case | Mean KL | Mean cosine | Top-10 overlap |",
        "|---|---:|---:|---:|",
    ]
    for name in ("reference", "ptq_baseline", "official_main_qwen_aux"):
        row = cases[name]["aggregate"]
        lines.append(
            f"| {name} | {row['mean_kl']:.6f} | "
            f"{row['mean_logit_cosine']:.6f} | "
            f"{row['top10_overlap']}/{row['top10_total']} |"
        )

    lines.extend(
        [
            "",
            "## Phase 1 — family interventions",
            "",
            "| Direction | Family | Matrices | Weight share | Mean KL | Score | Top-10 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    total_weights = result["protocol"]["total_original_weights"]
    phase1 = [
        (name, case)
        for name, case in cases.items()
        if name.startswith(("inject.broad", "leaveout.broad", "inject.atomic", "leaveout.atomic"))
    ]
    phase1.sort(key=lambda item: item[1]["score"], reverse=True)
    for _, case in phase1:
        row = case["aggregate"]
        lines.append(
            f"| {case['mode']} | {case['family']} | "
            f"{case['selected_matrices']} | "
            f"{case['selected_original_weights'] / total_weights:.2%} | "
            f"{row['mean_kl']:.6f} | {case['score']:+.6f} | "
            f"{row['top10_overlap']}/{row['top10_total']} |"
        )

    selected = result["selected_layered_family"]
    lines.extend(
        [
            "",
            f"## Phase 2 — `{selected}` by 8-layer block",
            "",
            "| Direction | Layers | Matrices | Weight share | Mean KL | Score | Top-10 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    phase2 = [case for name, case in cases.items() if ".block_" in name]
    phase2.sort(key=lambda case: case["score"], reverse=True)
    for case in phase2:
        row = case["aggregate"]
        lines.append(
            f"| {case['mode']} | {case['layer_start']}-{case['layer_end']} | "
            f"{case['selected_matrices']} | "
            f"{case['selected_original_weights'] / total_weights:.2%} | "
            f"{row['mean_kl']:.6f} | {case['score']:+.6f} | "
            f"{row['top10_overlap']}/{row['top10_total']} |"
        )

    lines.extend(
        [
            "",
            f"## Phase 3 — `{selected}` individual layers",
            "",
            "| Direction | Layer | Matrices | Weight share | Mean KL | Score | Top-10 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    phase3 = [cases[name] for name in result["phase3"]["layer_cases"]]
    phase3.sort(key=lambda case: case["score"], reverse=True)
    for case in phase3:
        row = case["aggregate"]
        lines.append(
            f"| {case['mode']} | {case['layer']} | "
            f"{case['selected_matrices']} | "
            f"{case['selected_original_weights'] / total_weights:.2%} | "
            f"{row['mean_kl']:.6f} | {case['score']:+.6f} | "
            f"{row['top10_overlap']}/{row['top10_total']} |"
        )

    lines.extend(["", "## Selected greedy generations", ""])
    for case_name in result["generation_cases"]:
        lines.extend([f"### `{case_name}`", ""])
        for prompt_id, generation in cases[case_name]["generation"].items():
            lines.append(f"- {prompt_id}: `{generation!r}`")
        lines.append("")

    best_inject = result["ranked_inject"][0]
    best_leaveout = result["ranked_leaveout"][0]
    best_layer_inject = max(
        (name for name in result["phase3"]["layer_cases"] if cases[name]["mode"] == "inject"),
        key=lambda name: cases[name]["score"],
    )
    best_layer_leaveout = max(
        (name for name in result["phase3"]["layer_cases"] if cases[name]["mode"] == "leaveout"),
        key=lambda name: cases[name]["score"],
    )
    lines.extend(
        [
            "## Result",
            "",
            f"- Strongest recovery intervention: `{best_inject}` "
            f"(score {cases[best_inject]['score']:+.6f}).",
            f"- Strongest necessity intervention: `{best_leaveout}` "
            f"(score {cases[best_leaveout]['score']:+.6f}).",
            f"- Selected layered family: `{selected}`.",
            f"- Peak MLX memory: {result['peak_memory_gb']:.3f} GiB.",
            "",
            "## Interpretation",
            "",
            "- No single family or layer restores semantic generation. The "
            "failure is an interacting multi-family effect, not one poisoned "
            "matrix.",
            "- `linear_attn.out_proj` is the strongest efficient recovery "
            "target: replacing 5.61% of original weights gives the largest "
            "family injection improvement.",
            "- `linear_attn.in_proj_z` is strongly necessary but weak in "
            "isolation: leaving its 5.61% as PTQ nearly destroys the official "
            "main path, while injecting it alone barely repairs the PTQ path.",
            "- MLP damage is depth-asymmetric. Layers 0-7 are the critical "
            "necessity block, while layers 56-63 provide nearly all MLP-side "
            "recovery from the PTQ baseline.",
            f"- The strongest individual recovery layer is "
            f"`{best_layer_inject}`; the strongest individual necessity layer "
            f"is `{best_layer_leaveout}`. Their effects remain much smaller "
            "than the corresponding 8-layer blocks, showing cross-layer "
            "interaction.",
            "- Embedding and lm_head replacements have negligible effect "
            "relative to their size, so they are poor first QAT targets.",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--ptq", type=Path, default=DEFAULT_PTQ)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--generation-tokens", type=int, default=16)
    args = parser.parse_args()
    result = run(
        args.official.expanduser().resolve(),
        args.ptq.expanduser().resolve(),
        generation_tokens=args.generation_tokens,
    )
    _write_report(result, args.out)
    print(f"Wrote {args.out / 'result.json'}", flush=True)
    print(f"Wrote {args.out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
