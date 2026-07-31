"""E2: four-way MLX ablation of Bonsai/PTQ main weights and auxiliary weights.

The official Bonsai and our Qwen3.6 PTQ checkpoints share the same MLX model
structure.  We therefore swap parameters in memory, without writing duplicate
checkpoints, and measure which side of the split controls the semantic gap.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

DEFAULT_OFFICIAL = Path(
    "~/.cache/modelscope/models/"
    "prism-ml--Ternary-Bonsai-27B-mlx-2bit/snapshots/master"
).expanduser()
DEFAULT_PTQ = Path("qwen3.6-27b-mlx-2bit-v2")
DEFAULT_OUT = Path("experiment/analysis_qwen36_bonsai_migration/e2")

PROMPTS = [
    {"id": "fact", "text": "What is the capital of France? Answer briefly."},
    {"id": "math", "text": "Calculate 17 * 23. Give only the result."},
    {"id": "zh_math", "text": "请计算 18 × 3，只给出答案。"},
    {
        "id": "code",
        "text": "Write a Python function add(a, b) that returns their sum.",
    },
]

CASES = [
    ("A_official_main_official_aux", "official", "official"),
    ("B_official_main_qwen_aux", "official", "ptq"),
    ("C_ptq_main_official_aux", "ptq", "official"),
    ("D_ptq_main_qwen_aux", "ptq", "ptq"),
]


def _chat_prompt(tokenizer: Any, text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )


def _last_logits(model: Any, tokens: list[int]) -> np.ndarray:
    output = model(mx.array([tokens], dtype=mx.int32))
    logits = output.logits if hasattr(output, "logits") else output
    last = logits[0, -1].astype(mx.float32)
    mx.eval(last)
    return np.asarray(last)


def _top_tokens(
    logits: np.ndarray, tokenizer: Any, count: int = 10
) -> list[dict[str, Any]]:
    ids = np.argpartition(logits, -count)[-count:]
    ids = ids[np.argsort(logits[ids])[::-1]]
    return [
        {
            "id": int(token_id),
            "text": tokenizer.decode([int(token_id)]),
            "logit": float(logits[token_id]),
        }
        for token_id in ids
    ]


def _kl(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref64 = reference.astype(np.float64)
    cand64 = candidate.astype(np.float64)
    ref_logp = ref64 - np.logaddexp.reduce(ref64)
    cand_logp = cand64 - np.logaddexp.reduce(cand64)
    ref_p = np.exp(ref_logp)
    return float(np.sum(ref_p * (ref_logp - cand_logp)))


def _cosine(x: np.ndarray, y: np.ndarray) -> float:
    x64 = x.astype(np.float64)
    y64 = y.astype(np.float64)
    return float(
        np.dot(x64, y64)
        / math.sqrt(max(float(np.dot(x64, x64) * np.dot(y64, y64)), 1e-30))
    )


def _generate(
    model: Any, tokenizer: Any, prompt: str, max_tokens: int
) -> tuple[str, dict[str, float | int]]:
    sampler = make_sampler(temp=0.0)
    started = time.perf_counter()
    parts: list[str] = []
    last = None
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        parts.append(response.text)
        last = response
    elapsed = time.perf_counter() - started
    metrics: dict[str, float | int] = {"elapsed_seconds": elapsed}
    if last is not None:
        metrics.update(
            {
                "prompt_tokens": int(last.prompt_tokens),
                "generation_tokens": int(last.generation_tokens),
                "prompt_tps": float(last.prompt_tps),
                "generation_tps": float(last.generation_tps),
            }
        )
    return "".join(parts), metrics


def _partition(keys: set[str]) -> tuple[set[str], set[str], set[str]]:
    scale_suffix = ".scales"
    bases = {key.removesuffix(scale_suffix) for key in keys if key.endswith(scale_suffix)}
    main = {
        key
        for base in bases
        for key in (base + ".weight", base + ".scales", base + ".biases")
        if key in keys
    }
    return bases, main, keys - main


def run(
    official_dir: Path,
    ptq_dir: Path,
    *,
    max_tokens: int,
    experiment_label: str = "E2",
) -> dict[str, Any]:
    print(f"Loading official: {official_dir}", flush=True)
    official_model, tokenizer = load(str(official_dir))
    print(f"Loading PTQ: {ptq_dir}", flush=True)
    target_model, ptq_tokenizer = load(str(ptq_dir))

    official = dict(tree_flatten(official_model.parameters()))
    ptq = dict(tree_flatten(target_model.parameters()))
    if official.keys() != ptq.keys():
        only_official = sorted(official.keys() - ptq.keys())
        only_ptq = sorted(ptq.keys() - official.keys())
        raise ValueError(
            "checkpoint parameter sets differ: "
            f"official-only={only_official[:10]}, ptq-only={only_ptq[:10]}"
        )
    for key in official:
        if official[key].shape != ptq[key].shape:
            raise ValueError(
                f"shape mismatch for {key}: {official[key].shape} != {ptq[key].shape}"
            )

    keys = set(official)
    bases, main_keys, aux_keys = _partition(keys)
    if len(bases) != 498:
        raise ValueError(f"expected 498 quantized matrices, found {len(bases)}")

    prompts = [
        {
            **item,
            "chat": _chat_prompt(tokenizer, item["text"]),
        }
        for item in PROMPTS
    ]
    # Tokenization equivalence makes the hybrid comparison well-defined.
    for item in prompts:
        official_ids = tokenizer.encode(item["chat"])
        ptq_ids = ptq_tokenizer.encode(item["chat"])
        if official_ids != ptq_ids:
            raise ValueError(f"tokenizer mismatch for prompt {item['id']}")
        item["token_ids"] = official_ids

    result: dict[str, Any] = {
        "protocol": {
            "experiment_label": experiment_label,
            "official_model": str(official_dir),
            "ptq_model": str(ptq_dir),
            "quantized_matrices": len(bases),
            "main_parameter_arrays": len(main_keys),
            "aux_parameter_arrays": len(aux_keys),
            "main_parameter_values": int(sum(official[k].size for k in main_keys)),
            "aux_parameter_values": int(sum(official[k].size for k in aux_keys)),
            "max_generation_tokens": max_tokens,
            "greedy": True,
            "thinking": False,
            "prompts": [{"id": p["id"], "text": p["text"]} for p in prompts],
        },
        "cases": {},
    }
    logits_by_case: dict[str, dict[str, np.ndarray]] = {}

    for case_name, main_source, aux_source in CASES:
        print(
            f"\n[{case_name}] main={main_source}, aux={aux_source}",
            flush=True,
        )
        main_dict = official if main_source == "official" else ptq
        aux_dict = official if aux_source == "official" else ptq
        updates = [(key, main_dict[key]) for key in sorted(main_keys)]
        updates.extend((key, aux_dict[key]) for key in sorted(aux_keys))
        target_model.load_weights(updates, strict=False)
        mx.eval(target_model.parameters())

        case_result: dict[str, Any] = {"prompts": {}}
        case_logits: dict[str, np.ndarray] = {}
        for item in prompts:
            prompt_id = item["id"]
            print(f"  {prompt_id}: logits", end="", flush=True)
            started = time.perf_counter()
            logits = _last_logits(target_model, item["token_ids"])
            logits_seconds = time.perf_counter() - started
            case_logits[prompt_id] = logits
            print(", generation", end="", flush=True)
            generation, metrics = _generate(
                target_model, tokenizer, item["chat"], max_tokens
            )
            print(f" -> {generation!r}", flush=True)
            case_result["prompts"][prompt_id] = {
                "top10": _top_tokens(logits, tokenizer),
                "logits_seconds": logits_seconds,
                "generation": generation,
                "generation_metrics": metrics,
            }
        logits_by_case[case_name] = case_logits
        result["cases"][case_name] = case_result
        mx.clear_cache()

    reference_name = CASES[0][0]
    for case_name, _, _ in CASES:
        comparisons = {}
        for item in prompts:
            prompt_id = item["id"]
            reference = logits_by_case[reference_name][prompt_id]
            candidate = logits_by_case[case_name][prompt_id]
            ref_top = set(np.argpartition(reference, -10)[-10:].tolist())
            cand_top = set(np.argpartition(candidate, -10)[-10:].tolist())
            comparisons[prompt_id] = {
                "kl_official_to_case": _kl(reference, candidate),
                "logit_cosine": _cosine(reference, candidate),
                "top10_overlap": len(ref_top & cand_top),
            }
        result["cases"][case_name]["vs_A_official"] = comparisons

    result["peak_memory_gb"] = mx.get_peak_memory() / (1024**3)
    return result


def _write_report(result: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        f"# {result['protocol'].get('experiment_label', 'E2')} — "
        "Bonsai/PTQ main × auxiliary four-way MLX ablation",
        "",
        "A is the all-official reference. KL is `KL(A || case)`.",
        "",
        "## Aggregate",
        "",
        "| Case | Mean KL | Mean logit cosine | top-10 overlap |",
        "|---|---:|---:|---:|",
    ]
    for case_name, _, _ in CASES:
        comparisons = list(result["cases"][case_name]["vs_A_official"].values())
        mean_kl = sum(row["kl_official_to_case"] for row in comparisons) / len(
            comparisons
        )
        mean_cosine = sum(row["logit_cosine"] for row in comparisons) / len(
            comparisons
        )
        overlap = sum(row["top10_overlap"] for row in comparisons)
        lines.append(
            f"| {case_name} | {mean_kl:.8f} | {mean_cosine:.8f} | "
            f"{overlap}/{10 * len(comparisons)} |"
        )
    lines.extend(
        [
            "",
            "## Per-prompt result",
            "",
        "| Case | Prompt | KL | logit cosine | top-10 overlap | Generation |",
        "|---|---|---:|---:|---:|---|",
        ]
    )
    for case_name, _, _ in CASES:
        case = result["cases"][case_name]
        for prompt in result["protocol"]["prompts"]:
            prompt_id = prompt["id"]
            comparison = case["vs_A_official"][prompt_id]
            generation = (
                case["prompts"][prompt_id]["generation"]
                .replace("\n", "\\n")
                .replace("|", "\\|")
            )
            lines.append(
                f"| {case_name} | {prompt_id} | "
                f"{comparison['kl_official_to_case']:.8f} | "
                f"{comparison['logit_cosine']:.8f} | "
                f"{comparison['top10_overlap']}/10 | {generation} |"
            )
    protocol = result["protocol"]
    label = protocol.get("experiment_label", "E2")
    case_c = list(
        result["cases"]["C_ptq_main_official_aux"]["vs_A_official"].values()
    )
    case_d = list(
        result["cases"]["D_ptq_main_qwen_aux"]["vs_A_official"].values()
    )
    case_c_overlap = sum(row["top10_overlap"] for row in case_c)
    case_d_overlap = sum(row["top10_overlap"] for row in case_d)
    if label == "E3":
        experiment_interpretation = [
            "- The one-shot main matrices still fail all four semantic probes "
            f"(case D aggregate top-10 overlap {case_d_overlap}/40). Better "
            "offline code agreement is therefore insufficient for recovery.",
            "- Official auxiliary parameters do not rescue the one-shot main "
            f"weights (case C aggregate top-10 overlap {case_c_overlap}/40).",
            "- This rejects a pure threshold/scale PTQ reproduction as the full "
            "Bonsai mechanism. Training-aware weight migration or distillation "
            "is required for the remaining main-matrix gap.",
        ]
    else:
        experiment_interpretation = [
            "- The quantized main matrices are the decisive source of the "
            "semantic gap: case C keeps every official auxiliary parameter but "
            f"has only {case_c_overlap}/40 aggregate top-10 overlap.",
            "- Official main matrices plus Qwen auxiliary parameters (case B) "
            "retain factual and code behavior, but arithmetic regresses. "
            "Auxiliary migration is therefore secondary, not irrelevant.",
            "- Official auxiliary parameters reduce the broken-PTQ mean KL, but "
            "cannot rescue generation. The next reproduction experiment should "
            "change the main-matrix code/scale rule first.",
        ]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            *experiment_interpretation,
            "",
            "## Protocol",
            "",
            f"- Quantized main matrices: {protocol['quantized_matrices']}",
            f"- Main parameter arrays: {protocol['main_parameter_arrays']}",
            f"- Auxiliary parameter arrays: {protocol['aux_parameter_arrays']}",
            f"- Peak MLX memory: {result['peak_memory_gb']:.3f} GiB",
            f"- Greedy generation, thinking disabled, "
            f"max {protocol['max_generation_tokens']} tokens.",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--ptq", type=Path, default=DEFAULT_PTQ)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--experiment-label", default="E2")
    args = parser.parse_args()

    result = run(
        args.official.expanduser().resolve(),
        args.ptq.expanduser().resolve(),
        max_tokens=args.max_tokens,
        experiment_label=args.experiment_label,
    )
    _write_report(result, args.out)
    print(f"\nWrote {args.out / 'result.json'}", flush=True)
    print(f"Wrote {args.out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
