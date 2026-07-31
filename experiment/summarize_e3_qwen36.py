"""Summarize the E1/E2/E3 progression without rerunning 27B inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path("experiment/analysis_qwen36_bonsai_migration")


def _read(name: str) -> dict[str, Any]:
    return json.loads((ROOT / name / "result.json").read_text(encoding="utf-8"))


def _aggregate(result: dict[str, Any], case: str) -> dict[str, float | int]:
    rows = list(result["cases"][case]["vs_A_official"].values())
    return {
        "mean_kl": sum(row["kl_official_to_case"] for row in rows) / len(rows),
        "mean_logit_cosine": sum(row["logit_cosine"] for row in rows) / len(rows),
        "top10_overlap": sum(row["top10_overlap"] for row in rows),
        "top10_total": 10 * len(rows),
    }


def main() -> None:
    e1 = _read("e1")
    e2 = _read("e2")
    e3 = _read("e3")
    summary = {
        "offline_code_reproduction": {
            "v2_8ls_code_agreement": e1["summary"]["report_code_agreement"],
            "v3_oneshot_code_agreement": e1["summary"][
                "one_shot_code_agreement"
            ],
            "official_zero_rate": e1["summary"]["official_zero_rate"],
            "v2_8ls_zero_rate": e1["summary"]["report_zero_rate"],
            "v3_oneshot_zero_rate": e1["summary"]["one_shot_zero_rate"],
        },
        "official_aux": {
            "v2_8ls": _aggregate(e2, "C_ptq_main_official_aux"),
            "v3_oneshot": _aggregate(e3, "C_ptq_main_official_aux"),
        },
        "qwen_aux": {
            "v2_8ls": _aggregate(e2, "D_ptq_main_qwen_aux"),
            "v3_oneshot": _aggregate(e3, "D_ptq_main_qwen_aux"),
        },
        "conclusion": (
            "One-shot absmean/0.5 improves offline code agreement and the "
            "all-PTQ first-token distribution slightly, but does not recover "
            "semantic generation. Pure threshold/scale PTQ is insufficient."
        ),
    }
    out = ROOT / "e3"
    (out / "comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    offline = summary["offline_code_reproduction"]
    lines = [
        "# E3 — v2 8-LS versus v3 one-shot",
        "",
        "## Offline reproduction",
        "",
        "| Candidate | Code agreement | Zero rate |",
        "|---|---:|---:|",
        f"| Official | 100.00% | {offline['official_zero_rate']:.2%} |",
        f"| v2 8-LS | {offline['v2_8ls_code_agreement']:.2%} | "
        f"{offline['v2_8ls_zero_rate']:.2%} |",
        f"| v3 one-shot | {offline['v3_oneshot_code_agreement']:.2%} | "
        f"{offline['v3_oneshot_zero_rate']:.2%} |",
        "",
        "## End-to-end first-token comparison",
        "",
        "| Aux parameters | Candidate | Mean KL | Mean cosine | Top-10 overlap |",
        "|---|---|---:|---:|---:|",
    ]
    for aux_name, aux_label in (
        ("official_aux", "Official"),
        ("qwen_aux", "Qwen"),
    ):
        for candidate, candidate_label in (
            ("v2_8ls", "v2 8-LS"),
            ("v3_oneshot", "v3 one-shot"),
        ):
            row = summary[aux_name][candidate]
            lines.append(
                f"| {aux_label} | {candidate_label} | {row['mean_kl']:.6f} | "
                f"{row['mean_logit_cosine']:.6f} | "
                f"{row['top10_overlap']}/{row['top10_total']} |"
            )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "One-shot `absmean + 0.5` is a better container-level reproduction, "
            "but it does not restore semantic generation. With Qwen auxiliary "
            "parameters it changes mean KL from "
            f"{summary['qwen_aux']['v2_8ls']['mean_kl']:.3f} to "
            f"{summary['qwen_aux']['v3_oneshot']['mean_kl']:.3f} and cosine "
            f"from {summary['qwen_aux']['v2_8ls']['mean_logit_cosine']:.3f} to "
            f"{summary['qwen_aux']['v3_oneshot']['mean_logit_cosine']:.3f}; "
            "all four generations remain broken.",
            "",
            "The next experiment must target training-aware main-matrix "
            "migration rather than another global threshold sweep.",
            "",
        ]
    )
    (out / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
