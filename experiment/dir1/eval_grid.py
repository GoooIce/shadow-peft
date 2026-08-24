"""Direction-① full-grid eval + 挽回率 verdict (ticket 09).

Loads the three per-tier adapters trained by ``train_tier.py``, runs the
6-cell × dual-path grid (05 harness), captures shadow-state spectra on the
real checkpoints (08 suite, independent report), computes per-cell ppl
挽回率 and the spec verdict (W4A16 主格 ≥50% 成功 / 30–50% 降叙事 / <30% 放
弃), and writes report.md + report.json + budget accounting. Auto power-off
+ WeChat notify at the end.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file as safetensors_load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dir1.ops import BudgetLedger, StageTimer, power_off_now, wechat_notify  # noqa: E402
from dir1.smoke_12b import load_causal_model  # noqa: E402
from lab import grid, spectral  # noqa: E402
from lab.atoms import load_teacher_data, load_wikitext2_test_text, make_batch  # noqa: E402

import random  # noqa: E402

from shadow_peft import ShadowConfig, get_shadow_model  # noqa: E402

VERDICTS = {"success": "成功(挽回 ≥50%)", "reduced": "降叙事(30–50%,「挽回可观部分」)", "abandon": "放弃(<30%,先复盘管线)"}


def ppl_recovery(ppl_quant: float, ppl_quant_shadow: float, ppl_bf16: float) -> float | None:
    denom = ppl_quant - ppl_bf16
    if denom <= 0:
        return None
    return (ppl_quant - ppl_quant_shadow) / denom


def verdict_of(recovery: float | None) -> str:
    if recovery is None:
        return "indeterminate"
    if recovery >= 0.5:
        return "success"
    if recovery >= 0.3:
        return "reduced"
    return "abandon"


def load_adapter(path: str | Path) -> dict:
    p = Path(path) / "shadow_adapter.safetensors"
    return dict(safetensors_load_file(str(p)))


def load_mmlu_probe(tokenizer, n: int):
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="validation")
    letter = ["A", "B", "C", "D"]
    out = []
    for row in list(ds)[:n]:
        prompt = (
            f"Question: {row['question']}\n"
            + "".join(f"{letter[i]}. {c}\n" for i, c in enumerate(row["choices"]))
            + "Answer:"
        )
        out.append({"prompt_ids": tokenizer.encode(prompt), "answer": letter[row["answer"]]})
    return out


def load_gsm8k_probe(tokenizer, n: int):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for row in list(ds)[:n]:
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"prompt_ids": tokenizer.encode(row["question"] + "\nAnswer:"), "answer": gold})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-12b-it")
    ap.add_argument("--runs-root", default="experiment/runs/dir1")
    ap.add_argument("--data", default="experiment/data/teacher_gemma4_12b.jsonl")
    ap.add_argument("--out", default="experiment/runs/dir1/eval")
    ap.add_argument("--budget-file", default="experiment/runs/dir1/budget.jsonl")
    ap.add_argument("--mmlu-n", type=int, default=32)
    ap.add_argument("--gsm8k-n", type=int, default=32)
    ap.add_argument("--kld-batches", type=int, default=4)
    ap.add_argument("--spectral-batches", type=int, default=4)
    ap.add_argument("--no-shutdown", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def fresh_base():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return load_causal_model(args.model, torch.bfloat16, device)

    # Trained adapters per tier (06/07 outputs). 铁律: only these fresh dir1
    # checkpoints are ever loaded — legacy adapters are banned.
    adapters = {t: load_adapter(Path(args.runs_root) / t / "final") for t in grid.tiers.TIERS}

    shadow_cfg = ShadowConfig(
        num_shadow_layers=2, injection_hidden_size=64, gate_hidden_size=32,
        alpha=0.1, dropout=0.0,
    )

    rng = random.Random(0)
    seqs = load_teacher_data(args.data)
    kld_batches = [
        make_batch(seqs, rng, 1, 256, str(device), pad_id=pad_id)
        for _ in range(args.kld_batches)
    ]
    eval_text = load_wikitext2_test_text()

    ledger = BudgetLedger(args.budget_file)

    with StageTimer(ledger, "eval_grid", note="6 cells × dual path + anchor + bare"):
        result = grid.run_grid(
            base_factory=fresh_base,
            shadow_config=shadow_cfg,
            adapter_state=adapters,
            tokenizer=tokenizer,
            eval_text=eval_text,
            device=device,
            kld_batches=kld_batches,
            mmlu_examples=load_mmlu_probe(tokenizer, args.mmlu_n),
            gsm8k_examples=load_gsm8k_probe(tokenizer, args.gsm8k_n),
            decode=tokenizer.decode,
            ppl_chunks=20,
        )

    # --- 挽回率 + 判定 --------------------------------------------------------
    anchor_ppl = result["anchor"]["ppl"]
    bare = {b["backbone_tier"]: b["ppl"] for b in result["bare_bases"] if b["kernel_path"] != "fake_quant"}
    verdicts = {}
    for cell in result["cells"]:
        tier = cell["cell"]["backbone_tier"]
        rec = ppl_recovery(bare[tier], cell["ppl"], anchor_ppl)
        cell["ppl_recovery"] = rec
        if tier == "w4a16" and cell["cell"]["shadow_precision"] == "bf16":
            verdicts[cell["cell"]["kernel_path"]] = {
                "recovery": rec, "verdict": verdict_of(rec)
            }

    # --- 谱诊断独立报告段(真实 checkpoint) ----------------------------------
    # Per tier: same trained adapter on a bf16 backbone (pre) vs its tiered
    # backbone (post) — the shadow-state spectral drift quantization causes.
    with StageTimer(ledger, "spectral_capture", note="pre/post per tier on real checkpoints"):
        try:
            from lab import tiers as _tiers

            drift, pre_metrics, post_metrics = {}, {}, {}
            for tier in grid.tiers.TIERS:
                base_pre = fresh_base()
                peft_pre = get_shadow_model(base_pre, shadow_cfg)
                peft_pre.load_adapter_state_dict(adapters[tier])
                peft_pre.eval()
                col_pre = spectral.ShadowStateCollector(peft_pre)
                col_pre.run(peft_pre, kld_batches[: args.spectral_batches])
                pre_metrics[tier] = spectral.metrics_from_capture(col_pre.captures)

                base_post = fresh_base()
                _tiers.apply_tier_conversion(base_post, tier, device)
                peft_post = get_shadow_model(base_post, shadow_cfg)
                peft_post.load_adapter_state_dict(adapters[tier])
                peft_post.eval()
                col_post = spectral.ShadowStateCollector(peft_post)
                col_post.run(peft_post, kld_batches[: args.spectral_batches])
                post_metrics[tier] = spectral.metrics_from_capture(col_post.captures)

                drift[tier] = spectral.drift_report(col_pre.captures, col_post.captures)["summary"]
                del base_pre, peft_pre, base_post, peft_post
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            spectral_report = spectral.render_report(
                {f"pre/{t}/{k}": v for t, d in pre_metrics.items() for k, v in d.items()},
                {f"post/{t}/{k}": v for t, d in post_metrics.items() for k, v in d.items()},
                title="dir1 shadow-state spectral diagnostics (real checkpoints)",
            ) + "\n\n## drift summary (pre → post, per tier)\n\n" + json.dumps(
                drift, indent=1, ensure_ascii=False
            )
        except Exception as exc:
            spectral_report = f"spectral capture failed: {exc!r}"

    # --- 汇总落盘 -------------------------------------------------------------
    budget = ledger.summary()
    report_lines = [
        "# 方向① 全网格实跑 + 挽回率判定",
        "",
        f"- 预算核算: {budget['total_hours']:.2f} GPU 时 ≈ ¥{budget['total_cny']} (上限 80 时 / ¥230)",
        f"- anchor (bf16+bf16 shadow) ppl: {anchor_ppl:.3f}",
        f"- bare ppls: {json.dumps({k: round(v, 3) for k, v in bare.items()})}",
        "",
        "## 挽回率判定(W4A16 主格)",
        "",
    ]
    for path, v in verdicts.items():
        report_lines.append(
            f"- kernel={path}: 挽回率 {v['recovery'] if v['recovery'] is None else round(v['recovery'], 4)} "
            f"→ {VERDICTS.get(v['verdict'], v['verdict'])}"
        )
    report_lines += ["", "## Pareto", "", result["pareto"], "", "## 谱诊断(独立报告)", "", spectral_report]

    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    grid.write_metrics(result, out_dir / "metrics.json")
    (out_dir / "spectral.md").write_text(spectral_report, encoding="utf-8")

    status = "ok"
    wechat_notify(
        "dir1 全网格评测完成",
        f"verdict: {verdicts}; budget ¥{budget['total_cny']}/{230}",
    )
    if not args.no_shutdown:
        power_off_now()
    print(json.dumps({"status": status, "budget": budget, "verdicts": verdicts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
