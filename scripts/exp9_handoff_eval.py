#!/usr/bin/env python3
"""Run the exp9 token-level handoff evaluation bench (交接评测台).

Examples::

    PYTHONPATH=src python scripts/exp9_handoff_eval.py --backend synthetic --n 32 \
        --seed 9 --out runs/exp9/handoff_eval

    PYTHONPATH=src python scripts/exp9_handoff_eval.py --backend mock \
        --dataset runs/exp9/dataset.jsonl --strategies legacy,probe,oracle \
        --probe runs/exp9/probe.json --out runs/exp9/handoff_eval_probe

    PYTHONPATH=src python scripts/exp9_handoff_eval.py --backend synthetic \
        --export-candidates runs/exp9/candidates.jsonl --out runs/exp9/smoke

The ``transformers`` and ``openai`` backends need ``--backend-config`` pointing
at a JSON object accepted by ``edge_cloud.backends.build_backend`` (for example
``{"kind": "transformers", "model_path": "...", "shadow_ckpt": "..."}``). The
cloud side is always the dataset's scripted ``cloud_answer`` so every backend
kind produces the same artifact schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from edge_cloud.backends import Backend, build_backend
from edge_cloud.handoff_eval import (
    DEFAULT_COST_MODEL,
    STRATEGIES,
    EvalSample,
    LatencyModel,
    export_candidates,
    load_dataset,
    run_evaluation,
    synthetic_dataset,
    write_candidates,
)


def _parse_strategies(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    unknown = sorted(set(names) - set(STRATEGIES))
    if not names:
        raise SystemExit("--strategies 不能为空")
    if unknown:
        raise SystemExit(f"未知策略: {', '.join(unknown)}; 可选 {', '.join(STRATEGIES)}")
    return names


def _real_backend_factory(spec_path: str, kind: str):
    payload = json.loads(Path(spec_path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("--backend-config 顶层必须是 JSON 对象")
    payload.setdefault("kind", kind)
    backend: Backend = build_backend(payload)
    return lambda sample: backend


def _load_samples(args: argparse.Namespace) -> tuple[list[EvalSample], str]:
    if args.dataset:
        return load_dataset(args.dataset), str(args.dataset)
    if args.backend == "synthetic":
        return synthetic_dataset(args.n, seed=args.seed), "synthetic"
    raise SystemExit(f"--backend {args.backend} 需要 --dataset（JSONL prompt 集）")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="exp9 token 级端云交接评测台（四策略、曲线、bootstrap、候选导出）"
    )
    parser.add_argument(
        "--backend",
        choices=("synthetic", "mock", "transformers", "openai"),
        default="synthetic",
        help="synthetic/mock 用确定性 mock；transformers/openai 复用 build_backend",
    )
    parser.add_argument("--dataset", help="JSONL prompt 集；未给出时 synthetic 自动生成")
    parser.add_argument("--n", type=int, default=32, help="synthetic prompt 数量")
    parser.add_argument("--seed", type=int, default=9, help="数据与 bootstrap 随机种子")
    parser.add_argument("--bootstrap", type=int, default=200, help="bootstrap 重采样次数，0 关闭")
    parser.add_argument(
        "--strategies",
        default=",".join(STRATEGIES),
        help=f"逗号分隔，可选 {', '.join(STRATEGIES)}",
    )
    parser.add_argument("--out", default="runs/exp9/handoff_eval", help="输出目录")
    parser.add_argument("--probe", help="probe JSON 路径；缺失时 probe 策略回退 legacy")
    parser.add_argument("--threshold", type=float, default=0.62, help="legacy 置信度阈值")
    parser.add_argument("--cloud-rtt-ms", type=float, default=None, help="覆盖合成 RTT")
    parser.add_argument(
        "--backend-config",
        help="transformers/openai 后端的 JSON 配置（传给 build_backend）",
    )
    parser.add_argument(
        "--export-candidates",
        help="额外导出 probe 训练候选 JSONL（prompt_id/features/label）",
    )
    args = parser.parse_args(argv)

    strategies = _parse_strategies(args.strategies)
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold 必须在 [0, 1] 内")
    if args.bootstrap < 0:
        raise SystemExit("--bootstrap 不能为负数")
    if args.backend in {"transformers", "openai"} and not args.backend_config:
        raise SystemExit(
            f"--backend {args.backend} 需要 --backend-config（模型路径、shadow_ckpt 等），"
            "当前评测台的云端答案由数据集脚本提供"
        )
    if args.backend in {"synthetic", "mock"} and args.backend_config:
        raise SystemExit("--backend-config 只用于 transformers/openai 后端")

    samples, source = _load_samples(args)
    if not samples:
        raise SystemExit("数据集为空，无法评测")

    rtt = args.cloud_rtt_ms
    cost_model = (
        DEFAULT_COST_MODEL
        if rtt is None
        else replace(DEFAULT_COST_MODEL, cloud_rtt_ms=max(0.0, float(rtt)))
    )
    latency_model = LatencyModel(
        cloud_rtt_ms=(cost_model.cloud_rtt_ms if rtt is not None else LatencyModel().cloud_rtt_ms)
    )
    factory = (
        _real_backend_factory(args.backend_config, args.backend)
        if args.backend in {"transformers", "openai"}
        else None
    )

    if args.export_candidates:
        rows = export_candidates(samples, backend_factory=factory)
        path = write_candidates(rows, args.export_candidates)
        positives = sum(row["label"] for row in rows)
        print(
            f"candidates: {len(rows)} rows, positives {positives} "
            f"({positives / len(rows):.1%}) -> {path}"
        )

    run = run_evaluation(
        samples,
        strategies=strategies,
        seed=args.seed,
        n_boot=args.bootstrap,
        probe_path=args.probe,
        escalation_threshold=args.threshold,
        backend_kind=args.backend,
        backend_factory=factory,
        cost_model=cost_model,
        latency_model=latency_model,
        dataset_source=source,
    )
    paths = run.write(args.out)
    results = run.results

    print(f"dataset : {len(samples)} prompts ({source})")
    print(
        f"backend : {results['backend']['kind']} "
        f"({results['backend']['edge_model'] or 'n/a'}), "
        f"streaming={results['backend']['streaming']}"
    )
    for strategy in strategies:
        metrics = results["strategies"][strategy]
        position = metrics["handoff_position"]["mean"]
        position_text = "n/a" if position is None else f"{position:.1f}"
        print(
            f"{strategy:>9}: acc={metrics['accuracy']:.3f} "
            f"exact={metrics['exact_match_rate']:.3f} "
            f"escalate={metrics['escalation_rate']:.3f} "
            f"handoff={metrics['handoff_rate']:.3f} @{position_text} "
            f"cost={metrics['total_cost']:.6f} "
            f"p95={metrics['latency_ms']['p95']:.1f}ms"
        )
    print(f"results : {paths['results']}")
    print(f"summary : {paths['summary']}")
    print(f"curves  : {paths['curves']}")
    print(f"prompts : {paths['per_prompt']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
