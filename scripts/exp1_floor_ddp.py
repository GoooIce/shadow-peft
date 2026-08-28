"""exp1-floor P2: two-node DDP token-ladder run (spec: exp1-ppl-floor).

Same verdict shape as exp1_rung.py (ppl_bf16 / ppl_quant / ppl_shadow /
recovery) plus shadow_layers and tokens, so ladder points land on one axis.
Rank 0 measures the two baselines while rank 1 walks straight into
train_shadow (the first grad all_reduce is the rendezvous).

Launch (both nodes, over the 200G direct link; see spec playbook):

    ssh goooinext 'cd ~/Code/shadow-peft && PYTHONUNBUFFERED=1 \
      NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_GID_INDEX=3 nohup .venv/bin/torchrun \
      --master-addr=10.100.144.2 --master-port=29500 --nnodes=2 --nproc-per-node=1 \
      --node-rank=0 scripts/exp1_floor_ddp.py --shadow-layers <L> --max-tokens <T> \
      > runs/logs/<date>-floor-ddp-L<T>.log 2>&1 < /dev/null &'
    # gooo: same with --node-rank=1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_lab.atoms import despace_wikitext, eval_ppl, load_wikitext2_test_text
from shadow_lab.trainer import TrainConfig, _maybe_init_distributed, _quantize_base, train_shadow

AXIS_CTX = 512
AXIS_CHUNKS = 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow-layers", type=int, default=2)
    ap.add_argument("--max-tokens", type=float, required=True)
    ap.add_argument("--model", default=str(Path.home() / "models" / "LFM2.5-350M-Base"))
    ap.add_argument("--data", default="runs/exp1/teacher_floor_pool.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-seq", type=int, default=8, help="GLOBAL batch (split across ranks)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--same-data", action="store_true",
                    help="SHADOW_DIST_SAME_DATA equivalence smoke")
    ap.add_argument("--skip-baselines", action="store_true",
                    help="skip bf16/quant baselines (equivalence smoke: judge on train_log losses)")
    args = ap.parse_args()

    ddp = _maybe_init_distributed()
    rank = ddp[0] if ddp else 0

    out_dir = Path(args.out_dir)
    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    axis_text = despace_wikitext(load_wikitext2_test_text())

    def ppl(model) -> float:
        return eval_ppl(model, tok, axis_text, chunks=AXIS_CHUNKS, device="cuda",
                        context_len=AXIS_CTX)

    ppl_bf16 = ppl_quant = float("nan")
    if rank == 0 and not args.skip_baselines:
        out_dir.mkdir(parents=True, exist_ok=True)
        print("=== baseline 1/2 bf16 (rank0) ===", flush=True)
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
        ppl_bf16 = ppl(base)
        print(f"ppl_bf16 = {ppl_bf16:.4f}", flush=True)
        del base
        torch.cuda.empty_cache()
        print("=== baseline 2/2 quant-only (rank0) ===", flush=True)
        qc = TrainConfig(bits=2, model=args.model, data=args.data)
        quant_copy = (AutoModelForCausalLM.from_pretrained(args.model, dtype=qc.profile.dtype)
                      .to("cuda").eval())
        _quantize_base(quant_copy, qc, len(tok))
        ppl_quant = ppl(quant_copy)
        print(f"ppl_quant = {ppl_quant:.4f}", flush=True)
        del quant_copy
        torch.cuda.empty_cache()

    cfg = TrainConfig(
        bits=2,
        loss_mode="ce_kl",
        model=args.model,
        data=args.data,
        max_tokens=args.max_tokens,
        seq_len=512,
        batch_seq=args.batch_seq,
        lr=args.lr,
        shadow_layers=args.shadow_layers,
        thr_ratio=0.5,
        ppl_context_len=AXIS_CTX,
        pad_id=pad_id,
        eval_every=args.eval_every,
        save_dir=str(out_dir / "train"),
    )
    if args.same_data:
        import os
        os.environ["SHADOW_DIST_SAME_DATA"] = "1"

    print("=== shadow distillation (DDP) ===", flush=True)
    t0 = time.time()
    report = train_shadow(cfg)
    wall = time.time() - t0

    if rank == 0:
        ppl_shadow = report.final_ppl_20chunks
        recovery = (ppl_quant - ppl_shadow) / (ppl_quant - ppl_bf16)
        verdict = {
            "bits": 2,
            "shadow_layers": args.shadow_layers,
            "tokens": report.tokens,
            "ppl_bf16": round(ppl_bf16, 4),
            "ppl_quant": round(ppl_quant, 4),
            "ppl_shadow": round(ppl_shadow, 4),
            "recovery": round(recovery, 4),
            "axis": "despaced wikitext, ctx=512, chunks=20",
            "ddp": True,
            "global_batch": args.batch_seq,
            "tok_per_s": round(report.tokens / wall, 1),
            "train_seconds": round(wall, 1),
            "trainable_params": report.trainable_params,
        }
        (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
        print(json.dumps(verdict, indent=2), flush=True)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
