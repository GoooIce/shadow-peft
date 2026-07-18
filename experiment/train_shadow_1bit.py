"""
E1: train a Shadow adapter on a 1-bit quantized Qwen3-1.7B with distillation.

Student: Qwen3-1.7B -> quantize_model_1bit (refined, trimmed) -> frozen.
Teacher: fp16 Qwen3-1.7B, same device, no_grad.

Loss modes (all include the standard shadow CE term):
  ce          : CE only                          (E1-B)
  ce_kl       : CE + KL(teacher||student)        (E1-C)
  ce_kl_hmse  : CE + KL + per-layer hidden MSE   (E1-D, OneBit recipe)

Run: .venv/bin/python experiment/train_shadow_1bit.py --loss-mode ce_kl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_peft import ShadowConfig, ShadowForCausalLM, get_shadow_model
from shadow_peft.quantization import quantize_model_1bit

MODEL = "Qwen/Qwen3-1.7B"
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
DTYPE = torch.float16
EOS = 151643  # <|endoftext|>, also used as pad id


def load_teacher_data(path: str, min_len: int = 32) -> list[list[int]]:
    seqs = []
    with open(path) as f:
        for line in f:
            ids = json.loads(line)["ids"]
            if len(ids) >= min_len:
                seqs.append(ids)
    return seqs


def make_batch(
    seqs: list[list[int]], rng: random.Random, batch_size: int, seq_len: int, device: str
) -> dict:
    picked = [rng.choice(seqs) for _ in range(batch_size)]
    cropped = [s[:seq_len] for s in picked]
    maxlen = max(len(s) for s in cropped)
    input_ids = torch.full((batch_size, maxlen), EOS, dtype=torch.long)
    attn = torch.zeros((batch_size, maxlen), dtype=torch.long)
    for i, s in enumerate(cropped):
        input_ids[i, : len(s)] = torch.tensor(s)
        attn[i, : len(s)] = 1
    labels = input_ids.clone()
    labels[attn == 0] = -100
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attn.to(device),
        "labels": labels.to(device),
    }


@torch.no_grad()
def eval_ppl(
    task: ShadowForCausalLM,
    tokenizer,
    text: str,
    chunks: int = 8,
    chunk_len: int = 512,
) -> float:
    ids = tokenizer.encode(text)
    was_training = task.training
    task.eval()
    total_nll, total_tok = 0.0, 0
    limit = min(len(ids), chunks * chunk_len) - chunk_len - 1
    for start in range(0, limit, chunk_len):
        x = torch.tensor(ids[start : start + chunk_len + 1], device=DEVICE)[None]
        logits = task(input_ids=x).logits.float()
        logp = logits[:, :-1] - torch.logsumexp(logits[:, :-1], -1, keepdim=True)
        tgt = x[:, 1:]
        total_nll += -logp.gather(-1, tgt.unsqueeze(-1)).sum().item()
        total_tok += tgt.numel()
    if was_training:
        task.train()
    return math.exp(total_nll / total_tok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss-mode", choices=["ce", "ce_kl", "ce_kl_hmse"], required=True)
    ap.add_argument("--data", default="experiment/data/teacher_qwen3_1p7b.jsonl")
    ap.add_argument("--max-tokens", type=float, default=5e6)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-seq", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta2", type=float, default=0.98)
    ap.add_argument("--distill-weight", type=float, default=1.0)
    ap.add_argument("--distill-temperature", type=float, default=1.0)
    ap.add_argument("--distill-hidden-weight", type=float, default=1.0)
    ap.add_argument("--shadow-loss-weight", type=float, default=0.05)
    ap.add_argument("--shadow-layers", type=int, default=2)
    ap.add_argument("--injection-hidden", type=int, default=16)
    ap.add_argument("--gate-hidden", type=int, default=10)
    ap.add_argument("--refine-iters", type=int, default=10)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    save_dir = Path(args.save_dir or f"experiment/runs/e1_{args.loss_mode}")
    save_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    student_vocab = len(tokenizer)

    # --- student: 1-bit base + shadow ---
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE)
    manifest = quantize_model_1bit(
        base,
        group_size=args.group_size,
        refine_iters=args.refine_iters,
        trim_vocab_to=student_vocab,
    )
    print(f"quantized={len(manifest['quantized'])} trimmed={manifest['trimmed']}", flush=True)
    base.eval()

    cfg = ShadowConfig(
        num_shadow_layers=args.shadow_layers,
        injection_hidden_size=args.injection_hidden,
        gate_hidden_size=args.gate_hidden,
        alpha=0.1,
        dropout=0.0,
    )
    task = ShadowForCausalLM(
        get_shadow_model(base, cfg),
        shadow_loss_weight=args.shadow_loss_weight,
        distill_weight=args.distill_weight if args.loss_mode != "ce" else 0.0,
        distill_temperature=args.distill_temperature,
        distill_hidden_weight=(
            args.distill_hidden_weight if args.loss_mode == "ce_kl_hmse" else 0.0
        ),
    )
    task.to(DEVICE)
    task.train()

    # --- teacher: fp16, frozen ---
    teacher = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    trainable = [p for p in task.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"trainable params: {n_train:,}", flush=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, args.beta2), weight_decay=0.0)

    # --- data & eval text ---
    seqs = load_teacher_data(args.data)
    print(f"teacher sequences: {len(seqs)}", flush=True)
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_text = "\n".join(line for line in ds["text"] if line.strip())[:200_000]

    log_f = open(save_dir / "train_log.jsonl", "w")
    ppl = eval_ppl(task, tokenizer, eval_text)
    teacher_ppl = eval_ppl(teacher, tokenizer, eval_text)
    print(f"initial student PPL {ppl:.1f} | teacher PPL {teacher_ppl:.2f}", flush=True)

    tokens_seen = 0
    step = 0
    recent_losses: list[float] = []
    best_ppl = ppl
    last_good_state = {k: v.detach().clone() for k, v in task.state_dict().items() if v.requires_grad}
    t0 = time.time()

    while tokens_seen < args.max_tokens:
        step += 1
        batch = make_batch(seqs, rng, args.batch_seq, args.seq_len, DEVICE)
        tokens_seen += int(batch["attention_mask"].sum().item())

        fwd = dict(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        with torch.no_grad():
            if args.loss_mode == "ce":
                pass
            else:
                t_out = teacher(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=(args.loss_mode == "ce_kl_hmse"),
                )
                # Student vocab is trimmed; teacher keeps the padding rows.
                fwd["teacher_logits"] = t_out.logits[..., :student_vocab]
                if args.loss_mode == "ce_kl_hmse":
                    fwd["teacher_hidden_states"] = t_out.hidden_states

        out = task(**fwd)
        loss_val = out.loss.item()

        # Spike guard (FBI-LLM protocol): roll back on extreme loss.
        if len(recent_losses) >= 20:
            median = sorted(recent_losses[-50:])[len(recent_losses[-50:]) // 2]
            if loss_val > 4.0 * median:
                task.load_state_dict(last_good_state, strict=False)
                print(f"step {step}: SPIKE {loss_val:.1f} > 4x median {median:.2f}, rolled back", flush=True)
                continue

        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        recent_losses.append(loss_val)

        if step % args.log_every == 0:
            msg = {
                "step": step,
                "tokens": tokens_seen,
                "loss": loss_val,
                "kd": out.kd_loss.item() if out.kd_loss is not None else None,
                "hmse": out.hidden_loss.item() if out.hidden_loss is not None else None,
                "tok_per_s": tokens_seen / (time.time() - t0),
            }
            print(msg, flush=True)
            log_f.write(json.dumps(msg) + "\n")
            log_f.flush()

        if step % args.eval_every == 0:
            last_good_state = {k: v.detach().clone() for k, v in task.state_dict().items() if v.requires_grad}
            ppl = eval_ppl(task, tokenizer, eval_text)
            print(f"step {step}: eval PPL {ppl:.1f} (best {best_ppl:.1f})", flush=True)
            if ppl < best_ppl:
                best_ppl = ppl
                task.peft_model.save_pretrained(save_dir / "best")

    task.peft_model.save_pretrained(save_dir / "final")
    final_ppl = eval_ppl(task, tokenizer, eval_text, chunks=20)
    summary = {
        "loss_mode": args.loss_mode,
        "tokens": tokens_seen,
        "steps": step,
        "final_ppl_20chunks": final_ppl,
        "best_ppl_8chunks": best_ppl,
        "teacher_ppl_8chunks": teacher_ppl,
        "trainable_params": n_train,
        "wall_time_s": time.time() - t0,
        "args": vars(args),
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
