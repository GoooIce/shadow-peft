"""Distillation trainer for shadow adapters over quantized bases.

The deep module of the experiment lab: one `train_shadow(cfg)` hides the
whole E1/E2 protocol — student construction (bit-width is a config field,
not a script), frozen teacher, KL/hidden-MSE distillation with teacher
logits sliced to the trimmed student vocab, the FBI-LLM spike-guard
rollback, chunked-PPL evaluation, and best/final checkpointing.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_peft import ShadowConfig, ShadowForCausalLM, get_shadow_model
from shadow_peft.quantization import quantize_model_1bit, quantize_model_2bit

from . import tiers
from .atoms import (
    despace_wikitext,
    QWEN3_ENDOFTEXT_ID,
    eval_ppl,
    load_teacher_data,
    load_wikitext2_test_text,
    make_batch,
    pick_device,
)

LOSS_MODES = ("ce", "ce_kl", "ce_kl_hmse")


@dataclass(frozen=True)
class BitProfile:
    """Per-bit-width facts that used to live as constants in twin scripts."""

    tag: str  # run-directory prefix
    label: str  # summary.json "quantization" value
    dtype: torch.dtype
    refine_iters: int


_PROFILES: dict[int, BitProfile] = {
    1: BitProfile("e1", "affine_1bit", torch.float16, 10),
    # fp16 training diverges (loss -> nan) at lr=1e-5 under 2-bit ternary.
    2: BitProfile("e2", "ternary_2bit", torch.bfloat16, 8),
}


@dataclass
class TrainConfig:
    bits: int = 1
    # Backbone tier seam (direction ①): "bits" = the E1/E2 shadow-side packed
    # path driven by `bits`; otherwise a torchao tier {bf16, w8a8_fp8, w4a16}
    # applied to the backbone (requires bits=0, never trims the vocab).
    backbone_tier: str = "bits"
    # Train against the tier's fake-quant numerics (real kernels have no
    # autograd); eval still measures both paths.
    tier_fake_quant: bool = False
    loss_mode: str = "ce_kl"
    model: str = "Qwen/Qwen3-1.7B"
    data: str = "experiment/data/teacher_qwen3_1p7b.jsonl"
    max_tokens: float = 5e6
    seq_len: int = 512
    batch_seq: int = 4
    lr: float = 1e-5
    beta2: float = 0.98
    distill_weight: float = 1.0
    distill_temperature: float = 1.0
    distill_hidden_weight: float = 1.0
    shadow_loss_weight: float = 0.05
    shadow_layers: int = 2
    injection_hidden: int = 16
    gate_hidden: int = 10
    group_size: int = 128
    thr_ratio: float = 0.5  # 2-bit ternary only
    refine_iters: int | None = None  # None -> per-bits profile default
    dtype: torch.dtype | None = None  # None -> per-bits profile default
    log_every: int = 25
    eval_every: int = 250
    # Context-carrying ppl windows: Gemma-it resists cold mid-stream windows.
    ppl_context_len: int = 0
    # Joint-calibration forwards for w8a8_fp8 (run with the shadow attached).
    calib_steps: int = 8
    # Activation-memory relief for 12B-class distillation (frozen base with
    # shadow grads flowing through); non-reentrant so no-grad inputs are fine.
    gradient_checkpointing: bool = False
    seed: int = 0
    save_dir: str | None = None  # None -> experiment/runs/{tag}_{loss_mode}
    device: str | None = None  # None -> pick_device()
    pad_id: int = QWEN3_ENDOFTEXT_ID

    def __post_init__(self):
        if self.backbone_tier == "bits":
            if self.bits not in _PROFILES:
                raise ValueError(f"unsupported bits={self.bits} (expected 1 or 2)")
        else:
            if self.bits != 0:
                raise ValueError(
                    "torchao backbone tiers require bits=0 "
                    "(the shadow-side packed path driven by bits>0 is the E1/E2 protocol)"
                )
            if self.backbone_tier not in tiers.TIERS:
                raise ValueError(
                    f"unsupported backbone_tier={self.backbone_tier!r} (expected 'bits' or one of {tiers.TIERS})"
                )
        if self.loss_mode not in LOSS_MODES:
            raise ValueError(f"unsupported loss_mode={self.loss_mode!r} (expected one of {LOSS_MODES})")

    @property
    def profile(self) -> BitProfile:
        if self.bits == 0:
            return BitProfile(f"d1_{self.backbone_tier}", self.backbone_tier, torch.bfloat16, 0)
        return _PROFILES[self.bits]

    def resolved_save_dir(self) -> Path:
        return Path(self.save_dir or f"experiment/runs/{self.profile.tag}_{self.loss_mode}")


@dataclass
class TrainReport:
    loss_mode: str
    quantization: str
    bits: int
    backbone_tier: str
    tokens: int
    steps: int
    final_ppl_20chunks: float
    best_ppl_8chunks: float
    teacher_ppl_8chunks: float
    trainable_params: int
    wall_time_s: float
    save_dir: str


def _json_safe(obj):
    """Stringify config values json.dumps cannot serialize (torch.dtype)."""
    if isinstance(obj, torch.dtype):
        return str(obj)
    return obj


def _summary_args_dump(cfg: TrainConfig) -> dict:
    """asdict(cfg) made JSON-serializable, for the summary.json "args" entry."""
    return {k: _json_safe(v) for k, v in asdict(cfg).items()}


def _quantize_base(base, cfg: TrainConfig, student_vocab: int) -> dict:
    """The backbone seam: torchao tiers (direction ①) or the E1/E2 packed path."""
    if cfg.bits == 0:
        device = next(base.parameters()).device
        return tiers.apply_training_conversion(
            base, cfg.backbone_tier, device, fake_quant=cfg.tier_fake_quant
        )
    if cfg.bits == 1:
        return quantize_model_1bit(
            base,
            group_size=cfg.group_size,
            refine_iters=cfg.refine_iters if cfg.refine_iters is not None else cfg.profile.refine_iters,
            trim_vocab_to=student_vocab,
        )
    return quantize_model_2bit(
        base,
        group_size=cfg.group_size,
        refine_iters=cfg.refine_iters if cfg.refine_iters is not None else cfg.profile.refine_iters,
        thr_ratio=cfg.thr_ratio,
        trim_vocab_to=student_vocab,
    )


def train_shadow(cfg: TrainConfig) -> TrainReport:
    """Run one E1/E2 distillation end-to-end and return the report."""
    device = cfg.device or pick_device()
    dtype = cfg.dtype if cfg.dtype is not None else cfg.profile.dtype
    refine_iters = cfg.refine_iters if cfg.refine_iters is not None else cfg.profile.refine_iters

    save_dir = cfg.resolved_save_dir()
    save_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    student_vocab = len(tokenizer)

    # --- student: quantized base + shadow ---
    base = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype).to(device)
    manifest = _quantize_base(base, cfg, student_vocab)
    if cfg.bits == 0:
        print(f"backbone tier={cfg.backbone_tier} format={manifest.get('format')}", flush=True)
    else:
        print(f"quantized={len(manifest['quantized'])} trimmed={manifest['trimmed']}", flush=True)
    base.eval()

    shadow_cfg = ShadowConfig(
        num_shadow_layers=cfg.shadow_layers,
        injection_hidden_size=cfg.injection_hidden,
        gate_hidden_size=cfg.gate_hidden,
        alpha=0.1,
        dropout=0.0,
    )
    task = ShadowForCausalLM(
        get_shadow_model(base, shadow_cfg),
        shadow_loss_weight=cfg.shadow_loss_weight,
        distill_weight=cfg.distill_weight if cfg.loss_mode != "ce" else 0.0,
        distill_temperature=cfg.distill_temperature,
        distill_hidden_weight=(
            cfg.distill_hidden_weight if cfg.loss_mode == "ce_kl_hmse" else 0.0
        ),
    )
    task.to(device)
    task.train()
    if cfg.gradient_checkpointing:
        task.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("gradient checkpointing enabled (use_reentrant=False)", flush=True)

    # --- teacher: full precision, frozen ---
    teacher = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    trainable = [p for p in task.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"trainable params: {n_train:,}", flush=True)
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, betas=(0.9, cfg.beta2), weight_decay=0.0)

    # --- data & eval text ---
    seqs = load_teacher_data(cfg.data)
    print(f"teacher sequences: {len(seqs)}", flush=True)
    eval_text = load_wikitext2_test_text()
    eval_text_raw = eval_text
    eval_text = despace_wikitext(eval_text)

    # Joint calibration (iron rule): W8A8 activation scales must be observed
    # on shadow-carrying activations, so the calibration forwards run through
    # the mounted student — never the bare quantized base.
    if cfg.backbone_tier == "w8a8_fp8" and cfg.calib_steps > 0:
        calib_batches = [
            make_batch(seqs, rng, cfg.batch_seq, cfg.seq_len, device, pad_id=cfg.pad_id)
            for _ in range(cfg.calib_steps)
        ]
        n_calib = tiers.joint_calibrate(task.peft_model, calib_batches)
        print(f"joint calibration: {n_calib} forwards with shadow attached", flush=True)

    log_f = open(save_dir / "train_log.jsonl", "w")
    ppl = eval_ppl(task, tokenizer, eval_text, device=device, context_len=cfg.ppl_context_len)
    teacher_ppl = eval_ppl(teacher, tokenizer, eval_text, device=device, context_len=cfg.ppl_context_len)
    print(f"initial student PPL {ppl:.1f} | teacher PPL {teacher_ppl:.2f}", flush=True)

    tokens_seen = 0
    step = 0
    recent_losses: list[float] = []
    best_ppl = ppl
    last_good_state = {k: v.detach().clone() for k, v in task.state_dict().items() if v.requires_grad}
    t0 = time.time()

    while tokens_seen < cfg.max_tokens:
        step += 1
        batch = make_batch(seqs, rng, cfg.batch_seq, cfg.seq_len, device, pad_id=cfg.pad_id)
        tokens_seen += int(batch["attention_mask"].sum().item())

        fwd = dict(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        with torch.no_grad():
            if cfg.loss_mode == "ce":
                pass
            else:
                t_out = teacher(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=(cfg.loss_mode == "ce_kl_hmse"),
                )
                # Student vocab is trimmed; teacher keeps the padding rows.
                fwd["teacher_logits"] = t_out.logits[..., :student_vocab]
                if cfg.loss_mode == "ce_kl_hmse":
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

        if step % cfg.log_every == 0:
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

        if step % cfg.eval_every == 0:
            last_good_state = {k: v.detach().clone() for k, v in task.state_dict().items() if v.requires_grad}
            ppl = eval_ppl(
                task, tokenizer, eval_text, device=device, context_len=cfg.ppl_context_len
            )
            print(f"step {step}: eval PPL {ppl:.1f} (best {best_ppl:.1f})", flush=True)
            if ppl < best_ppl:
                best_ppl = ppl
                task.peft_model.save_pretrained(save_dir / "best")

    task.peft_model.save_pretrained(save_dir / "final")
    final_ppl = eval_ppl(
        task, tokenizer, eval_text, chunks=20, device=device, context_len=cfg.ppl_context_len
    )
    report = TrainReport(
        loss_mode=cfg.loss_mode,
        quantization=cfg.profile.label,
        bits=cfg.bits,
        backbone_tier=cfg.backbone_tier,
        tokens=tokens_seen,
        steps=step,
        final_ppl_20chunks=final_ppl,
        best_ppl_8chunks=best_ppl,
        teacher_ppl_8chunks=teacher_ppl,
        trainable_params=n_train,
        wall_time_s=time.time() - t0,
        save_dir=str(save_dir),
    )
    summary = {**asdict(report), "refine_iters": refine_iters, "args": _summary_args_dump(cfg)}
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}, indent=2), flush=True)
    log_f.close()
    return report
