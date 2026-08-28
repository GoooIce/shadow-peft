"""Distillation trainer for shadow adapters over quantized bases.

The deep module of the experiment lab: one `train_shadow(cfg)` hides the
whole E1/E2 protocol — student construction (bit-width is a config field,
not a script), frozen teacher, KL/hidden-MSE distillation with teacher
logits sliced to the trimmed student vocab, the FBI-LLM spike-guard
rollback, chunked-PPL evaluation, and best/final checkpointing.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_peft import ShadowConfig, ShadowForCausalLM, get_shadow_model
from shadow_peft.quantization import quantize_model_1bit, quantize_model_2bit, quantize_model_8bit

from . import tiers
from .atoms import (
    QWEN3_ENDOFTEXT_ID,
    despace_wikitext,
    eval_ppl,
    load_teacher_data,
    load_wikitext2_test_text,
    make_batch,
    pick_device,
)
from .grace import (
    CKA_MIN_TOKENS,
    DKD_NCKD_WEIGHT,
    DKD_TCKD_WEIGHT,
    DKD_TEMPERATURE,
    GATE_LAMBDA,
    IB_BETA_INIT,
    IB_BETA_MAX,
    IB_BETA_MIN,
    IB_EMA_DECAY,
    IB_ETA,
    IB_TAU,
    GateMonitor,
    IBController,
    gated_dkd_loss,
    span_cka_loss,
)

LOSS_MODES = ("ce", "ce_kl", "ce_kl_hmse", "dkd_gate", "dkd_gate_ib", "dkd_gate_ib_cka")
# GRACE loss ladder (exp1-grace): the trainer owns these distill terms — the
# task wrapper only contributes CE + shadow-CE (wrapper distill weight is 0).
GRACE_MODES = ("dkd_gate", "dkd_gate_ib", "dkd_gate_ib_cka")


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
    # Near-lossless sanity rung (exp1 rung A): shadow must not hurt the base.
    8: BitProfile("rungA_8bit", "affine_8bit", torch.bfloat16, 0),
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
    lr_schedule: str = "constant"  # constant | cosine (decay to 10% of lr by max_tokens)
    beta2: float = 0.98
    distill_weight: float = 1.0
    distill_temperature: float = 1.0
    distill_hidden_weight: float = 1.0
    # --- GRACE loss-mode knobs (paper Table 8 defaults; exp1-grace spec) ---
    # DKD decomposition: L = tckd_weight * TCKD + nckd_weight * NCKD, at
    # dkd_temperature. distill_weight keeps its role: total distill strength.
    dkd_temperature: float = DKD_TEMPERATURE
    dkd_tckd_weight: float = DKD_TCKD_WEIGHT
    dkd_nckd_weight: float | str = DKD_NCKD_WEIGHT
    # Soft gate g = exp(-gate_lambda * h_norm) on the T=1 teacher entropy;
    # 0 disables the gate (g == 1).
    gate_lambda: float = GATE_LAMBDA
    # Gate monitoring: stats every gate_log_every steps; degradation latch
    # (annotation only) when batch CV < cv or mean > mean for degrade_steps
    # consecutive steps.
    gate_log_every: int = 200
    gate_degrade_cv: float = 0.05
    gate_degrade_mean: float = 0.95
    gate_degrade_steps: int = 500
    # IB controller (dkd_gate_ib(+_cka)): dual ascent on the distill weight.
    ib_tau: float = IB_TAU
    ib_eta: float = IB_ETA
    ib_beta_init: float = IB_BETA_INIT
    ib_beta_min: float = IB_BETA_MIN
    ib_beta_max: float = IB_BETA_MAX
    ib_ema_decay: float = IB_EMA_DECAY
    # span-CKA (dkd_gate_ib_cka): L = cka_weight * (1 - CKA) on the
    # continuation span of the penultimate hidden layer.
    cka_weight: float = 1.0
    cka_min_tokens: int = CKA_MIN_TOKENS
    shadow_loss_weight: float = 0.05
    shadow_layers: int = 2
    # Width knobs for the implicit shadow trunk (exp1-floor-2 structure sweep;
    # None = derive from base like before). hidden_size itself is NOT a knob —
    # the shadow reads/writes the base residual stream.
    shadow_intermediate_size: int | None = None
    shadow_num_attention_heads: int | None = None
    shadow_num_key_value_heads: int | None = None
    # Native shadow-weight precision (direction ① deployability): "nvfp8"/"nvfp4"
    # keeps every shadow Linear weight ON the format grid for the whole run
    # (projected-gradient training — no bf16 master, no STE).
    shadow_quant: str | None = None
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
    # Explicit shadow layer composition (hybrid backbones; exp4 ablation knob).
    shadow_layer_types: list[str] | None = None
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
                raise ValueError(f"unsupported bits={self.bits} (expected one of {sorted(_PROFILES)})")
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
        return Path(self.save_dir or f"runs/{self.profile.tag}_{self.loss_mode}")


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
    # GRACE modes only: gate/IB monitoring summary for the arm verdict
    # (gate_degraded, final gate stats, beta trajectory bounds). None for
    # the legacy three modes — verdict fields stay isomorphic to the floor
    # series when absent.
    grace: dict | None = None


def _json_safe(obj):
    """Stringify config values json.dumps cannot serialize (torch.dtype)."""
    if isinstance(obj, torch.dtype):
        return str(obj)
    return obj


def _summary_args_dump(cfg: TrainConfig) -> dict:
    """asdict(cfg) made JSON-serializable, for the summary.json "args" entry."""
    return {k: _json_safe(v) for k, v in asdict(cfg).items()}


def _quantize_base(base, cfg: TrainConfig, student_vocab: int | None) -> dict:
    """The backbone seam: torchao tiers (direction ①) or the E1/E2 packed path.

    ``student_vocab`` is kept for signature compatibility but vocab trimming is
    disabled: on LFM ``len(tokenizer)`` (64,400) < embed rows (65,536), and
    teacher-sampled ids cover the full softmax — trimming the tail rows makes
    training batches assert in ``F.embedding``. The ~2% embedding saving is
    not worth a silently wrong student support.
    """
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
            trim_vocab_to=None,
        )
    if cfg.bits == 8:
        return quantize_model_8bit(
            base,
            group_size=cfg.group_size,
            refine_iters=cfg.refine_iters if cfg.refine_iters is not None else cfg.profile.refine_iters,
            trim_vocab_to=None,
        )
    return quantize_model_2bit(
        base,
        group_size=cfg.group_size,
        refine_iters=cfg.refine_iters if cfg.refine_iters is not None else cfg.profile.refine_iters,
        thr_ratio=cfg.thr_ratio,
        trim_vocab_to=None,
    )


def _sync_grads(params, world: int) -> None:
    """Average grads in-place across ranks so every replica steps identically.

    Coalesced into ONE all_reduce (per-param calls cost ~0.5ms latency each —
    ~70ms/step at 140 tensors on the GB10 link; one 140MB flat buffer is
    7.6ms end-to-end).
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat)
    flat /= world
    for buf, g in zip(torch._utils._unflatten_dense_tensors(flat, grads), grads, strict=True):
        g.copy_(buf)


def _maybe_init_distributed() -> tuple[int, int, int] | None:
    """torchrun context -> (rank, world, local_rank); None when single-process.

    Backend from DIST_BACKEND (default nccl; gloo+cpu for local tests).
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return None
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = os.environ.get("DIST_BACKEND", "nccl")
    if not dist.is_initialized():
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend, device_id=torch.cuda.current_device())  # type: ignore[arg-type]
        else:
            dist.init_process_group(backend)
    return rank, world, local_rank


def train_shadow(cfg: TrainConfig, shadow_model=None) -> TrainReport:
    """Run one E1/E2 distillation end-to-end and return the report."""
    device = cfg.device or pick_device()
    ddp = _maybe_init_distributed()
    rank, world = (ddp[0], ddp[1]) if ddp else (0, 1)

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    dtype = cfg.dtype if cfg.dtype is not None else cfg.profile.dtype
    refine_iters = cfg.refine_iters if cfg.refine_iters is not None else cfg.profile.refine_iters

    save_dir = cfg.resolved_save_dir()
    save_dir.mkdir(parents=True, exist_ok=True)
    # Data sampling must be rank-sharded (rank-identical draws would make the
    # averaged gradient a copy of one rank's batch — half the GPUs do nothing).
    # SHADOW_DIST_SAME_DATA=1 forces identical draws for the DDP-vs-single
    # bit-equivalence smoke only.
    if ddp and not os.environ.get("SHADOW_DIST_SAME_DATA"):
        rng = random.Random(cfg.seed + rank)
    else:
        rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    student_vocab = len(tokenizer)  # pre-quantization width (trim input only)

    # --- student: quantized base + shadow ---
    base = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype).to(device)
    manifest = _quantize_base(base, cfg, student_vocab)
    student_vocab = base.get_output_embeddings().weight.shape[0]  # true student width
    if cfg.bits == 0:
        log(f"backbone tier={cfg.backbone_tier} format={manifest.get('format')}")
    else:
        log(f"quantized={len(manifest['quantized'])} trimmed={manifest['trimmed']}")
    base.eval()

    shadow_cfg = ShadowConfig(
        num_shadow_layers=cfg.shadow_layers,
        injection_hidden_size=cfg.injection_hidden,
        gate_hidden_size=cfg.gate_hidden,
        alpha=0.1,
        dropout=0.0,
        shadow_intermediate_size=cfg.shadow_intermediate_size,
        shadow_num_attention_heads=cfg.shadow_num_attention_heads,
        shadow_num_key_value_heads=cfg.shadow_num_key_value_heads,
        shadow_layer_types=cfg.shadow_layer_types,
    )
    # --- loss-mode dispatch, resolved once (all downstream branching reads
    # these flags; adding a mode touches only this block) ---
    grace_mode = cfg.loss_mode in GRACE_MODES
    want_all_hidden = cfg.loss_mode == "ce_kl_hmse"  # every layer (hidden-MSE)
    want_penult = cfg.loss_mode == "dkd_gate_ib_cka"  # penultimate only (span-CKA)
    use_ib = cfg.loss_mode in ("dkd_gate_ib", "dkd_gate_ib_cka")
    task = ShadowForCausalLM(
        get_shadow_model(base, shadow_cfg, shadow_model=shadow_model),
        shadow_loss_weight=cfg.shadow_loss_weight,
        # GRACE modes own their distill terms trainer-side (gated DKD + IB
        # beta + span-CKA); the wrapper must not add its own plain KL.
        distill_weight=(
            0.0 if (cfg.loss_mode == "ce" or grace_mode) else cfg.distill_weight
        ),
        distill_temperature=cfg.distill_temperature,
        distill_hidden_weight=(
            cfg.distill_hidden_weight if want_all_hidden else 0.0
        ),
    )
    task.to(device)
    task.train()
    if cfg.gradient_checkpointing:
        task.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        log("gradient checkpointing enabled (use_reentrant=False)")

    # --- teacher: full precision, frozen ---
    teacher = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    trainable = [p for p in task.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    log(f"trainable params: {n_train:,}")

    # Native shadow-weight precision: project the random init onto the format
    # grid BEFORE the optimizer/first eval exist, so training, eval and the
    # saved checkpoint all see exactly the on-grid weights.
    shadow_linears: list = []
    if cfg.shadow_quant:
        from shadow_lab import shadow_quant

        shadow_linears = shadow_quant.collect_shadow_linears(task.peft_model)
        n_proj = shadow_quant.project_linears(shadow_linears, cfg.shadow_quant)
        log(f"shadow weights native {cfg.shadow_quant}: {n_proj} linears projected on-grid")
    if cfg.batch_seq % world != 0:
        raise ValueError(f"batch_seq={cfg.batch_seq} not divisible by world={world}")
    local_batch = cfg.batch_seq // world
    if ddp:
        log(f"DDP: world={world} rank={rank} local_batch={local_batch} (global {cfg.batch_seq})")
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, betas=(0.9, cfg.beta2), weight_decay=0.0)
    sched = None
    if cfg.lr_schedule == "cosine":
        total_steps = max(1, int(-(-cfg.max_tokens // (local_batch * cfg.seq_len * world))))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: 0.1 + 0.45 * (1.0 + torch.cos(torch.tensor(torch.pi * s / total_steps)).item()))
    elif cfg.lr_schedule != "constant":
        raise ValueError(f"unsupported lr_schedule={cfg.lr_schedule!r}")

    # --- GRACE runtime (exp1-grace): gate monitor + optional IB controller ---
    monitor: GateMonitor | None = None
    ib: IBController | None = None
    if grace_mode:
        monitor = GateMonitor(
            log_every=cfg.gate_log_every,
            degrade_cv=cfg.gate_degrade_cv,
            degrade_mean=cfg.gate_degrade_mean,
            degrade_steps=cfg.gate_degrade_steps,
        )
        if use_ib:
            ib = IBController(
                tau=cfg.ib_tau,
                eta=cfg.ib_eta,
                beta=cfg.ib_beta_init,
                beta_min=cfg.ib_beta_min,
                beta_max=cfg.ib_beta_max,
                ema_decay=cfg.ib_ema_decay,
            )

    # --- data & eval text ---
    seqs = load_teacher_data(cfg.data)
    log(f"teacher sequences: {len(seqs)}")
    eval_text = load_wikitext2_test_text()
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
        log(f"joint calibration: {n_calib} forwards with shadow attached")

    if rank == 0:
        log_f = open(save_dir / "train_log.jsonl", "w")  # noqa: SIM115
    ppl = eval_ppl(task, tokenizer, eval_text, device=device, context_len=cfg.ppl_context_len) if rank == 0 else float("nan")
    teacher_ppl = (eval_ppl(teacher, tokenizer, eval_text, device=device, context_len=cfg.ppl_context_len)
                   if rank == 0 else float("nan"))
    log(f"initial student PPL {ppl:.1f} | teacher PPL {teacher_ppl:.2f}")

    tokens_seen = 0
    step = 0
    recent_losses: list[float] = []
    best_ppl = ppl
    last_good_state = {k: v.detach().clone() for k, v in task.state_dict().items() if v.requires_grad}
    t0 = time.time()

    while tokens_seen < cfg.max_tokens:
        step += 1
        batch = make_batch(seqs, rng, local_batch, cfg.seq_len, device, pad_id=cfg.pad_id)
        # Exit counter MUST be rank-invariant: counting actual unmasked tokens
        # desyncs the ranks at the loop edge (EOS-truncated batches differ per
        # rank) — the rank that exits early leaves the other's last all_reduce
        # unanswered and the NCCL watchdog kills the job. The padded count is
        # off by <0.1% from the real token total.
        tokens_seen += local_batch * cfg.seq_len * world

        fwd = dict(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        teacher_penult = None
        with torch.no_grad():
            if cfg.loss_mode == "ce":
                pass
            else:
                t_out = teacher(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=want_all_hidden or want_penult,
                )
                # Student vocab is trimmed; teacher keeps the padding rows.
                fwd["teacher_logits"] = t_out.logits[..., :student_vocab]
                if want_all_hidden:
                    fwd["teacher_hidden_states"] = t_out.hidden_states
                if want_penult:
                    # Extract before `t_out` goes out of scope: only the
                    # penultimate slice is kept (no full hidden tuple alive
                    # across the student forward).
                    teacher_penult = t_out.hidden_states[-2]
        if want_penult:
            # Student-side same-position hidden comes from the task wrapper.
            fwd["output_hidden_states"] = True

        out = task(**fwd)
        # GRACE modes compose their distill terms here, trainer-side; the
        # wrapper's loss carries CE + shadow-CE only (its distill weight is 0).
        loss = out.loss
        gd = None
        cka_out = None
        beta = ib.beta if ib is not None else 1.0
        if grace_mode:
            gd = gated_dkd_loss(
                out.logits,
                fwd["teacher_logits"],
                batch["labels"],
                temperature=cfg.dkd_temperature,
                tckd_weight=cfg.dkd_tckd_weight,
                nckd_weight=cfg.dkd_nckd_weight,
                gate_lambda=cfg.gate_lambda,
            )
            loss = loss + cfg.distill_weight * beta * gd.loss
            if want_penult:
                student_penult = getattr(out, "hidden_states", None)
                if student_penult is None:
                    raise RuntimeError(
                        "task output missing hidden_states despite output_hidden_states=True; "
                        "cannot compute span-CKA."
                    )
                cka_out = span_cka_loss(
                    student_penult[-2],
                    teacher_penult,
                    batch["labels"],
                    min_tokens=cfg.cka_min_tokens,
                )
                loss = loss + cfg.cka_weight * cka_out.loss
        loss_val = loss.item()
        if ddp:
            # Average the loss BEFORE the guard so every rank takes the same
            # rollback decision (a rank-local decision would fork the replicas
            # and deadlock the next all_reduce).
            lv = torch.tensor(loss_val, device=device)
            dist.all_reduce(lv)
            loss_val = lv.item() / world

        # Spike guard (FBI-LLM protocol): roll back on extreme loss.
        if len(recent_losses) >= 20:
            median = sorted(recent_losses[-50:])[len(recent_losses[-50:]) // 2]
            if loss_val > 4.0 * median:
                task.load_state_dict(last_good_state, strict=False)
                log(f"step {step}: SPIKE {loss_val:.1f} > 4x median {median:.2f}, rolled back")
                continue

        opt.zero_grad()
        loss.backward()
        if ddp:
            # Manual sync: average grads so all replicas step identically.
            # Must precede clip (rank-local norms would fork the replicas).
            _sync_grads(trainable, world)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if shadow_linears:
            # Native-format training: snap the stepped weights back onto the
            # grid so forward/eval/deployment always see on-grid values.
            shadow_quant.project_linears(shadow_linears, cfg.shadow_quant)
        if sched is not None:
            sched.step()
        if ib is not None:
            # Dual ascent once per optimizer step, on the EMA of the observed
            # gated-DKD (rank-averaged so replicas keep identical betas).
            gdkd_val = gd.loss.item()
            if ddp:
                gv = torch.tensor(gdkd_val, device=device)
                dist.all_reduce(gv)
                gdkd_val = gv.item() / world
            beta = ib.step(gdkd_val)
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
            if gd is not None:
                # GRACE component breakdown (full beta trajectory in the log).
                msg.update(
                    {
                        "gdkd": gd.loss.item(),
                        "tckd": gd.tckd,
                        "nckd": gd.nckd,
                        "gate_mean": gd.gate_mean,
                        "beta": beta,
                    }
                )
            if cka_out is not None:
                msg["cka"] = cka_out.loss.item()
                msg["cka_skipped"] = cka_out.n_skipped
            log(str(msg))
            if rank == 0:
                log_f.write(json.dumps(msg) + "\n")
                log_f.flush()

        if monitor is not None:
            gate_rec = monitor.record(step, gd.gate, beta)
            if gate_rec is not None and rank == 0:
                log_f.write(json.dumps(gate_rec) + "\n")
                log_f.flush()

        if step % cfg.eval_every == 0:
            # Snapshot on EVERY rank (spike-guard rollback restores this).
            last_good_state = {k: v.detach().clone() for k, v in task.state_dict().items() if v.requires_grad}
            if rank == 0:
                ppl = eval_ppl(
                    task, tokenizer, eval_text, device=device, context_len=cfg.ppl_context_len
                )
                log(f"step {step}: eval PPL {ppl:.1f} (best {best_ppl:.1f})")
                if ppl < best_ppl:
                    best_ppl = ppl
                    task.peft_model.save_pretrained(save_dir / "best")

    final_ppl = float("nan")
    if rank == 0:
        task.peft_model.save_pretrained(save_dir / "final")
        final_ppl = eval_ppl(
            task, tokenizer, eval_text, chunks=20, device=device, context_len=cfg.ppl_context_len
        )
    grace_summary = None
    if grace_mode:
        assert monitor is not None
        # No-IB modes hold beta at the fixed identity value 1 (the B1 arm's
        # semantics), so their summary must not report ib_beta_init.
        grace_summary = {
            "gate_degraded": monitor.degraded,
            "gate_final_mean": monitor.last_mean,
            "gate_final_var": monitor.last_var,
            "gate_final_cv": monitor.last_cv,
            "gate_max_degrade_streak": monitor.max_streak,
            "beta_init": ib.beta if ib is not None else 1.0,
            "beta_final": ib.beta if ib is not None else 1.0,
            "beta_min": ib.seen_min if ib is not None else 1.0,
            "beta_max": ib.seen_max if ib is not None else 1.0,
        }
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
        grace=grace_summary,
    )
    if rank == 0:
        summary = {**asdict(report), "refine_iters": refine_iters, "args": _summary_args_dump(cfg)}
        (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        log(json.dumps({k: v for k, v in summary.items() if k != "args"}, indent=2))
        log_f.close()
    return report
