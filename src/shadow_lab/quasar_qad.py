"""End-to-end QUASAR NVFP4 quantization-aware distillation campaign.

The campaign follows the paper's healing protocol at the framework's existing
teacher-token JSONL seam: a full-precision teacher, a simulated NVFP4 W4A4
student, forward KL, and only quantized Linear latent weights trainable.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .atoms import QWEN3_ENDOFTEXT_ID, load_teacher_data, make_batch, pick_device
from .quasar import (
    QuasarNVFP4Config,
    materialize_quasar_nvfp4,
    prepare_quasar_nvfp4,
)


@dataclass
class QuasarQADConfig:
    """One runnable NVFP4-only QAD experiment."""

    model: str
    data: str
    save_dir: str
    teacher_model: str | None = None
    steps: int = 4096
    seq_len: int = 4096
    batch_seq: int = 1
    gradient_accumulation_steps: int = 32
    lr: float = 1e-6
    lr_schedule: str = "cosine"
    beta2: float = 0.999
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    dtype: torch.dtype = torch.bfloat16
    device: str | None = None
    seed: int = 0
    pad_id: int = QWEN3_ENDOFTEXT_ID
    log_every: int = 10
    max_shard_size: str = "5GB"
    gradient_checkpointing: bool = False
    nvfp4: QuasarNVFP4Config = field(default_factory=QuasarNVFP4Config)

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        if self.batch_seq < 1:
            raise ValueError("batch_seq must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError("lr_schedule must be 'constant' or 'cosine'")
        if not 0.0 < self.beta2 < 1.0:
            raise ValueError("beta2 must be in (0, 1)")
        if self.log_every < 1:
            raise ValueError("log_every must be >= 1")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if not self.nvfp4.quantize_activations:
            raise ValueError("the first QUASAR phase is native NVFP4 W4A4; activations must be enabled")


@dataclass
class QuasarQADReport:
    algorithm: str
    format: str
    steps: int
    tokens: int
    final_loss: float
    trainable_params: int
    quantized_layers: int
    saliency_layers: int
    wall_time_s: float
    save_dir: str


def forward_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Forward KL(P_teacher || P_student) over valid next-token positions."""

    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"teacher/student logits differ: {teacher_logits.shape} vs {student_logits.shape}"
        )
    if attention_mask.shape != student_logits.shape[:2]:
        raise ValueError("attention_mask shape does not match logits")
    student_logp = F.log_softmax(student_logits[:, :-1].float(), dim=-1)
    teacher_logp = F.log_softmax(teacher_logits[:, :-1].float(), dim=-1)
    per_token = F.kl_div(
        student_logp,
        teacher_logp,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)
    valid = attention_mask[:, 1:].to(per_token.dtype)
    return (per_token * valid).sum() / valid.sum().clamp_min(1.0)


def _json_safe_config(config: QuasarQADConfig) -> dict:
    value = asdict(config)
    value["dtype"] = str(config.dtype)
    return value


def run_quasar_qad(config: QuasarQADConfig) -> QuasarQADReport:
    """Train, materialize, and save one QUASAR NVFP4 QAD checkpoint."""

    device = config.device or pick_device()
    output_dir = Path(config.save_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"QUASAR output directory is not empty: {output_dir}")
    model_path = Path(config.model).expanduser()
    if model_path.exists() and model_path.resolve() == output_dir.expanduser().resolve():
        raise ValueError("save_dir must not overwrite the input model directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    sequences = load_teacher_data(config.data)
    if not sequences:
        raise ValueError(f"no token sequences found in {config.data!r}")

    student = AutoModelForCausalLM.from_pretrained(config.model, dtype=config.dtype).to(device)
    controller = prepare_quasar_nvfp4(student, config.nvfp4, freeze_non_quantized=True)
    for name, layer in controller.layers.items():
        if layer.in_features % 16:
            raise ValueError(
                f"native compressed-tensors NVFP4 requires in_features divisible by 16; "
                f"{name} has {layer.in_features}"
            )
    if config.gradient_checkpointing:
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(student.config, "use_cache"):
            student.config.use_cache = False
    student.train()

    teacher_path = config.teacher_model or config.model
    teacher = AutoModelForCausalLM.from_pretrained(teacher_path, dtype=config.dtype).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    trainable = controller.parameters()
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.lr,
        betas=(0.9, config.beta2),
        weight_decay=config.weight_decay,
    )
    controller.bind_optimizer(optimizer)
    scheduler = None
    if config.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.steps,
            eta_min=config.lr * 0.1,
        )

    trainable_params = sum(parameter.numel() for parameter in trainable)
    tokens = 0
    final_loss = float("nan")
    started = time.time()
    log_path = output_dir / "train_log.jsonl"

    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(1, config.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for _ in range(config.gradient_accumulation_steps):
                batch = make_batch(
                    sequences,
                    rng,
                    config.batch_seq,
                    config.seq_len,
                    device,
                    pad_id=config.pad_id,
                )
                with torch.no_grad():
                    teacher_logits = teacher(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).logits
                student_logits = student(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
                loss = forward_kl_loss(
                    student_logits,
                    teacher_logits,
                    batch["attention_mask"],
                )
                (loss / config.gradient_accumulation_steps).backward()
                accumulated_loss += loss.detach().item()
                tokens += int(batch["attention_mask"].sum().item())

            torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            final_loss = accumulated_loss / config.gradient_accumulation_steps

            if step % config.log_every == 0 or step == 1 or step == config.steps:
                record = {
                    "step": step,
                    "tokens": tokens,
                    "forward_kl": final_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "saliency_layers": controller.saliency_layer_count,
                    "tok_per_s": tokens / max(time.time() - started, 1e-9),
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                print(f"[quasar-nvfp4] {json.dumps(record)}", flush=True)

    saliency_layers = controller.saliency_layer_count
    student.eval()
    del teacher, teacher_logits, student_logits, loss, batch
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    manifest = materialize_quasar_nvfp4(controller)
    del optimizer, scheduler, trainable, controller
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    student.save_pretrained(output_dir, max_shard_size=config.max_shard_size)
    tokenizer.save_pretrained(output_dir)

    report = QuasarQADReport(
        algorithm="quasar",
        format="nvfp4-pack-quantized",
        steps=config.steps,
        tokens=tokens,
        final_loss=final_loss,
        trainable_params=trainable_params,
        quantized_layers=len(manifest["layers"]),
        saliency_layers=saliency_layers,
        wall_time_s=time.time() - started,
        save_dir=str(output_dir),
    )
    summary = {
        "report": asdict(report),
        "args": _json_safe_config(config),
        "quantization": manifest,
    }
    with (output_dir / "quasar_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return report
