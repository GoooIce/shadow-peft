"""Direction-① 3×2 eval-grid harness: one cell -> every metric.

A cell is ``(backbone tier, shadow precision, kernel path)``:

- backbone tier: ``bf16`` / ``w8a8_fp8`` / ``w4a16`` via :mod:`lab.tiers`
  (conversion always precedes shadow mounting — iron rule 1).
- shadow precision: ``bf16`` / ``int8``. INT8 is **eval-time PTQ of the
  trained shadow backbone** through the affine-8bit scheme row
  (``quantize_model_8bit`` on ``peft.shadow_model``); training is untouched
  and the grid's INT8 columns share one training with their BF16 twin. Our
  packed modules dequantize-then-matmul, so the shadow side is inherently
  reference-numerics on both kernel paths — the fake-quant/kernel split only
  concerns the torchao backbone tiers.
- kernel path: ``kernel`` keeps torchao's quantized tensor implementations
  (real kernels; on CUDA these are the FP8/Int4 kernels); ``fake_quant``
  dequantizes the converted weights back to plain tensors so matmuls run in
  bf16 with the quantization error baked in (reference numerics). torchao
  documents the two paths are not always numerically equal — hence both.

Per-cell metrics: wikitext chunked ppl (main axis, :func:`lab.atoms.eval_ppl`),
MMLU / GSM8K generation scores (extraction seams from :mod:`lab.eval`),
KLD(reference ‖ cell) logits divergence, and the memory three-rows (weight
residency / activation+KV peak / total peak, allocated & reserved columns via
``reset_peak_memory_stats`` — CUDA only; CPU cells record null peaks with a
device note). The vocabulary is never trimmed (Gemma keeps 262,144).

``run_grid`` additionally evaluates each tier's *bare* quantized backbone
(no shadow) so the report stage can compute the ppl 挽回率
``(ppl_quant − ppl_quant+shadow) / (ppl_quant − ppl_bf16)`` from raw JSON.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from transformers import PreTrainedModel

from shadow_peft import ShadowConfig, ShadowForCausalLM, get_shadow_model
from shadow_peft.peft_model import ShadowPeftModel
from shadow_peft.quantization import quantize_model_8bit

from . import tiers
from .atoms import eval_ppl
from .eval import extract_answer_from_text, extract_gsm8k_answer_from_text

SHADOW_PRECISIONS = ("bf16", "int8")
KERNEL_PATHS = ("fake_quant", "kernel")
# The 3×2 grid: (backbone_tier, shadow_precision).
GRID: list[tuple[str, str]] = [(t, s) for t in tiers.TIERS for s in SHADOW_PRECISIONS]


# ---------------------------------------------------------------------------
# Cell mechanics
# ---------------------------------------------------------------------------


def ptq_shadow_int8(peft_model: ShadowPeftModel, group_size: int = 128) -> dict[str, Any]:
    """Eval-time PTQ of the trained shadow backbone via the affine-8bit row."""
    return quantize_model_8bit(peft_model.shadow_model, group_size=group_size)


def to_fake_quant(module: torch.nn.Module) -> int:
    """Swap torchao-quantized weight tensors for dequantized plain tensors.

    After this, every matmul runs in ordinary bf16 with the quantization
    error baked into the weights — torchao's "fake quant" reference numerics.
    Returns the number of swapped weights. Modules already replaced by our
    own packed quantization (e.g. the INT8 shadow) dequantize in their
    forward and are left untouched.
    """
    swapped = 0
    for _, mod in module.named_modules():
        w = getattr(mod, "weight", None)
        if (
            w is not None
            and type(w).__module__.split(".")[0] == "torchao"
            and hasattr(type(w), "dequantize")
        ):
            # torchao tensor-impl (Int8Tensor/Int4Tensor/Float8Tensor...);
            # plain tensors also expose a no-op `dequantize` that only raises,
            # so the module check is what selects the real implementations.
            deq = w.dequantize()  # back to the original high-precision dtype
            mod.weight = torch.nn.Parameter(deq, requires_grad=False)
            swapped += 1
    return swapped


@torch.no_grad()
def eval_kld(
    ref_model: torch.nn.Module | Callable[[dict[str, torch.Tensor]], torch.Tensor],
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
) -> float:
    """Mean per-token KL(P_ref ‖ P_model) over ``batches`` logits.

    ``ref_model`` may also be a callable mapping a batch to reference logits —
    lets the anchor's logits be cached once and its weights freed (two 12B
    models do not fit on one 32GB card together).
    """
    model.eval()
    if hasattr(ref_model, "eval"):
        ref_model.eval()
    total_kl, total_tok = 0.0, 0
    for batch in batches:
        if callable(ref_model) and not isinstance(ref_model, torch.nn.Module):
            r = ref_model(batch).float()
        else:
            r = ref_model(
                input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask")
            ).logits.float()
        q = model(
            input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask")
        ).logits.float()
        logp = torch.log_softmax(r, dim=-1)
        logq = torch.log_softmax(q, dim=-1)
        kl = (logp.exp() * (logp - logq)).sum(-1)  # [b, t]
        mask = batch.get("attention_mask")
        if mask is not None:
            kl = kl * mask.float()
            ntok = int(mask.sum().item())
        else:
            ntok = kl.numel()
        total_kl += kl.sum().item()
        total_tok += ntok
    return total_kl / max(total_tok, 1)


def weight_residency_bytes(module: torch.nn.Module) -> int:
    """Sum of unique tensor storages (params, buffers, quantized weight attrs).

    Storage-address dedup keeps tied weights (e.g. lm_head ↔ embed_tokens)
    counted once. Works for torchao tensor-impl weights that no longer appear
    in ``named_parameters``.
    """
    seen: set[int] = set()
    total = 0

    def add(t: torch.Tensor) -> None:
        nonlocal total
        if isinstance(t, torch.Tensor) and t is not None:
            try:
                storage = t.untyped_storage()
                nbytes = storage.nbytes()
                key = ("storage", storage.data_ptr())
            except (RuntimeError, ValueError):
                # Tensor-impl subclasses (torchao) may not expose a valid
                # python storage; fall back to shape-based accounting keyed
                # by the tensor's own id.
                nbytes = t.numel() * t.element_size()
                key = ("tensor", id(t))
            if key not in seen:
                seen.add(key)
                total += nbytes

    for _, mod in module.named_modules():
        for t in list(mod._parameters.values()) + list(mod._buffers.values()):
            if t is not None:
                add(t)
        # Quantized weights live as plain tensor attributes (tensor-impl
        # subclasses that nn.Module does not track as parameters).
        from torch.nn import Parameter

        for v in vars(mod).values():
            if isinstance(v, torch.Tensor) and not isinstance(v, Parameter):
                add(v)
    return total


@torch.no_grad()
def generation_score(
    model: torch.nn.Module,
    examples: list[dict[str, Any]],
    decode: Callable[[list[int]], str],
    *,
    task: str,
    max_new_tokens: int = 12,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Greedy-generation score for a tiny benchmark probe.

    ``examples`` carry ``prompt_ids`` and ``answer``; ``task`` selects the
    canonical extraction seam from :mod:`lab.eval` (``mmlu`` → letter,
    ``gsm8k`` → numeric final answer). Returns ``{"score", "n"}``.
    """
    if not examples:
        return {"score": None, "n": 0}
    model.eval()
    hits = 0
    for ex in examples:
        ids = torch.tensor([ex["prompt_ids"]], dtype=torch.long)
        if device is not None:
            ids = ids.to(device)
        attn = torch.ones_like(ids)
        out = model.generate(
            input_ids=ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        text = decode(out[0].tolist()[len(ex["prompt_ids"]) :])
        if task == "mmlu":
            pred = extract_answer_from_text(text)
            ok = pred is not None and pred == str(ex["answer"]).upper()
        elif task == "gsm8k":
            pred = extract_gsm8k_answer_from_text(text)
            ok = str(pred).replace(",", "").strip() == str(ex["answer"]).replace(",", "").strip()
        else:
            raise ValueError(f"unknown task {task!r}")
        hits += int(ok)
    return {"score": hits / len(examples), "n": len(examples)}


# ---------------------------------------------------------------------------
# Cell / grid evaluation
# ---------------------------------------------------------------------------


def evaluate_cell(
    *,
    base: PreTrainedModel,
    shadow_config: ShadowConfig,
    adapter_state: dict[str, torch.Tensor] | None,
    backbone_tier: str,
    shadow_precision: str,
    kernel_path: str,
    tokenizer: Any,
    eval_text: str,
    device: torch.device,
    ref_model: torch.nn.Module | None = None,
    kld_batches: list[dict[str, torch.Tensor]] | None = None,
    mmlu_examples: list[dict[str, Any]] | None = None,
    gsm8k_examples: list[dict[str, Any]] | None = None,
    decode: Callable[[list[int]], str] | None = None,
    ppl_chunks: int = 4,
    max_new_tokens: int = 12,
) -> dict[str, Any]:
    """Evaluate one grid cell; returns its metrics dict (JSON-serializable)."""
    if backbone_tier not in tiers.TIERS:
        raise ValueError(f"unknown backbone tier {backbone_tier!r}")
    if shadow_precision not in SHADOW_PRECISIONS:
        raise ValueError(f"unknown shadow precision {shadow_precision!r}")
    if kernel_path not in KERNEL_PATHS and backbone_tier != "bf16":
        raise ValueError(f"unknown kernel path {kernel_path!r}")

    tier_manifest = tiers.apply_tier_conversion(base, backbone_tier, device)
    base.eval()

    peft_model = get_shadow_model(base, shadow_config)
    if adapter_state is not None:
        peft_model.load_adapter_state_dict(adapter_state)
    peft_model.eval()
    # ShadowForCausalLM is the generate-capable wrapper: HF generate must run
    # through the adapter's forward (the wrapped base alone has no shadow
    # state), and its prepare_inputs_for_generation does exactly that.
    wrapped = ShadowForCausalLM(peft_model)
    wrapped.eval()

    fake_swaps = 0
    if kernel_path == "fake_quant" and backbone_tier != "bf16":
        fake_swaps = to_fake_quant(peft_model)

    shadow_ptq = None
    if shadow_precision == "int8":
        shadow_ptq = ptq_shadow_int8(peft_model)

    embed = base.get_input_embeddings()
    vocab_rows = embed.weight.shape[0] if embed is not None else None

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    ppl = eval_ppl(
        wrapped, tokenizer, eval_text, chunks=ppl_chunks, device=str(device)
    )
    mmlu = (
        generation_score(
            wrapped, mmlu_examples, decode, task="mmlu",
            max_new_tokens=max_new_tokens, device=device,
        )
        if mmlu_examples and decode
        else {"score": None, "n": 0}
    )
    gsm8k = (
        generation_score(
            wrapped, gsm8k_examples, decode, task="gsm8k",
            max_new_tokens=max_new_tokens, device=device,
        )
        if gsm8k_examples and decode
        else {"score": None, "n": 0}
    )
    kld = (
        eval_kld(ref_model, wrapped, kld_batches)
        if ref_model is not None and kld_batches
        else None
    )

    weights_b = weight_residency_bytes(peft_model)
    peak_alloc = peak_reserved = None
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)

    return {
        "cell": {
            "backbone_tier": backbone_tier,
            "shadow_precision": shadow_precision,
            "kernel_path": backbone_tier == "bf16" and "n/a" or kernel_path,
        },
        "backbone_format": tier_manifest.get("format"),
        "fake_quant_swaps": fake_swaps,
        "shadow_ptq_modules": len(shadow_ptq["quantized"]) if shadow_ptq else 0,
        "ppl": ppl,
        "mmlu": mmlu,
        "gsm8k": gsm8k,
        "kld_vs_baseline": kld,
        "memory": {
            "device": str(device),
            "weight_residency_bytes": weights_b,
            "activation_kv_peak_bytes": (
                peak_alloc - weights_b if peak_alloc is not None else None
            ),
            "total_peak_allocated_bytes": peak_alloc,
            "total_peak_reserved_bytes": peak_reserved,
            "note": None if peak_alloc is not None else "peak stats are CUDA-only",
        },
        "vocab": {"embedding_rows": vocab_rows, "trimmed": False},
    }


def evaluate_bare_base(
    *,
    base: PreTrainedModel,
    backbone_tier: str,
    kernel_path: str,
    tokenizer: Any,
    eval_text: str,
    device: torch.device,
    ppl_chunks: int = 4,
) -> dict[str, Any]:
    """Per-tier bare (no-shadow) backbone ppl — the ppl_quant of 挽回率."""
    tier_manifest = tiers.apply_tier_conversion(base, backbone_tier, device)
    base.eval()
    fake_swaps = 0
    if kernel_path == "fake_quant" and backbone_tier != "bf16":
        fake_swaps = to_fake_quant(base)
    ppl = eval_ppl(base, tokenizer, eval_text, chunks=ppl_chunks, device=str(device))
    return {
        "backbone_tier": backbone_tier,
        "kernel_path": backbone_tier == "bf16" and "n/a" or kernel_path,
        "backbone_format": tier_manifest.get("format"),
        "fake_quant_swaps": fake_swaps,
        "ppl": ppl,
    }


def run_grid(
    *,
    base_factory: Callable[[], PreTrainedModel],
    shadow_config: ShadowConfig,
    adapter_state: dict[str, torch.Tensor] | None,
    tokenizer: Any,
    eval_text: str,
    device: torch.device,
    kld_batches: list[dict[str, torch.Tensor]] | None = None,
    mmlu_examples: list[dict[str, Any]] | None = None,
    gsm8k_examples: list[dict[str, Any]] | None = None,
    decode: Callable[[list[int]], str] | None = None,
    kernel_paths: tuple[str, ...] = KERNEL_PATHS,
    ppl_chunks: int = 4,
    max_new_tokens: int = 12,
) -> dict[str, Any]:
    """Run the 3×2 grid × kernel paths + anchor + bare baselines.

    The anchor (backbone bf16 + shadow bf16) provides the BF16 baseline for
    MMLU/GSM8K recovery (cell score / anchor score) and the KLD reference.
    Raw ppls for every cell, the anchor, and each tier's bare backbone land in
    the result so the report stage can compute ppl 挽回率 without re-running.
    """
    # Anchor first: reference for KLD + downstream recovery rates.
    def adapter_for(tier: str):
        if isinstance(adapter_state, dict) and set(adapter_state) & set(tiers.TIERS):
            return adapter_state.get(tier)
        return adapter_state

    anchor = evaluate_cell(
        base=base_factory(),
        shadow_config=shadow_config,
        adapter_state=adapter_for("bf16"),
        backbone_tier="bf16",
        shadow_precision="bf16",
        kernel_path="n/a",
        tokenizer=tokenizer,
        eval_text=eval_text,
        device=device,
        kld_batches=kld_batches,
        mmlu_examples=mmlu_examples,
        gsm8k_examples=gsm8k_examples,
        decode=decode,
        ppl_chunks=ppl_chunks,
        max_new_tokens=max_new_tokens,
    )
    # KLD reference = the anchor model itself (bf16 backbone + bf16 shadow).
    ref_logits = None
    if kld_batches:
        ref_inner = get_shadow_model(base_factory(), shadow_config)
        state = adapter_for("bf16")
        if state is not None:
            ref_inner.load_adapter_state_dict(state)
        ref_inner.eval()
        cached = []
        with torch.no_grad():
            for batch in kld_batches:
                cached.append(
                    ref_inner(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                    ).logits.float().cpu()
                )
        lookup = {id(b): c for b, c in zip(kld_batches, cached)}

        def ref_logits(batch):
            return lookup[id(batch)].to(batch["input_ids"].device)

        del ref_inner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cells: list[dict[str, Any]] = []
    for tier, shadow_prec in GRID:
        for path in kernel_paths:
            if tier == "bf16" and path != kernel_paths[0]:
                continue  # bf16 has no torchao conversion: one evaluation
            cell = evaluate_cell(
                base=base_factory(),
                shadow_config=shadow_config,
                adapter_state=adapter_for(tier),
                backbone_tier=tier,
                shadow_precision=shadow_prec,
                kernel_path=path,
                tokenizer=tokenizer,
                eval_text=eval_text,
                device=device,
                ref_model=ref_logits,
                kld_batches=kld_batches,
                mmlu_examples=mmlu_examples,
                gsm8k_examples=gsm8k_examples,
                decode=decode,
                ppl_chunks=ppl_chunks,
                max_new_tokens=max_new_tokens,
            )
            for bench in ("mmlu", "gsm8k"):
                a, c = anchor[bench]["score"], cell[bench]["score"]
                cell[bench]["recovery_vs_bf16"] = (c / a) if (a and c is not None) else None
            cells.append(cell)

    bare: list[dict[str, Any]] = [
        evaluate_bare_base(
            base=base_factory(),
            backbone_tier=tier,
            kernel_path=path,
            tokenizer=tokenizer,
            eval_text=eval_text,
            device=device,
            ppl_chunks=ppl_chunks,
        )
        for tier in tiers.TIERS
        for path in (kernel_paths if tier != "bf16" else ("n/a",))
    ]

    result = {
        "anchor": anchor,
        "cells": cells,
        "bare_bases": bare,
        "pareto": pareto_table(cells),
        "grid_definition": {
            "tiers": list(tiers.TIERS),
            "shadow_precisions": list(SHADOW_PRECISIONS),
            "kernel_paths": list(kernel_paths),
            "ppl_recovery_note": (
                "挽回率 = (ppl_quant − ppl_quant+shadow) / (ppl_quant − ppl_bf16) "
                "computable from cells[].ppl, bare_bases[].ppl, anchor.ppl"
            ),
        },
    }
    return result


def write_metrics(result: dict[str, Any], path: str | Path) -> None:
    """Persist a run_grid result as the cell metrics JSON (runbook entry)."""
    import json as _json

    from pathlib import Path as _Path

    out = _Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def pareto_table(cells: list[dict[str, Any]]) -> str:
    """Markdown Pareto table: x = total peak allocated (CPU proxy: weight
    residency), y = ppl, kernel path labeled per row."""
    def x_of(cell: dict[str, Any]) -> tuple[int, bool]:
        peak = cell["memory"]["total_peak_allocated_bytes"]
        if peak is not None:
            return peak, True
        return cell["memory"]["weight_residency_bytes"], False

    rows = sorted(cells, key=lambda c: x_of(c)[0])
    lines = [
        "| cell | kernel_path | x (total peak alloc) | ppl |",
        "|---|---|---|---|",
    ]
    for c in rows:
        x, real = x_of(c)
        x_label = f"{x:,}" + ("" if real else " (weight-residency proxy)")
        cc = c["cell"]
        name = f"{cc['backbone_tier']}+{cc['shadow_precision']}"
        lines.append(f"| {name} | {cc['kernel_path']} | {x_label} | {c['ppl']:.3f} |")
    return "\n".join(lines)
