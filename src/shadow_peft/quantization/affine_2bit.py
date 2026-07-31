"""
Ternary 2-bit affine quantization: weights are projected to ``{-1, 0, +1}``
with a per-group scale, then packed into 2-bit slots.

This mirrors ``affine_1bit.py`` but targets 3 levels instead of 2.  Two
projection entry points are provided:

  - :func:`quantize_2bit_ternary` is the original configurable baseline.  Per
    group of ``group_size`` weights, ``s_g = mean(|w|)``, then
    ``t_i = 0`` if ``|w_i| < thr_ratio * s_g`` else ``sign(w_i)``.
  - :func:`quantize_2bit_bonsai` is the reproducible Bonsai-style surrogate
    from ``deep-research-report.md``: g128, absmean initialization, strict
    threshold assignment, and eight group-wise least-squares updates by
    default.  Prism ML's proprietary migration/training solver is not public;
    this function reproduces the published representation, not that solver.

Code/affine mapping (the only valid one for 3 ternary levels in a 4-level
2-bit affine grid)::

    ternary  code   dequant (scale=s_g, bias=-s_g)
      -1  -->  0  --> -s_g
       0  -->  1  -->  0
      +1  -->  2  --> +s_g
               3  --> unused (would dequantize to +2*s_g)

Bits are packed 16 per word, LSB-first (weight j occupies bits 2*(j%16)
and 2*(j%16)+1), stored as int32 bit patterns. Returns
``(packed_int32 [out, in/16], scales [out, in/group_size], biases [out,
in/group_size])``; scales/biases keep the input dtype (non-fp16/bf16 inputs
are cast to fp16 first).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8
# Keep the temporary fp32/ternary/int64 working set bounded when quantizing
# vocabulary tables and other very large matrices. Bonsai statistics are
# independent per row/group, so row chunking is bit-identical to processing
# the full tensor at once.
_BONSAI_MAX_CHUNK_ELEMENTS = 8 * 1024 * 1024


def _check_group_size(group_size: int) -> None:
    if group_size < 16 or group_size % 16 != 0:
        raise ValueError(f"group_size must be a positive multiple of 16, got {group_size}")


def _check_dims(in_features: int, group_size: int) -> None:
    _check_group_size(group_size)
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        )


def quantize_2bit_ternary(
    weight: torch.Tensor,
    group_size: int = 128,
    refine_iters: int = 0,
    thr_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a 2D weight ``[out, in]`` to packed ternary 2-bit affine form.

    Per group of ``group_size`` values along the last dim:
      s_g = mean(|w|);  t = 0 if |w| < thr_ratio*s_g else sign(w)
      dequant: w_hat = s_g * t, stored as scale=s_g, bias=-s_g.

    With ``refine_iters > 0``, ``s_g`` is refined by Lloyd-style alternation
    (least-squares refit, then re-threshold), which reduces group MSE.

    Bits are packed 16 per word, LSB-first. Returns ``(packed_int32 [out,
    in/16], scales [out, in/group_size], biases [out, in/group_size])``.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D [out, in], got shape {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    _check_dims(in_features, group_size)
    if refine_iters < 0:
        raise ValueError(f"refine_iters must be >= 0, got {refine_iters}")
    if not 0.0 < thr_ratio <= 1.0:
        raise ValueError(f"thr_ratio must be in (0, 1], got {thr_ratio}")

    dtype = weight.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        weight = weight.to(torch.float16)
        dtype = weight.dtype

    # Group stats in fp32 so the eps clamp survives (1e-8 underflows fp16).
    groups = weight.reshape(out_features, in_features // group_size, group_size).float()

    # absmean init (BitNet b1.58 style).
    scales = groups.abs().mean(dim=-1).clamp_min(_EPS)

    # Ternary assignment: |w| < thr*s -> 0, else sign(w).
    thr = thr_ratio * scales.unsqueeze(-1)
    ternary = torch.zeros_like(groups)
    ternary[groups > thr] = 1.0
    ternary[groups < -thr] = -1.0

    if refine_iters > 0:
        for _ in range(refine_iters):
            denom = (ternary * ternary).sum(dim=-1).clamp_min(1.0)
            new_scales = (groups * ternary).sum(dim=-1) / denom
            new_scales = new_scales.abs().clamp_min(_EPS)
            scales = new_scales
            thr = thr_ratio * scales.unsqueeze(-1)
            ternary = torch.zeros_like(groups)
            ternary[groups > thr] = 1.0
            ternary[groups < -thr] = -1.0

    # Map ternary {-1,0,+1} -> code {0,1,2}. Code 3 is never produced.
    code = (ternary + 1.0).to(torch.int64)  # -1->0, 0->1, +1->2

    # Pack 16 2-bit codes per int32 word, LSB-first.
    n_words = in_features // 16
    code = code.reshape(out_features, in_features)
    code = code.reshape(out_features, n_words, 16)
    shifts = 2 ** (2 * torch.arange(16, dtype=torch.int64, device=weight.device))
    packed = (code * shifts).sum(dim=-1)  # values in [0, 2^32)
    packed = (packed & 0xFFFFFFFF).to(torch.int32)  # wrap to int32 bit pattern

    # Affine params: scale=s_g, bias=-s_g  =>  dequant = s_g * (code-1) = s_g * ternary
    biases = (-scales).to(dtype)

    return packed, scales.to(dtype), biases


def quantize_2bit_bonsai(
    weight: torch.Tensor,
    group_size: int = 128,
    refine_iters: int = 8,
    thr_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``weight [out, in]`` with the report's Bonsai-style surrogate.

    Each group is initialized with ``mean(abs(w))``.  A refinement iteration
    assigns strict ternary levels at ``+-thr_ratio * scale`` (values exactly on
    the boundary remain zero), then least-squares refits the scale::

        q_i = -1 if w_i < -thr_ratio*s else 1 if w_i > thr_ratio*s else 0
        s   = abs(sum(w_i*q_i) / max(sum(q_i*q_i), 1))

    The returned MLX-affine representation uses codes ``{0,1,2}`` for ternary
    ``{-1,0,+1}``, ``scale=s`` and ``bias=-s``; code 3 is never emitted.  Codes
    are packed 16 per int32 word, least-significant code first.

    ``group_size=128``, ``refine_iters=8`` and ``thr_ratio=0.5`` are the report
    defaults.  Other group sizes remain available for small-model tests and
    experiments, but are not Bonsai g128 checkpoint compatible.

    This is a format-compatible, deterministic baseline.  It must not be
    interpreted as a reproduction of Prism ML's unpublished optimization or
    training procedure.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D [out, in], got shape {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    _check_dims(in_features, group_size)
    if refine_iters < 0:
        raise ValueError(f"refine_iters must be >= 0, got {refine_iters}")
    if not 0.0 < thr_ratio <= 1.0:
        raise ValueError(f"thr_ratio must be in (0, 1], got {thr_ratio}")

    dtype = weight.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        weight = weight.to(torch.float16)
        dtype = weight.dtype

    rows_per_chunk = max(1, _BONSAI_MAX_CHUNK_ELEMENTS // in_features)
    packed_chunks: list[torch.Tensor] = []
    scale_chunks: list[torch.Tensor] = []
    bias_chunks: list[torch.Tensor] = []

    def assign(
        chunk_groups: torch.Tensor, current_scales: torch.Tensor
    ) -> torch.Tensor:
        threshold = thr_ratio * current_scales.unsqueeze(-1)
        # Strict comparisons are intentional: values exactly at either
        # threshold belong to the zero level, matching the report pseudocode.
        return torch.where(
            chunk_groups > threshold,
            torch.ones_like(chunk_groups),
            torch.where(
                chunk_groups < -threshold, -torch.ones_like(chunk_groups), 0.0
            ),
        )

    for row_start in range(0, out_features, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, out_features)
        groups = (
            weight[row_start:row_end]
            .reshape(row_end - row_start, in_features // group_size, group_size)
            .float()
        )

        scales = groups.abs().mean(dim=-1).clamp_min(_EPS)

        # refine_iters=0 is a useful no-refit baseline. For N>0 this is exactly
        # N assignment -> least-squares steps; the returned scale is fitted to
        # the returned assignment rather than to a subsequently changed one.
        ternary = assign(groups, scales)
        for iteration in range(refine_iters):
            denom = (ternary * ternary).sum(dim=-1).clamp_min(1.0)
            scales = ((groups * ternary).sum(dim=-1) / denom).abs().clamp_min(_EPS)
            if iteration + 1 < refine_iters:
                ternary = assign(groups, scales)

        # Map ternary {-1,0,+1} -> code {0,1,2}. Code 3 is never produced.
        code = (ternary + 1.0).to(torch.int64)

        # Pack 16 2-bit codes per int32 word, LSB-first.
        n_words = in_features // 16
        code = code.reshape(row_end - row_start, n_words, 16)
        shifts = 2 ** (
            2 * torch.arange(16, dtype=torch.int64, device=weight.device)
        )
        packed = (code * shifts).sum(dim=-1)
        packed = (packed & 0xFFFFFFFF).to(torch.int32)

        # 1e-8 rounds to zero in fp16. Replace only values that actually
        # rounded to zero; valid fp16 subnormal scales must not be inflated.
        stored_scales = scales.to(dtype)
        stored_scales = torch.where(
            stored_scales == 0,
            torch.full_like(stored_scales, torch.finfo(dtype).tiny),
            stored_scales,
        )
        packed_chunks.append(packed)
        scale_chunks.append(stored_scales)
        bias_chunks.append(-stored_scales)

    if len(packed_chunks) == 1:
        return packed_chunks[0], scale_chunks[0], bias_chunks[0]
    return (
        torch.cat(packed_chunks, dim=0),
        torch.cat(scale_chunks, dim=0),
        torch.cat(bias_chunks, dim=0),
    )


def dequantize_2bit_ternary(
    packed: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Inverse of :func:`quantize_2bit_ternary`: unpack ``packed [out, in/16]``
    (int32, LSB-first, 16 2-bit codes per word) and broadcast
    ``scale * q + bias`` per group. Output dtype follows ``scales``.
    """
    out_features, num_words = packed.shape
    in_features = num_words * 16
    shifts = 2 * torch.arange(16, dtype=torch.int32, device=packed.device)
    codes = ((packed.unsqueeze(-1) >> shifts) & 0b11).to(scales.dtype)
    codes = codes.reshape(out_features, in_features // group_size, group_size)
    w = scales.unsqueeze(-1) * codes + biases.unsqueeze(-1)
    return w.reshape(out_features, in_features)


class QuantizedLinear2Bit(nn.Module):
    """
    Drop-in replacement for ``nn.Linear`` storing a packed ternary 2-bit weight.

    The packed weight and group scales/biases are buffers (no gradient); the
    optional output bias is a frozen Parameter. In eval mode the dequantized
    weight is cached and reused; in train mode it is recomputed every forward.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        _check_dims(in_features, group_size)
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.register_buffer(
            "weight", torch.zeros(out_features, in_features // 16, dtype=torch.int32)
        )
        self.register_buffer(
            "scales",
            torch.zeros(out_features, in_features // group_size, dtype=torch.float16),
        )
        self.register_buffer(
            "biases",
            torch.zeros(out_features, in_features // group_size, dtype=torch.float16),
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self._cached_weight: torch.Tensor | None = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 128,
        refine_iters: int = 0,
        thr_ratio: float = 0.7,
        quant_fn=None,
    ) -> QuantizedLinear2Bit:
        if quant_fn is None:

            def _default(w):
                return quantize_2bit_ternary(
                    w, group_size, refine_iters=refine_iters, thr_ratio=thr_ratio
                )

            quant_fn = _default
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
        )
        with torch.no_grad():
            packed, scales, biases = quant_fn(linear.weight.detach())
            module.weight = packed
            module.scales = scales
            module.biases = biases
            if linear.bias is not None:
                # Match the dequantized weight dtype so F.linear dtype-checks pass
                # even when the source linear was fp32 (weight becomes fp16).
                module.bias = nn.Parameter(
                    linear.bias.detach().clone().to(scales.dtype), requires_grad=False
                )
        return module

    def dequantize_weight(self) -> torch.Tensor:
        return dequantize_2bit_ternary(
            self.weight, self.scales, self.biases, self.group_size
        )

    def train(self, mode: bool = True) -> QuantizedLinear2Bit:
        if mode:
            self._cached_weight = None
        return super().train(mode)

    def _apply(self, fn, recurse=True):
        # Invalidate the cache when buffers move (`.to()`, `.cuda()`, ...).
        self._cached_weight = None
        return super()._apply(fn, recurse)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # A parent model's load_state_dict() bypasses this module's public
        # load_state_dict method, so invalidate at the recursive hook point.
        self._cached_weight = None
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.dequantize_weight()
        else:
            if self._cached_weight is None:
                self._cached_weight = self.dequantize_weight()
            weight = self._cached_weight
        # Float32 source models are quantized into an FP16 container. Cast the
        # dequantized compute view back to the activation dtype so the module
        # remains usable without requiring callers to cast their inputs.
        if weight.dtype != x.dtype:
            weight = weight.to(x.dtype)
        bias = self.bias
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(x.dtype)
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, group_size={self.group_size}"
        )


class QuantizedEmbedding2Bit(nn.Module):
    """Drop-in replacement for ``nn.Embedding`` storing a packed ternary 2-bit weight."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        _check_dims(embedding_dim, group_size)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.group_size = group_size
        self.register_buffer(
            "weight", torch.zeros(num_embeddings, embedding_dim // 16, dtype=torch.int32)
        )
        self.register_buffer(
            "scales",
            torch.zeros(num_embeddings, embedding_dim // group_size, dtype=torch.float16),
        )
        self.register_buffer(
            "biases",
            torch.zeros(num_embeddings, embedding_dim // group_size, dtype=torch.float16),
        )

    @classmethod
    def from_embedding(
        cls,
        embedding: nn.Embedding,
        group_size: int = 128,
        refine_iters: int = 0,
        thr_ratio: float = 0.7,
        quant_fn=None,
    ) -> QuantizedEmbedding2Bit:
        if quant_fn is None:

            def _default(w):
                return quantize_2bit_ternary(
                    w, group_size, refine_iters=refine_iters, thr_ratio=thr_ratio
                )

            quant_fn = _default
        module = cls(
            embedding.num_embeddings,
            embedding.embedding_dim,
            padding_idx=embedding.padding_idx,
            group_size=group_size,
        )
        with torch.no_grad():
            packed, scales, biases = quant_fn(embedding.weight.detach())
            module.weight = packed
            module.scales = scales
            module.biases = biases
        return module

    def dequantize_weight(self) -> torch.Tensor:
        return dequantize_2bit_ternary(
            self.weight, self.scales, self.biases, self.group_size
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Reference implementation: dequantize the whole table, then gather.
        # A gather-then-dequantize kernel would avoid materializing the table.
        weight = dequantize_2bit_ternary(
            self.weight, self.scales, self.biases, self.group_size
        )
        return F.embedding(input, weight, padding_idx=self.padding_idx)

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"padding_idx={self.padding_idx}, group_size={self.group_size}"
        )


def quantize_model_2bit(
    model: nn.Module,
    group_size: int = 128,
    skip_patterns: tuple[str, ...] | list[str] = (),
    refine_iters: int | None = None,
    trim_vocab_to: int | None = None,
    thr_ratio: float = 0.5,
    method: str = "affine",
    quantize_tied_weights: bool | None = None,
) -> dict[str, Any]:
    """
    Replace every ``nn.Linear`` / ``nn.Embedding`` in ``model`` (in place) with
    its ternary 2-bit quantized counterpart. Returns a JSON-serializable manifest.

    - Modules whose full name contains any of ``skip_patterns`` are left as-is.
    - A ``nn.Linear`` whose weight is shared with an already-replaced module
      (for example a tied lm_head) follows ``quantize_tied_weights``; skipped
      tied modules are listed under ``"tied"``.
    - Layers whose last dim is not divisible by ``group_size`` are skipped and
      listed under "skipped" with the reason.
    - ``method="bonsai"`` selects the report's g128 ternary/least-squares
      surrogate.  It reproduces the published representation, not Prism ML's
      unpublished migration/training solver.
    - When ``refine_iters`` is omitted, it defaults to 8 for ``method="bonsai"``
      and 0 for the legacy ``method="affine"`` baseline.
    - Tied output heads are quantized as a separate triplet by default in
      Bonsai mode, matching the published embedding/lm-head coverage.  The
      legacy affine mode keeps its previous skip behavior.  Override either
      default with ``quantize_tied_weights``.
    - ``trim_vocab_to`` drops dead padding rows (e.g. Qwen3's 151936 -> 151669)
      from ``nn.Embedding`` tables and any module with "lm_head" in its name
      before quantization. Pass the FULL tokenizer length (including special
      tokens, e.g. ``len(AutoTokenizer)``).
    """
    _check_group_size(group_size)
    if not 0.0 < thr_ratio <= 1.0:
        raise ValueError(f"thr_ratio must be in (0, 1], got {thr_ratio}")
    if method not in ("affine", "bonsai"):
        raise ValueError(f"method must be 'affine' or 'bonsai', got {method!r}")
    if trim_vocab_to is not None and trim_vocab_to < 1:
        raise ValueError(f"trim_vocab_to must be >= 1, got {trim_vocab_to}")
    if refine_iters is not None and refine_iters < 0:
        raise ValueError(f"refine_iters must be >= 0, got {refine_iters}")

    # Preserve the legacy affine default while giving the report-defined
    # Bonsai-style entry point its documented eight LS updates.
    resolved_refine_iters = (
        (8 if method == "bonsai" else 0) if refine_iters is None else refine_iters
    )
    resolved_quantize_tied = (
        method == "bonsai" if quantize_tied_weights is None else quantize_tied_weights
    )

    if method == "bonsai":

        def quant_fn(w):
            return quantize_2bit_bonsai(
                w,
                group_size,
                refine_iters=resolved_refine_iters,
                thr_ratio=thr_ratio,
            )

    else:
        quant_fn = None  # use the default in from_linear/from_embedding
    quantized: list[str] = []
    skipped: dict[str, str] = {}
    tied: list[str] = []
    quantized_tied: list[str] = []
    trimmed: dict[str, list[int]] = {}
    # Keep only names and original data pointers. Holding a materialized
    # ``list(model.named_modules())`` or the tensors themselves prevents
    # replaced BF16 weights from being released, which is prohibitive for a
    # 27B checkpoint. Capturing all pointers up front also avoids false tied
    # detections if the allocator later reuses a freed address.
    module_weights = [
        (name, module.weight.data_ptr())
        for name, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.Embedding))
    ]
    replaced_weight_ptrs: set[int] = set()

    for name, original_weight_ptr in module_weights:
        module = model.get_submodule(name)
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            raise RuntimeError(f"quantizable module {name!r} changed during replacement")
        matched = next((p for p in skip_patterns if p in name), None)
        if matched is not None:
            skipped[name] = f"matched skip pattern '{matched}'"
            continue
        weight = module.weight
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model

        is_tied = original_weight_ptr in replaced_weight_ptrs
        if is_tied and not resolved_quantize_tied:
            tied.append(name)
            # Keep a tied lm_head dimensionally consistent with a trimmed
            # embedding: re-point it at the trimmed slice of the fp weight.
            if (
                trim_vocab_to is not None
                and isinstance(module, nn.Linear)
                and weight.shape[0] > trim_vocab_to
            ):
                with torch.no_grad():
                    new_head = nn.Linear(
                        weight.shape[1],
                        trim_vocab_to,
                        bias=module.bias is not None,
                        device=weight.device,
                        dtype=weight.dtype,
                    )
                    new_head.weight.copy_(weight[:trim_vocab_to])
                    if module.bias is not None:
                        new_head.bias.copy_(module.bias[:trim_vocab_to])
                setattr(parent, child_name, new_head)
                trimmed[name] = [weight.shape[0], trim_vocab_to]
            continue
        if weight.shape[-1] % group_size != 0:
            skipped[name] = (
                f"last dim {weight.shape[-1]} not divisible by group_size {group_size}"
            )
            continue

        # Row-trim vocab-sized modules before packing.
        source = module
        if (
            trim_vocab_to is not None
            and weight.shape[0] > trim_vocab_to
            and (isinstance(module, nn.Embedding) or "lm_head" in name)
        ):
            with torch.no_grad():
                if isinstance(module, nn.Embedding):
                    source = nn.Embedding(
                        trim_vocab_to, weight.shape[1], device=weight.device
                    )
                else:
                    source = nn.Linear(
                        weight.shape[1],
                        trim_vocab_to,
                        bias=module.bias is not None,
                        device=weight.device,
                        dtype=weight.dtype,
                    )
                source.weight = nn.Parameter(
                    weight[:trim_vocab_to].clone(), requires_grad=False
                )
                if getattr(module, "bias", None) is not None:
                    source.bias = nn.Parameter(
                        module.bias[:trim_vocab_to].clone(), requires_grad=False
                    )
            trimmed[name] = [weight.shape[0], trim_vocab_to]

        if isinstance(source, nn.Linear):
            new_module: nn.Module = QuantizedLinear2Bit.from_linear(
                source,
                group_size,
                refine_iters=resolved_refine_iters,
                thr_ratio=thr_ratio,
                quant_fn=quant_fn,
            )
        else:
            new_module = QuantizedEmbedding2Bit.from_embedding(
                source,
                group_size,
                refine_iters=resolved_refine_iters,
                thr_ratio=thr_ratio,
                quant_fn=quant_fn,
            )
        new_module.train(module.training)
        setattr(parent, child_name, new_module)
        replaced_weight_ptrs.add(original_weight_ptr)
        quantized.append(name)
        if is_tied:
            quantized_tied.append(name)

    if (
        trimmed
        and getattr(model, "config", None) is not None
        and getattr(model.config, "vocab_size", 0) > trim_vocab_to
    ):
        # Keep the model config consistent with the trimmed vocab so e.g. HF's
        # internal CE loss (`logits.view(-1, config.vocab_size)`) still works.
        model.config.vocab_size = trim_vocab_to

    return {
        "quantization": {
            "mode": "affine",
            "bits": 2,
            "ternary": True,
            "method": method,
            "group_size": group_size,
            "refine_iters": resolved_refine_iters,
            "thr_ratio": thr_ratio,
            "quantize_tied_weights": resolved_quantize_tied,
            "trim_vocab_to": trim_vocab_to,
        },
        "quantized": quantized,
        "skipped": skipped,
        "tied": tied,
        "quantized_tied": quantized_tied,
        "trimmed": trimmed,
    }
