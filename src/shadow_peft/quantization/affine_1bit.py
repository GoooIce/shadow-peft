from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file

_EPS = 1e-8


def _check_group_size(group_size: int) -> None:
    if group_size < 32 or group_size % 32 != 0:
        raise ValueError(f"group_size must be a positive multiple of 32, got {group_size}")


def _check_dims(in_features: int, group_size: int) -> None:
    _check_group_size(group_size)
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        )


def quantize_1bit_affine(
    weight: torch.Tensor, group_size: int = 128, refine_iters: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a 2D weight `[out, in]` to packed 1-bit affine form.

    Per group of `group_size` values along the last dim:
      w_min = min(w), w_max = max(w), scale = max(w_max - w_min, eps), bias = w_min
      q = round((w - w_min) / scale) in {0, 1}; dequant: w_hat = scale * q + bias

    With `refine_iters > 0`, the min/max assignment is improved by Lloyd-style
    alternation (least-squares fit of the two levels, then rethreshold at their
    midpoint). Group MSE decreases monotonically; outliers are clipped onto the
    nearer level instead of stretching the grid. Output format is unchanged.

    Bits are packed 32 per word, LSB-first (weight j occupies bit j % 32), stored
    as int32 bit patterns. Returns `(packed_int32 [out, in/32], scales [out,
    in/group_size], biases [out, in/group_size])`; scales/biases keep the input
    dtype (non-fp16/bf16 inputs are cast to fp16 first).
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D [out, in], got shape {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    _check_dims(in_features, group_size)
    if refine_iters < 0:
        raise ValueError(f"refine_iters must be >= 0, got {refine_iters}")

    dtype = weight.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        weight = weight.to(torch.float16)
        dtype = weight.dtype

    # Group stats in fp32 so the eps clamp survives (1e-8 underflows fp16).
    groups = weight.reshape(out_features, in_features // group_size, group_size).float()
    w_min = groups.amin(dim=-1)
    w_max = groups.amax(dim=-1)
    scales = (w_max - w_min).clamp_min(_EPS)
    biases = w_min

    if refine_iters > 0:
        bits = groups > (biases.unsqueeze(-1) + 0.5 * scales.unsqueeze(-1))
        for _ in range(refine_iters):
            qf = bits.float()
            n1 = qf.sum(dim=-1)
            n0 = group_size - n1
            sum1 = (groups * qf).sum(dim=-1)
            sum0 = groups.sum(dim=-1) - sum1
            mean1 = sum1 / n1.clamp_min(1)
            mean0 = sum0 / n0.clamp_min(1)
            new_scale = mean1 - mean0
            valid = (n1 > 0) & (n0 > 0) & (new_scale > _EPS)
            scales = torch.where(valid, new_scale, scales)
            biases = torch.where(valid, mean0, biases)
            bits = groups > (biases.unsqueeze(-1) + 0.5 * scales.unsqueeze(-1))
        q = bits.to(torch.int64)
    else:
        q = torch.round((groups - w_min.unsqueeze(-1)) / scales.unsqueeze(-1)).to(torch.int64)

    q = q.reshape(out_features, in_features // 32, 32)
    powers = 2 ** torch.arange(32, dtype=torch.int64, device=weight.device)
    packed = (q * powers).sum(dim=-1)  # values in [0, 2^32)
    packed = (packed & 0xFFFFFFFF).to(torch.int32)  # wrap to int32 bit pattern
    return packed, scales.to(dtype), biases.to(dtype)


def dequantize_1bit_affine(
    packed: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Inverse of `quantize_1bit_affine`: unpack `packed [out, in/32]` (int32,
    LSB-first) and broadcast `scale * q + bias` per group. Output dtype follows
    `scales`.

    Also the dequantizer for the Bonsai/sign variant
    (`quantize_1bit_bonsai` / `quantize_1bit_sign`): all variants share the
    identical packed-code/scale/bias container, so the same unpacking recovers
    {-d, +d} when ``scale=2d, bias=-d``.
    """
    out_features, num_words = packed.shape
    in_features = num_words * 32
    shifts = torch.arange(32, dtype=torch.int32, device=packed.device)
    # Arithmetic right shift sign-extends, but & 1 recovers the original bit.
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    q = bits.reshape(out_features, in_features // group_size, group_size).to(scales.dtype)
    w = scales.unsqueeze(-1) * q + biases.unsqueeze(-1)
    return w.reshape(out_features, in_features)


def quantize_1bit_bonsai(
    weight: torch.Tensor, group_size: int = 128, refine_iters: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight with the Bonsai-style binary surrogate.

    Per group of group_size values along the last dim:
      d_g = mean(|w|);  t = -1 if w < 0 else +1
      code bit: 0 -> -d_g,  1 -> +d_g
      stored as scale = 2*d_g, bias = -d_g  =>  dequant: 2*d_g*q - d_g = {-d_g, +d_g}

    Zero is assigned to the positive bit as a deterministic tie convention.
    The optimal magnitude for a fixed sign assignment is already
    mean(abs(w)). Therefore refine_iters is accepted for API parity but does
    not alter the result: this surrogate is at its closed-form fixed point
    immediately.

    Codes are packed 32 per int32 word, least-significant bit first. Group
    statistics and signs are computed from an FP32 view of the input. FP16 or
    BF16 input keeps its scale storage dtype; other input stores scale/bias in
    FP16. group_size=128 is the published Bonsai/Q1_0 layout. Other multiples
    of 32 remain available for torch reference tests but are not
    g128-compatible.

    This reproduces the public binary representation and a deterministic
    dense-to-binary baseline. It is not Prism ML's unpublished
    migration/training solver.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D [out, in], got shape {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    _check_dims(in_features, group_size)
    if refine_iters < 0:
        raise ValueError(f"refine_iters must be >= 0, got {refine_iters}")

    dtype = (
        weight.dtype
        if weight.dtype in (torch.float16, torch.bfloat16)
        else torch.float16
    )

    # Group statistics use fp32; the container dtype is restored at the end.
    groups = weight.reshape(out_features, in_features // group_size, group_size).float()

    # For binary signs, the least-squares magnitude is exactly mean(abs(w)).
    # Do not clamp zero groups: scale=bias=0 reconstructs them exactly.
    d = groups.abs().mean(dim=-1)

    # sign -> bit: positive (+d) -> 1, negative (-d) -> 0.
    q = (groups >= 0).to(torch.int64)

    q = q.reshape(out_features, in_features // 32, 32)
    powers = 2 ** torch.arange(32, dtype=torch.int64, device=weight.device)
    packed = (q * powers).sum(dim=-1)  # values in [0, 2^32)
    packed = (packed & 0xFFFFFFFF).to(torch.int32)  # wrap to int32 bit pattern

    # Derive bias from the stored scale so bias == -scale/2 remains exact in
    # the container dtype, matching real Bonsai MLX checkpoints.
    stored_scales = (2.0 * d).to(dtype)
    if not torch.isfinite(stored_scales).all().item():
        raise ValueError("Bonsai scale is not finite after conversion to storage dtype")
    stored_biases = -stored_scales / 2
    return packed, stored_scales, stored_biases


def quantize_1bit_sign(
    weight: torch.Tensor, group_size: int = 128, refine_iters: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward-compatible name for :func:`quantize_1bit_bonsai`."""
    return quantize_1bit_bonsai(
        weight,
        group_size=group_size,
        refine_iters=refine_iters,
    )


class QuantizedLinear1Bit(nn.Module):
    """
    Drop-in replacement for `nn.Linear` storing a packed 1-bit affine weight.

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
            "weight", torch.zeros(out_features, in_features // 32, dtype=torch.int32)
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
        cls, linear: nn.Linear, group_size: int = 128, refine_iters: int = 0
    ) -> QuantizedLinear1Bit:
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
        )
        with torch.no_grad():
            packed, scales, biases = quantize_1bit_affine(
                linear.weight.detach(), group_size, refine_iters=refine_iters
            )
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
        return dequantize_1bit_affine(self.weight, self.scales, self.biases, self.group_size)

    def train(self, mode: bool = True) -> QuantizedLinear1Bit:
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
        # Parent-model load_state_dict() reaches this recursive hook directly.
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
        # FP32 source models are stored in an FP16 quantization container.
        # Match the compute view to the activation dtype for a usable drop-in.
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


def _embedding_output_scale(embedding: nn.Embedding) -> float:
    """Read scaled-embedding metadata without dereferencing a meta tensor."""
    scalar = getattr(embedding, "scalar_embed_scale", None)
    if isinstance(scalar, (int, float)):
        return float(scalar)
    scale = getattr(embedding, "embed_scale", 1.0)
    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1 or scale.device.type == "meta":
            if scale.device.type == "meta":
                raise ValueError(
                    "scaled embedding on meta must expose scalar_embed_scale"
                )
            raise ValueError("embedding embed_scale must be a scalar")
        return float(scale.item())
    if isinstance(scale, (int, float)):
        return float(scale)
    return 1.0


class QuantizedEmbedding1Bit(nn.Module):
    """Drop-in replacement for `nn.Embedding` storing a packed 1-bit affine weight."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        group_size: int = 128,
        output_scale: float = 1.0,
    ) -> None:
        super().__init__()
        _check_dims(embedding_dim, group_size)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.group_size = group_size
        self.scalar_embed_scale = output_scale
        self.register_buffer(
            "embed_scale", torch.tensor(output_scale), persistent=False
        )
        self.register_buffer(
            "weight", torch.zeros(num_embeddings, embedding_dim // 32, dtype=torch.int32)
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
        cls, embedding: nn.Embedding, group_size: int = 128, refine_iters: int = 0
    ) -> QuantizedEmbedding1Bit:
        module = cls(
            embedding.num_embeddings,
            embedding.embedding_dim,
            padding_idx=embedding.padding_idx,
            group_size=group_size,
            output_scale=_embedding_output_scale(embedding),
        )
        with torch.no_grad():
            packed, scales, biases = quantize_1bit_affine(
                embedding.weight.detach(), group_size, refine_iters=refine_iters
            )
            module.weight = packed
            module.scales = scales
            module.biases = biases
            module.embed_scale = module.embed_scale.to(scales.device)
        return module

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Decode only rows used by this batch.  This matters for large-vocabulary
        # models (Gemma 4 has 262144 rows): materializing the full dense table for
        # every short prompt would otherwise dominate both memory and latency.
        if input.numel() == 0:
            return self.scales.new_empty((*input.shape, self.embedding_dim))
        rows, inverse = torch.unique(input, sorted=False, return_inverse=True)
        weight = dequantize_1bit_affine(
            self.weight.index_select(0, rows),
            self.scales.index_select(0, rows),
            self.biases.index_select(0, rows),
            self.group_size,
        )
        # padding_idx only changes gradient accumulation in F.embedding.  The
        # packed table is frozen, so gathering through the remapped indices is
        # value-equivalent without translating padding_idx into `rows` space.
        output = F.embedding(inverse, weight)
        return output * self.embed_scale.to(dtype=output.dtype, device=output.device)

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"padding_idx={self.padding_idx}, group_size={self.group_size}, "
            f"output_scale={self.scalar_embed_scale}"
        )


class QuantizedLinear1BitSign(QuantizedLinear1Bit):
    """
    Sign variant of :class:`QuantizedLinear1Bit`: same packed-code/scale/bias
    container, but the weight is binarized as ``sign(w)*mean(|w|)`` (BitNet /
    Bonsai Q1_0 ``y = bit ? d : -d``), so dequantization yields ``{-d, +d}``.
    Only the factory differs (calls :func:`quantize_1bit_sign`).
    """

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, group_size: int = 128, refine_iters: int = 0
    ) -> QuantizedLinear1BitSign:
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
        )
        with torch.no_grad():
            packed, scales, biases = quantize_1bit_sign(
                linear.weight.detach(), group_size, refine_iters=refine_iters
            )
            module.weight = packed
            module.scales = scales
            module.biases = biases
            if linear.bias is not None:
                module.bias = nn.Parameter(
                    linear.bias.detach().clone().to(scales.dtype), requires_grad=False
                )
        return module


class QuantizedEmbedding1BitSign(QuantizedEmbedding1Bit):
    """Sign variant of :class:`QuantizedEmbedding1Bit` (see :class:`QuantizedLinear1BitSign`)."""

    @classmethod
    def from_embedding(
        cls, embedding: nn.Embedding, group_size: int = 128, refine_iters: int = 0
    ) -> QuantizedEmbedding1BitSign:
        module = cls(
            embedding.num_embeddings,
            embedding.embedding_dim,
            padding_idx=embedding.padding_idx,
            group_size=group_size,
            output_scale=_embedding_output_scale(embedding),
        )
        with torch.no_grad():
            packed, scales, biases = quantize_1bit_sign(
                embedding.weight.detach(), group_size, refine_iters=refine_iters
            )
            module.weight = packed
            module.scales = scales
            module.biases = biases
            module.embed_scale = module.embed_scale.to(scales.device)
        return module


def quantize_model_1bit(
    model: nn.Module,
    group_size: int = 128,
    skip_patterns: tuple[str, ...] | list[str] = (),
    refine_iters: int = 0,
    trim_vocab_to: int | None = None,
    variant: str | None = None,
    *,
    method: str | None = None,
    quantize_tied_weights: bool | None = None,
) -> dict[str, Any]:
    """
    Replace every `nn.Linear`/`nn.Embedding` in `model` (in place) with its
    1-bit quantized counterpart. Returns a JSON-serializable manifest.

    - Modules whose full name contains any of `skip_patterns` are left as-is.
    - A `nn.Linear` whose weight is the same tensor as an already-replaced
      module's weight (e.g. a tied lm_head) follows `quantize_tied_weights`.
    - Layers whose last dim is not divisible by `group_size` are skipped and
      listed under "skipped" with the reason.
    - `refine_iters > 0` enables Lloyd-style scale/bias refinement for the
      affine path. Bonsai/sign validates this argument for API parity but is
      already a closed-form fixed point and therefore does not change.
    - Legacy `variant` selects ``"affine"`` (default, min/max two-level grid)
      or ``"sign"`` (the binary sign recipe). The preferred explicit Bonsai
      entry is `method="bonsai"`, which selects the same public {-d,+d}
      representation through :func:`quantize_1bit_bonsai`.
    - Explicit Bonsai mode quantizes tied output heads as separate triplets by
      default so ordinary Hugging Face Torch forwards remain fully quantized.
      These are independent runtime copies; the MLX exporter deduplicates them
      to match shared-embedding checkpoint storage. Legacy affine/sign calls
      keep the previous skip behavior. Override either policy with
      `quantize_tied_weights`.
    - `trim_vocab_to` drops dead padding rows (e.g. Qwen3's 151936 -> 151669)
      from `nn.Embedding` tables and any module with "lm_head" in its name
      before quantization. A tied lm_head is re-pointed at the trimmed fp
      weight so its output dim stays consistent with the trimmed embedding.
      Trims are recorded under "trimmed" as `{name: [orig_rows, new_rows]}`.
      Pass the FULL tokenizer length (including special tokens, e.g.
      `len(AutoTokenizer)`) — a bare `vocab_size` attribute may exclude
      special tokens and truncate their rows.
    """
    _check_group_size(group_size)
    if refine_iters < 0:
        raise ValueError(f"refine_iters must be >= 0, got {refine_iters}")
    if variant not in (None, "affine", "sign"):
        raise ValueError(f"variant must be 'affine' or 'sign', got {variant!r}")
    if method not in (None, "affine", "bonsai"):
        raise ValueError(f"method must be 'affine' or 'bonsai', got {method!r}")
    if trim_vocab_to is not None and trim_vocab_to < 1:
        raise ValueError(f"trim_vocab_to must be >= 1, got {trim_vocab_to}")

    variant_method = "bonsai" if variant == "sign" else "affine"
    if method is not None and variant is not None and method != variant_method:
        raise ValueError(
            f"variant={variant!r} conflicts with method={method!r}; "
            "use affine/affine or sign/bonsai"
        )
    resolved_method = method if method is not None else variant_method
    resolved_variant = "sign" if resolved_method == "bonsai" else "affine"
    explicit_bonsai = method == "bonsai"
    resolved_quantize_tied = (
        explicit_bonsai if quantize_tied_weights is None else quantize_tied_weights
    )

    quantized: list[str] = []
    skipped: dict[str, str] = {}
    tied: list[str] = []
    quantized_tied: list[str] = []
    tied_to: dict[str, str] = {}
    trimmed: dict[str, list[int]] = {}
    replaced_weights: list[tuple[str, torch.Tensor]] = []

    for name, module in list(model.named_modules()):
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            continue
        matched = next((p for p in skip_patterns if p in name), None)
        if matched is not None:
            skipped[name] = f"matched skip pattern '{matched}'"
            continue
        weight = module.weight
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model

        tied_to_name = next(
            (
                previous_name
                for previous_name, previous_weight in replaced_weights
                if weight is previous_weight
                or weight.data_ptr() == previous_weight.data_ptr()
            ),
            None,
        )
        is_tied = tied_to_name is not None
        if tied_to_name is not None:
            tied_to[name] = tied_to_name
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
                new_head.train(module.training)
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
                    if (
                        module.padding_idx is not None
                        and module.padding_idx >= trim_vocab_to
                    ):
                        raise ValueError(
                            f"padding_idx {module.padding_idx} is outside trimmed "
                            f"vocabulary size {trim_vocab_to} for {name!r}"
                        )
                    source = nn.Embedding(
                        trim_vocab_to,
                        weight.shape[1],
                        padding_idx=module.padding_idx,
                        device=weight.device,
                        dtype=weight.dtype,
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
            if resolved_variant == "sign":
                new_module: nn.Module = QuantizedLinear1BitSign.from_linear(
                    source, group_size, refine_iters=refine_iters
                )
            else:
                new_module = QuantizedLinear1Bit.from_linear(
                    source, group_size, refine_iters=refine_iters
                )
        else:
            if resolved_variant == "sign":
                new_module = QuantizedEmbedding1BitSign.from_embedding(
                    source, group_size, refine_iters=refine_iters
                )
            else:
                new_module = QuantizedEmbedding1Bit.from_embedding(
                    source, group_size, refine_iters=refine_iters
                )
        new_module.train(module.training)
        setattr(parent, child_name, new_module)
        replaced_weights.append((name, weight))
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

    quantization = {
        "mode": "affine",
        "bits": 1,
        "variant": resolved_variant,
        "group_size": group_size,
        "refine_iters": refine_iters,
        "trim_vocab_to": trim_vocab_to,
    }
    if method is not None or quantize_tied_weights is not None:
        quantization["method"] = resolved_method
        quantization["quantize_tied_weights"] = resolved_quantize_tied

    return {
        "quantization": quantization,
        "quantized": quantized,
        "skipped": skipped,
        "tied": tied,
        "quantized_tied": quantized_tied,
        "tied_to": tied_to,
        "trimmed": trimmed,
    }


def _validate_1bit_manifest(
    manifest: Mapping[str, Any],
) -> tuple[int, str, list[str], dict[str, tuple[int, int]]]:
    """Return normalized loader fields after validating the 1-bit manifest."""
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    quantization = manifest.get("quantization")
    if not isinstance(quantization, Mapping):
        raise ValueError("manifest.quantization must be a mapping")
    bits = quantization.get("bits")
    if isinstance(bits, bool) or bits != 1:
        raise ValueError(f"manifest quantization bits must be 1, got {bits!r}")
    if quantization.get("mode") != "affine":
        raise ValueError("manifest quantization mode must be 'affine'")
    group_size = quantization.get("group_size")
    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise ValueError("manifest group_size must be an integer")
    _check_group_size(group_size)

    variant = quantization.get("variant")
    if variant not in ("affine", "sign"):
        raise ValueError(f"manifest variant must be 'affine' or 'sign', got {variant!r}")
    method = quantization.get("method")
    if method is not None and method not in ("affine", "bonsai"):
        raise ValueError(f"manifest method must be 'affine' or 'bonsai', got {method!r}")
    if method == "bonsai" and variant != "sign":
        raise ValueError("manifest method='bonsai' requires variant='sign'")
    if method == "affine" and variant != "affine":
        raise ValueError("manifest method='affine' requires variant='affine'")

    quantized = manifest.get("quantized")
    if not isinstance(quantized, list) or not all(
        isinstance(name, str) and name for name in quantized
    ):
        raise ValueError("manifest.quantized must be a list of non-empty module names")
    if len(quantized) != len(set(quantized)):
        raise ValueError("manifest.quantized contains duplicate module names")

    raw_trimmed = manifest.get("trimmed", {})
    if not isinstance(raw_trimmed, Mapping):
        raise ValueError("manifest.trimmed must be a mapping")
    trimmed: dict[str, tuple[int, int]] = {}
    for name, sizes in raw_trimmed.items():
        if (
            not isinstance(name, str)
            or not isinstance(sizes, list)
            or len(sizes) != 2
            or any(isinstance(size, bool) or not isinstance(size, int) for size in sizes)
            or not 0 < sizes[1] <= sizes[0]
        ):
            raise ValueError(f"invalid trimmed entry for {name!r}: {sizes!r}")
        trimmed[name] = (sizes[0], sizes[1])

    quantized_tied = manifest.get("quantized_tied", [])
    tied_to = manifest.get("tied_to", {})
    if not isinstance(quantized_tied, list) or not all(
        isinstance(name, str) for name in quantized_tied
    ):
        raise ValueError("manifest.quantized_tied must be a list of module names")
    if not isinstance(tied_to, Mapping) or not all(
        isinstance(name, str) and isinstance(target, str)
        for name, target in tied_to.items()
    ):
        raise ValueError("manifest.tied_to must map module names to module names")
    for name in quantized_tied:
        if name not in quantized or name not in tied_to or tied_to[name] not in quantized:
            raise ValueError(
                f"quantized tied module {name!r} must reference another quantized module"
            )
    return group_size, variant, quantized, trimmed


def _replacement_1bit_module(
    module: nn.Module,
    name: str,
    group_size: int,
    variant: str,
    trimmed: Mapping[str, tuple[int, int]],
    *,
    use_meta: bool,
) -> nn.Module:
    """Build an empty quantized container from an FP skeleton module."""
    if isinstance(module, nn.Linear):
        rows, cols = module.out_features, module.in_features
        linear = True
    elif isinstance(module, nn.Embedding):
        rows, cols = module.num_embeddings, module.embedding_dim
        linear = False
    else:
        raise TypeError(
            f"manifest module {name!r} must resolve to nn.Linear or nn.Embedding, "
            f"got {type(module).__name__}"
        )
    if tuple(module.weight.shape) != (rows, cols):
        raise ValueError(
            f"module {name!r} weight shape {tuple(module.weight.shape)} does not "
            f"match its declared shape {(rows, cols)}"
        )
    if name in trimmed:
        old_rows, new_rows = trimmed[name]
        if rows not in (old_rows, new_rows):
            raise ValueError(
                f"module {name!r} has {rows} rows, incompatible with trim "
                f"{old_rows} -> {new_rows}"
            )
        rows = new_rows
    _check_dims(cols, group_size)

    cls: type[nn.Module]
    if linear:
        cls = QuantizedLinear1BitSign if variant == "sign" else QuantizedLinear1Bit
        args = (cols, rows)
        kwargs: dict[str, Any] = {
            "bias": module.bias is not None,
            "group_size": group_size,
        }
    else:
        if module.padding_idx is not None and module.padding_idx >= rows:
            raise ValueError(f"module {name!r} padding_idx is outside the trimmed table")
        cls = QuantizedEmbedding1BitSign if variant == "sign" else QuantizedEmbedding1Bit
        args = (rows, cols)
        kwargs = {
            "padding_idx": module.padding_idx,
            "group_size": group_size,
            "output_scale": _embedding_output_scale(module),
        }

    device = torch.device("meta") if use_meta else module.weight.device
    with torch.device(device):
        replacement = cls(*args, **kwargs)
    replacement.train(module.training)
    return replacement


def rebuild_quantized_model_1bit(
    model: nn.Module,
    manifest: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor] | None = None,
) -> nn.Module:
    """Rebuild packed 1-bit modules in ``model`` and optionally load their state.

    ``model`` is normally an FP skeleton created on the meta device.  No source
    weight is quantized: module names and shapes come from the skeleton while
    the packed triplets come from ``state_dict``.  State loading is deliberately
    ``strict=True, assign=True`` so BF16 tensors are assigned without a cast.
    """
    group_size, variant, quantized, trimmed = _validate_1bit_manifest(manifest)
    plans: list[tuple[str, nn.Module]] = []
    for name in quantized:
        try:
            module = model.get_submodule(name)
        except AttributeError as error:
            raise ValueError(f"manifest module {name!r} does not exist in model") from error
        plans.append(
            (
                name,
                _replacement_1bit_module(
                    module,
                    name,
                    group_size,
                    variant,
                    trimmed,
                    use_meta=state_dict is not None,
                ),
            )
        )

    if state_dict is not None:
        expected = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
        for name, replacement in plans:
            prefix = f"{name}."
            expected = {key: shape for key, shape in expected.items() if not key.startswith(prefix)}
            expected.update(
                {f"{prefix}{key}": tuple(tensor.shape) for key, tensor in replacement.state_dict().items()}
            )
        actual_keys = set(state_dict)
        missing = sorted(set(expected) - actual_keys)
        unexpected = sorted(actual_keys - set(expected))
        if missing or unexpected:
            raise ValueError(
                f"checkpoint state keys do not match rebuilt model; missing={missing[:5]}, "
                f"unexpected={unexpected[:5]}"
            )
        for key, expected_shape in expected.items():
            tensor = state_dict[key]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"checkpoint entry {key!r} is not a tensor")
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"checkpoint tensor {key!r} has shape {tuple(tensor.shape)}, "
                    f"expected {expected_shape}"
                )
        for name in quantized:
            weight = state_dict[f"{name}.weight"]
            scales = state_dict[f"{name}.scales"]
            biases = state_dict[f"{name}.biases"]
            if weight.dtype != torch.int32:
                raise ValueError(f"checkpoint tensor {name}.weight must have dtype torch.int32")
            if scales.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError(f"checkpoint tensor {name}.scales must be FP16 or BF16")
            if biases.dtype != scales.dtype:
                raise ValueError(f"checkpoint tensors {name}.scales and biases must share dtype")

    for name, replacement in plans:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, replacement)

    trim_vocab_to = manifest["quantization"].get("trim_vocab_to")
    if (
        isinstance(trim_vocab_to, int)
        and getattr(model, "config", None) is not None
        and getattr(model.config, "vocab_size", 0) > trim_vocab_to
    ):
        model.config.vocab_size = trim_vocab_to

    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True, assign=True)
        # Non-persistent metadata is intentionally absent from safetensors and
        # therefore remains meta after an assign-load. Recreate the scaled
        # embedding scalar beside its newly assigned packed buffers.
        for name in quantized:
            module = model.get_submodule(name)
            if isinstance(module, QuantizedEmbedding1Bit):
                module.embed_scale = torch.tensor(
                    module.scalar_embed_scale,
                    dtype=torch.float32,
                    device=module.scales.device,
                )
        first_tensor = next(iter(state_dict.values()), None)
        if isinstance(first_tensor, torch.Tensor) and first_tensor.device.type != "meta":
            _materialize_gemma4_rotary_buffers(model, first_tensor.device)
        remaining_meta = [
            name for name, buffer in model.named_buffers() if buffer.device.type == "meta"
        ]
        if remaining_meta:
            raise RuntimeError(
                "checkpoint load left non-persistent buffers on meta: "
                f"{remaining_meta[:5]}"
            )
    return model


def _transformers_meta_model(checkpoint_dir: Path) -> nn.Module:
    """Construct the architecture named by config.json without importing HF eagerly."""
    try:
        import transformers
    except ImportError as error:  # pragma: no cover - required project dependency
        raise ImportError("transformers is required to construct a model from config.json") from error

    config = transformers.AutoConfig.from_pretrained(checkpoint_dir, local_files_only=True)
    architectures = getattr(config, "architectures", None)
    if not architectures or not isinstance(architectures[0], str):
        raise ValueError("config.json must declare an architecture for automatic loading")
    model_class = getattr(transformers, architectures[0], None)
    if not isinstance(model_class, type) or not issubclass(model_class, nn.Module):
        raise ValueError(f"unsupported Transformers architecture {architectures[0]!r}")
    with torch.device("meta"):
        return model_class(config)


def _materialize_gemma4_rotary_buffers(model: nn.Module, device: torch.device) -> None:
    """Recompute Gemma 4's non-persistent RoPE buffers after meta construction."""
    candidates = [
        module
        for module in model.modules()
        if type(module).__name__ == "Gemma4UnifiedTextRotaryEmbedding"
    ]
    if not candidates:
        return
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedTextRotaryEmbedding,
    )

    for module in candidates:
        if not isinstance(module, Gemma4UnifiedTextRotaryEmbedding):
            continue
        for layer_type in module.layer_types:
            rope_params = module.config.rope_parameters[layer_type]
            if rope_params is None:
                continue
            kwargs: dict[str, Any] = {"device": device, "layer_type": layer_type}
            if layer_type == "full_attention" and module.rope_type[layer_type] == "proportional":
                kwargs["head_dim_key"] = "global_head_dim"
            inv_freq, attention_scaling = module.rope_init_fns[layer_type](
                module.config, **kwargs
            )
            setattr(module, f"{layer_type}_inv_freq", inv_freq)
            setattr(module, f"{layer_type}_original_inv_freq", inv_freq.clone())
            setattr(module, f"{layer_type}_attention_scaling", attention_scaling)


def load_quantized_checkpoint_1bit(
    checkpoint_or_state: str | Path | Mapping[str, torch.Tensor],
    *,
    model: nn.Module | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Load a Torch 1-bit checkpoint from a directory or an in-memory state.

    Directory mode reads ``config.json``, ``quantization.json`` and
    ``model.safetensors``; pass ``model`` to bypass automatic Transformers
    construction.  Mapping mode requires both an existing ``model`` and its
    ``manifest``.
    """
    created_model = model is None
    if isinstance(checkpoint_or_state, (str, Path)):
        checkpoint_dir = Path(checkpoint_or_state)
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint_dir}")
        manifest_path = checkpoint_dir / "quantization.json"
        weights_path = checkpoint_dir / "model.safetensors"
        if not manifest_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                "checkpoint directory must contain quantization.json and model.safetensors"
            )
        loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest is not None and dict(manifest) != loaded_manifest:
            raise ValueError("provided manifest does not match quantization.json")
        manifest = loaded_manifest
        state_dict = safetensors_load_file(str(weights_path), device="cpu")
        if model is None:
            if not (checkpoint_dir / "config.json").is_file():
                raise FileNotFoundError("automatic loading requires config.json")
            model = _transformers_meta_model(checkpoint_dir)
    else:
        state_dict = checkpoint_or_state
        if model is None or manifest is None:
            raise ValueError("in-memory state loading requires both model and manifest")

    assert model is not None and manifest is not None
    rebuild_quantized_model_1bit(model, manifest, state_dict)
    if created_model:
        model.eval()
    return model


def save_quantized_checkpoint(
    model: nn.Module, manifest: dict[str, Any], dir_path: str | Path
) -> None:
    """Save `model.state_dict()` as `model.safetensors` plus `quantization.json`."""
    save_dir = Path(dir_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    safetensors_save_file(state_dict, str(save_dir / "model.safetensors"))
    (save_dir / "quantization.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
