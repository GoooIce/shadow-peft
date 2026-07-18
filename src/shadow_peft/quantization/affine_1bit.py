from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file as safetensors_save_file

_EPS = 1e-8


def _check_dims(in_features: int, group_size: int) -> None:
    if group_size < 32 or group_size % 32 != 0:
        raise ValueError(f"group_size must be a positive multiple of 32, got {group_size}")
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
    """
    out_features, num_words = packed.shape
    in_features = num_words * 32
    shifts = torch.arange(32, dtype=torch.int32, device=packed.device)
    # Arithmetic right shift sign-extends, but & 1 recovers the original bit.
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    q = bits.reshape(out_features, in_features // group_size, group_size).to(scales.dtype)
    w = scales.unsqueeze(-1) * q + biases.unsqueeze(-1)
    return w.reshape(out_features, in_features)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.dequantize_weight()
        else:
            if self._cached_weight is None:
                self._cached_weight = self.dequantize_weight()
            weight = self._cached_weight
        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, group_size={self.group_size}"
        )


class QuantizedEmbedding1Bit(nn.Module):
    """Drop-in replacement for `nn.Embedding` storing a packed 1-bit affine weight."""

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
        )
        with torch.no_grad():
            packed, scales, biases = quantize_1bit_affine(
                embedding.weight.detach(), group_size, refine_iters=refine_iters
            )
            module.weight = packed
            module.scales = scales
            module.biases = biases
        return module

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Reference implementation: dequantize the whole table, then gather.
        # A gather-then-dequantize kernel would avoid materializing the table.
        weight = dequantize_1bit_affine(self.weight, self.scales, self.biases, self.group_size)
        return F.embedding(input, weight, padding_idx=self.padding_idx)

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"padding_idx={self.padding_idx}, group_size={self.group_size}"
        )


def quantize_model_1bit(
    model: nn.Module,
    group_size: int = 128,
    skip_patterns: tuple[str, ...] | list[str] = (),
    refine_iters: int = 0,
    trim_vocab_to: int | None = None,
) -> dict[str, Any]:
    """
    Replace every `nn.Linear`/`nn.Embedding` in `model` (in place) with its
    1-bit quantized counterpart. Returns a JSON-serializable manifest.

    - Modules whose full name contains any of `skip_patterns` are left as-is.
    - A `nn.Linear` whose weight is the same tensor as an already-replaced
      module's weight (e.g. a tied lm_head) is skipped and listed under "tied".
    - Layers whose last dim is not divisible by `group_size` are skipped and
      listed under "skipped" with the reason.
    - `refine_iters > 0` enables Lloyd-style scale/bias refinement
      (see `quantize_1bit_affine`).
    - `trim_vocab_to` drops dead padding rows (e.g. Qwen3's 151936 -> 151669)
      from `nn.Embedding` tables and any module with "lm_head" in its name
      before quantization. A tied lm_head is re-pointed at the trimmed fp
      weight so its output dim stays consistent with the trimmed embedding.
      Trims are recorded under "trimmed" as `{name: [orig_rows, new_rows]}`.
      Pass the FULL tokenizer length (including special tokens, e.g.
      `len(AutoTokenizer)`) — a bare `vocab_size` attribute may exclude
      special tokens and truncate their rows.
    """
    if trim_vocab_to is not None and trim_vocab_to < 1:
        raise ValueError(f"trim_vocab_to must be >= 1, got {trim_vocab_to}")

    quantized: list[str] = []
    skipped: dict[str, str] = {}
    tied: list[str] = []
    trimmed: dict[str, list[int]] = {}
    replaced_weights: list[torch.Tensor] = []

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

        if any(weight is w or weight.data_ptr() == w.data_ptr() for w in replaced_weights):
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
            new_module: nn.Module = QuantizedLinear1Bit.from_linear(
                source, group_size, refine_iters=refine_iters
            )
        else:
            new_module = QuantizedEmbedding1Bit.from_embedding(
                source, group_size, refine_iters=refine_iters
            )
        setattr(parent, child_name, new_module)
        replaced_weights.append(weight)
        quantized.append(name)

    return {
        "quantization": {
            "mode": "affine",
            "bits": 1,
            "group_size": group_size,
            "refine_iters": refine_iters,
            "trim_vocab_to": trim_vocab_to,
        },
        "quantized": quantized,
        "skipped": skipped,
        "tied": tied,
        "trimmed": trimmed,
    }


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
