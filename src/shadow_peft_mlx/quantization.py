"""Affine quantization entry points (1-bit / ternary 2-bit) for the MLX backend.

Layout produced here is the standard MLX quantized format: packed ``uint32``
weights (LSB-first — 32 values per word for 1-bit, 16 for 2-bit), one
``scale``/``bias`` pair per ``group_size`` block along the last dim,
``w_hat = scale * q + bias``.

``bits=1`` support is not yet in a PyPI release of MLX; it lives in
ml-explore/mlx PR #3161 ("Add 1-bit affine quantization support (Metal)").
Build MLX from that PR (or any branch containing it) before calling
:func:`quantize_model_1bit`::

    git clone https://github.com/ml-explore/mlx.git
    cd mlx && git fetch origin refs/pull/3161/head && git checkout FETCH_HEAD
    pip install -e .
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

_1BIT_PR_URL = "https://github.com/ml-explore/mlx/pull/3161"


def ensure_1bit_support() -> None:
    """Raise a RuntimeError unless the installed MLX supports ``bits=1``."""
    probe = mx.zeros((1, 128), dtype=mx.float16)
    try:
        mx.quantize(probe, group_size=128, bits=1)
    except ValueError as exc:
        raise RuntimeError(
            "The installed MLX does not support bits=1 affine quantization. "
            f"Build MLX from {_1BIT_PR_URL} (or any branch containing it) and "
            "re-install it into this environment."
        ) from exc


def pack_1bit_refined(
    weight: mx.array,
    group_size: int = 128,
    iters: int = 10,
    eps: float = 1e-8,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Lloyd-style optimized 1-bit affine packing, MLX-format compatible.

    Starts from the min/max assignment (same as ``mx.quantize(bits=1)``) and
    alternates, per group of ``group_size`` weights along the last dim:

      1. least-squares fit of the two levels: ``bias = mean(w[q==0])``,
         ``scale = mean(w[q==1]) - bias``;
      2. reassignment ``q_i = 1`` iff ``w_i > bias + scale / 2``.

    Group MSE decreases monotonically. Outliers are clipped onto the nearer
    level instead of stretching the grid — the main defect of min/max packing.

    Returns ``(packed uint32 [..., K/32], scales fp16, biases fp16)``, exactly
    the layout consumed by ``mx.dequantize`` / ``mx.quantized_matmul``.
    """
    if iters < 0:
        raise ValueError(f"iters must be >= 0, got {iters}")
    if weight.ndim != 2 or weight.shape[-1] % group_size != 0:
        raise ValueError(
            f"weight must be [out, K] with K % {group_size} == 0, "
            f"got {tuple(weight.shape)}"
        )

    w = weight.astype(mx.float32)
    out_features, k = w.shape
    n_words = k // 32
    wg = w.reshape(out_features, k // group_size, group_size)

    # Min/max init, mirroring mx.quantize(bits=1).
    lo = wg.min(axis=-1)
    hi = wg.max(axis=-1)
    scale = mx.maximum(hi - lo, eps)
    bias = lo
    bits = wg > (bias[..., None] + 0.5 * scale[..., None])

    for _ in range(iters):
        qf = bits.astype(mx.float32)
        n1 = qf.sum(axis=-1)
        n0 = group_size - n1
        sum1 = (wg * qf).sum(axis=-1)
        sum0 = wg.sum(axis=-1) - sum1
        mean1 = sum1 / mx.maximum(n1, 1)
        mean0 = sum0 / mx.maximum(n0, 1)
        new_scale = mean1 - mean0
        valid = (n1 > 0) & (n0 > 0) & (new_scale > eps)
        scale = mx.where(valid, new_scale, scale)
        bias = mx.where(valid, mean0, bias)
        bits = wg > (bias[..., None] + 0.5 * scale[..., None])

    # Pack 32 bits per uint32 word, LSB-first.
    words = bits.reshape(out_features, n_words, 32).astype(mx.uint32)
    packed = (words * (2 ** mx.arange(32, dtype=mx.uint32))).sum(axis=-1)

    return (
        packed,
        scale.reshape(out_features, k // group_size).astype(mx.float16),
        bias.reshape(out_features, k // group_size).astype(mx.float16),
    )


def quantize_model_1bit(
    model: nn.Module,
    group_size: int = 128,
    skip_patterns: tuple[str, ...] = (),
    refine_iters: int = 0,
    trim_vocab_to: int | None = None,
) -> dict:
    """
    Quantize every eligible ``nn.Linear`` / ``nn.Embedding`` of ``model`` to
    1-bit affine, in place.

    Args:
        model: The model to quantize.
        group_size: Weights per affine group along the last dim. Layers whose
            last weight dim is not divisible by it are left untouched.
        skip_patterns: Fully-qualified module names containing any of these
            substrings are left untouched.
        refine_iters: If > 0, repack every quantized weight with
            :func:`pack_1bit_refined` (Lloyd-style optimized scale/bias/bits),
            recovering accuracy at zero inference cost.
        trim_vocab_to: Drop dead padding rows (e.g. Qwen3's 151936 -> 151669)
            from ``nn.Embedding`` tables and any module with "lm_head" in its
            path before quantization. With tied lm_head (mlx-lm reads logits
            from the embedding table) trimming the embedding trims the head
            automatically. Trims are recorded under "trimmed". Pass the FULL
            tokenizer length (including special tokens, e.g.
            ``len(AutoTokenizer)``) — a bare ``vocab_size`` attribute may
            exclude special tokens and truncate their rows.

    Returns:
        A manifest describing what was quantized::

            {
              "quantization": {"mode": "affine", "bits": 1, "group_size": ...,
                               "refine_iters": ..., "trim_vocab_to": ...},
              "quantized": [module paths ...],
              "skipped": {module path: reason},
              "tied": [module paths sharing an already-quantized weight],
              "trimmed": {module path: [orig_rows, new_rows]},
            }
    """
    ensure_1bit_support()
    return _quantize_model_affine(
        model,
        bits=1,
        group_size=group_size,
        skip_patterns=skip_patterns,
        refine_iters=refine_iters,
        trim_vocab_to=trim_vocab_to,
        thr_ratio=None,
        repack=pack_1bit_refined,
    )


def pack_2bit_ternary(
    weight: mx.array,
    group_size: int = 128,
    iters: int = 0,
    thr_ratio: float = 0.5,
    eps: float = 1e-8,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Ternary 2-bit affine packing ({-1, 0, +1}), MLX-format compatible.

    Mirrors the torch ``quantize_2bit_ternary``: per group of ``group_size``
    weights along the last dim, ``s_g = mean(|w|)``, then
    ``t_i = 0`` if ``|w_i| < thr_ratio * s_g`` else ``sign(w_i)``; stored as
    ``scale = s_g``, ``bias = -s_g`` so ``w_hat = s_g * t``. ``iters > 0``
    refits ``s_g`` by Lloyd-style alternation.

    Returns ``(packed uint32 [..., K/16], scales fp16, biases fp16 = -scales)``,
    exactly the layout consumed by ``mx.dequantize`` / ``mx.quantized_matmul``.
    """
    if iters < 0:
        raise ValueError(f"iters must be >= 0, got {iters}")
    if not 0.0 < thr_ratio <= 1.0:
        raise ValueError(f"thr_ratio must be in (0, 1], got {thr_ratio}")
    if weight.ndim != 2 or weight.shape[-1] % group_size != 0:
        raise ValueError(
            f"weight must be [out, K] with K % {group_size} == 0, "
            f"got {tuple(weight.shape)}"
        )

    w = weight.astype(mx.float32)
    out_features, k = w.shape
    n_words = k // 16
    wg = w.reshape(out_features, k // group_size, group_size)

    # absmean init (BitNet b1.58 style).
    scales = mx.maximum(wg.abs().mean(axis=-1), eps)

    def ternary_assign(s):
        thr = thr_ratio * s[..., None]
        t = mx.zeros_like(wg)
        t = mx.where(wg > thr, 1.0, t)
        t = mx.where(wg < -thr, -1.0, t)
        return t

    ternary = ternary_assign(scales)
    for _ in range(iters):
        denom = mx.maximum((ternary * ternary).sum(axis=-1), 1.0)
        scales = (wg * ternary).sum(axis=-1) / denom
        scales = mx.maximum(mx.abs(scales), eps)
        ternary = ternary_assign(scales)

    # Map ternary {-1,0,+1} -> code {0,1,2}; code 3 is never produced.
    codes = (ternary + 1.0).astype(mx.uint32)

    # Pack 16 2-bit codes per uint32 word, LSB-first.
    codes = codes.reshape(out_features, n_words, 16)
    packed = (codes * (4 ** mx.arange(16, dtype=mx.uint32))).sum(axis=-1)

    scales16 = scales.astype(mx.float16)
    return packed, scales16, (-scales.astype(mx.float32)).astype(mx.float16)


def quantize_model_2bit(
    model: nn.Module,
    group_size: int = 128,
    skip_patterns: tuple[str, ...] = (),
    refine_iters: int = 0,
    trim_vocab_to: int | None = None,
    thr_ratio: float = 0.5,
) -> dict:
    """
    Quantize every eligible ``nn.Linear`` / ``nn.Embedding`` of ``model`` to
    ternary 2-bit affine ({-1, 0, +1}), in place — the MLX twin of the torch
    ``quantize_model_2bit``.

    ``bits=2`` is natively supported by MLX, so no special build is required.
    Args mirror :func:`quantize_model_1bit`; additionally ``thr_ratio`` is the
    ternary threshold (``|w| < thr_ratio * s_g -> 0``), default 0.5, matching
    the torch scheme table default.
    """
    return _quantize_model_affine(
        model,
        bits=2,
        group_size=group_size,
        skip_patterns=skip_patterns,
        refine_iters=refine_iters,
        trim_vocab_to=trim_vocab_to,
        thr_ratio=thr_ratio,
        repack=pack_2bit_ternary,
    )


def _quantize_model_affine(
    model: nn.Module,
    *,
    bits: int,
    group_size: int,
    skip_patterns: tuple[str, ...],
    refine_iters: int,
    trim_vocab_to: int | None,
    thr_ratio: float | None,
    repack,
) -> dict:
    """Shared walker for the affine quantization entry points (1-bit / 2-bit)."""
    if trim_vocab_to is not None and trim_vocab_to < 1:
        raise ValueError(f"trim_vocab_to must be >= 1, got {trim_vocab_to}")

    manifest: dict = {
        "quantization": {
            "mode": "affine",
            "bits": bits,
            "group_size": group_size,
            "refine_iters": refine_iters,
            "trim_vocab_to": trim_vocab_to,
        },
        "quantized": [],
        "skipped": {},
        "tied": [],
        "trimmed": {},
    }
    if thr_ratio is not None:
        manifest["quantization"]["ternary"] = True
        manifest["quantization"]["thr_ratio"] = thr_ratio

    seen_weight_ids: set[int] = set()
    originals: dict[str, mx.array] = {}

    def predicate(path: str, module: nn.Module) -> bool:
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return False
        if any(pattern in path for pattern in skip_patterns):
            manifest["skipped"][path] = "skip_patterns"
            return False
        if id(module.weight) in seen_weight_ids:
            # e.g. an lm_head sharing its weight with the token embedding.
            manifest["tied"].append(path)
            return False
        if module.weight.shape[-1] % group_size != 0:
            manifest["skipped"][path] = (
                f"last dim {module.weight.shape[-1]} not divisible by {group_size}"
            )
            return False
        seen_weight_ids.add(id(module.weight))
        if (
            trim_vocab_to is not None
            and module.weight.shape[0] > trim_vocab_to
            and (isinstance(module, nn.Embedding) or "lm_head" in path)
        ):
            manifest["trimmed"][path] = [module.weight.shape[0], trim_vocab_to]
            module.weight = module.weight[:trim_vocab_to]
            seen_weight_ids.add(id(module.weight))
        manifest["quantized"].append(path)
        originals[path] = module.weight
        return True

    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=predicate)

    if manifest["trimmed"]:
        # Keep ModelArgs consistent with the trimmed vocab.
        args = getattr(model, "args", None)
        if args is not None and getattr(args, "vocab_size", 0) > trim_vocab_to:
            args.vocab_size = trim_vocab_to

    if refine_iters > 0:
        for path, module in model.named_modules():
            if path not in originals:
                continue
            if thr_ratio is None:
                packed, scales, biases = repack(
                    originals[path], group_size=group_size, iters=refine_iters
                )
            else:
                packed, scales, biases = repack(
                    originals[path],
                    group_size=group_size,
                    iters=refine_iters,
                    thr_ratio=thr_ratio,
                )
            module.weight, module.scales, module.biases = packed, scales, biases
        mx.eval(model.parameters())
        originals.clear()

    return manifest
