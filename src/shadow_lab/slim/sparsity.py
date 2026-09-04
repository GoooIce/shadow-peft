"""Sparse-attention patching (AngelSlim ``Sparsity``).

AngelSlim's sparsity is a convert-time attention patch (idempotent, no
calibration): ``run()`` is a no-op, ``convert()`` applies the algorithm's
patch, ``save()`` re-emits HF weights plus the pattern JSON. Pattern-based
algorithms need a pre-computed pattern file (``attn_kwargs={"pattern_path":
...}``); keep this in mind when picking an algorithm for a family the
upstream pattern tooling has not seen.
"""

from __future__ import annotations

from . import engine as _engine

SPARSE_ALGORITHMS = (
    "stem",
    "minference",
    "a_shape",
    "tri_shape",
    "xattention",
    "flexprefill",
    "flashprefill",
)


def apply_sparse(
    base,
    tokenizer,
    algorithm: str,
    *,
    save_dir: str | None = None,
    model_name: str | None = None,
    attn_kwargs: dict | None = None,
    allow_pseudo_sparse: bool = False,
) -> dict:
    """Patch ``base`` in place with a sparse-attention algorithm; returns the manifest.

    Unlike the quant tiers this is an inference-time transform — patch,
    measure, deploy. No shadow interaction.
    """
    if algorithm not in SPARSE_ALGORITHMS:
        raise ValueError(
            f"unsupported sparse algorithm {algorithm!r} (expected one of {SPARSE_ALGORITHMS})"
        )
    from angelslim.utils.config_parser import CompressionConfig, SparsityConfig

    compress_config = CompressionConfig(
        name="Sparsity",
        sparsity=SparsityConfig(
            name=algorithm,
            attn_kwargs=attn_kwargs or {},
            allow_pseudo_sparse=allow_pseudo_sparse,
        ),
    )

    slim_engine = _engine.build_engine(base, tokenizer, model_name=model_name)
    slim_engine.prepare_compressor(
        "Sparsity",
        global_config=_engine.build_global_config(base, None, save_path=save_dir or "./output"),
        compress_config=compress_config,
    )
    slim_engine.run()  # no-op for sparsity (kept for the canonical order)
    slim_engine.convert()
    sequence = ["convert"]
    if save_dir:
        slim_engine.save(save_dir)
        sequence.append("save")

    return {
        "backend": "angelslim",
        "tier": f"sparse:{algorithm}",
        "format": "sparse_patch",
        "sequence": sequence,
        "model_name": model_name or _engine.resolve_family(base),
    }
