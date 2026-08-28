"""Cached single-forward block drafter (spec exp2 v2-block: DSpark vs DFlash2).

A pretrained causal-LM trunk (LFM2.5-350M) plus k low-rank position heads
sharing the trunk's tied lm_head. The trunk keeps its own DynamicCache and is
driven incrementally, so a round costs a *constant* number of small forwards
instead of the v1 chain's k sequential ones:

- DSpark structure (``r=0``, optional confidence head): the block comes from
  the heads applied to ``h_last`` — the final hidden state left over from the
  previous commit forward — so drafting itself needs **zero** extra trunk
  forwards. The confidence head predicts per-position draft==teacher argmax
  agreement (thresholded dynamic-k is an eval-side ablation).
- DFlash2 structure (``r>=1``): ``r`` causal refinement rounds; each embeds
  the current block estimate as input (trunk's own next-token head re-scores,
  head-0 autoregressive convention from exp5) and rolls the trunk cache back
  afterwards with the O(1) crop snapshot.

Detached by design: no target-feature injection (exp5 verdict). The trainer
fine-tunes the trunk with the v1 causal recipe (CE + KL, all positions) —
which is also exactly what refinement consumes at inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from shadow_lab.specdec import _cached_kwargs, _restore, _snapshot


class BlockHeads(nn.Module):
    """k low-rank position transforms (+ optional per-position confidence)."""

    def __init__(self, hidden: int, k: int, with_confidence: bool = False):
        super().__init__()
        self.k = k
        self.tf = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(k)])
        self.conf = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(k)]) \
            if with_confidence else None

    def forward(self, h_last: torch.Tensor, lm_head: nn.Module, k_use: int | None = None):
        """Block logits [b, k_use, vocab] from the last context position [b, 1, hidden]."""
        n = k_use if k_use is not None else self.k
        h = h_last[:, -1]  # [b, hidden] — drop the length-1 seq dim
        return torch.stack([lm_head(self.tf[j](h)) for j in range(n)], dim=1)

    def confidence(self, h_last: torch.Tensor, k_use: int | None = None) -> torch.Tensor:
        """Per-position agreement logit [b, k_use] (structure=dspark only)."""
        if self.conf is None:
            raise RuntimeError("confidence heads not configured")
        n = k_use if k_use is not None else self.k
        h = h_last[:, -1]
        return torch.cat([self.conf[j](self.tf[j](h)) for j in range(n)], dim=1)


class CachedBlockDrafter(nn.Module):
    """Pretrained trunk + BlockHeads with an incremental cache and O(1) rollback.

    Engine protocol (consumed by ``specdec.block_speculative_generate``):
    ``prefill`` / ``draft(k, r)`` / ``commit(tokens)`` / ``snapshot`` / ``restore``.
    """

    def __init__(self, lm, k: int = 8, with_confidence: bool = False):
        super().__init__()
        self.lm = lm
        self.k = k
        hidden = lm.config.hidden_size
        # heads adopt the trunk's dtype (bf16 trunk + fp32 heads would crash)
        self.heads = BlockHeads(hidden, k, with_confidence).to(next(lm.parameters()).dtype)
        self._cache = None
        self._h_last: torch.Tensor | None = None

    # ---- engine protocol ----

    def prefill(self, input_ids: torch.Tensor) -> None:
        mask = torch.ones(1, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        out = self.lm(input_ids=input_ids, use_cache=True, attention_mask=mask,
                      output_hidden_states=True)
        self._cache = out.past_key_values
        self._h_last = out.hidden_states[-1][:, -1:]

    def draft(self, k_use: int | None = None, r: int = 0,
              conf_threshold: float | None = None) -> torch.Tensor:
        """Draft a block [1, n_eff]; refinement rounds never disturb the cache.

        ``conf_threshold`` (dspark structure): truncate the block at the first
        position whose predicted agreement probability falls below the
        threshold — dynamic draft length.
        """
        n = k_use if k_use is not None else self.k
        lm_head = self.lm.get_output_embeddings()
        block = self.heads(self._h_last, lm_head, n).argmax(dim=-1)  # [1, n]
        if conf_threshold is not None and self.heads.conf is not None:
            conf = torch.sigmoid(self.heads.confidence(self._h_last, n))[0]  # [n]
            below = (conf < conf_threshold).nonzero()
            n_eff = int(below[0]) if len(below) else n
            block = block[:, :max(n_eff, 1)]
        for _ in range(r):
            snap = self.snapshot()
            # embed the current estimates; next-token head at position
            # n_ctx-1+j predicts position n_ctx+j (never its own position).
            ids = block[:, :-1]
            out = self.lm(input_ids=ids, past_key_values=self._cache,
                          output_hidden_states=True,
                          **_cached_kwargs(self._cache, ids.shape[1], ids.device))
            h_new = out.hidden_states[-1]  # positions n_ctx .. n_ctx+n-2
            h_all = torch.cat([self._h_last, h_new], dim=1)  # n_ctx-1 .. n_ctx+n-2
            block = lm_head(h_all).argmax(dim=-1)  # [1, n]
            self.restore(snap)
        return block

    def commit(self, tokens: torch.Tensor) -> torch.Tensor:
        """Feed committed tokens through the trunk; returns the new h_last [1,1,hidden]."""
        out = self.lm(input_ids=tokens[None], past_key_values=self._cache,
                      output_hidden_states=True,
                      **_cached_kwargs(self._cache, int(tokens.numel()), tokens.device))
        self._cache = out.past_key_values
        self._h_last = out.hidden_states[-1][:, -1:]
        return self._h_last

    def snapshot(self):
        """O(1): crop snapshot of the trunk cache + the current h_last reference."""
        return (_snapshot(self._cache) if self._cache is not None else None, self._h_last)

    def restore(self, snap) -> None:
        cache_snap, h_last = snap
        if self._cache is not None:
            _restore(self._cache, cache_snap)
        self._h_last = h_last
