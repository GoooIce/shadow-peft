"""DFlash2-structured shadow drafter (spec exp5): KV injection + block drafting.

Two defining DFlash2 traits, adapted (freedoms recorded in the spec):

- **KV injection (every layer)**: drafter layer i receives target hidden
  states from source layer s(i) (evenly spaced coverage of the target depth)
  additively in its K and V projections. Context-position features come free
  from the verify forward's ``output_hidden_states``; drafted block positions
  are self-conditioned (zero injection — block-diffusion convention).
- **Block drafting**: k parallel heads predict the whole block in one pass,
  then ``r`` rounds of causal-masked refinement re-embed the current block
  estimate. Position heads are low-rank transforms sharing the tied lm_head
  (a full vocab-sized head per position would cost k x 33M params).

v1 simplicity: the drafter re-runs the full (context + block) sequence every
refinement round — a 4-layer hidden-448 model at chat lengths is cheap and
this removes the entire drafter-cache correctness surface. The wall-clock
win over sequential drafting (k python-loop steps -> r+1 batched passes) is
preserved. Training is plain causal (CE + KL); refinement is inference-time
only (zero-shot — the r ablation shows whether it earns its keep).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class InjectedLayer(nn.Module):
    """Pre-norm transformer layer with additive target-feature KV injection."""

    def __init__(self, hidden: int, n_heads: int, n_kv_heads: int, target_hidden: int,
                 intermediate: int):
        super().__init__()
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = hidden // n_heads
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, n_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(hidden, n_kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)
        self.k_inj = nn.Linear(target_hidden, n_kv_heads * self.head_dim, bias=False)
        self.v_inj = nn.Linear(target_hidden, n_kv_heads * self.head_dim, bias=False)
        self.attn_norm = RMSNorm(hidden)
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)
        self.mlp_norm = RMSNorm(hidden)

    def forward(self, x, tgt):
        h = self.attn_norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        if tgt is not None:
            k = k + self.k_inj(tgt)
            v = v + self.v_inj(tgt)
        b, s, _ = q.shape
        q = q.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        rep = self.n_heads // self.n_kv_heads
        if rep > 1:  # GQA emulation (portable across torch versions)
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(b, s, -1))
        m = self.mlp_norm(x)
        x = x + self.down(F.silu(self.gate(m)) * self.up(m))
        return x


class KVInjectedDrafter(nn.Module):
    """Causal drafter with per-layer KV injection and parallel block heads."""

    def __init__(self, *, vocab: int, hidden: int = 448, n_layers: int = 4,
                 n_heads: int = 8, n_kv_heads: int = 4, target_hidden: int = 2048,
                 target_n_layers: int = 16, k: int = 4, intermediate: int | None = None):
        super().__init__()
        self.k = k
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([
            InjectedLayer(hidden, n_heads, n_kv_heads, target_hidden,
                          intermediate or hidden * 3)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.emb.weight  # tied
        # Low-rank per-position transforms -> shared tied head
        self.block_tf = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(k)])
        # Drafter layer i <- target source layer s(i), evenly spaced (1-based,
        # hidden_states tuple has L+1 entries with embedding at 0).
        self.sources = [(i + 1) * target_n_layers // (n_layers + 1) for i in range(n_layers)]
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)

    def inject_features(self, target_hiddens: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        """Per-drafter-layer target features [b, s, target_hidden] (free captures)."""
        return [target_hiddens[s] for s in self.sources]

    def _trunk(self, input_ids: torch.Tensor, feats: list[torch.Tensor] | None,
               n_ctx: int) -> torch.Tensor:
        """Causal forward; feats cover the first n_ctx positions (zeroed after)."""
        x = self.emb(input_ids)
        b, s, hd = x.shape
        for i, layer in enumerate(self.layers):
            if feats is not None and n_ctx < s:
                zeros = torch.zeros(b, s - n_ctx, feats[i].shape[-1], device=x.device,
                                    dtype=feats[i].dtype)
                tgt = torch.cat([feats[i], zeros], dim=1)
            else:
                tgt = feats[i] if feats is not None else None
            x = layer(x, tgt)
        return self.final_norm(x)

    def forward_causal(self, input_ids: torch.Tensor, feats: list[torch.Tensor]) -> torch.Tensor:
        """Training/inference causal logits [b, s, vocab] (full-position injection)."""
        return self.lm_head(self._trunk(input_ids, feats, n_ctx=input_ids.shape[1]))

    def forward_block(self, input_ids: torch.Tensor, feats: list[torch.Tensor]) -> torch.Tensor:
        """Parallel block logits [b, k, vocab] from the last context position."""
        h_last = self._trunk(input_ids, feats, n_ctx=input_ids.shape[1])[:, -1]
        return torch.stack([self.lm_head(tf(h_last)) for tf in self.block_tf], dim=1)

    @torch.no_grad()
    def draft_block(self, prefix_ids: torch.Tensor, prefix_feats: list[torch.Tensor],
                    *, r: int) -> torch.Tensor:
        """Draft k tokens: parallel heads + r causal-masked refinement rounds.

        Each round re-embeds the current block estimate; block positions see
        context (injected) and earlier estimates (causal), never the future.
        """
        block = self.forward_block(prefix_ids, prefix_feats).argmax(dim=-1)  # [1, k]
        n_ctx = prefix_ids.shape[1]
        for _ in range(r):
            # Re-score autoregressively with the current estimates as context:
            # the token at n_ctx+j is predicted by the *next-token* head at
            # position n_ctx+j-1 (head-j predicts j+1 ahead — using it on its
            # own position is an off-by-one that shreds the block).
            ids = torch.cat([prefix_ids, block[:, :-1]], dim=1)
            h = self._trunk(ids, prefix_feats, n_ctx=n_ctx)
            block = self.lm_head(h[:, n_ctx - 1:]).argmax(dim=-1)
        return block


class InjectionSeam:
    """Holds the current per-layer target features for an injected attention.

    Explicit protocol (the engine/trainer owns the semantics):
    ``set(feats)`` with [b, s, target_hidden] aligned to the NEXT forward's
    positions, or ``set(None)`` for zero injection (unverified draft tokens
    are self-conditioned — block-diffusion convention).
    """

    def __init__(self, proj_k, proj_v):
        self.proj_k = proj_k
        self.proj_v = proj_v
        self.feats: torch.Tensor | None = None

    def set(self, feats):
        self.feats = feats

    def delta(self, s: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.feats is None:
            return None
        f = self.feats
        if f.shape[1] != s:
            raise ValueError(f"injection feats len {f.shape[1]} != forward len {s}")
        return self.proj_k(f), self.proj_v(f)


class _SeamedKProj(nn.Module):
    """k_proj wrapper: k(x) + k_inj(target_feats)."""

    def __init__(self, proj, seam):
        super().__init__()
        self.proj = proj
        self.seam = seam

    def forward(self, x):
        k = self.proj(x)
        d = self.seam.delta(x.shape[1], x.device, x.dtype)
        return k if d is None else k + d[0]


class _SeamedVProj(nn.Module):
    def __init__(self, proj, seam):
        super().__init__()
        self.proj = proj
        self.seam = seam

    def forward(self, x):
        v = self.proj(x)
        d = self.seam.delta(x.shape[1], x.device, x.dtype)
        return v if d is None else v + d[1]


class InjectedLFMDrafter:
    """A pretrained LFM2 causal LM with surgical per-layer KV injection.

    Duck-types the drafter side of the specdec engines: callable like the
    model, plus ``sources`` / ``set_feats`` for the engine's feature
    plumbing. Same-depth sources: drafter layer i <- target layer i.
    """

    def __init__(self, lm, target_hidden: int):
        self.lm = lm
        self.seams: list[InjectionSeam] = []
        backbone = lm.model if hasattr(lm, "model") else lm
        for layer in backbone.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None or not hasattr(attn, "k_proj"):
                continue
            seam = InjectionSeam(
                nn.Linear(target_hidden, attn.k_proj.out_features, bias=False,
                          device=next(lm.parameters()).device, dtype=next(attn.k_proj.parameters()).dtype),
                nn.Linear(target_hidden, attn.v_proj.out_features, bias=False,
                          device=next(lm.parameters()).device, dtype=next(attn.v_proj.parameters()).dtype),
            )
            attn.k_proj = _SeamedKProj(attn.k_proj, seam)
            attn.v_proj = _SeamedVProj(attn.v_proj, seam)
            self.seams.append(seam)
        self.sources = list(range(1, len(self.seams) + 1))

    def set_feats(self, hs_tuple, n_new: int | None = None):
        """Feed captured target hidden states; n_new slices the last positions."""
        for seam, src in zip(self.seams, self.sources, strict=True):
            f = hs_tuple[src]
            seam.set(f[:, -n_new:] if n_new is not None else f)

    def clear_feats(self):
        for seam in self.seams:
            seam.set(None)

    def __call__(self, *args, **kwargs):
        return self.lm(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.lm, name)

    def inject_params(self):
        for seam in self.seams:
            yield seam.proj_k
            yield seam.proj_v
