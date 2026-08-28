"""Lossless speculative decoding engine (greedy, chain-draft + batched-verify).

v1 contract (spec exp2):
- greedy acceptance: draft token i is accepted iff it equals the target's
  argmax at that position; on mismatch the target's own token is emitted —
  the output is therefore **token-identical to plain greedy decoding of the
  target for ANY draft** (the lossless guarantee doubles as the correctness
  test).
- hybrid-cache safety: before every draft chain both caches are snapshotted;
  on partial acceptance they are restored and the committed tokens replayed,
  so conv short-state (LFM2 ``conv_L_cache=3``) and GQA KV stay exact.
  Full acceptance takes the no-replay fast path (caches already consistent).
- wall-clock path (v1.1): O(1) crop snapshots replace the per-round full
  clone of the KV history, and the full-accept bonus token is deferred into
  the next verify forward — one fewer full target pass per round (detached
  mode; feature-injected drafts keep per-position alignment instead).

Batch size 1 (chat τ evaluation and edge deployment are the bs=1 regime;
the bs∈{4,8} matrix comes later with a batched variant).
"""

from __future__ import annotations

import time

import torch


def _clone_state(obj):
    """Deep-clone tensors, recursing through dict/list/tuple containers.

    Cache layers store conv states as ``dict[state_idx -> Tensor]`` — a bare
    tensor walk would silently skip them and leave stale conv context after
    a rollback.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: _clone_state(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clone_state(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_state(v) for v in obj)
    return obj


def _deep_snapshot(cache) -> list[dict]:
    """Full deep clone of every attribute of every cache layer (O(L) per round)."""
    return [{n: _clone_state(v) for n, v in vars(layer).items()} for layer in cache.layers]


def _deep_restore(cache, snaps) -> None:
    """Write a deep snapshot back into the live cache (clone to keep it reusable)."""
    for layer, snap in zip(cache.layers, snaps, strict=True):
        for n, v in snap.items():
            setattr(layer, n, _clone_state(v))


def _snapshot(cache) -> list[dict]:
    """O(1) rollback snapshot (crop mode).

    ``DynamicLayer.update`` concatenates new positions onto *fresh* tensors
    and never mutates the cached prefix, so a shallow reference to the
    pre-round ``keys``/``values`` is a bit-exact rollback point — restoring
    writes the old object back (no copy). Only in-place short states (conv
    dict containers, tiny by construction: the ``conv_L_cache`` window) need
    real clones. Invariant relied on: every mutable-in-place cache state
    lives in a dict attribute; direct tensor attributes are cat-updated.
    """
    snaps = []
    for layer in cache.layers:
        snaps.append({n: (_clone_state(v) if isinstance(v, dict) else v)
                      for n, v in vars(layer).items()})
    return snaps


def _restore(cache, snaps) -> None:
    """Roll the live cache back to a crop snapshot.

    Tensors: write the pre-round object back (the live one is a later cat
    result — restoring by reference is bit-exact and free). Dicts: clone from
    the snapshot so it stays reusable.
    """
    for layer, snap in zip(cache.layers, snaps, strict=True):
        for n, v in snap.items():
            if isinstance(v, torch.Tensor):
                if getattr(layer, n, None) is not v:
                    setattr(layer, n, v)
            elif isinstance(v, dict):
                setattr(layer, n, _clone_state(v))
            else:
                setattr(layer, n, v)


def _cached_kwargs(cache, n: int, device) -> dict:
    """Mirror GenerationMixin: explicit cache_position + attention_mask."""
    L = cache.get_seq_length()
    return {
        "use_cache": True,
        "cache_position": torch.arange(L, L + n, device=device),
        "attention_mask": torch.ones(1, L + n, dtype=torch.long, device=device),
    }

@torch.no_grad()
def greedy_speculative_generate(
    target,
    draft,
    input_ids: torch.Tensor,
    *,
    k: int = 4,
    max_new_tokens: int = 64,
    snapshot: str = "crop",
    defer_bonus: bool = True,
    collect_timing: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Greedy lossless speculative generation. Returns (ids, stats).

    ``snapshot="crop"`` (default): O(1) shallow-reference cache snapshots —
    see :func:`_snapshot`. ``"deep"`` keeps the legacy full-clone path as an
    ablation/escape hatch.

    ``defer_bonus=True`` (detached mode only; ignored when the draft takes
    feature injection): the bonus token of a full-accept round is *not*
    pushed through the target in its own forward — it leads the next
    round's verify input, so one full target pass per round is saved.
    """
    assert input_ids.shape[0] == 1, "engine v1 is batch-size-1"
    assert snapshot in ("crop", "deep")
    target.eval()
    draft.eval()
    snap_of = _snapshot if snapshot == "crop" else _deep_snapshot
    restore_of = _restore if snapshot == "crop" else _deep_restore

    def _clock():
        if collect_timing:
            torch.cuda.synchronize()
        return time.perf_counter()

    dev = input_ids.device
    prefill_mask = torch.ones(1, input_ids.shape[1], dtype=torch.long, device=dev)
    wants_feats = hasattr(draft, "set_feats")
    use_defer = defer_bonus and not wants_feats
    out_t = target(input_ids=input_ids, use_cache=True, attention_mask=prefill_mask,
                   output_hidden_states=wants_feats)
    if wants_feats:
        draft.set_feats(out_t.hidden_states)  # full-coverage injection at prefill
    out_d = draft(input_ids=input_ids, use_cache=True, attention_mask=prefill_mask)
    if wants_feats:
        draft.clear_feats()
    t_cache, d_cache = out_t.past_key_values, out_d.past_key_values
    cur_t_logits = out_t.logits[:, -1:]
    cur_d_logits = out_d.logits[:, -1:]

    seq = [input_ids[0]]
    produced = 0
    rounds = 0
    accept_hist = [0] * (k + 2)
    pending = None  # committed token the target cache has not processed yet
    t_ms = {"snapshot": 0.0, "draft": 0.0, "verify": 0.0, "commit": 0.0, "replay": 0.0}

    while produced < max_new_tokens:
        rounds += 1
        t0 = _clock()
        t_snap = snap_of(t_cache)
        d_snap = snap_of(d_cache)
        d_logits = cur_d_logits

        # 1. chain draft: k greedy tokens.
        t1 = _clock()
        draft_toks = []
        for _ in range(k):
            nxt = d_logits.argmax(dim=-1)  # [1, 1]
            draft_toks.append(nxt)
            o = draft(input_ids=nxt, past_key_values=d_cache, **_cached_kwargs(d_cache, 1, dev))
            d_cache = o.past_key_values
            d_logits = o.logits[:, -1:]
        d_tok = torch.cat(draft_toks, dim=1)  # [1, k]

        # 2. batched verify: one target forward over [pending?] + the chain.
        # With a deferred bonus the pending token leads the input, so the
        # forward's own logits already cover every prediction we need.
        t2 = _clock()
        if use_defer and pending is not None:
            v_in = torch.cat([pending[None], d_tok], dim=1)
            o_t = target(input_ids=v_in, past_key_values=t_cache, **_cached_kwargs(t_cache, k + 1, dev))
            t_cache = o_t.past_key_values
            v_argmax = o_t.logits.argmax(dim=-1)[0]  # [k+1]
        else:
            o_t = target(input_ids=d_tok, past_key_values=t_cache,
                         output_hidden_states=wants_feats,
                         **_cached_kwargs(t_cache, k, dev))
            t_cache = o_t.past_key_values
            # predictions for positions [cur .. cur+k] (k+1 of them).
            v_argmax = torch.cat([cur_t_logits, o_t.logits], dim=1).argmax(dim=-1)[0]  # [k+1]

        # 3. accept the matching prefix.
        match = d_tok[0] == v_argmax[:k]
        n = int(torch.cumprod(match, dim=0).sum())
        accept_hist[n] += 1
        if n == k:
            t3 = _clock()
            committed = torch.cat([d_tok[0], v_argmax[k : k + 1]])  # k + bonus
            if use_defer:
                # target cache already covers [pending + k drafts]; the bonus
                # stays pending — it leads the next verify. Draft side: feed it.
                pending = v_argmax[k : k + 1]
                o_db = draft(input_ids=pending[None], past_key_values=d_cache,
                             **_cached_kwargs(d_cache, 1, dev))
                d_cache = o_db.past_key_values
                cur_d_logits = o_db.logits[:, -1:]
                cur_t_logits = o_t.logits[:, -1:]
            else:
                pending = None
                # fast path: target cache already covers the k drafts; feed it the
                # bonus token. draft cache covers the k drafts; feed it the bonus too.
                o_tb = target(input_ids=v_argmax[k : k + 1][None], past_key_values=t_cache,
                              output_hidden_states=wants_feats,
                              **_cached_kwargs(t_cache, 1, dev))
                t_cache = o_tb.past_key_values
                cur_t_logits = o_tb.logits[:, -1:]
                if wants_feats:
                    # feed the k accepted draft positions first, then the bonus
                    draft.set_feats(o_t.hidden_states)
                    o = draft(input_ids=d_tok, past_key_values=d_cache,
                              **_cached_kwargs(d_cache, k, dev))
                    d_cache = o.past_key_values
                    draft.set_feats(o_tb.hidden_states)
                o_db = draft(input_ids=v_argmax[k : k + 1][None], past_key_values=d_cache,
                             **_cached_kwargs(d_cache, 1, dev))
                if wants_feats:
                    draft.clear_feats()
                d_cache = o_db.past_key_values
                cur_d_logits = o_db.logits[:, -1:]
        else:
            t3 = _clock()
            committed = torch.cat([d_tok[0, :n], v_argmax[n : n + 1]])  # n + corrective
            # roll both caches back to the snapshot, then replay the committed
            # tokens (hybrid-cache exact: conv short-state + GQA KV). In defer
            # mode the target cache is one token behind the draft cache (the
            # pending token), so its replay input leads with that token.
            had_pending = pending is not None
            t_pending_tok = pending
            pending = None
            restore_of(t_cache, t_snap)
            restore_of(d_cache, d_snap)
            t_replay = torch.cat([t_pending_tok, committed]) if had_pending else committed
            o_t2 = target(input_ids=t_replay[None], past_key_values=t_cache,
                          **_cached_kwargs(t_cache, int(t_replay.numel()), dev))
            t_cache = o_t2.past_key_values
            cur_t_logits = o_t2.logits[:, -1:]
            if wants_feats:
                # verify states cover the k draft positions; committed uses n+1
                draft.set_feats(tuple(h[:, : int(committed.numel())] for h in o_t.hidden_states))
            o_d2 = draft(input_ids=committed[None], past_key_values=d_cache,
                         **_cached_kwargs(d_cache, int(committed.numel()), dev))
            d_cache = o_d2.past_key_values
            cur_d_logits = o_d2.logits[:, -1:]
            if wants_feats:
                draft.clear_feats()
        if collect_timing:
            torch.cuda.synchronize()
            t_ms["snapshot"] += t1 - t0
            t_ms["draft"] += t2 - t1
            t_ms["verify"] += t3 - t2
            t_ms["commit" if n == k else "replay"] += time.perf_counter() - t3

        seq.append(committed)
        produced += int(committed.numel())

    # tau uses the raw committed count (pre-trim overshoot), or rounds with
    # all-accept would under-report k+1.
    stats = {
        "rounds": rounds,
        "produced": produced,
        "tau": round(produced / rounds, 4),
        "accept_hist": accept_hist,
        "k": k,
    }
    if collect_timing:
        stats["t_ms"] = {kk: round(v * 1000, 2) for kk, v in t_ms.items()}
    L0 = input_ids.shape[1]
    gen = torch.cat(seq)[L0 : L0 + max_new_tokens]
    return torch.cat([input_ids[0], gen])[None], stats


@torch.no_grad()
def block_speculative_generate_fullctx(target, drafter, input_ids, *, k=4, r=2,
                                       max_new_tokens=64):
    """DFlash2-structured block drafting, full-recompute drafter (spec exp5, archived).

    Draft: ``drafter.draft_block`` (parallel heads + r refinement passes,
    full-recompute — no drafter cache to roll back). Verify/accept: the same
    greedy prefix-match guarantee; only the target cache is snapshotted.
    ``drafter`` is a ``KVInjectedDrafter`` fed by free hidden-state captures
    from the verify forwards. Output is lossless w.r.t. target greedy.
    """
    assert input_ids.shape[0] == 1
    target.eval(), drafter.eval()
    dev = input_ids.device

    L = input_ids.shape[1]
    out = target(input_ids=input_ids, use_cache=True, output_hidden_states=True,
                 attention_mask=torch.ones(1, L, dtype=torch.long, device=dev))
    t_cache = out.past_key_values
    # Full-context target features per layer (cached forwards only return the
    # new positions' states — keep the full list and extend on every commit).
    hs_all = list(out.hidden_states)
    feats = [hs_all[s] for s in drafter.sources]
    cur_t_logits = out.logits[:, -1:]
    seq = [input_ids[0]]
    produced, rounds = 0, 0

    while produced < max_new_tokens:
        rounds += 1
        t_snap = _snapshot(t_cache)

        d_tok = drafter.draft_block(input_ids, feats, r=r)  # [1, k]
        o_t = target(input_ids=d_tok, past_key_values=t_cache,
                     output_hidden_states=True, **_cached_kwargs(t_cache, k, dev))
        t_cache = o_t.past_key_values
        v_argmax = torch.cat([cur_t_logits, o_t.logits], dim=1).argmax(dim=-1)[0]  # [k+1]

        match = d_tok[0] == v_argmax[:k]
        n = int(torch.cumprod(match, dim=0).sum())
        if n == k:
            committed = torch.cat([d_tok[0], v_argmax[k:k + 1]])
            o_tb = target(input_ids=v_argmax[k:k + 1][None], past_key_values=t_cache,
                          output_hidden_states=True, **_cached_kwargs(t_cache, 1, dev))
            t_cache = o_tb.past_key_values
            cur_t_logits = o_tb.logits[:, -1:]
            # merged hidden capture: verify (k accepted) + bonus (1)
            hs_new = [torch.cat([h, hb], dim=1) for h, hb in
                      zip(o_t.hidden_states, o_tb.hidden_states, strict=True)]
        else:
            committed = torch.cat([d_tok[0, :n], v_argmax[n:n + 1]])
            _restore(t_cache, t_snap)
            o_t2 = target(input_ids=committed[None], past_key_values=t_cache,
                          output_hidden_states=True,
                          **_cached_kwargs(t_cache, int(committed.numel()), dev))
            t_cache = o_t2.past_key_values
            cur_t_logits = o_t2.logits[:, -1:]
            hs_new = list(o_t2.hidden_states)

        # drafter context grows by exactly the committed tokens (free features)
        input_ids = torch.cat([input_ids, committed[None]], dim=1)
        hs_all = [torch.cat([h_old, h_new], dim=1) for h_old, h_new in
                  zip(hs_all, hs_new, strict=True)]
        feats = [hs_all[s] for s in drafter.sources]
        seq.append(committed)
        produced += int(committed.numel())

    L0 = input_ids.shape[1] - produced
    gen = torch.cat(seq)[L0 : L0 + max_new_tokens]
    return torch.cat([seq[0], gen])[None], {
        "rounds": rounds, "produced": produced, "tau": round(produced / rounds, 4), "k": k, "r": r,
    }


@torch.no_grad()
def block_speculative_generate(target, drafter, input_ids, *, k=4, r=0,
                               max_new_tokens=64, snapshot: str = "crop",
                               defer_bonus: bool = True, collect_timing: bool = False,
                               conf_threshold: float | None = None,
                               ) -> tuple[torch.Tensor, dict]:
    """Cached single-forward block drafting (spec exp2 v2-block).

    ``drafter`` implements the :class:`shadow_lab.block_drafter.CachedBlockDrafter`
    protocol (``prefill`` / ``draft`` / ``commit`` / ``snapshot`` / ``restore``).
    Verify/accept semantics are the chain engine's: greedy prefix-match,
    deferred bonus on full accepts (the target lags one token in defer mode;
    the drafter never lags — its commit includes the round's own bonus).
    """

    assert input_ids.shape[0] == 1
    assert snapshot in ("crop", "deep")
    target.eval()
    drafter.eval()
    snap_of = _snapshot if snapshot == "crop" else _deep_snapshot
    restore_of = _restore if snapshot == "crop" else _deep_restore

    def _clock():
        if collect_timing:
            torch.cuda.synchronize()
        return time.perf_counter()

    dev = input_ids.device
    prefill_mask = torch.ones(1, input_ids.shape[1], dtype=torch.long, device=dev)
    out_t = target(input_ids=input_ids, use_cache=True, attention_mask=prefill_mask)
    t_cache = out_t.past_key_values
    cur_t_logits = out_t.logits[:, -1:]
    drafter.prefill(input_ids)

    seq = [input_ids[0]]
    produced = 0
    rounds = 0
    accept_hist = [0] * (k + 2)
    k_eff_sum = 0
    pending = None  # committed token the target cache has not processed yet
    t_ms = {"snapshot": 0.0, "draft": 0.0, "verify": 0.0, "commit": 0.0, "replay": 0.0}

    while produced < max_new_tokens:
        rounds += 1
        t0 = _clock()
        t_snap = snap_of(t_cache)
        d_snap = drafter.snapshot()

        t1 = _clock()
        d_tok = drafter.draft(k, r, conf_threshold=conf_threshold)  # [1, k_eff]
        assert d_tok.dim() == 2 and d_tok.shape[0] == 1, \
            f"drafter returned {tuple(d_tok.shape)}, expected [1, n]"

        t2 = _clock()
        if defer_bonus and pending is not None:
            v_in = torch.cat([pending[None], d_tok], dim=1)
            o_t = target(input_ids=v_in, past_key_values=t_cache,
                         **_cached_kwargs(t_cache, int(v_in.shape[1]), dev))
            t_cache = o_t.past_key_values
            v_argmax = o_t.logits.argmax(dim=-1)[0]  # [k_eff+1]
        else:
            o_t = target(input_ids=d_tok, past_key_values=t_cache,
                         **_cached_kwargs(t_cache, int(d_tok.shape[1]), dev))
            t_cache = o_t.past_key_values
            v_argmax = torch.cat([cur_t_logits, o_t.logits], dim=1).argmax(dim=-1)[0]

        k_eff = int(d_tok.shape[1])  # dynamic-k drafters (confidence truncation)
        k_eff_sum += k_eff
        match = d_tok[0] == v_argmax[:k_eff]
        n = int(torch.cumprod(match, dim=0).sum())
        accept_hist[n] += 1
        if n == k_eff:
            t3 = _clock()
            committed = torch.cat([d_tok[0], v_argmax[k_eff : k_eff + 1]])  # k_eff + bonus
            if defer_bonus:
                pending = v_argmax[k_eff : k_eff + 1]
                cur_t_logits = o_t.logits[:, -1:]
            else:
                pending = None
                o_tb = target(input_ids=v_argmax[k_eff : k_eff + 1][None], past_key_values=t_cache,
                              **_cached_kwargs(t_cache, 1, dev))
                t_cache = o_tb.past_key_values
                cur_t_logits = o_tb.logits[:, -1:]
            drafter.commit(committed)
        else:
            t3 = _clock()
            committed = torch.cat([d_tok[0, :n], v_argmax[n : n + 1]])  # n + corrective
            had_pending = pending is not None
            t_pending_tok = pending
            pending = None
            restore_of(t_cache, t_snap)
            drafter.restore(d_snap)
            t_replay = torch.cat([t_pending_tok, committed]) if had_pending else committed
            o_t2 = target(input_ids=t_replay[None], past_key_values=t_cache,
                          **_cached_kwargs(t_cache, int(t_replay.numel()), dev))
            t_cache = o_t2.past_key_values
            cur_t_logits = o_t2.logits[:, -1:]
            drafter.commit(committed)
        if collect_timing:
            torch.cuda.synchronize()
            t_ms["snapshot"] += t1 - t0
            t_ms["draft"] += t2 - t1
            t_ms["verify"] += t3 - t2
            t_ms["commit" if n == k else "replay"] += time.perf_counter() - t3

        seq.append(committed)
        produced += int(committed.numel())

    stats = {
        "rounds": rounds,
        "produced": produced,
        "tau": round(produced / rounds, 4),
        "accept_hist": accept_hist,
        "k": k,
        "r": r,
        "k_eff_mean": round(k_eff_sum / rounds, 3),
    }
    if collect_timing:
        stats["t_ms"] = {kk: round(v * 1000, 2) for kk, v in t_ms.items()}
    L0 = input_ids.shape[1]
    gen = torch.cat(seq)[L0 : L0 + max_new_tokens]
    return torch.cat([input_ids[0], gen])[None], stats


@torch.no_grad()
def greedy_reference(model, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Cached incremental greedy loop — the deployment-comparable ground truth.

    Full-sequence recompute drifts from cached forward in low precision
    (bf16 kernel shapes differ), so the lossless property is asserted against
    this reference, matching what serving-time greedy decoding produces.
    """
    dev = input_ids.device
    out = model(input_ids=input_ids, use_cache=True)
    cache = out.past_key_values
    gen = [out.logits[:, -1].argmax(dim=-1, keepdim=True)]
    for _ in range(max_new_tokens - 1):
        nxt = gen[-1]
        o = model(input_ids=nxt, past_key_values=cache, **_cached_kwargs(cache, 1, dev))
        cache = o.past_key_values
        gen.append(o.logits[:, -1].argmax(dim=-1, keepdim=True))
    return torch.cat([input_ids] + gen, dim=1)


def measure_tau(target, draft, prompts_ids: list[torch.Tensor], *, k: int = 4,
                max_new_tokens: int = 64) -> dict:
    """Mean greedy τ over a prompt set (identical-setting comparability)."""
    taus, produced = [], 0
    for ids in prompts_ids:
        _, s = greedy_speculative_generate(target, draft, ids, k=k, max_new_tokens=max_new_tokens)
        taus.append(s["tau"])
        produced += s["produced"]
    return {"tau_mean": round(sum(taus) / len(taus), 4), "n_prompts": len(taus),
            "k": k, "max_new_tokens": max_new_tokens, "total_tokens": produced}
