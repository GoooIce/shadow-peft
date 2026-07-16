"""
ShadowPEFT Cached Decode 对齐验证 PoC.

验证: 对纯 full-attention 模型 (Llama/Qwen2), ShadowPEFT 的 cached 单 token decode
与 full-sequence 重算的逐 token logits 是否等价。

核心结论 (已验证): ShadowPEFT 的 injection/update 是逐 token 纯函数, 无跨 token 递归,
因此只需 base KV-cache + shadow backbone KV-cache 即可实现正确的增量解码。

用法:
    .venv/bin/python experiment/verify_cached_decode_alignment.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import DynamicCache, PreTrainedModel

from shadow_peft import ShadowConfig, ShadowPeftModel, get_shadow_model
from shadow_peft.model_utils import _get_backbone

# ---------------------------------------------------------------------------
# Tiny model factories (mirrors tests/conftest.py patterns)
# ---------------------------------------------------------------------------


def _tiny_llama(vocab_size: int = 128, num_layers: int = 4, hidden_size: int = 32) -> PreTrainedModel:
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(cfg)


def _tiny_qwen2(vocab_size: int = 128, num_layers: int = 4, hidden_size: int = 32) -> PreTrainedModel:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    return Qwen2ForCausalLM(cfg)


# ---------------------------------------------------------------------------
# Shadow cached decode engine
# ---------------------------------------------------------------------------


@dataclass
class ShadowCache:
    """复合 cache: 同时保存 base backbone 和 shadow backbone 的 KV-cache."""

    base_cache: DynamicCache = field(default_factory=DynamicCache)
    shadow_cache: DynamicCache = field(default_factory=DynamicCache)


class ShadowCachedDecoder:
    """
    手动驱动 ShadowPEFT 的 cached decode, 绕过 ShadowPeftModel.forward (强制 use_cache=False)。

    流程:
      prefill: shadow_backbone(seq) -> shadow_hidden -> base_backbone(seq, shadow)
      step:    shadow_backbone(token) -> shadow_token -> base_backbone(token, shadow)
    """

    def __init__(self, peft: ShadowPeftModel) -> None:
        self.peft = peft
        self.peft.eval()
        self.base_backbone = _get_backbone(peft.base_model)
        self.shadow_backbone = _get_backbone(peft.shadow_model)
        self.base_embed = peft.base_model.get_input_embeddings()
        self.lm_head = peft.base_model.get_output_embeddings()
        self.shadow_projection = peft.shadow_hidden_projection

        # Shadow embedding policy (mirrors ShadowPeftModel._compute_initial_shadow_hidden):
        # - implicit shadow model / explicit w/ removed embed_tokens: share base embeddings
        # - explicit shadow model w/ own embed_tokens: use shadow's own embeddings
        self._shadow_share_base = (not peft._explicit_shadow_model) or peft._explicit_share_base_embeddings
        self._shadow_embed = None
        if not self._shadow_share_base:
            shadow_get = getattr(peft.shadow_model, "get_input_embeddings", None)
            if callable(shadow_get):
                self._shadow_embed = shadow_get()

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.base_embed(input_ids)
        if emb is None:
            raise RuntimeError("Base model has no input embeddings.")
        return emb

    def _embed_shadow(self, input_ids: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Return (inputs_embeds, input_ids) for the shadow backbone.

        If sharing base embeddings, return (embeds, None).
        If shadow has its own embeddings, return (None, input_ids).
        """
        if self._shadow_share_base:
            return self._embed(input_ids), None
        return None, input_ids

    def _project_shadow(self, shadow_hidden: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.shadow_projection, torch.nn.Identity):
            return self.shadow_projection(shadow_hidden)
        return shadow_hidden

    def _run_shadow(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache: DynamicCache,
    ) -> torch.Tensor:
        """Run the shadow backbone with the correct embedding source."""
        inputs_embeds, shadow_input_ids = self._embed_shadow(input_ids)
        out = self.shadow_backbone(
            input_ids=shadow_input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=cache,
        )
        return self._project_shadow(out.last_hidden_state)

    def _run_base(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache: DynamicCache,
        shadow_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Run base backbone with shadow injection/update active."""
        self.peft._shadow_hidden_states = shadow_hidden
        try:
            out = self.base_backbone(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=cache,
            )
        finally:
            self.peft._shadow_hidden_states = None
        return out.last_hidden_state

    def prefill(self, input_ids: torch.Tensor, cache: ShadowCache) -> torch.Tensor:
        """Prefill the full prompt; returns logits [batch, seq, vocab]."""
        batch_size, seq_len = input_ids.shape
        inputs_embeds = self._embed(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=input_ids.device)

        shadow_hidden = self._run_shadow(input_ids, position_ids, cache.shadow_cache)
        hidden = self._run_base(inputs_embeds, position_ids, attention_mask, cache.base_cache, shadow_hidden)
        return self.lm_head(hidden)

    def step(self, input_ids: torch.Tensor, cache: ShadowCache, step_idx: int) -> torch.Tensor:
        """Decode a single token; returns logits [batch, 1, vocab]."""
        batch_size = input_ids.shape[0]
        inputs_embeds = self._embed(input_ids)  # [batch, 1, hidden]
        position_ids = torch.full((batch_size, 1), step_idx, dtype=torch.long, device=input_ids.device)
        # Attention mask must cover all seen tokens (prefix + this token).
        total_len = step_idx + 1
        attention_mask = torch.ones(batch_size, total_len, dtype=torch.long, device=input_ids.device)

        shadow_token = self._run_shadow(input_ids, position_ids, cache.shadow_cache)
        hidden = self._run_base(inputs_embeds, position_ids, attention_mask, cache.base_cache, shadow_token)
        return self.lm_head(hidden)


# ---------------------------------------------------------------------------
# Alignment verification
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    step: int
    token_cached: int
    token_full: int
    maxdiff: float
    logits_match: bool


def greedy_next(logits: torch.Tensor) -> torch.Tensor:
    """Return argmax token id from logits [batch, seq, vocab] or [batch, vocab]."""
    return logits[:, -1, :].argmax(dim=-1)


def verify_alignment(
    peft: ShadowPeftModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 8,
    *,
    threshold: float = 1e-3,
    label: str = "",
) -> list[StepResult]:
    """
    Compare cached decode vs full-sequence recompute, step by step.

    Path A (cached): prefill once, then single-token decode steps.
    Path B (full):   recompute the entire sequence at each step (use_cache=False).
    """
    decoder = ShadowCachedDecoder(peft)
    results: list[StepResult] = []

    prefix_len = input_ids.shape[1]

    if prefix_len + max_new_tokens > peft.base_model.config.max_position_embeddings:
        max_new_tokens = peft.base_model.config.max_position_embeddings - prefix_len
        print(f"  [warn] clamped max_new_tokens to {max_new_tokens} (max_position_embeddings)")

    # === Path A: cached ===
    cache = ShadowCache()
    with torch.no_grad():
        logits_prefill_cached = decoder.prefill(input_ids, cache)
    # First generated token comes from prefill's last position.
    cached_tokens = [greedy_next(logits_prefill_cached)]

    # === Path B: full-seq (ground truth) ===
    with torch.no_grad():
        logits_full_prefix = peft(input_ids=input_ids).logits[:, -1, :]
    full_tokens = [logits_full_prefix.argmax(dim=-1)]

    # Check prefill alignment.
    prefill_maxdiff = (logits_prefill_cached[:, -1, :] - logits_full_prefix).abs().max().item()
    prefill_match = greedy_next(logits_prefill_cached) == logits_full_prefix.argmax(dim=-1)
    results.append(
        StepResult(
            step=0,
            token_cached=cached_tokens[0].item(),
            token_full=full_tokens[0].item(),
            maxdiff=prefill_maxdiff,
            logits_match=bool(prefill_match.all().item()),
        )
    )

    # Greedy decode loop.
    for step in range(1, max_new_tokens):
        pos = prefix_len + step - 1  # position of the token we're decoding
        token_cached = cached_tokens[-1]
        token_full = full_tokens[-1]

        # Path A: cached single-token step.
        with torch.no_grad():
            logits_cached = decoder.step(token_cached.unsqueeze(-1), cache, step_idx=pos)
        new_cached = greedy_next(logits_cached)
        cached_tokens.append(new_cached)

        # Path B: full-sequence recompute.
        full_ids = torch.cat([input_ids, torch.stack(full_tokens, dim=1)], dim=1)
        with torch.no_grad():
            logits_full = peft(input_ids=full_ids).logits[:, -1, :]
        new_full = logits_full.argmax(dim=-1)
        full_tokens.append(new_full)

        maxdiff = (logits_cached[:, -1, :] - logits_full).abs().max().item()
        results.append(
            StepResult(
                step=step,
                token_cached=new_cached.item(),
                token_full=new_full.item(),
                maxdiff=maxdiff,
                logits_match=bool((new_cached == new_full).all().item()),
            )
        )

    # Print results table.
    header = f"{'step':>4}  {'cached':>7}  {'full':>7}  {'maxdiff':>12}  {'match':>5}"
    print(f"\n  [{label}] Alignment results (threshold={threshold:.0e}):")
    print(f"  {header}")
    print(f"  {'-' * len(header)}")
    all_match = True
    for r in results:
        ok = r.maxdiff < threshold and r.logits_match
        all_match = all_match and ok
        status = "OK" if ok else "FAIL"
        print(f"  {r.step:>4}  {r.token_cached:>7}  {r.token_full:>7}  {r.maxdiff:>12.2e}  {status:>5}")

    cached_seq = [input_ids[0].tolist()] + [t.item() for t in cached_tokens]
    full_seq = [input_ids[0].tolist()] + [t.item() for t in full_tokens]
    seq_match = cached_seq == full_seq
    print(f"\n  Sequence match: {seq_match}")
    if not seq_match:
        print(f"    cached: {cached_seq}")
        print(f"    full:   {full_seq}")

    max_overall = max(r.maxdiff for r in results)
    verdict = "PASS" if all_match and seq_match else "FAIL"
    print(f"  Overall max diff: {max_overall:.2e}  ->  {verdict}")

    return results


# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------


def build_llama_implicit() -> tuple[ShadowPeftModel, str]:
    torch.manual_seed(42)
    base = _tiny_llama(num_layers=4)
    cfg = ShadowConfig(num_shadow_layers=1, injection_hidden_size=8, gate_hidden_size=10, alpha=0.1, dropout=0.0)
    return get_shadow_model(base, cfg), "Llama-4L implicit shadow (hidden match)"


def build_qwen2_implicit() -> tuple[ShadowPeftModel, str]:
    torch.manual_seed(42)
    base = _tiny_qwen2(num_layers=4)
    cfg = ShadowConfig(num_shadow_layers=1, injection_hidden_size=8, gate_hidden_size=10, alpha=0.1, dropout=0.0)
    return get_shadow_model(base, cfg), "Qwen2-4L implicit shadow (hidden match)"


def build_llama_explicit_projection() -> tuple[ShadowPeftModel, str]:
    """Explicit shadow model with DIFFERENT hidden size -> requires projection."""
    torch.manual_seed(42)
    base = _tiny_llama(num_layers=4, hidden_size=32)
    # Shadow model with smaller hidden size (16) -> projection 16->32 kicks in.
    # Keep shadow's own embed_tokens (16-dim) since base embeddings (32-dim) won't fit.
    shadow_base = _tiny_llama(num_layers=1, hidden_size=16)
    cfg = ShadowConfig(num_shadow_layers=1, injection_hidden_size=8, gate_hidden_size=10, alpha=0.1, dropout=0.0)
    from shadow_peft import prepare_shadow_model

    shadow_backbone = prepare_shadow_model(shadow_base, remove_embed_tokens=False)
    peft = get_shadow_model(base, cfg, shadow_model=shadow_backbone)
    return peft, "Llama-4L explicit shadow (hidden mismatch, projection 16->32)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify ShadowPEFT cached decode alignment")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["llama", "qwen2", "projection"],
        choices=["llama", "qwen2", "projection"],
        help="Which model configurations to test",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Number of decode steps")
    parser.add_argument("--threshold", type=float, default=1e-3, help="Max logits diff threshold")
    parser.add_argument("--prompt-len", type=int, default=4, help="Prompt length")
    args = parser.parse_args()

    builders: dict[str, Any] = {
        "llama": build_llama_implicit,
        "qwen2": build_qwen2_implicit,
        "projection": build_llama_explicit_projection,
    }

    print("=" * 70)
    print("ShadowPEFT Cached Decode Alignment Verification")
    print("=" * 70)
    print(f"Config: prompt_len={args.prompt_len}, max_new_tokens={args.max_new_tokens}, threshold={args.threshold:.0e}")

    all_pass = True
    for key in args.models:
        peft, label = builders[key]()
        torch.manual_seed(123)  # deterministic input
        vocab = peft.base_model.config.vocab_size
        input_ids = torch.randint(0, vocab, (1, args.prompt_len))
        print(f"\n{'─' * 70}")
        print(f"Model: {label}")
        print(f"Input: {input_ids[0].tolist()}")
        results = verify_alignment(peft, input_ids, args.max_new_tokens, threshold=args.threshold, label=label)
        max_diff = max(r.maxdiff for r in results)
        passed = max_diff < args.threshold and all(r.logits_match for r in results)
        all_pass = all_pass and passed

    print(f"\n{'=' * 70}")
    verdict = "ALL PASS" if all_pass else "SOME FAILED"
    print(f"Final verdict: {verdict}")
    print("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
