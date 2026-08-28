"""Print trainable params for candidate exp1-floor-2 shadow structures.

    ssh goooinext 'cd ~/Code/shadow-peft && .venv/bin/python scripts/exp1_floor2_probe.py'
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_lab.trainer import TrainConfig, _quantize_base
from shadow_peft import ShadowConfig, ShadowForCausalLM, get_shadow_model

MODEL = "/home/goooice/models/LFM2.5-350M-Base"

CANDS = [
    ("L4-quarterwide", dict(num_shadow_layers=4, shadow_intermediate_size=1664,
                            shadow_num_attention_heads=4, shadow_num_key_value_heads=2)),
    ("L8-eighthwide", dict(num_shadow_layers=8, shadow_intermediate_size=832,
                           shadow_num_attention_heads=2, shadow_num_key_value_heads=1)),
]


def main() -> None:
    cfg = TrainConfig(bits=2, loss_mode="ce_kl", model=MODEL, data="unused")
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    _quantize_base(base, cfg, len(tok))
    base.eval()
    for name, kw in CANDS:
        sc = ShadowConfig(injection_hidden_size=16, gate_hidden_size=10,
                          alpha=0.1, dropout=0.0, **kw)
        task = ShadowForCausalLM(get_shadow_model(base, sc))
        n = sum(p.numel() for p in task.parameters() if p.requires_grad)
        print(f"{name:>18}: {n:>12,}", flush=True)
        del task


if __name__ == "__main__":
    main()
