"""ShadowConfig lives in shadow_core (backend-free); re-exported here.

`shadow_peft_mlx.ShadowConfig` and `shadow_peft.ShadowConfig` are the same
class, which is what makes adapter checkpoints portable across backends.
"""

from __future__ import annotations

from shadow_core.config import ShadowConfig

__all__ = ["ShadowConfig"]
