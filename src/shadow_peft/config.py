"""ShadowConfig lives in shadow_core (backend-free); re-exported here.

Both `shadow_peft.ShadowConfig` and `shadow_peft_mlx.ShadowConfig` are this
same class, which is what makes adapter checkpoints portable across backends.
"""

from __future__ import annotations

from shadow_core.config import ShadowConfig

__all__ = ["ShadowConfig"]
