"""shadow_core: backend-free core shared by the torch and MLX packages.

Pure Python, zero third-party dependencies. Owns the checkpoint-portable
schema (``ShadowConfig``) and the harness router/session/config trio. Tensor
math stays in the backend packages.
"""

from .config import ShadowConfig

__all__ = ["ShadowConfig"]
