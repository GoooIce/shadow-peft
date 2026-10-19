"""EdgeCloud-CI: edge small model + cloud large model collaborative inference.

产线端部署小模型负责低时延本地推理，云端大模型在低置信度或高价值任务上
兜底增强；难例与人工纠错回流云端，训练后的适配器按灰度版本下发回端侧。
"""

from __future__ import annotations

from .cloud_server import CloudService
from .config import CloudConfig, EdgeConfig
from .edge_server import EdgeService
from .router import CloudClient, Router

__all__ = [
    "CloudClient",
    "CloudConfig",
    "CloudService",
    "EdgeConfig",
    "EdgeService",
    "Router",
]

__version__ = "0.1.0"
