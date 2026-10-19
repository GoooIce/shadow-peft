"""Command line entry points: edge agent, cloud service and local demo."""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import threading
from pathlib import Path
from typing import Any

from .cloud_server import CloudService
from .config import CloudConfig, EdgeConfig, load_config
from .edge_server import EdgeService


def _section(path: str | None, name: str) -> dict[str, Any]:
    payload = load_config(path)
    section = payload.get(name, {})
    if not isinstance(section, dict):
        raise SystemExit(f"配置节 {name!r} 必须是 JSON 对象")
    return section


def _wait_forever() -> None:
    event = threading.Event()
    with contextlib.suppress(KeyboardInterrupt):
        event.wait()


def serve_edge(args: argparse.Namespace) -> int:
    config = EdgeConfig.from_dict(_section(args.config, "edge"))
    if args.port:
        config.agent_port = args.port
    service = EdgeService(config)
    server = service.start(host=args.host or "0.0.0.0", port=args.port or config.agent_port)
    print(f"edge agent  : {server.url}")
    print(f"console     : {server.url}/")
    print(f"cloud       : {config.cloud_url}")
    try:
        _wait_forever()
    finally:
        service.stop()
    return 0


def serve_cloud(args: argparse.Namespace) -> int:
    config = CloudConfig.from_dict(_section(args.config, "cloud"))
    service = CloudService(config)
    server = service.start(host=args.host or "0.0.0.0", port=args.port or config.port)
    print(f"cloud service: {server.url}")
    print(f"model        : {service.backend.model_id}")
    try:
        _wait_forever()
    finally:
        service.stop()
    return 0


def run_demo(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).expanduser().resolve()
    if args.reset and workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    cloud_config = CloudConfig(
        backend={"kind": "mock_cloud", "latency_ms": args.cloud_latency_ms},
        feedback_path=str(workdir / "cloud_feedback.jsonl"),
        registry_path=str(workdir / "registry.json"),
        dataset_path=str(workdir / "sft_hard_cases.jsonl"),
    )
    cloud_service = CloudService(cloud_config)
    if cloud_service.registry.state()["active"] is None:
        cloud_service.registry.publish(
            "v1",
            artifact="runs/edge_cloud/adapters/v1",
            notes="演示基线适配器版本",
        )
    cloud_server = cloud_service.start(host="127.0.0.1", port=args.cloud_port)

    edge_config = EdgeConfig(
        backend={"kind": "mock_edge", "latency_ms": args.edge_latency_ms},
        cloud_url=cloud_server.url,
        outbox_path=str(workdir / "edge_outbox.jsonl"),
        feedback_path=str(workdir / "edge_feedback.jsonl"),
        outbox_interval_s=args.outbox_interval_s,
    )
    edge_service = EdgeService(edge_config)
    edge_server = edge_service.start(host=args.host, port=args.port)

    print("=" * 68)
    print(" 端云协同推理控制台（演示模式）")
    print(f" 控制台   : {edge_server.url}/")
    print(f" 端侧服务 : {edge_server.url}/healthz")
    print(f" 云端服务 : {cloud_server.url}/healthz")
    print(f" 工作目录 : {workdir}")
    print(" 按 Ctrl+C 退出")
    print("=" * 68)
    try:
        _wait_forever()
    finally:
        edge_service.stop()
        cloud_service.stop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edge-cloud",
        description="产线端小模型 + 云端大模型协同推理参考实现",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="启动端侧 + 云端 + 控制台（模拟模型）")
    demo.add_argument("--host", default="127.0.0.1")
    demo.add_argument("--port", type=int, default=8080)
    demo.add_argument("--cloud-port", type=int, default=9100)
    demo.add_argument("--workdir", default="runs/edge_cloud/demo")
    demo.add_argument("--reset", action="store_true", help="启动前清空演示数据")
    demo.add_argument("--edge-latency-ms", type=float, default=35.0)
    demo.add_argument("--cloud-latency-ms", type=float, default=520.0)
    demo.add_argument("--outbox-interval-s", type=float, default=5.0)
    demo.set_defaults(func=run_demo)

    edge = subparsers.add_parser("serve-edge", help="启动端侧代理服务")
    edge.add_argument("--config", default=None, help="JSON 配置（读取 edge 节）")
    edge.add_argument("--host", default="0.0.0.0")
    edge.add_argument("--port", type=int, default=None)
    edge.set_defaults(func=serve_edge)

    cloud = subparsers.add_parser("serve-cloud", help="启动云端增强服务")
    cloud.add_argument("--config", default=None, help="JSON 配置（读取 cloud 节）")
    cloud.add_argument("--host", default="0.0.0.0")
    cloud.add_argument("--port", type=int, default=None)
    cloud.set_defaults(func=serve_cloud)

    dump = subparsers.add_parser("config", help="打印默认配置模板")
    dump.add_argument("--kind", choices=["edge", "cloud"], default="edge")
    dump.set_defaults(func=print_config)
    return parser


def print_config(args: argparse.Namespace) -> int:
    payload = EdgeConfig().to_dict() if args.kind == "edge" else CloudConfig().to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
