"""Two-node NCCL/RDMA benchmark over the GB10 200G direct link.

Measures all_reduce latency/bandwidth at the tensor sizes our distillation
campaign actually produces (shadow grads ~35M params fp32 = 140MB), plus a
compute-coupled probe: local matmul step + all_reduce, to estimate DDP
speedup on a real training step.

Launch (one per node, over the direct link):

    MASTER_ADDR=10.100.144.2 MASTER_PORT=29500 \
    torchrun --nnodes=2 --nproc-per-node=1 --node-rank=<0|1> \
        scripts/bench_nccl_dist.py
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group(
        backend=os.environ.get("BENCH_BACKEND", "nccl"),
        device_id=torch.cuda.current_device(),  # type: ignore[arg-type]
    )
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda")
    if rank == 0:
        print(f"backend={dist.get_backend()} world={dist.get_world_size()} "
              f"torch={torch.__version__} nccl={torch.cuda.nccl.version()}", flush=True)

    def bench_allreduce(n_elem: int, iters: int = 20) -> tuple[float, float]:
        x = torch.randn(n_elem, device=dev)
        for _ in range(5):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        # bus bw (GB/s): 2*(N-1)/N * size / time, N=2 -> size/dt
        return dt, (n_elem * x.element_size()) / dt / 1e9

    results = {}
    for label, n_elem in [
        ("140MB (35M fp32 grads)", 35_000_000),
        ("56MB", 14_000_000),
        ("5.6MB", 1_400_000),
    ]:
        dt, bw = bench_allreduce(n_elem)
        results[label] = (dt, bw)
        if rank == 0:
            print(f"all_reduce {label}: {dt * 1e3:.2f} ms, bus-bw {bw:.1f} GB/s", flush=True)

    # Compute-coupled probe: emulate a distillation step (teacher fwd + student
    # fwd/bwd on 4x512 tokens) followed by the grad all_reduce.
    if rank == 0:
        print("--- coupled step probe (local batch 4x512, 35M-param grad) ---", flush=True)
    teacher = torch.nn.Linear(1024, 1024, dtype=torch.bfloat16, device=dev)
    student = torch.nn.Linear(1024, 1024, dtype=torch.bfloat16, device=dev)
    grad_like = torch.randn(35_000_000, device=dev)

    def one_step() -> None:
        h = torch.randn(4, 512, 1024, device=dev, dtype=torch.bfloat16)
        with torch.no_grad():
            _ = teacher(h)
        out = student(h)
        out.sum().backward()
        dist.all_reduce(grad_like)

    for _ in range(5):
        one_step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        one_step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 20
    if rank == 0:
        print(f"coupled step: {dt * 1e3:.1f} ms/step", flush=True)

    dist.destroy_process_group()
    if rank == 0:
        print("BENCH-OK", flush=True)


if __name__ == "__main__":
    main()
