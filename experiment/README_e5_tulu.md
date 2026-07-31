# E5 Tulu Shadow recovery

The E5 scripts distill the official Bonsai Qwen3.6-27B checkpoint into a
one-layer Shadow adapter attached to the locally produced 2-bit checkpoint.
Teacher and student are loaded sequentially, so they do not need to coexist in
unified memory.

## Requirements

- macOS/Apple Silicon with the PrismML fork of MLX required by the Bonsai
  1-bit/2-bit kernels.
- The official Bonsai checkpoint at:
  `~/.cache/modelscope/models/prism-ml--Ternary-Bonsai-27B-mlx-2bit/snapshots/master`
- The local self-quantized checkpoint at:
  `./qwen3.6-27b-mlx-2bit-v3-oneshot`
- Tulu 3:

```bash
modelscope download --dataset allenai/tulu-3-sft-mixture
```

Install the project environment:

```bash
uv sync --extra mlx --group dev
```

## E5-R2 target run

The defaults are the requested protocol:

- 1024 train / 64 held-out conversations;
- up to 4096 context tokens;
- 128 optimizer steps with batch size 8, covering all 1024 train examples;
- assistant-content positions only;
- 16 assistant target tokens per microbatch;
- exact teacher top-32 probabilities plus one aggregated tail bucket.

Prepare the deterministic dataset:

```bash
.venv/bin/python experiment/e5_tulu_assistant_recovery_qwen36.py \
  --prepare-only \
  --refresh-data
```

Build the teacher cache:

```bash
.venv/bin/python experiment/e5_tulu_assistant_recovery_qwen36.py \
  --teacher-only
```

The teacher cache is memory-mapped and flushed every eight examples. If the
process stops, run the same command again without
`--refresh-teacher-cache`; it resumes from `teacher_cache/completed.npy`.

After the cache reaches 1088/1088, train and evaluate:

```bash
.venv/bin/python experiment/e5_tulu_assistant_recovery_qwen36.py
```

Use `--dataset`, `--official`, `--ptq`, and `--out` when the checkpoints or
dataset are stored elsewhere. Generated datasets, caches, and adapters are
gitignored; only source, manifests, and compact reports belong in Git.

The 4096-token prefix uses chunked causal cache prefill. Backpropagation is
restricted to the selected assistant window, preventing the full 4096-token
activation graph and a `4096 × vocabulary` logit tensor from residing in
memory.
