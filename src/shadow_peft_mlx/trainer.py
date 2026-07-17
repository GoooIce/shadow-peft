from __future__ import annotations

from collections.abc import Iterable

import mlx.core as mx
import mlx.nn as nn
from mlx import optimizers


def loss_fn(model: nn.Module, input_ids: mx.array, labels: mx.array) -> tuple[mx.array, mx.array]:
    """Combined base+shadow loss; returns (loss, n_supervised_tokens)."""
    out = model(input_ids, labels=labels)
    if out.loss is None:
        raise RuntimeError("Task wrapper returned loss=None; labels are required for training.")
    n_tokens = (labels[:, 1:] != -100).sum()
    return out.loss, n_tokens


def train(
    model: nn.Module,
    dataset: Iterable[tuple[mx.array, mx.array]],
    *,
    lr: float = 1e-4,
    epochs: int = 1,
    log_every: int = 10,
    optimizer=None,
) -> list[tuple[int, float, int]]:
    """
    Minimal training loop for Shadow task wrappers (replaces HF Trainer).

    Parameters
    ----------
    model:
        A Shadow task wrapper (e.g. `ShadowForCausalLM`) holding a `ShadowPeftModel`.
        Only unfrozen (trainable) parameters receive gradients — typically the shadow
        backbone, injection/update adapters and any heads opted into via
        `ShadowConfig.modules_to_save`.
    dataset:
        Iterable of `(input_ids, labels)` `mx.array` pairs, batch dim first.
        NOTE: mlx-lm models do not consume attention masks — batches are assumed to be
        padding-free. Mask positions you want to ignore with `labels = -100`.
    lr / optimizer:
        Learning rate for the default `AdamW`, or pass a custom `mlx.optimizers` optimizer.
    log_every:
        Print progress every N steps (0 disables printing).

    Returns
    -------
    list of (step, loss, n_tokens) tuples.
    """
    model.train()  # enable dropout
    opt = optimizer if optimizer is not None else optimizers.AdamW(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    history: list[tuple[int, float, int]] = []
    step = 0
    for _ in range(epochs):
        for input_ids, labels in dataset:
            (loss, ntok), grads = loss_and_grad(model, input_ids, labels)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss, ntok)
            step += 1
            entry = (step, float(loss), int(ntok))
            history.append(entry)
            if log_every and step % log_every == 0:
                print(f"step {step}: loss={entry[1]:.4f} tokens={entry[2]}")
    model.eval()
    return history
