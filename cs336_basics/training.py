from __future__ import annotations

from collections.abc import Iterable
from os import PathLike
from typing import BinaryIO, IO

import numpy as np
import torch
from einops import reduce
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(
    logits: Float[Tensor, "... vocab_size"],
    targets: Int[Tensor, "..."],
) -> Float[Tensor, ""]:
    """
    Compute average cross-entropy from raw logits.

    Shape contract:
    - `logits` has shape (..., vocab_size)
    - `targets` has shape (...) and must match the batch-like prefix of `logits`

    Important:
    - Do not apply softmax first.
    - The input is raw logits from the model.
    - The returned scalar is the mean over every batch-like position.
    """
    if logits.ndim < 1:
        raise ValueError("Expected logits to have at least 1 dimension.")
    if targets.shape != logits.shape[:-1]:
        raise ValueError(
            "Expected targets.shape to match logits.shape[:-1], "
            f"got targets.shape={targets.shape}, logits.shape={logits.shape}."
        )
    if targets.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"Expected targets dtype int32 or int64, got {targets.dtype}."
        )

    # Shape checkpoint:
    # - max_logits: (..., 1)
    # - stable_logits: (..., vocab_size)
    max_logits: Float[Tensor, "... 1"] = torch.amax(logits, dim=-1, keepdim=True)
    stable_logits: Float[Tensor, "... vocab_size"] = logits - max_logits

    exp_sum: Float[Tensor, "... 1"] = reduce(
        torch.exp(stable_logits),
        "... vocab_size -> ... 1",
        "sum",
    )
    logsumexp_logits: Float[Tensor, "..."] = max_logits.squeeze(-1) + torch.log(
        exp_sum.squeeze(-1)
    )

    target_logits: Float[Tensor, "..."] = torch.gather(
        logits,
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    per_example_loss: Float[Tensor, "..."] = logsumexp_logits - target_logits
    return per_example_loss.mean()


def perplexity(losses: Float[Tensor, "..."]) -> Float[Tensor, ""]:
    """
    Compute perplexity from cross-entropy losses.

    This follows the handout definition:
        perplexity = exp(mean(losses))

    The input may be either:
    - a tensor of per-token / per-position cross-entropy losses, or
    - an already-averaged scalar cross-entropy loss.
    """
    if losses.numel() == 0:
        raise ValueError("Expected at least one loss value to compute perplexity.")

    mean_loss: Float[Tensor, ""] = losses.mean()
    return torch.exp(mean_loss)


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    """
    Clip the combined gradient norm of all parameters to `max_l2_norm`.

    Important:
    - This is global gradient clipping, not per-parameter clipping.
    - Only parameters with `grad is not None` should participate.
    - Gradients must be modified in place.
    """
    if max_l2_norm < 0:
        raise ValueError(f"Expected max_l2_norm >= 0, got {max_l2_norm}.")

    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if len(grads) == 0:
        return

    total_norm: Float[Tensor, ""] = torch.sqrt(
        sum(grad.square().sum() for grad in grads)
    )
    if total_norm <= max_l2_norm:
        return

    clip_coef: Float[Tensor, ""] = max_l2_norm / (total_norm + eps)
    for grad in grads:
        grad.mul_(clip_coef)


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Int[Tensor, "batch_size context_length"], Int[Tensor, "batch_size context_length"]]:
    """
    Sample a batch of next-token-prediction training examples from a 1D token stream.

    Contract:
    - `dataset` is a 1D numpy integer array (or array-like such as `np.memmap`)
    - sample `batch_size` random starting positions
    - for each start `s`:
      - `x = dataset[s : s + context_length]`
      - `y = dataset[s + 1 : s + 1 + context_length]`
    - return both tensors on the requested device

    Performance note:
    - This interface is compatible with `np.memmap`, so the later training pipeline
      can use memory-mapped token arrays without changing the batching API.
    """
    if dataset.ndim != 1:
        raise ValueError(f"Expected a 1D dataset, got shape {dataset.shape}.")
    if batch_size <= 0:
        raise ValueError(f"Expected batch_size > 0, got {batch_size}.")
    if context_length <= 0:
        raise ValueError(f"Expected context_length > 0, got {context_length}.")
    if len(dataset) <= context_length:
        raise ValueError(
            "Expected len(dataset) > context_length so labels can be shifted by one, "
            f"got len(dataset)={len(dataset)}, context_length={context_length}."
        )

    num_possible_starts = len(dataset) - context_length
    start_indices = np.random.randint(num_possible_starts, size=batch_size)

    x_np = np.stack([dataset[s : s + context_length] for s in start_indices])
    y_np = np.stack([dataset[s + 1 : s + 1 + context_length] for s in start_indices])

    x = torch.tensor(x_np, device=device, dtype=torch.long)
    y = torch.tensor(y_np, device=device, dtype=torch.long)
    return x, y


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    Save model state, optimizer state, and iteration count to a checkpoint.

    Suggested checkpoint payload:
    {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    """
    if iteration < 0:
        raise ValueError(f"Expected iteration >= 0, got {iteration}.")

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load model state, optimizer state, and iteration count from a checkpoint.

    Expected steps:
    1. `checkpoint = torch.load(src)`
    2. `model.load_state_dict(checkpoint["model"])`
    3. `optimizer.load_state_dict(checkpoint["optimizer"])`
    4. return `checkpoint["iteration"]`
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
