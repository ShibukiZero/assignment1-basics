from __future__ import annotations

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
