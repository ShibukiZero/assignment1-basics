from __future__ import annotations

from collections.abc import Callable, Iterable
import math

import torch


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Cosine learning-rate schedule with linear warmup.

    Expected behavior:
    - if `it < warmup_iters`, linearly warm up from `0` to `max_learning_rate`
    - if `warmup_iters <= it <= cosine_cycle_iters`, cosine-decay from
      `max_learning_rate` to `min_learning_rate`
    - if `it > cosine_cycle_iters`, stay at `min_learning_rate`

    Key checkpoints:
    - at `it = 0`, the warmup branch should return `0`
    - at `it = warmup_iters`, the cosine branch should return `max_learning_rate`
    - at `it = cosine_cycle_iters`, the cosine branch should return `min_learning_rate`
    """
    if warmup_iters < 0:
        raise ValueError(f"Expected warmup_iters >= 0, got {warmup_iters}.")
    if cosine_cycle_iters < warmup_iters:
        raise ValueError(
            "Expected cosine_cycle_iters >= warmup_iters, "
            f"got cosine_cycle_iters={cosine_cycle_iters}, warmup_iters={warmup_iters}."
        )

    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate

    if it <= cosine_cycle_iters:
        if cosine_cycle_iters == warmup_iters:
            return max_learning_rate

        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (
            max_learning_rate - min_learning_rate
        )

    return min_learning_rate


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer for Assignment 1."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        if not 0 <= beta1 < 1:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0 <= beta2 < 1:
            raise ValueError(f"Invalid beta2: {beta2}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                t = state["t"] + 1
                state["t"] = t

                m = beta1 * state["m"] + (1 - beta1) * grad
                v = beta2 * state["v"] + (1 - beta2) * grad.square()
                state["m"] = m
                state["v"] = v

                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # Apply decoupled weight decay to the pre-update parameters.
                p.data -= lr * weight_decay * p.data
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)

        return loss
