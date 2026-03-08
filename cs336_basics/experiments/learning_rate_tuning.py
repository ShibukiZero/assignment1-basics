from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class LearningRateRun:
    """Serializable result for one learning-rate sweep run."""

    learning_rate: float
    steps: int
    losses: list[float]
    initial_loss: float
    final_loss: float


class SGD(torch.optim.Optimizer):
    """
    Handout-style SGD with learning rate decay:
        theta_{t+1} = theta_t - lr / sqrt(t + 1) * grad
    """

    def __init__(self, params, lr: float = 1e-3) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], torch.Tensor] | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= (lr / math.sqrt(t + 1)) * grad
                state["t"] = t + 1
        return loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Assignment 1 Chapter 4 learning-rate tuning toy SGD experiment.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        action="append",
        dest="learning_rates",
        help="Learning rate to evaluate. Pass multiple times. Defaults to 1e1, 1e2, 1e3.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of optimization steps per learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to reset the toy setup before each run.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to serialize the full sweep report as JSON.",
    )
    return parser.parse_args()


def run_one_learning_rate(
    learning_rate: float,
    steps: int,
    seed: int,
) -> LearningRateRun:
    torch.manual_seed(seed)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    optimizer = SGD([weights], lr=learning_rate)

    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = (weights**2).mean()
        losses.append(float(loss.detach().cpu().item()))
        loss.backward()
        optimizer.step()

    return LearningRateRun(
        learning_rate=learning_rate,
        steps=steps,
        losses=losses,
        initial_loss=losses[0],
        final_loss=losses[-1],
    )


def main() -> None:
    args = parse_args()
    learning_rates = args.learning_rates or [1e1, 1e2, 1e3]
    runs = [
        run_one_learning_rate(
            learning_rate=learning_rate,
            steps=args.steps,
            seed=args.seed,
        )
        for learning_rate in learning_rates
    ]

    for run in runs:
        losses_str = ", ".join(f"{loss:.6g}" for loss in run.losses)
        print(f"lr={run.learning_rate:.0e}")
        print(f"losses=[{losses_str}]")
        print(f"initial_loss={run.initial_loss:.6g} final_loss={run.final_loss:.6g}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "steps": args.steps,
                    "runs": [asdict(run) for run in runs],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved report to: {args.output_json}")


if __name__ == "__main__":
    main()
