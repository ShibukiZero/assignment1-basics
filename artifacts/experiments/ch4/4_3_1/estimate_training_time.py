"""Estimate AdamW training time for GPT-2 XL under an MFU assumption."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate GPT-2 XL AdamW training time from FLOP accounting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1_024,
        help="Batch size used for the training-step FLOP estimate.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=400_000,
        help="Number of training steps.",
    )
    parser.add_argument(
        "--mfu",
        type=float,
        default=0.5,
        help="Model FLOPs utilization as a fraction of peak throughput.",
    )
    parser.add_argument(
        "--peak-flops",
        type=float,
        default=19.5e12,
        help="Hardware peak throughput in FLOP/s (default: A100 FP32 peak).",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    vocab_size = 50_257
    context_length = 1_024
    num_layers = 48
    d_model = 1_600
    num_heads = 25
    batch_size = args.batch_size

    parameter_count = (
        2 * vocab_size * d_model
        + num_layers * (16 * d_model * d_model + 2 * d_model)
        + d_model
    )

    # Reuse the writeup accounting:
    # F_forward = L(32BTD^2 + 4BT^2D) + 2BTDV
    forward_flops = (
        num_layers
        * (
            32 * batch_size * context_length * d_model * d_model
            + 4 * batch_size * context_length * context_length * d_model
        )
        + 2 * batch_size * context_length * d_model * vocab_size
    )
    backward_flops = 2 * forward_flops
    optimizer_flops = 15 * parameter_count
    step_flops = forward_flops + backward_flops + optimizer_flops

    effective_flops_per_second = args.mfu * args.peak_flops
    total_flops = args.steps * step_flops
    total_seconds = total_flops / effective_flops_per_second
    total_days = total_seconds / (60 * 60 * 24)

    print("GPT-2 XL AdamW training-time estimate")
    print(f"batch_size={batch_size}")
    print(f"steps={args.steps}")
    print(f"mfu={args.mfu}")
    print(f"peak_flops_per_second={args.peak_flops:.6f}")
    print(f"effective_flops_per_second={effective_flops_per_second:.6f}")
    print(f"parameter_count={parameter_count}")
    print(f"forward_flops={forward_flops}")
    print(f"backward_flops={backward_flops}")
    print(f"optimizer_flops={optimizer_flops}")
    print(f"step_flops={step_flops}")
    print(f"total_flops={total_flops}")
    print(f"total_seconds={total_seconds:.6f}")
    print(f"total_days={total_days:.6f}")


if __name__ == "__main__":
    main(parse_args())
