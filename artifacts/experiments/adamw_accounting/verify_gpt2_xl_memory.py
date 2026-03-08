"""Verify GPT-2 XL AdamW memory accounting for writeup.md part 4.3(b)."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the AdamW training-memory formula for GPT-2 XL.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size to evaluate directly.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    vocab_size = 50_257
    context_length = 1_024
    num_layers = 48
    d_model = 1_600
    num_heads = 25

    # Parameter count from part (a):
    # P = 2VD + L(16D^2 + 2D) + D
    parameter_count = (
        2 * vocab_size * d_model
        + num_layers * (16 * d_model * d_model + 2 * d_model)
        + d_model
    )

    parameter_bytes = 4 * parameter_count
    gradient_bytes = parameter_bytes
    optimizer_state_bytes = 2 * parameter_bytes
    constant_bytes = parameter_bytes + gradient_bytes + optimizer_state_bytes

    # Activation count from part (a):
    # A = L(16BTD + 2BHT^2) + BTD + 2BTV
    activation_elements_per_batch = (
        num_layers
        * (
            16 * context_length * d_model
            + 2 * num_heads * context_length * context_length
        )
        + context_length * d_model
        + 2 * context_length * vocab_size
    )
    activation_bytes_per_batch = 4 * activation_elements_per_batch

    decimal_limit_bytes = 80_000_000_000
    binary_limit_bytes = 80 * 1024**3

    max_batch_decimal = (decimal_limit_bytes - constant_bytes) // activation_bytes_per_batch
    max_batch_binary = (binary_limit_bytes - constant_bytes) // activation_bytes_per_batch

    print("GPT-2 XL AdamW memory accounting")
    print(f"parameter_count={parameter_count}")
    print(f"parameter_bytes={parameter_bytes}")
    print(f"gradient_bytes={gradient_bytes}")
    print(f"optimizer_state_bytes={optimizer_state_bytes}")
    print(f"constant_bytes={constant_bytes}")
    print(f"activation_elements_per_batch={activation_elements_per_batch}")
    print(f"activation_bytes_per_batch={activation_bytes_per_batch}")
    print(
        "total_memory_bytes(batch_size)="
        f"{activation_bytes_per_batch} * batch_size + {constant_bytes}"
    )
    print(f"max_batch_size_under_80GB_decimal={max_batch_decimal}")
    print(f"max_batch_size_under_80GiB_binary={max_batch_binary}")

    if args.batch_size is not None:
        batch_size = args.batch_size
        total_bytes = activation_bytes_per_batch * batch_size + constant_bytes
        print(f"total_bytes_for_batch_size_{batch_size}={total_bytes}")
        print(f"total_gb_for_batch_size_{batch_size}={total_bytes / 1e9:.6f}")
        print(f"total_gib_for_batch_size_{batch_size}={total_bytes / (1024 ** 3):.6f}")


if __name__ == "__main__":
    main(parse_args())
