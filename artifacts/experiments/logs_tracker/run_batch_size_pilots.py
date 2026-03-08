from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_SWEEP_CONFIG = {
    "batch_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 544],
    "learning_rates": {
        "1": [2e-4, 5e-4, 1e-3, 2e-3, 3e-3],
        "2": [2e-4, 5e-4, 1e-3, 2e-3, 3e-3],
        "4": [5e-4, 1e-3, 2e-3, 3e-3, 4e-3],
        "8": [5e-4, 1e-3, 2e-3, 3e-3, 4e-3],
        "16": [1e-3, 1.5e-3, 2e-3, 3e-3, 4e-3],
        "32": [1e-3, 1.5e-3, 2e-3, 2.4e-3, 3e-3],
        "64": [1e-3, 2e-3, 3e-3, 4e-3, 5e-3],
        "128": [1e-3, 2e-3, 4e-3, 6e-3, 8e-3],
        "256": [2e-3, 4e-3, 6e-3, 8e-3, 1e-2],
        "512": [2e-3, 4e-3, 6e-3, 8e-3, 1.2e-2],
        "544": [2e-3, 4e-3, 6e-3, 8e-3, 1.2e-2],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or print the Chapter 7 batch-size x learning-rate pilot sweep.",
    )
    parser.add_argument(
        "--mode",
        choices=["print", "run"],
        default="print",
        help="Print commands or run them sequentially.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON config overriding the default 2D sweep grid.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of batch sizes to run from the config.",
    )
    parser.add_argument(
        "--train-npy",
        type=Path,
        default=Path("/root/autodl-tmp/tokenizer_exp_encoded/tiny_train_ids.npy"),
    )
    parser.add_argument(
        "--val-npy",
        type=Path,
        default=Path("/root/autodl-tmp/tokenizer_exp_encoded/tiny_valid_ids.npy"),
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path("/root/autodl-tmp/training_runs"),
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path(".agents/logs"),
    )
    parser.add_argument(
        "--tensorboard-root",
        type=Path,
        default=Path("/root/tf-logs"),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--checkpoint-interval", type=int, default=400)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--cosine-cycle-iters", type=int, default=400)

    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args()


def load_sweep_config(path: Path | None) -> dict:
    if path is None:
        return DEFAULT_SWEEP_CONFIG
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_lr_tag(lr: float) -> str:
    raw = f"{lr:.1e}"
    return raw.replace("+", "").replace(".", "p")


def selected_batch_sizes(args: argparse.Namespace, config: dict) -> list[int]:
    configured = [int(batch_size) for batch_size in config["batch_sizes"]]
    if args.batch_sizes is None:
        return configured
    requested = list(dict.fromkeys(args.batch_sizes))
    unknown = [bs for bs in requested if bs not in configured]
    if unknown:
        raise ValueError(
            f"Unsupported batch sizes {unknown}. Known sizes: {configured}"
        )
    return requested


def learning_rates_for_batch_size(config: dict, batch_size: int) -> list[float]:
    raw = config["learning_rates"].get(str(batch_size))
    if raw is None:
        raise ValueError(f"Missing learning-rate list for batch_size={batch_size}")
    return [float(value) for value in raw]


def build_command(args: argparse.Namespace, *, batch_size: int, learning_rate: float) -> list[str]:
    min_learning_rate = learning_rate / 10.0
    lr_tag = format_lr_tag(learning_rate)
    run_name = f"tinystories_bs{batch_size}_lr{lr_tag}_pilot"
    output_dir = args.training_root / run_name
    log_dir = args.log_root / run_name

    return [
        sys.executable,
        "cs336_basics/experiments/train_transformer_lm.py",
        "--train-npy",
        str(args.train_npy),
        "--val-npy",
        str(args.val_npy),
        "--output-dir",
        str(output_dir),
        "--log-dir",
        str(log_dir),
        "--tensorboard-root",
        str(args.tensorboard_root),
        "--run-name",
        run_name,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--vocab-size",
        str(args.vocab_size),
        "--context-length",
        str(args.context_length),
        "--d-model",
        str(args.d_model),
        "--num-layers",
        str(args.num_layers),
        "--num-heads",
        str(args.num_heads),
        "--d-ff",
        str(args.d_ff),
        "--rope-theta",
        str(args.rope_theta),
        "--batch-size",
        str(batch_size),
        "--max-steps",
        str(args.max_steps),
        "--eval-interval",
        str(args.eval_interval),
        "--eval-batches",
        str(args.eval_batches),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--learning-rate",
        str(learning_rate),
        "--min-learning-rate",
        str(min_learning_rate),
        "--warmup-iters",
        str(args.warmup_iters),
        "--cosine-cycle-iters",
        str(args.cosine_cycle_iters),
        "--betas",
        str(args.beta1),
        str(args.beta2),
        "--eps",
        str(args.eps),
        "--weight-decay",
        str(args.weight_decay),
        "--grad-clip",
        str(args.grad_clip),
    ]


def shell_quote(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def main() -> None:
    args = parse_args()
    config = load_sweep_config(args.config)
    batch_sizes = selected_batch_sizes(args, config)

    commands: list[list[str]] = []
    for batch_size in batch_sizes:
        for learning_rate in learning_rates_for_batch_size(config, batch_size):
            commands.append(
                build_command(
                    args,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                )
            )

    if args.mode == "print":
        for command in commands:
            print(shell_quote(command))
        return

    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {' '.join(command)}")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
