from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_SWEEP_CONFIG = {
    "learning_rates": [1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3]
}


def default_config_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "sweep_configs"
        / "layer_norm_ablation_coarse_bs128.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or print the LayerNorm-ablation learning-rate sweep.",
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
        default=default_config_path(),
        help="JSON config containing the learning-rate grid.",
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="*",
        default=None,
        help="Optional subset of learning rates to run from the config.",
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
    parser.add_argument(
        "--enable-tensorboard",
        action="store_true",
        help="Write TensorBoard event files for sweep runs.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--seed", type=int, default=42)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--disable-checkpoints",
        action="store_true",
        help="Skip writing best/latest checkpoint files during sweep runs.",
    )
    checkpoint_group.add_argument(
        "--enable-checkpoints",
        action="store_true",
        help="Write best/latest checkpoint files during sweep runs.",
    )

    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument(
        "--position-encoding",
        choices=["rope", "none"],
        default="rope",
        help="Position encoding used during the ablation sweep.",
    )
    parser.add_argument(
        "--ffn-variant",
        choices=["swiglu", "silu"],
        default="swiglu",
        help="Feed-forward variant used during the ablation sweep.",
    )

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--cosine-cycle-iters", type=int, default=10000)
    parser.add_argument(
        "--min-lr-scale",
        type=float,
        default=0.1,
        help="Set min_learning_rate = learning_rate * min_lr_scale.",
    )

    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="tinystories_no_norm_bs128_lrscan",
        help="Prefix for generated run names.",
    )
    return parser.parse_args()


def load_sweep_config(path: Path) -> dict:
    if not path.exists():
        return DEFAULT_SWEEP_CONFIG
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_lr_tag(lr: float) -> str:
    raw = f"{lr:.1e}"
    return raw.replace("+", "").replace(".", "p")


def selected_learning_rates(args: argparse.Namespace, config: dict) -> list[float]:
    configured = [float(value) for value in config["learning_rates"]]
    if args.learning_rates is None:
        return configured
    requested = list(dict.fromkeys(args.learning_rates))
    unknown = [lr for lr in requested if lr not in configured]
    if unknown:
        raise ValueError(
            f"Unsupported learning rates {unknown}. Known values: {configured}"
        )
    return requested


def build_command(args: argparse.Namespace, *, learning_rate: float) -> list[str]:
    min_learning_rate = learning_rate * args.min_lr_scale
    lr_tag = format_lr_tag(learning_rate)
    run_name = f"{args.run_prefix}_lr{lr_tag}"
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
        "--enable-tensorboard" if args.enable_tensorboard else "",
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
        "--norm-style",
        "none",
        "--disable-final-norm",
        "--position-encoding",
        args.position_encoding,
        "--ffn-variant",
        args.ffn_variant,
        "--batch-size",
        str(args.batch_size),
        "--max-steps",
        str(args.max_steps),
        "--eval-interval",
        str(args.eval_interval),
        "--eval-batches",
        str(args.eval_batches),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--disable-checkpoints" if args.disable_checkpoints or not args.enable_checkpoints else "",
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


def compact_command(parts: list[str]) -> list[str]:
    return [part for part in parts if part]


def shell_quote(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in compact_command(parts))


def main() -> None:
    args = parse_args()
    config = load_sweep_config(args.config)
    learning_rates = selected_learning_rates(args, config)

    commands = [
        build_command(args, learning_rate=learning_rate)
        for learning_rate in learning_rates
    ]

    if args.mode == "print":
        for command in commands:
            print(shell_quote(command))
        return

    for index, command in enumerate(commands, start=1):
        normalized = compact_command(command)
        print(f"[{index}/{len(commands)}] {' '.join(normalized)}")
        subprocess.run(normalized, check=True)


if __name__ == "__main__":
    main()
