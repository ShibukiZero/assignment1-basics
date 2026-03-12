from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_SUITE_CONFIG = {
    "experiments": [
        {
            "name": "baseline",
            "run_name": "tinystories_ablation_baseline_bs128_5k",
            "norm_style": "pre",
            "position_encoding": "rope",
            "ffn_variant": "swiglu",
            "use_final_norm": True,
            "d_ff": 1344,
        },
        {
            "name": "post_norm",
            "run_name": "tinystories_ablation_post_norm_bs128_5k",
            "norm_style": "post",
            "position_encoding": "rope",
            "ffn_variant": "swiglu",
            "use_final_norm": True,
            "d_ff": 1344,
        },
        {
            "name": "nope",
            "run_name": "tinystories_ablation_nope_bs128_5k",
            "norm_style": "pre",
            "position_encoding": "none",
            "ffn_variant": "swiglu",
            "use_final_norm": True,
            "d_ff": 1344,
        },
        {
            "name": "silu",
            "run_name": "tinystories_ablation_silu_bs128_5k",
            "norm_style": "pre",
            "position_encoding": "rope",
            "ffn_variant": "silu",
            "use_final_norm": True,
            "d_ff": 2048,
        },
    ]
}


def default_config_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "sweep_configs"
        / "chapter7_arch_ablation_suite_bs128_5k.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or print the fixed-hyperparameter Chapter 7 ablation suite.",
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
        help="JSON config containing the ablation suite definition.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of experiment names to run from the config.",
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
        help="Write TensorBoard event files for suite runs.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--seed", type=int, default=42)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--disable-checkpoints",
        action="store_true",
        help="Skip writing best/latest checkpoint files during suite runs.",
    )
    checkpoint_group.add_argument(
        "--enable-checkpoints",
        action="store_true",
        help="Write best/latest checkpoint files during suite runs.",
    )

    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=4e-3)
    parser.add_argument("--min-learning-rate", type=float, default=4e-4)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--cosine-cycle-iters", type=int, default=10000)

    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args()


def load_suite_config(path: Path) -> dict:
    if not path.exists():
        return DEFAULT_SUITE_CONFIG
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def selected_experiments(args: argparse.Namespace, config: dict) -> list[dict]:
    configured = list(config["experiments"])
    if args.experiments is None:
        return configured

    available = {entry["name"]: entry for entry in configured}
    requested_names = list(dict.fromkeys(args.experiments))
    unknown = [name for name in requested_names if name not in available]
    if unknown:
        raise ValueError(
            f"Unsupported experiments {unknown}. Known experiments: {sorted(available)}"
        )
    return [available[name] for name in requested_names]


def build_command(args: argparse.Namespace, *, experiment: dict) -> list[str]:
    run_name = experiment["run_name"]
    output_dir = args.training_root / run_name
    log_dir = args.log_root / run_name
    d_ff = int(experiment.get("d_ff", 1344))
    use_final_norm = bool(experiment.get("use_final_norm", True))

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
        str(d_ff),
        "--rope-theta",
        str(args.rope_theta),
        "--norm-style",
        str(experiment.get("norm_style", "pre")),
        "--disable-final-norm" if not use_final_norm else "",
        "--position-encoding",
        str(experiment.get("position_encoding", "rope")),
        "--ffn-variant",
        str(experiment.get("ffn_variant", "swiglu")),
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
        str(args.learning_rate),
        "--min-learning-rate",
        str(args.min_learning_rate),
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
    config = load_suite_config(args.config)
    experiments = selected_experiments(args, config)
    commands = [
        build_command(args, experiment=experiment)
        for experiment in experiments
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
