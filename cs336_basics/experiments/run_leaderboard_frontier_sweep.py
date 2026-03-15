from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_FRONTIER_CONFIG = {
    "experiments": [
        {
            "name": "baseline_l4_d512",
            "run_name": "owt_frontier_bs32_l4_d512_lr1p5e-03",
            "d_model": 512,
            "num_layers": 4,
            "num_heads": 16,
            "d_ff": 1344,
        },
        {
            "name": "depth_l6_d512",
            "run_name": "owt_frontier_bs32_l6_d512_lr1p5e-03",
            "d_model": 512,
            "num_layers": 6,
            "num_heads": 16,
            "d_ff": 1344,
        },
        {
            "name": "depth_l8_d512",
            "run_name": "owt_frontier_bs32_l8_d512_lr1p5e-03",
            "d_model": 512,
            "num_layers": 8,
            "num_heads": 16,
            "d_ff": 1344,
        },
        {
            "name": "width_l4_d768",
            "run_name": "owt_frontier_bs32_l4_d768_lr1p5e-03",
            "d_model": 768,
            "num_layers": 4,
            "num_heads": 24,
            "d_ff": 2048,
        },
        {
            "name": "width_l4_d1024",
            "run_name": "owt_frontier_bs32_l4_d1024_lr1p5e-03",
            "d_model": 1024,
            "num_layers": 4,
            "num_heads": 32,
            "d_ff": 2752,
        },
        {
            "name": "tradeoff_l6_d768",
            "run_name": "owt_frontier_bs32_l6_d768_lr1p5e-03",
            "d_model": 768,
            "num_layers": 6,
            "num_heads": 24,
            "d_ff": 2048,
        },
        {
            "name": "tradeoff_l8_d768",
            "run_name": "owt_frontier_bs32_l8_d768_lr1p5e-03",
            "d_model": 768,
            "num_layers": 8,
            "num_heads": 24,
            "d_ff": 2048,
        },
        {
            "name": "tradeoff_l6_d1024",
            "run_name": "owt_frontier_bs32_l6_d1024_lr1p5e-03",
            "d_model": 1024,
            "num_layers": 6,
            "num_heads": 32,
            "d_ff": 2752,
        },
    ]
}


def default_config_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "sweep_configs"
        / "leaderboard_frontier_bs32.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or print a width-depth frontier sweep for the Assignment 1 leaderboard."
        ),
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
        help="JSON config containing experiment definitions.",
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
        default=Path("/root/autodl-tmp/tokenizer_exp_encoded/owt_train_ids.npy"),
    )
    parser.add_argument(
        "--val-npy",
        type=Path,
        default=Path("/root/autodl-tmp/tokenizer_exp_encoded/owt_valid_ids.npy"),
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
        help="Write TensorBoard event files for frontier runs.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable-checkpoints",
        action="store_true",
        help="Skip writing best/latest checkpoint files during frontier pilots.",
    )

    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--norm-style", choices=["pre", "post", "none"], default="pre")
    parser.add_argument(
        "--position-encoding",
        choices=["rope", "none"],
        default="rope",
    )
    parser.add_argument(
        "--ffn-variant",
        choices=["swiglu", "silu"],
        default="swiglu",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--checkpoint-interval", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1.5e-3)
    parser.add_argument("--min-learning-rate", type=float, default=None)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--cosine-cycle-iters", type=int, default=400)

    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args()


def load_frontier_config(path: Path) -> dict:
    if not path.exists():
        return DEFAULT_FRONTIER_CONFIG
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


def experiment_value(experiment: dict, key: str, default):
    return experiment.get(key, default)


def build_command(args: argparse.Namespace, *, experiment: dict) -> list[str]:
    run_name = experiment["run_name"]
    output_dir = args.training_root / run_name
    log_dir = args.log_root / run_name
    learning_rate = float(experiment_value(experiment, "learning_rate", args.learning_rate))
    min_learning_rate = experiment_value(
        experiment,
        "min_learning_rate",
        args.min_learning_rate if args.min_learning_rate is not None else learning_rate / 10.0,
    )
    use_final_norm = bool(experiment_value(experiment, "use_final_norm", True))
    resume_from = experiment_value(experiment, "resume_from", None)

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
        str(experiment_value(experiment, "context_length", args.context_length)),
        "--d-model",
        str(experiment_value(experiment, "d_model", 512)),
        "--num-layers",
        str(experiment_value(experiment, "num_layers", 4)),
        "--num-heads",
        str(experiment_value(experiment, "num_heads", 16)),
        "--d-ff",
        str(experiment_value(experiment, "d_ff", 1344)),
        "--rope-theta",
        str(args.rope_theta),
        "--norm-style",
        str(experiment_value(experiment, "norm_style", args.norm_style)),
        "--disable-final-norm" if not use_final_norm else "",
        "--position-encoding",
        str(experiment_value(experiment, "position_encoding", args.position_encoding)),
        "--ffn-variant",
        str(experiment_value(experiment, "ffn_variant", args.ffn_variant)),
        "--batch-size",
        str(experiment_value(experiment, "batch_size", args.batch_size)),
        "--max-steps",
        str(experiment_value(experiment, "max_steps", args.max_steps)),
        "--eval-interval",
        str(experiment_value(experiment, "eval_interval", args.eval_interval)),
        "--eval-batches",
        str(experiment_value(experiment, "eval_batches", args.eval_batches)),
        "--checkpoint-interval",
        str(experiment_value(experiment, "checkpoint_interval", args.checkpoint_interval)),
        "--disable-checkpoints" if args.disable_checkpoints else "",
        "--learning-rate",
        str(learning_rate),
        "--min-learning-rate",
        str(min_learning_rate),
        "--warmup-iters",
        str(experiment_value(experiment, "warmup_iters", args.warmup_iters)),
        "--cosine-cycle-iters",
        str(experiment_value(experiment, "cosine_cycle_iters", args.cosine_cycle_iters)),
        "--betas",
        str(args.beta1),
        str(args.beta2),
        "--eps",
        str(args.eps),
        "--weight-decay",
        str(args.weight_decay),
        "--grad-clip",
        str(args.grad_clip),
        "--resume-from",
        str(resume_from) if resume_from is not None else "",
    ]


def compact_command(parts: list[str]) -> list[str]:
    return [part for part in parts if part]


def shell_quote(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in compact_command(parts))


def main() -> None:
    args = parse_args()
    config = load_frontier_config(args.config)
    experiments = selected_experiments(args, config)
    commands = [build_command(args, experiment=experiment) for experiment in experiments]

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
