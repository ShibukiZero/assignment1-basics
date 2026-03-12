from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate configured learning-rate figures from automated sweep runs.",
    )
    parser.add_argument(
        "--figure-config",
        type=Path,
        required=True,
        help="Path to the figure-set JSON configuration.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=None,
        help="Fallback root directory containing per-run logs.",
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=None,
        help="Optional learning_rate_runs.json produced by summarize_learning_rate_sweep.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory to place figure subdirectories into.",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="tinystories_lr_auto",
        help="Run prefix used by run_learning_rate_sweep.py",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="loss",
        choices=["loss", "perplexity"],
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
    )
    return parser.parse_args()


def format_lr_tag(lr: float) -> str:
    raw = f"{lr:.1e}"
    return raw.replace("+", "").replace(".", "p")


def run_name_for_lr(run_prefix: str, learning_rate: float) -> str:
    return f"{run_prefix}_lr{format_lr_tag(learning_rate)}"


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    with args.figure_config.open("r", encoding="utf-8") as f:
        config = json.load(f)
    manifest_by_lr: dict[float, Path] = {}
    if args.run_manifest is not None:
        with args.run_manifest.open("r", encoding="utf-8") as f:
            manifest_rows = json.load(f)
        for row in manifest_rows:
            artifact_metrics_path = row.get("artifact_metrics_path")
            if artifact_metrics_path:
                manifest_by_lr[float(row["learning_rate"])] = Path(artifact_metrics_path)

    plot_script = Path(__file__).with_name("plot_learning_rate_curves.py")

    for figure in config["figures"]:
        output_dir = args.output_root / figure["name"]
        command = [
            sys.executable,
            str(plot_script),
            "--output-dir",
            str(output_dir),
            "--split",
            args.split,
            "--metric",
            args.metric,
            "--title",
            figure["title"],
            "--dpi",
            str(args.dpi),
        ]

        for learning_rate in figure["learning_rates"]:
            lr_value = float(learning_rate)
            if lr_value in manifest_by_lr:
                metrics_path = manifest_by_lr[lr_value]
            elif args.log_root is not None:
                run_name = run_name_for_lr(args.run_prefix, lr_value)
                metrics_path = args.log_root / run_name / "metrics.jsonl"
            else:
                raise ValueError(
                    f"No artifact metrics path for learning_rate={lr_value} and no --log-root fallback was provided."
                )
            command.extend(
                [
                    "--run",
                    f"{learning_rate:.6g}={metrics_path}",
                ]
            )

        print(" ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
