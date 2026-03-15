from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_PLOT_CONFIG = {
    "comparisons": [
        {
            "name": "pre_norm_ablation",
            "title": "Pre-norm vs post-norm on TinyStories",
            "baseline_label": "pre-norm baseline",
            "baseline_run_name": "tinystories_ablation_baseline_bs128_5k",
            "ablation_label": "post-norm",
            "ablation_run_name": "tinystories_ablation_post_norm_bs128_5k",
        },
        {
            "name": "no_pos_emb",
            "title": "RoPE vs NoPE on TinyStories",
            "baseline_label": "RoPE baseline",
            "baseline_run_name": "tinystories_ablation_baseline_bs128_5k",
            "ablation_label": "NoPE",
            "ablation_run_name": "tinystories_ablation_nope_bs128_5k",
        },
        {
            "name": "swiglu_ablation",
            "title": "SwiGLU vs SiLU on TinyStories",
            "baseline_label": "SwiGLU baseline",
            "baseline_run_name": "tinystories_ablation_baseline_bs128_5k",
            "ablation_label": "SiLU",
            "ablation_run_name": "tinystories_ablation_silu_bs128_5k",
        },
    ]
}


def default_config_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "ablation_suite_plots_bs128_5k.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate baseline-vs-ablation learning curves for the Chapter 7 ablation suite.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="JSON config containing the comparison list.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path(".agents/logs"),
        help="Directory containing per-run metrics.jsonl files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory to place comparison figure subdirectories into.",
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


def load_plot_config(path: Path) -> dict:
    if not path.exists():
        return DEFAULT_PLOT_CONFIG
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = load_plot_config(args.config)
    plot_script = Path(__file__).with_name("plot_learning_rate_curves.py")

    for comparison in config["comparisons"]:
        output_dir = args.output_root / comparison["name"]
        baseline_metrics = args.log_root / comparison["baseline_run_name"] / "metrics.jsonl"
        ablation_metrics = args.log_root / comparison["ablation_run_name"] / "metrics.jsonl"
        command = [
            sys.executable,
            str(plot_script),
            "--run",
            f"{comparison['baseline_label']}={baseline_metrics}",
            "--run",
            f"{comparison['ablation_label']}={ablation_metrics}",
            "--output-dir",
            str(output_dir),
            "--split",
            args.split,
            "--metric",
            args.metric,
            "--title",
            comparison["title"],
            "--dpi",
            str(args.dpi),
        ]
        print(" ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
