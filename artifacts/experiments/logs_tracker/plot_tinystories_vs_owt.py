from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot TinyStories vs OpenWebText learning curves from two metrics.jsonl files.",
    )
    parser.add_argument(
        "--tinystories-metrics",
        type=Path,
        required=True,
        help="Path to the TinyStories metrics.jsonl file.",
    )
    parser.add_argument(
        "--owt-metrics",
        type=Path,
        default=Path(
            "artifacts/experiments/logs_tracker/results/owt_main_bs128_lr2p5e-03/run/metrics.jsonl"
        ),
        help="Path to the OpenWebText metrics.jsonl file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write comparison figures into.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which split to compare.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="loss",
        choices=["loss", "perplexity"],
        help="Which metric to compare.",
    )
    parser.add_argument(
        "--tinystories-label",
        type=str,
        default="TinyStories",
        help="Legend label for the TinyStories curve.",
    )
    parser.add_argument(
        "--owt-label",
        type=str,
        default="OpenWebText",
        help="Legend label for the OpenWebText curve.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="TinyStories vs OpenWebText learning curves",
        help="Shared title for both figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI.",
    )
    return parser.parse_args()


def load_points(metrics_path: Path, *, split: str, metric: str) -> list[dict]:
    points_by_step: dict[int, dict] = {}
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split") != split:
                continue
            step = int(row["step"])
            points_by_step[step] = {
                "step": step,
                "wallclock_seconds": float(row["wallclock_seconds"]),
                metric: float(row[metric]),
            }
    points = [points_by_step[step] for step in sorted(points_by_step)]
    if not points:
        raise ValueError(f"No rows for split={split!r} found in {metrics_path}.")
    return points


def plot_curves(
    *,
    runs: list[tuple[str, list[dict]]],
    metric: str,
    x_key: str,
    output_path: Path,
    title: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, points in runs:
        xs = [point[x_key] for point in points]
        ys = [point[metric] for point in points]
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=4, label=label)

    ax.set_title(title)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("Training Step" if x_key == "step" else "Wallclock Time (seconds)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_summary(
    *,
    runs: list[tuple[str, list[dict]]],
    metric: str,
    split: str,
    output_path: Path,
) -> None:
    lines = [
        f"# TinyStories vs OpenWebText summary ({split} {metric})",
        "",
        "| Label | Final step | Final value | Best value | Best step | Final wallclock (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, points in runs:
        final_point = points[-1]
        best_point = min(points, key=lambda point: point[metric])
        lines.append(
            "| "
            f"{label} | "
            f"{final_point['step']} | "
            f"{final_point[metric]:.6f} | "
            f"{best_point[metric]:.6f} | "
            f"{best_point['step']} | "
            f"{final_point['wallclock_seconds']:.2f} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        (
            args.tinystories_label,
            load_points(
                args.tinystories_metrics,
                split=args.split,
                metric=args.metric,
            ),
        ),
        (
            args.owt_label,
            load_points(
                args.owt_metrics,
                split=args.split,
                metric=args.metric,
            ),
        ),
    ]

    plot_curves(
        runs=runs,
        metric=args.metric,
        x_key="step",
        output_path=args.output_dir / f"{args.split}_{args.metric}_vs_step.png",
        title=args.title,
        dpi=args.dpi,
    )
    plot_curves(
        runs=runs,
        metric=args.metric,
        x_key="wallclock_seconds",
        output_path=args.output_dir / f"{args.split}_{args.metric}_vs_wallclock.png",
        title=args.title,
        dpi=args.dpi,
    )
    write_summary(
        runs=runs,
        metric=args.metric,
        split=args.split,
        output_path=args.output_dir / f"{args.split}_{args.metric}_summary.md",
    )


if __name__ == "__main__":
    main()
