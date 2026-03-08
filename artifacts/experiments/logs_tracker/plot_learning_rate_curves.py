from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot learning curves from Chapter 7 metrics.jsonl files.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=METRICS_JSONL",
        help="Curve label and metrics path, e.g. 2e-3=.agents/logs/foo/metrics.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the figures into.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which split to plot.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="loss",
        choices=["loss", "perplexity"],
        help="Which metric to plot.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title shared by both figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI.",
    )
    return parser.parse_args()


def parse_run_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(
            f"Expected LABEL=METRICS_JSONL format, got {raw!r}."
        )
    label, raw_path = raw.split("=", maxsplit=1)
    return label, Path(raw_path)


def load_points(metrics_path: Path, *, split: str, metric: str) -> list[dict]:
    points: list[dict] = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split") != split:
                continue
            points.append(
                {
                    "step": int(row["step"]),
                    "wallclock_seconds": float(row["wallclock_seconds"]),
                    metric: float(row[metric]),
                }
            )

    if not points:
        raise ValueError(
            f"No rows for split={split!r} found in {metrics_path}."
        )
    return points


def plot_curves(
    *,
    runs: list[tuple[str, list[dict]]],
    metric: str,
    x_key: str,
    output_path: Path,
    title: str | None,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, points in runs:
        xs = [point[x_key] for point in points]
        ys = [point[metric] for point in points]
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=4, label=label)

    if title:
        ax.set_title(title)
    ax.set_ylabel(metric.replace("_", " ").title())
    if x_key == "step":
        ax.set_xlabel("Training Step")
    elif x_key == "wallclock_seconds":
        ax.set_xlabel("Wallclock Time (seconds)")
    else:
        ax.set_xlabel(x_key.replace("_", " ").title())
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
        f"# Learning curve summary ({split} {metric})",
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

    runs: list[tuple[str, list[dict]]] = []
    for raw_run in args.run:
        label, metrics_path = parse_run_spec(raw_run)
        points = load_points(metrics_path, split=args.split, metric=args.metric)
        runs.append((label, points))

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
