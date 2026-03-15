from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the best learning curve for each batch size.",
    )
    parser.add_argument(
        "--best-runs-json",
        type=Path,
        required=True,
        help="Path to batch_size_best_runs.json produced by summarize_batch_size_sweep.py",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=None,
        help="Fallback log root containing per-run metrics.jsonl files when artifact paths are unavailable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write figures into.",
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


def load_best_runs(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No batch-size best-run rows found in {path}.")
    return rows


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
        raise ValueError(f"No rows for split={split!r} found in {metrics_path}.")
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
    ax.set_xlabel("Training Step" if x_key == "step" else "Wallclock Time (seconds)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_summary(
    *,
    best_rows: list[dict],
    output_path: Path,
) -> None:
    lines = [
        "# Batch-size curve summary",
        "",
        "| Batch size | Best LR | Best val loss | Best run |",
        "|---:|---:|---:|---|",
    ]
    for row in best_rows:
        lines.append(
            f"| {row['batch_size']} | {row['best_learning_rate']:.6g} | "
            f"{row['best_val_loss']:.6f} | {row['best_run_name']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_rows = sorted(
        load_best_runs(args.best_runs_json),
        key=lambda row: int(row["batch_size"]),
    )

    runs: list[tuple[str, list[dict]]] = []
    for row in best_rows:
        run_name = row["best_run_name"]
        raw_metrics_path = row.get("artifact_metrics_path")
        if raw_metrics_path:
            metrics_path = Path(raw_metrics_path)
        elif args.log_root is not None:
            metrics_path = args.log_root / run_name / "metrics.jsonl"
        else:
            raise ValueError(
                f"Missing artifact_metrics_path for {run_name} and no --log-root fallback provided."
            )
        points = load_points(metrics_path, split=args.split, metric=args.metric)
        label = f"bs={row['batch_size']} (lr={row['best_learning_rate']:.3g})"
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
        best_rows=best_rows,
        output_path=args.output_dir / "batch_size_best_curve_summary.md",
    )


if __name__ == "__main__":
    main()
