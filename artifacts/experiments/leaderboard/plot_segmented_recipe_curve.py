from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass
class StageSpec:
    label: str
    metrics_path: Path
    start_step: int | None
    end_step: int | None


@dataclass
class StageSeries:
    label: str
    points: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a multi-stage training recipe as one cumulative learning curve "
            "with stage-specific colors."
        ),
    )
    parser.add_argument(
        "--stage",
        action="append",
        required=True,
        metavar="LABEL=METRICS_JSONL",
        help="Stage label and metrics path.",
    )
    parser.add_argument(
        "--stage-window",
        action="append",
        default=[],
        metavar="START:END",
        help=(
            "Optional inclusive step window for each stage, aligned with "
            "--stage order. Use an empty side to leave it unbounded, e.g. "
            "':12400' or '12600:'."
        ),
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
        help="Optional shared title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI.",
    )
    return parser.parse_args()


def parse_stage_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Expected LABEL=METRICS_JSONL format, got {raw!r}.")
    label, raw_path = raw.split("=", maxsplit=1)
    path = Path(raw_path)
    if path.is_dir():
        path = path / "metrics.jsonl"
    return label, path


def parse_stage_window(raw: str) -> tuple[int | None, int | None]:
    if ":" not in raw:
        raise ValueError(
            f"Expected START:END format for --stage-window, got {raw!r}."
        )
    raw_start, raw_end = raw.split(":", maxsplit=1)
    start_step = int(raw_start) if raw_start else None
    end_step = int(raw_end) if raw_end else None
    return start_step, end_step


def build_stage_specs(args: argparse.Namespace) -> list[StageSpec]:
    stage_count = len(args.stage)
    stage_windows = [(None, None)] * stage_count
    if args.stage_window:
        if len(args.stage_window) != stage_count:
            raise ValueError(
                "If provided, --stage-window must appear exactly once per --stage."
            )
        stage_windows = [parse_stage_window(raw) for raw in args.stage_window]

    specs: list[StageSpec] = []
    for idx, raw_stage in enumerate(args.stage):
        label, metrics_path = parse_stage_spec(raw_stage)
        start_step, end_step = stage_windows[idx]
        specs.append(
            StageSpec(
                label=label,
                metrics_path=metrics_path,
                start_step=start_step,
                end_step=end_step,
            )
        )
    return specs


def load_points(
    metrics_path: Path,
    *,
    split: str,
    metric: str,
    start_step: int | None,
    end_step: int | None,
) -> list[dict]:
    points: list[dict] = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split") != split:
                continue
            step = int(row["step"])
            if start_step is not None and step < start_step:
                continue
            if end_step is not None and step > end_step:
                continue
            points.append(
                {
                    "step": step,
                    "wallclock_seconds": float(row["wallclock_seconds"]),
                    metric: float(row[metric]),
                }
            )

    if not points:
        raise ValueError(
            f"No rows for split={split!r} found in {metrics_path} after filtering."
        )

    points.sort(key=lambda point: point["step"])
    return points


def stitch_stages(
    specs: list[StageSpec],
    *,
    split: str,
    metric: str,
) -> list[StageSeries]:
    stitched: list[StageSeries] = []
    cumulative_wallclock_offset = 0.0
    previous_final_point: dict | None = None

    for spec in specs:
        raw_points = load_points(
            spec.metrics_path,
            split=split,
            metric=metric,
            start_step=spec.start_step,
            end_step=spec.end_step,
        )
        stage_points: list[dict] = []
        if previous_final_point is not None:
            # Add a synthetic anchor at the stage boundary so the new segment
            # visibly starts from the checkpoint that it resumed from.
            stage_points.append(
                {
                    "step": previous_final_point["step"],
                    "wallclock_seconds": 0.0,
                    metric: previous_final_point[metric],
                    "cumulative_wallclock_seconds": cumulative_wallclock_offset,
                }
            )
        for point in raw_points:
            point = dict(point)
            point["cumulative_wallclock_seconds"] = (
                cumulative_wallclock_offset + point["wallclock_seconds"]
            )
            stage_points.append(point)
        cumulative_wallclock_offset = stage_points[-1]["cumulative_wallclock_seconds"]
        previous_final_point = stage_points[-1]
        stitched.append(StageSeries(label=spec.label, points=stage_points))

    return stitched


def add_stage_boundaries(ax: plt.Axes, stages: list[StageSeries], x_key: str) -> None:
    ymin, ymax = ax.get_ylim()
    text_y = ymax - 0.03 * (ymax - ymin)
    for idx in range(1, len(stages)):
        boundary_x = stages[idx - 1].points[-1][x_key]
        ax.axvline(
            boundary_x,
            linestyle="--",
            linewidth=1.2,
            color="gray",
            alpha=0.55,
        )
        ax.text(
            boundary_x,
            text_y,
            f"Stage {idx + 1}",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="gray",
        )


def plot_stitched_curves(
    *,
    stages: list[StageSeries],
    metric: str,
    x_key: str,
    output_path: Path,
    title: str | None,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    cmap = plt.get_cmap("tab10")

    for idx, stage in enumerate(stages):
        xs = [point[x_key] for point in stage.points]
        ys = [point[metric] for point in stage.points]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2,
            markersize=4,
            color=cmap(idx % 10),
            label=stage.label,
        )

    if title:
        ax.set_title(title)
    ax.set_ylabel(metric.replace("_", " ").title())
    if x_key == "step":
        ax.set_xlabel("Training Step")
    elif x_key == "cumulative_wallclock_seconds":
        ax.set_xlabel("Cumulative Wallclock Time (seconds)")
    else:
        ax.set_xlabel(x_key.replace("_", " ").title())
    ax.grid(alpha=0.3)
    add_stage_boundaries(ax, stages, x_key)
    ax.legend(title="Training stage")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_summary(
    *,
    stages: list[StageSeries],
    metric: str,
    split: str,
    output_path: Path,
) -> None:
    flattened = [point for stage in stages for point in stage.points]
    best_point = min(flattened, key=lambda point: point[metric])

    lines = [
        f"# Segmented recipe summary ({split} {metric})",
        "",
        (
            f"Overall best {metric}: `{best_point[metric]:.6f}` "
            f"at step `{best_point['step']}` and cumulative wallclock "
            f"`{best_point['cumulative_wallclock_seconds']:.2f}s`."
        ),
        "",
        "| Stage | Step range | Final value | Best value | Best step | Final cumulative wallclock (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for stage in stages:
        final_point = stage.points[-1]
        stage_best = min(stage.points, key=lambda point: point[metric])
        step_range = f"{stage.points[0]['step']} -> {stage.points[-1]['step']}"
        lines.append(
            "| "
            f"{stage.label} | "
            f"{step_range} | "
            f"{final_point[metric]:.6f} | "
            f"{stage_best[metric]:.6f} | "
            f"{stage_best['step']} | "
            f"{final_point['cumulative_wallclock_seconds']:.2f} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = build_stage_specs(args)
    stages = stitch_stages(specs, split=args.split, metric=args.metric)

    plot_stitched_curves(
        stages=stages,
        metric=args.metric,
        x_key="step",
        output_path=args.output_dir / f"{args.split}_{args.metric}_vs_step.png",
        title=args.title,
        dpi=args.dpi,
    )
    plot_stitched_curves(
        stages=stages,
        metric=args.metric,
        x_key="cumulative_wallclock_seconds",
        output_path=args.output_dir / f"{args.split}_{args.metric}_vs_wallclock.png",
        title=args.title,
        dpi=args.dpi,
    )
    write_summary(
        stages=stages,
        metric=args.metric,
        split=args.split,
        output_path=args.output_dir / f"{args.split}_{args.metric}_summary.md",
    )


if __name__ == "__main__":
    main()
