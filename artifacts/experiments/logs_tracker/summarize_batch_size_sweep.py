from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


RUN_PATTERN = re.compile(
    r"^tinystories_bs(?P<batch_size>\d+)_lr(?P<lr_tag>[0-9ep\-]+)_pilot$"
)


@dataclass
class RunResult:
    run_name: str
    batch_size: int
    learning_rate: float
    best_val_loss: float
    best_val_step: int | None
    final_step: int
    total_wallclock_seconds: float
    summary_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the Chapter 7 batch-size x learning-rate sweep.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path(".agents/logs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def parse_lr_tag(raw: str) -> float:
    return float(raw.replace("p", "."))


def load_summary(run_dir: Path) -> RunResult | None:
    match = RUN_PATTERN.match(run_dir.name)
    if match is None:
        return None

    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    return RunResult(
        run_name=run_dir.name,
        batch_size=int(match.group("batch_size")),
        learning_rate=parse_lr_tag(match.group("lr_tag")),
        best_val_loss=float(summary["best_val_loss"]),
        best_val_step=(
            int(summary["best_val_step"])
            if summary["best_val_step"] is not None
            else None
        ),
        final_step=int(summary["final_step"]),
        total_wallclock_seconds=float(summary["total_wallclock_seconds"]),
        summary_path=summary_path,
    )


def geometric_midpoint(a: float, b: float) -> float:
    return (a * b) ** 0.5


def suggest_next_grid(results: list[RunResult]) -> list[float]:
    ordered = sorted(results, key=lambda result: result.learning_rate)
    best_index = min(range(len(ordered)), key=lambda idx: ordered[idx].best_val_loss)
    best_lr = ordered[best_index].learning_rate

    if len(ordered) == 1:
        return [best_lr]

    if best_index == 0:
        right = ordered[1].learning_rate
        return [best_lr, geometric_midpoint(best_lr, right), right]

    if best_index == len(ordered) - 1:
        left = ordered[-2].learning_rate
        return [left, geometric_midpoint(left, best_lr), best_lr]

    left = ordered[best_index - 1].learning_rate
    right = ordered[best_index + 1].learning_rate
    return [left, geometric_midpoint(left, best_lr), best_lr, geometric_midpoint(best_lr, right), right]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    for run_dir in sorted(args.log_root.iterdir()):
        if not run_dir.is_dir():
            continue
        result = load_summary(run_dir)
        if result is not None:
            results.append(result)

    results.sort(key=lambda result: (result.batch_size, result.learning_rate))

    by_batch_size: dict[int, list[RunResult]] = {}
    for result in results:
        by_batch_size.setdefault(result.batch_size, []).append(result)

    lines = [
        "# Batch-size sweep summary",
        "",
        "| Batch size | LR | Best val loss | Best step | Final step | Wallclock (s) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.batch_size} | {result.learning_rate:.6g} | "
            f"{result.best_val_loss:.6f} | {result.best_val_step} | "
            f"{result.final_step} | {result.total_wallclock_seconds:.2f} |"
        )

    lines.extend(["", "## Best LR per batch size", ""])
    lines.extend(
        [
            "| Batch size | Best LR | Best val loss | Suggested next LR grid |",
            "|---:|---:|---:|---|",
        ]
    )

    best_rows = []
    for batch_size in sorted(by_batch_size):
        bucket = by_batch_size[batch_size]
        best_result = min(bucket, key=lambda result: result.best_val_loss)
        next_grid = ", ".join(
            f"{lr:.6g}" for lr in suggest_next_grid(bucket)
        )
        lines.append(
            f"| {batch_size} | {best_result.learning_rate:.6g} | "
            f"{best_result.best_val_loss:.6f} | {next_grid} |"
        )
        best_rows.append(
            {
                "batch_size": batch_size,
                "best_learning_rate": best_result.learning_rate,
                "best_val_loss": best_result.best_val_loss,
                "suggested_next_grid": suggest_next_grid(bucket),
                "best_run_name": best_result.run_name,
                "summary_path": str(best_result.summary_path),
            }
        )

    (args.output_dir / "batch_size_sweep_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "batch_size_best_runs.json").open("w", encoding="utf-8") as f:
        json.dump(best_rows, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
