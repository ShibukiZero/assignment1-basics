from __future__ import annotations

import argparse
import json
import re
import shutil
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
    run_dir: Path


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
        run_dir=run_dir,
    )


def geometric_midpoint(a: float, b: float) -> float:
    return (a * b) ** 0.5


def persist_run_artifacts(result: RunResult, destination_root: Path) -> dict[str, str]:
    destination_dir = destination_root / result.run_name
    destination_dir.mkdir(parents=True, exist_ok=True)

    copied_paths: dict[str, str] = {"artifact_run_dir": str(destination_dir)}
    for filename in ("metrics.jsonl", "summary.json", "diagnostics.jsonl", "config.json"):
        source = result.run_dir / filename
        if not source.exists():
            continue
        target = destination_dir / filename
        shutil.copy2(source, target)
        key = f"artifact_{filename.replace('.', '_')}"
        copied_paths[key] = str(target)
    return copied_paths


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
    persisted_runs_root = args.output_dir / "runs"

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
        persisted_bucket = {}
        for result in bucket:
            persisted_bucket[result.run_name] = persist_run_artifacts(result, persisted_runs_root)
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
                **persisted_bucket[best_result.run_name],
                "artifact_metrics_path": persisted_bucket[best_result.run_name].get("artifact_metrics_jsonl"),
                "artifact_summary_path": persisted_bucket[best_result.run_name].get("artifact_summary_json"),
                "artifact_diagnostics_path": persisted_bucket[best_result.run_name].get("artifact_diagnostics_jsonl"),
                "artifact_config_path": persisted_bucket[best_result.run_name].get("artifact_config_json"),
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
