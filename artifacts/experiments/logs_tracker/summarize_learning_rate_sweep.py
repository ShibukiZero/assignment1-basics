from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


RUN_PATTERN = re.compile(
    r"^tinystories_lr_auto_lr(?P<lr_tag>[0-9ep\-]+)$"
)


@dataclass
class RunResult:
    run_name: str
    learning_rate: float
    best_val_loss: float
    best_val_step: int | None
    final_step: int
    total_wallclock_seconds: float
    summary_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the Chapter 7 learning-rate sweep.",
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
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="tinystories_lr_auto",
    )
    return parser.parse_args()


def parse_lr_tag(raw: str) -> float:
    return float(raw.replace("p", "."))


def load_summary(run_dir: Path, run_prefix: str) -> RunResult | None:
    pattern = re.compile(
        rf"^{re.escape(run_prefix)}_lr(?P<lr_tag>[0-9ep\-]+)$"
    )
    match = pattern.match(run_dir.name)
    if match is None:
        return None

    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    return RunResult(
        run_name=run_dir.name,
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
        result = load_summary(run_dir, args.run_prefix)
        if result is not None:
            results.append(result)

    results.sort(key=lambda result: result.learning_rate)

    lines = [
        "# Learning-rate sweep summary",
        "",
        "| LR | Best val loss | Best step | Final step | Wallclock (s) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.learning_rate:.6g} | {result.best_val_loss:.6f} | "
            f"{result.best_val_step} | {result.final_step} | {result.total_wallclock_seconds:.2f} |"
        )

    if results:
        best_result = min(results, key=lambda result: result.best_val_loss)
        suggested = suggest_next_grid(results)
        lines.extend(
            [
                "",
                "## Best LR",
                "",
                f"- Best LR: `{best_result.learning_rate:.6g}`",
                f"- Best validation loss: `{best_result.best_val_loss:.6f}`",
                f"- Suggested next LR grid: `{', '.join(f'{lr:.6g}' for lr in suggested)}`",
            ]
        )

        payload = {
            "best_learning_rate": best_result.learning_rate,
            "best_val_loss": best_result.best_val_loss,
            "best_run_name": best_result.run_name,
            "summary_path": str(best_result.summary_path),
            "suggested_next_grid": suggested,
        }
        with (args.output_dir / "learning_rate_best_run.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

    (args.output_dir / "learning_rate_sweep_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
