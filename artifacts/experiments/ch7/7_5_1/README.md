# Leaderboard Artifacts

This directory stores durable artifacts for the optional leaderboard section.

The current writeup-facing segmented recipe artifacts are from the H100 rerun on
2026-06-27. The best validation loss was `3.617796` at step `13080`, after
`3487.55` seconds on the cumulative evaluation-time curve. The full staged
training run completed in `4010.65` seconds (`1.11` hours), within the 1.5
H100-hour budget.

Recommended structure:

- `figures/`
  - final writeup-facing plots for the leaderboard section
- `results/`
  - lightweight copied summaries or supporting markdown tables

This directory is intentionally separate from:

- `artifacts/experiments/ch7/`

because the leaderboard experiments follow a different, continuation-heavy
workflow and are easier to review when isolated from the main Chapter 7.4
experiment artifacts under `artifacts/experiments/ch7/7_4_1/`.

Plotting helpers for continuation-heavy recipes live here as well, including:

- `plot_segmented_recipe_curve.py`

which stitches multiple training stages into one cumulative wallclock curve while
keeping stage-specific colors and labels.
