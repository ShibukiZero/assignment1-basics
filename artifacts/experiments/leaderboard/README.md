# Leaderboard Artifacts

This directory stores durable artifacts for the optional leaderboard section.

Recommended structure:

- `figures/`
  - final writeup-facing plots for the leaderboard section
- `results/`
  - lightweight copied summaries or supporting markdown tables

This directory is intentionally separate from:

- `artifacts/experiments/logs_tracker/`

because the leaderboard experiments follow a different, continuation-heavy workflow and are easier to review when isolated from the main Chapter 7.4 experiment log.

Plotting helpers for continuation-heavy recipes live here as well, including:

- `plot_segmented_recipe_curve.py`

which stitches multiple training stages into one cumulative wallclock curve while
keeping stage-specific colors and labels.
