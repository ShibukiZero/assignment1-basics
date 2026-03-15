# Experiments Artifacts

This directory stores reproducible outputs from assignment experiments.

Some tracked experiment records intentionally preserve provenance from the original
remote execution environment. Paths such as `.agents/logs/...`,
`/root/autodl-tmp/...`, or `/root/tf-logs/...` should be read as historical run
metadata, not as repository-local paths that are expected to exist on every
machine. For repository-local evidence, prefer paths rooted under
`artifacts/experiments/`.

## Layout
- `ch2/`: Chapter 2 artifacts grouped by handout section.
- `ch3/`: Chapter 3 artifacts grouped by handout section.
- `ch4/`: Chapter 4 artifacts grouped by handout section.
- `ch7/`: Chapter 7 artifacts grouped by handout section.

Within each chapter directory, the first nested folder uses the corresponding
handout numbering so the writeup and the repository layout stay aligned. For
example:

- `ch2/2_5_1/`: artifacts for the first question under Section 2.5
- `ch7/7_4_1/`: artifacts for the main experiment under Section 7.4

Question folders may contain `results/`, `figures/`, or small helper scripts
when those files are specific to that handout item.

## Related code
- Experiment runner scripts and sweep configs live under:
  - `cs336_basics/experiments/`
  - `cs336_basics/experiments/sweep_configs/`
