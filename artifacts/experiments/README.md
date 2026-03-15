# Experiments Artifacts

This directory stores reproducible outputs from assignment experiments.

Some tracked experiment records intentionally preserve provenance from the original
remote execution environment. Paths such as `.agents/logs/...`,
`/root/autodl-tmp/...`, or `/root/tf-logs/...` should be read as historical run
metadata, not as repository-local paths that are expected to exist on every
machine. For repository-local evidence, prefer paths rooted under
`artifacts/experiments/`.

## Subdirectories
- `logs_tracker/`: Chapter 7 durable run ledger, persisted summarized results, figures, and plotting utilities.
- `tokenizer/`: tokenizer training and tokenizer-related experiment artifacts.

## Related code
- Experiment runner scripts and sweep configs live under:
  - `cs336_basics/experiments/`
  - `cs336_basics/experiments/sweep_configs/`
