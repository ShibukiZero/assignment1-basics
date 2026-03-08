# Chapter 7 Experiment Log

This is the canonical in-repo tracker for Chapter 7 experiments.

Use it to:
- record each run with a timestamp
- summarize the intent of the run
- capture the most important metrics and outcomes
- point to the artifact locations

Conventions:
- Add new entries to the top
- Record both local write time and any run timestamp available from logs
- Keep raw checkpoints on the data disk
- Keep lightweight logs in `.agents/logs/<run_name>/`

## Entry Template

### `<run_name>`
- Logged at (local): `YYYY-MM-DD HH:MM:SS TZ`
- Run timestamp (from logs/config): `...`
- Status: `PASS | FAIL | RUNNING`
- Purpose:
- Dataset:
- Model summary:
- Optimization summary:
- Large-artifact path:
- Lightweight-log path:
- TensorBoard path:
- Key metrics:
- Main observations:
- Decision:
- Next action:

---

## Runs

### `ch7_logging_smoke`
- Logged at (local): `2026-03-08 18:27:59 CST`
- Run timestamp (from logs/config): `2026-03-08T10:22:13.691775+00:00`
- Status: `PASS`
- Purpose: Validate the Chapter 7.1 logging infrastructure after splitting large artifacts and lightweight logs into separate locations.
- Dataset: TinyStories token IDs (`tiny_train_ids.npy`, `tiny_valid_ids.npy`)
- Model summary: smoke config with `vocab_size=10000`, `context_length=32`, `d_model=64`, `num_layers=2`, `num_heads=4`, `d_ff=256`
- Optimization summary: `batch_size=4`, `max_steps=3`, `eval_interval=1`, `eval_batches=1`, `learning_rate=1e-3`, `min_learning_rate=1e-4`, `warmup_iters=1`, `cosine_cycle_iters=3`
- Large-artifact path: `/root/autodl-tmp/training_runs/ch7_logging_smoke`
- Lightweight-log path: `.agents/logs/ch7_logging_smoke`
- TensorBoard path: `.agents/logs/ch7_logging_smoke/tensorboard`
- Key metrics:
  - best validation loss: `9.192152` at step `3`
  - final step: `3`
  - total wallclock: `0.567279s`
  - observed validation losses:
    - step 1: `9.214347`
    - step 2: `9.214599`
    - step 3: `9.192152`
- Main observations:
  - `config.json` captures run metadata, model config, optimization config, and directory layout
  - `metrics.jsonl` includes `step`, `wallclock_seconds`, `tokens_seen`, `split`, `loss`, `perplexity`, `learning_rate`, `batch_size`, and `context_length`
  - `summary.json` records best-checkpoint and final-wallclock information
  - TensorBoard event logging succeeded under the lightweight log directory
  - The checkpoint/log split now matches the intended workflow
- Decision: The Chapter 7.1 logging infrastructure is sufficient for baseline experiments.
- Next action: Start the TinyStories baseline and learning-rate sweep, and append each subsequent run to this file.
