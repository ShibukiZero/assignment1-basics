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

### `tinystories_lr_sweep_b_pilot`
- Logged at (local): `2026-03-08 18:45:01 CST`
- Run timestamp (from logs/config): `2026-03-08T10:43:33.432752+00:00`
- Status: `PASS`
- Purpose: First TinyStories LR-sweep pilot run to verify the 7.2 baseline configuration on H800 and check whether the baseline training pipeline remains stable at larger model scale.
- Dataset: TinyStories token IDs (`tiny_train_ids.npy`, `tiny_valid_ids.npy`)
- Model summary: TinyStories baseline architecture with `vocab_size=10000`, `context_length=256`, `d_model=512`, `num_layers=4`, `num_heads=16`, `d_ff=1344`, `rope_theta=10000`
- Optimization summary: `batch_size=32`, `max_steps=20`, `eval_interval=5`, `eval_batches=4`, `learning_rate=3e-4`, `min_learning_rate=3e-5`, `warmup_iters=200`, `cosine_cycle_iters=40000`, `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.1`
- Large-artifact path: `/root/autodl-tmp/training_runs/tinystories_lr_sweep_b_pilot`
- Lightweight-log path: `.agents/logs/tinystories_lr_sweep_b_pilot`
- TensorBoard path: `/root/tf-logs/tinystories_lr_sweep_b_pilot`
- Key metrics:
  - best validation loss: `8.925974` at step `20`
  - final step: `20`
  - total wallclock: `5.860483s`
  - observed validation losses:
    - step 5: `9.234610`
    - step 10: `9.176618`
    - step 15: `9.072989`
    - step 20: `8.925974`
- Main observations:
  - The baseline-scale model runs successfully on the current H800 environment without immediate instability.
  - Validation loss decreases monotonically over the first 20 steps, so there is no evidence of divergence at this setting.
  - This run is still primarily a pipeline/pilot check, not a conclusive LR comparison:
    - with `warmup_iters=200`, the effective LR only reached `2.85e-05` by step 20
    - so the nominal target LR `3e-4` has not actually been exercised yet
  - The wallclock signal looks reasonable for a baseline-scale smoke test:
    - about `3.97s` to reach step 20 eval
    - about `5.86s` total wallclock
- Decision: Keep this configuration as a valid baseline-scale pilot, but do not use it yet to rank LR choices.
- Next action: Run the same pilot template for the other LR points, then either reduce warmup for pilot sweeps or run longer pilots so the configured LR is actually reached.

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
- TensorBoard path: `/root/tf-logs/ch7_logging_smoke`
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
  - TensorBoard event logging succeeded under `/root/tf-logs/ch7_logging_smoke`
  - The checkpoint/log split now matches the intended workflow
- Decision: The Chapter 7.1 logging infrastructure is sufficient for baseline experiments.
- Next action: Start the TinyStories baseline and learning-rate sweep, and append each subsequent run to this file.
