# Chapter 7 Log Tracker

This directory stores durable experiment records, generated summaries, and plotting utilities for Chapter 7.

Scope:
- Store durable summarized outputs that are useful for the final writeup
- Keep compact per-run summary snapshots copied from temporary remote logs
- Keep plotting scripts close to the result folders they consume

This directory should not contain experiment runner scripts. Training / sweep execution code lives under:
- `cs336_basics/experiments/`
- `cs336_basics/experiments/sweep_configs/`

Copied run metadata, retained `summary.json` snapshots, and generation metadata
in this directory preserve original run-time provenance from the remote execution
environment.
Fields such as `log_dir`, `summary_path`, `output_dir`, `tensorboard_dir`,
`checkpoint_path`, and `config_path` may therefore point to `.agents/logs/...`
or `/root/...` locations that are not part of the tracked repository. Treat
those fields as historical context only; the durable in-repo evidence is the
corresponding file under `artifacts/experiments/logs_tracker/` and any
`artifact_*` fields that explicitly point back into this directory.

Current structure:
- `plot_*.py`: plotting entrypoints
- `results/`: summarized outputs and persisted per-run summary snapshots copied from temporary logs
- `figures/`: figures intended for the writeup

Current durable result sets:
- `results/bs128_lr_round2/`: TinyStories `batch_size=128` learning-rate sweep summary and retained run summaries
- `results/batch_size_sweep_round2/`: batch-size sweep summary
- `results/generate_tinystories_bs128/`: generated TinyStories sample used for writeup
- `results/generate_owt_bs128/`: generated OpenWebText sample used for writeup
- `results/layer_norm_ablation_coarse_lr/`: no-RMSNorm coarse lower-LR sweep summary
- `results/owt_bs128_lr_coarse/`: OpenWebText coarse learning-rate sweep summary and retained run summaries
- `results/owt_bs128_lr_refined/`: OpenWebText refined learning-rate sweep summary and retained run summaries
- `results/owt_main_bs128_lr2p5e-03/`: writeup-facing OpenWebText main run summary and retained run summaries

Current writeup-facing figures:
- `figures/arch_ablation_suite_bs128_5k/`
- `figures/bs128_learning_rate/`
- `figures/batch_size_round2/`
- `figures/layer_norm_ablation_final/`
- `figures/tinystories_vs_owt_bs128_main/`

Submission-facing summaries:
- `experiment_log.md`: human-readable Chapter 7 experiment log

Retired files:
- `phase_summary.md`

The source of truth in this directory is the structured summaries plus retained per-run
`summary.json` snapshots under `results/`. Heavier per-run `config.json`,
`metrics.jsonl`, and `diagnostics.jsonl` files are intentionally trimmed from the
submission-oriented repository once the durable summaries and figures are materialized.
The human-readable `experiment_log.md` is a compact index over those durable artifacts
for submission and review.
