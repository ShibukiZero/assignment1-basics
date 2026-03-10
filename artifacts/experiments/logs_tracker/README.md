# Chapter 7 Log Tracker

This directory stores durable experiment records, generated summaries, and plotting utilities for Chapter 7.

Scope:
- Store durable summarized outputs that are useful for the final writeup
- Keep lightweight copied artifacts from temporary remote logs
- Keep plotting scripts close to the result folders they consume

This directory should not contain experiment runner scripts. Training / sweep execution code lives under:
- `cs336_basics/experiments/`
- `cs336_basics/experiments/sweep_configs/`

Current structure:
- `plot_*.py`: plotting entrypoints
- `results/`: summarized outputs and persisted lightweight run artifacts copied from temporary logs
- `figures/`: figures intended for the writeup

Current durable result sets:
- `results/bs128_lr_round2/`: TinyStories `batch_size=128` learning-rate sweep summary and copied run artifacts
- `results/batch_size_sweep_round2/`: batch-size sweep summary
- `results/generate_tinystories_bs128/`: generated TinyStories sample used for writeup
- `results/layer_norm_ablation_coarse_lr/`: no-RMSNorm coarse lower-LR sweep summary

Current writeup-facing figures:
- `figures/bs128_learning_rate/`
- `figures/batch_size_round2/`
- `figures/layer_norm_ablation_stage1/`

Retired files:
- `experiment_log.md`
- `phase_summary.md`

Those free-form ledgers were intentionally removed after they fell out of sync with the newer result manifests under `results/`. Going forward, the source of truth in this directory is the structured summaries plus copied per-run artifacts.
