# Chapter 7 Experiment Log

This document is the human-readable experiment log for the Chapter 7 language-model experiments. It complements the structured run artifacts under `results/` and the logging code in `cs336_basics/experiments/train_transformer_lm.py`.

## Logging infrastructure

The Chapter 7 training entrypoint records experiment state in terms of both gradient steps and wall-clock time, as required by the handout.

For each run, the training script writes:

- `config.json`: the full run configuration, including model, optimization, logging, and checkpoint settings
- `metrics.jsonl`: train and validation metrics with `step`, `wallclock_seconds`, `tokens_seen`, `loss`, `perplexity`, and learning-rate metadata
- `diagnostics.jsonl`: optimization diagnostics such as gradient norms and parameter norms
- `summary.json`: a compact end-of-run summary with best validation loss, best step, final step, and checkpoint paths
- optional TensorBoard event files under the configured TensorBoard root

Relevant implementation points:

- metric dataclasses are defined in `cs336_basics/experiments/train_transformer_lm.py`
- metrics are emitted during training at every evaluation interval
- each record includes both `step` and `wallclock_seconds`, so all learning curves can be plotted against either x-axis
- the same logging path is used for TinyStories, OpenWebText, learning-rate sweeps, batch-size sweeps, and architecture ablations
- copied run metadata may retain original remote paths such as `.agents/logs/...` or `/root/...`; for repository-local references, prefer the `artifact_*` fields and the files linked under `artifacts/experiments/logs_tracker/`

## Durable result layout

Durable Chapter 7 artifacts are stored under:

- `artifacts/experiments/logs_tracker/results/`
- `artifacts/experiments/logs_tracker/figures/`

The `results/` directory stores copied lightweight run artifacts and machine-readable summaries. The `figures/` directory stores writeup-facing plots generated from those summaries.
Repository-local durable evidence always lives under those two directories, even
when copied JSON metadata still mentions the original remote execution paths.

## Experiment ledger

### 1. TinyStories learning-rate tuning

Goal:
- tune the base TinyStories model to reach the target validation loss
- identify a good learning-rate region and characterize the failure side

What was tried:
- a broad pilot sweep over increasing learning rates for `batch_size=128`
- a refined sweep centered near the best region
- a longer confirmation run using the selected learning rate

Where to find the results:
- summary: `artifacts/experiments/logs_tracker/results/bs128_lr_round2/learning_rate_sweep_summary.md`
- machine-readable run table: `artifacts/experiments/logs_tracker/results/bs128_lr_round2/learning_rate_runs.json`
- selected best-run metadata: `artifacts/experiments/logs_tracker/results/bs128_lr_round2/learning_rate_best_run.json`
- learning-curve figure: `artifacts/experiments/logs_tracker/figures/bs128_learning_rate/val_loss_vs_step.png`
- learning-curve figure: `artifacts/experiments/logs_tracker/figures/bs128_learning_rate/val_loss_vs_wallclock.png`

Outcome used in the writeup:
- selected TinyStories training setting: `batch_size=128`, `learning_rate=4.0e-3`
- best pilot validation loss in the refined sweep: `3.501499`
- the final long run with this setting is the TinyStories reference checkpoint used for later comparisons

### 2. Batch-size sweep on TinyStories

Goal:
- study how the best learning rate changes with batch size
- compare quality and runtime tradeoffs across batch sizes

What was tried:
- batch sizes from `1` through `512`
- a locally re-tuned learning-rate grid for each batch size
- fixed-horizon pilot runs for cross-batch comparison

Where to find the results:
- summary: `artifacts/experiments/logs_tracker/results/batch_size_sweep_round2/batch_size_sweep_summary.md`
- best-run table: `artifacts/experiments/logs_tracker/results/batch_size_sweep_round2/batch_size_best_runs.json`
- step-axis figure: `artifacts/experiments/logs_tracker/figures/batch_size_round2/val_loss_vs_step.png`
- wall-clock figure: `artifacts/experiments/logs_tracker/figures/batch_size_round2/val_loss_vs_wallclock.png`
- short figure summary: `artifacts/experiments/logs_tracker/figures/batch_size_round2/batch_size_best_curve_summary.md`

Outcome used in the writeup:
- best local learning rates increase with batch size
- `batch_size=128` was retained as the main TinyStories setting
- `batch_size=256` and `batch_size=512` gave better fixed-step pilot losses but at substantially higher runtime cost

### 3. TinyStories generation check

Goal:
- decode from the tuned TinyStories checkpoint and verify that the trained model produces fluent in-domain text

What was tried:
- sampling from the final TinyStories checkpoint using nucleus sampling
- prompt: `"Once upon a time, there was"`
- decoding with `temperature=0.8` and `top_p=0.9`

Where to find the results:
- generated text: `artifacts/experiments/logs_tracker/results/generate_tinystories_bs128/ts_generate_bs128_final.txt`
- generation metadata: `artifacts/experiments/logs_tracker/results/generate_tinystories_bs128/ts_generate_bs128_final.json`

Outcome used in the writeup:
- this generation artifact was used as the final TinyStories sample

### 4. LayerNorm ablation

Goal:
- test the impact of removing RMSNorm and determine whether lower learning rates can restore stable training

What was tried:
- removed block RMSNorms and the final RMSNorm used in the writeup-facing no-norm run
- coarse lower-learning-rate sweep on the no-RMSNorm model
- a longer confirmation run at the best stable lower learning rate

Where to find the results:
- summary: `artifacts/experiments/logs_tracker/results/layer_norm_ablation_coarse_lr/learning_rate_sweep_summary.md`
- machine-readable run table: `artifacts/experiments/logs_tracker/results/layer_norm_ablation_coarse_lr/learning_rate_runs.json`
- selected best-run metadata: `artifacts/experiments/logs_tracker/results/layer_norm_ablation_coarse_lr/learning_rate_best_run.json`
- figure: `artifacts/experiments/logs_tracker/figures/layer_norm_ablation_final/val_loss_vs_step.png`
- figure: `artifacts/experiments/logs_tracker/figures/layer_norm_ablation_final/val_loss_vs_wallclock.png`
- figure summary: `artifacts/experiments/logs_tracker/figures/layer_norm_ablation_final/val_loss_summary.md`

Outcome used in the writeup:
- the previous baseline learning rate became unstable without RMSNorm
- the best stable lower-learning-rate setting in the coarse sweep was `1.0e-3`

### 5. Architecture ablation suite

Goal:
- compare the baseline TinyStories model with three controlled architectural modifications:
  - post-norm instead of pre-norm
  - NoPE instead of RoPE
  - SiLU FFN instead of SwiGLU, with approximately matched parameter count

What was tried:
- a matched-horizon suite of runs with the same optimizer settings and training budget
- one baseline control plus one run for each ablation

Where to find the results:
- plotting manifest: `artifacts/experiments/logs_tracker/ablation_suite_plots_bs128_5k.json`
- post-norm figure directory: `artifacts/experiments/logs_tracker/figures/arch_ablation_suite_bs128_5k/pre_norm_ablation/`
- NoPE figure directory: `artifacts/experiments/logs_tracker/figures/arch_ablation_suite_bs128_5k/no_pos_emb/`
- SwiGLU vs. SiLU figure directory: `artifacts/experiments/logs_tracker/figures/arch_ablation_suite_bs128_5k/swiglu_ablation/`

Outcome used in the writeup:
- pre-norm outperformed post-norm
- RoPE outperformed NoPE
- SwiGLU outperformed SiLU by a smaller margin than the normalization or position-encoding changes

### 6. OpenWebText learning-rate tuning

Goal:
- retune the learning rate for OpenWebText while keeping the same model architecture and batch size used for the TinyStories main run

What was tried:
- a coarse pilot sweep at `batch_size=128`
- a refined pilot sweep centered on the best coarse region

Where to find the results:
- coarse summary: `artifacts/experiments/logs_tracker/results/owt_bs128_lr_coarse/learning_rate_sweep_summary.md`
- coarse run table: `artifacts/experiments/logs_tracker/results/owt_bs128_lr_coarse/learning_rate_runs.json`
- refined summary: `artifacts/experiments/logs_tracker/results/owt_bs128_lr_refined/learning_rate_sweep_summary.md`
- refined run table: `artifacts/experiments/logs_tracker/results/owt_bs128_lr_refined/learning_rate_runs.json`
- refined best-run metadata: `artifacts/experiments/logs_tracker/results/owt_bs128_lr_refined/learning_rate_best_run.json`

Outcome used in the writeup:
- coarse best region: around `2e-3` to `3e-3`
- refined selected setting: `batch_size=128`, `learning_rate=2.5e-3`

### 7. OpenWebText main experiment

Goal:
- train on OpenWebText with the same architecture and total training-iteration budget as the TinyStories main run

What was tried:
- a full `10000`-step OpenWebText run with the retuned learning rate from the pilot sweeps
- matched TinyStories vs. OpenWebText comparisons in both step space and wall-clock space

Where to find the results:
- main-run summary: `artifacts/experiments/logs_tracker/results/owt_main_bs128_lr2p5e-03/owt_main_run_summary.md`
- copied run artifacts: `artifacts/experiments/logs_tracker/results/owt_main_bs128_lr2p5e-03/run/`
- comparison figure: `artifacts/experiments/logs_tracker/figures/tinystories_vs_owt_bs128_main/val_loss_vs_step.png`
- comparison figure: `artifacts/experiments/logs_tracker/figures/tinystories_vs_owt_bs128_main/val_loss_vs_wallclock.png`
- comparison summary: `artifacts/experiments/logs_tracker/figures/tinystories_vs_owt_bs128_main/val_loss_summary.md`

Outcome used in the writeup:
- selected OWT main-run setting: `batch_size=128`, `learning_rate=2.5e-3`
- best validation loss: `3.888615`
- total wall-clock time: `3470.46s`

### 8. OpenWebText generation check

Goal:
- decode from the final OpenWebText checkpoint and inspect qualitative fluency

What was tried:
- sampling from the final OpenWebText checkpoint with the same prompt and decoding hyperparameters used for the TinyStories sample

Where to find the results:
- generated text: `artifacts/experiments/logs_tracker/results/generate_owt_bs128/owt_generate_bs128_final.txt`
- generation metadata: `artifacts/experiments/logs_tracker/results/generate_owt_bs128/owt_generate_bs128_final.json`

Outcome used in the writeup:
- this generation artifact was used as the final OpenWebText sample

## How to trace an individual run

For any run copied into `results/.../runs/` or `results/.../run/`, the typical files are:

- `config.json`
- `metrics.jsonl`
- `diagnostics.jsonl`
- `summary.json`

These are the canonical inputs used by the plotting scripts under `artifacts/experiments/logs_tracker/plot_*.py`.

## Submission intent

This file is the narrative experiment log requested by the handout. The detailed numerical evidence remains in the structured summaries, per-run JSONL files, and plotted figures linked above.
