# Chapter 7.1 Experiment Log

This directory stores the Chapter 7.1 experiment-log artifact and explains how
the rest of `artifacts/experiments/ch7/` is organized.

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
corresponding file under `artifacts/experiments/ch7/` and any
`artifact_*` fields that explicitly point back into this directory.

Current Chapter 7 structure:
- `7_1_1/`: narrative experiment log
- `7_2_1/`: TinyStories learning-rate tuning artifacts
- `7_2_2/`: batch-size sweep artifacts
- `7_2_3/`: TinyStories generation artifact
- `7_3_0/`: shared manifest for the architecture-ablation suite
- `7_3_1/`: no-RMSNorm ablation artifacts
- `7_3_2/`: post-norm ablation figures
- `7_3_3/`: NoPE ablation figures
- `7_3_4/`: SiLU-vs-SwiGLU ablation figures
- `7_4_1/`: OpenWebText retuning, main run, and generation artifacts
- `7_5_1/`: optional leaderboard artifacts
- `tools/`: Chapter 7 plotting helpers shared across multiple question folders

Submission-facing summary in this directory:
- `experiment_log.md`: human-readable Chapter 7 experiment log

Retired files:
- `phase_summary.md`

The source of truth for Chapter 7 is the structured summaries plus retained
per-run `summary.json` snapshots under the numbered question folders. Heavier
per-run `config.json`, `metrics.jsonl`, and `diagnostics.jsonl` files are
intentionally trimmed from the submission-oriented repository once the durable
summaries and figures are materialized. The human-readable `experiment_log.md`
is a compact index over those durable artifacts for submission and review.
