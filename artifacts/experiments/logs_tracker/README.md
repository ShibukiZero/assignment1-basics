# Chapter 7 Log Tracker

This directory stores durable experiment records, generated summaries, and plotting utilities for Chapter 7.

Scope:
- Keep a running experiment ledger outside `.agents/`
- Store summarized outputs that are useful for the final writeup
- Keep plotting scripts close to the result folders they consume

This directory should not contain experiment runner scripts. Training / sweep execution code lives under:
- `cs336_basics/experiments/`
- `cs336_basics/experiments/sweep_configs/`

Files:
- `experiment_log.md`: per-run durable ledger with copied key metrics
- `phase_summary.md`: phase-level conclusions and next-step decisions
- `plot_*.py`: plotting entrypoints
- `*_round*/`: summarized results for a completed sweep round
- `figures_*/`: figures intended for the writeup

Update rule:
- Add each new run to the top of `experiment_log.md`
- Record the local system time when the entry is written
- Copy enough raw evidence into `experiment_log.md` that later cleanup of `.agents/logs/` does not erase the record
- Keep phase-level interpretation in `phase_summary.md`
