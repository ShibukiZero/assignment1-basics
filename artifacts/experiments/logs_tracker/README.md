# Chapter 7 Experiment Tracking

This directory stores durable, in-repo tracking documents for Chapter 7.

Scope:
- Keep a running experiment ledger outside `.agents/`
- Record why each run happened, when it happened, and what changed
- Point to the real artifacts:
  - large artifacts on the data disk
  - lightweight logs under `.agents/logs/<run_name>/`

Files:
- `experiment_log.md`: per-run durable ledger with copied key metrics
- `phase_summary.md`: phase-level conclusions and next-step decisions

Update rule:
- Add each new run to the top of `experiment_log.md`
- Record the local system time when the entry is written
- Copy enough raw evidence into `experiment_log.md` that later cleanup of `.agents/logs/` does not erase the record
- Keep phase-level interpretation in `phase_summary.md`
