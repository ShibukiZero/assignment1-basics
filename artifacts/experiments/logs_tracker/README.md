# Chapter 7 Experiment Tracking

This directory stores durable, in-repo tracking documents for Chapter 7.

Scope:
- Keep a running experiment ledger outside `.agents/`
- Record why each run happened, when it happened, and what changed
- Point to the real artifacts:
  - large artifacts on the data disk
  - lightweight logs under `.agents/logs/<run_name>/`

Primary file:
- `experiment_log.md`

Update rule:
- Add each new run to the top of `experiment_log.md`
- Record the local system time when the entry is written
- Keep entries concise and evidence-based
