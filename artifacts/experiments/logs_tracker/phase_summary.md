# Chapter 7 Phase Summary

This file stores phase-level conclusions derived from the run ledger in `experiment_log.md`.

Use it to:
- summarize a sweep after all runs in that sweep are logged
- record decisions that guide the next phase
- avoid mixing high-level interpretation into the raw run ledger

## Phase A: LR pilot sweep

- Logged at (local): `2026-03-08 19:05:30 CST`
- Scope:
  - `tinystories_lr_sweep_n_pilot`
  - `tinystories_lr_sweep_o_pilot`
  - `tinystories_lr_sweep_l_pilot`
  - `tinystories_lr_sweep_m_pilot`
  - `tinystories_lr_sweep_a_pilot`
  - `tinystories_lr_sweep_b_pilot (phase-a retake)`
  - `tinystories_lr_sweep_c_pilot`
  - `tinystories_lr_sweep_d_pilot`
  - `tinystories_lr_sweep_e_pilot`
  - `tinystories_lr_sweep_f_pilot`
  - `tinystories_lr_sweep_g_pilot`
  - `tinystories_lr_sweep_h_pilot`
  - `tinystories_lr_sweep_i_pilot`
  - `tinystories_lr_sweep_j_pilot`
  - `tinystories_lr_sweep_k_pilot`

- Fixed settings:
  - model: TinyStories baseline (`vocab_size=10000`, `context_length=256`, `d_model=512`, `num_layers=4`, `num_heads=16`, `d_ff=1344`, `rope_theta=10000`)
  - optimizer: `beta1=0.9`, `beta2=0.999`, `eps=1e-8`, `weight_decay=0.1`
  - pilot template: `batch_size=32`, `warmup_iters=20`, `max_steps=60`, `eval_interval=10`, `eval_batches=4`

- LR grid and best validation losses at step 60:
  - `1e-4`: `5.934037`
  - `3e-4`: `4.362573`
  - `6e-4`: `3.865036`
  - `1e-3`: `3.709330`
  - `2e-3`: `3.640632`
  - `4e-3`: `3.850892`
  - `6e-3`: `3.905126`
  - `8e-3`: `4.142055`
  - `1.2e-2`: `4.345952`
  - `1.6e-2`: `4.643384`
  - `2e-2`: `4.393443`
  - `5e-2`: `5.234851`
  - `1e-1`: `5.912755`
  - `2e-1`: `6.727881`
  - `5e-1`: `25.741734` (best recorded step; final curve degraded further)

- Ranking:
  - `2e-3` > `1e-3` > `4e-3` ≈ `6e-4` > `6e-3` > `8e-3` > `1.2e-2` > `2e-2` > `1.6e-2` > `5e-2` > `1e-1` > `2e-1` > `5e-1` > `3e-4` > `1e-4`

- Conclusions:
  - All tested LR values through `5e-1` remained numerically finite in this 60-step pilot setting.
  - `1e-4` is clearly too small for this setup.
  - Quality improved up to `2e-3`, then degraded steadily across the entire upper tail from `4e-3` through `5e-1`.
  - The best-quality region is now likely centered around `1e-3` to `2e-3`, with `2e-3` currently best.
  - At `2e-1` and `5e-1`, gradient clipping was saturated at every logged eval point (`grad_post` pinned at about `1.0`), and losses stayed catastrophically high.
  - `5e-1` is best interpreted as a practical divergence / failure case even though it did not emit NaN.

- Decision:
  - The upper-tail search is complete for 7.2 purposes: the best LR sits far below the region where training becomes practically unusable.
  - Preserve `1e-3` to `2e-3` as the leading region for later fine search and longer confirmation runs.
  - Use `5e-1` as the failure-side reference curve in the learning-rate analysis if the writeup does not require literal NaN divergence.

- Next planned action:
  - Move to fine search and longer confirmation runs near `1e-3` to `2e-3`.
  - Candidate next comparison grid: `1.2e-3`, `1.6e-3`, `2e-3`, `2.4e-3`.

## Phase B: LR fine sweep

- Logged at (local): `2026-03-08 19:48:00 CST`
- Scope:
  - `tinystories_lr_fine_a` (`1.2e-3`)
  - `tinystories_lr_fine_b` (`1.6e-3`)
  - `tinystories_lr_fine_c` (`2.0e-3`)
  - `tinystories_lr_fine_d` (`2.4e-3`)

- Fixed settings:
  - model: TinyStories baseline (`vocab_size=10000`, `context_length=256`, `d_model=512`, `num_layers=4`, `num_heads=16`, `d_ff=1344`, `rope_theta=10000`)
  - optimizer: `beta1=0.9`, `beta2=0.999`, `eps=1e-8`, `weight_decay=0.1`
  - pilot template: `batch_size=32`, `warmup_iters=20`, `max_steps=60`, `eval_interval=10`, `eval_batches=4`

- Best validation losses at step 60:
  - `1.2e-3`: `3.679679`
  - `1.6e-3`: `3.704379`
  - `2.0e-3`: `3.640632`
  - `2.4e-3`: `3.658737`

- Ranking:
  - `2.0e-3` > `2.4e-3` > `1.2e-3` > `1.6e-3`

- Conclusions:
  - `2.0e-3` remains the best LR in the refined local region.
  - `2.4e-3` is close enough to justify one longer confirmation run.
  - `1.6e-3` is now the weakest point in this local sweep and can be dropped.
  - Diagnostics look healthy after warmup for all four runs; clipping is not persistently active in the post-warmup regime.

- Decision:
  - Use `2.0e-3` as the primary longer-run confirmation candidate.
  - Use `2.4e-3` as the secondary confirmation candidate.

- Next planned action:
  - Run longer confirmation experiments for `2.0e-3` and `2.4e-3`.
  - Restore the assignment-style warmup in the confirmation stage.

## Phase C: Medium-length confirmation

- Logged at (local): `2026-03-08 20:05:00 CST`
- Scope:
  - `tinystories_lr_final_2e3_5k`

- Fixed settings:
  - model: TinyStories baseline (`vocab_size=10000`, `context_length=256`, `d_model=512`, `num_layers=4`, `num_heads=16`, `d_ff=1344`, `rope_theta=10000`)
  - optimizer: `learning_rate=2.0e-3`, `min_learning_rate=2.0e-4`, `beta1=0.9`, `beta2=0.999`, `eps=1e-8`, `weight_decay=0.1`
  - schedule: `warmup_iters=200`, `cosine_cycle_iters=5000`
  - runtime template: `batch_size=32`, `max_steps=5000`, `eval_interval=250`, `eval_batches=8`

- Result:
  - best validation loss: `1.579280` at step `5000`
  - total wallclock: `365.846570s`
  - total tokens processed: `40,960,000`

- Conclusions:
  - The selected LR `2.0e-3` remains stable and effective over a much longer horizon.
  - The training curve is still improving, though gains are slower late in the run.
  - This configuration is already close to the downscaled TinyStories target and does not show a clear reason to reopen LR tuning.

- Decision:
  - Do not tune LR further right now.
  - Prefer more training time over changing optimizer hyperparameters at this stage.

- Next planned action:
  - Extend the same configuration to a roughly 30-minute run.
  - Only revisit other hyperparameters if the longer run plateaus well above the target.

## Phase D: 25k-step TinyStories run

- Logged at (local): `2026-03-08 20:18:00 CST`
- Scope:
  - `tinystories_lr_final_2e3_25k`

- Fixed settings:
  - model: TinyStories baseline (`vocab_size=10000`, `context_length=256`, `d_model=512`, `num_layers=4`, `num_heads=16`, `d_ff=1344`, `rope_theta=10000`)
  - optimizer: `learning_rate=2.0e-3`, `min_learning_rate=2.0e-4`, `beta1=0.9`, `beta2=0.999`, `eps=1e-8`, `weight_decay=0.1`
  - schedule: `warmup_iters=200`, `cosine_cycle_iters=25000`
  - runtime template: `batch_size=32`, `max_steps=25000`, `eval_interval=1000`, `eval_batches=8`

- Result:
  - best validation loss: `1.362352` at step `24000`
  - final validation loss: `1.371345` at step `25000`
  - total wallclock: `1756.146916s` (about 29.3 minutes)
  - total tokens processed: `204,800,000`

- Conclusions:
  - The run satisfies the 7.2 TinyStories validation target (`<= 1.45`).
  - Training remained stable and healthy throughout; diagnostics do not suggest a reason to reopen optimizer tuning.
  - The best checkpoint occurred before the final step, but the tail remained close enough that there is no obvious failure mode or collapse.

- Decision:
  - The learning-rate tuning task for 7.2 can be considered complete with `learning_rate=2.0e-3`.
  - Do not spend more time tuning LR, warmup, or other AdamW hyperparameters before moving to the next assignment requirement.

- Next planned action:
  - Use the best checkpoint from `tinystories_lr_final_2e3_25k` for generation and for later Chapter 7 comparisons.
  - Move on to the next Chapter 7 deliverable unless a stronger final TinyStories checkpoint is specifically desired.
