# Chapter 7 TinyStories tuning plan (2026-03-08)

## Source read this turn
- `.agents/reference/MinerU_markdown_cs336_spring2025_assignment1_basics_2027395555163308032.md`
- `.agents/reference/MinerU_markdown_1412.6980v9_2030537029501517824.md` (Adam paper)

## What the Adam paper directly recommends

The Adam paper gives one clear default set for Adam:
- `alpha = 0.001`
- `beta1 = 0.9`
- `beta2 = 0.999`
- `epsilon = 1e-8`

Important nuance:
- The paper gives these as good defaults across tested ML problems.
- The paper does **not** directly recommend:
  - warmup
  - decoupled weight decay / AdamW-style weight decay
  - language-model-specific LR schedules

So for Chapter 7:
- use the Adam defaults as the first anchor point
- but still treat LR, warmup, and weight decay as assignment-tuning targets

## What 7.2 asks before the first question

TinyStories baseline setup from the handout:
- `vocab_size = 10000`
- `context_length = 256`
- `d_model = 512`
- `d_ff = 1344`
- `rope_theta = 10000`
- `num_layers = 4`
- `num_heads = 16`
- total tokens processed about `327,680,000`

Hyperparameters explicitly left for tuning:
- learning rate
- warmup
- Adam hyperparameters (`beta1`, `beta2`, `epsilon`)
- weight decay

## Practical interpretation

Use the Adam-paper defaults as the first baseline optimizer.
Then vary only one or two knobs at a time.

Order of tuning should be:
1. learning rate
2. warmup
3. weight decay
4. `beta2`
5. `beta1`
6. `epsilon` only if there is instability or no clear gain

Reason:
- LR and warmup usually dominate early LM stability
- weight decay can noticeably shift final val loss
- `beta2` often matters more than `beta1` for Transformer LM training stability
- `epsilon` usually stays fixed unless numerics suggest otherwise

## First batch of runs to try

### Fixed model baseline
- `vocab_size = 10000`
- `context_length = 256`
- `d_model = 512`
- `d_ff = 1344`
- `rope_theta = 10000`
- `num_layers = 4`
- `num_heads = 16`

### Fixed optimizer defaults for the first sweep
- `beta1 = 0.9`
- `beta2 = 0.999`
- `epsilon = 1e-8`
- `weight_decay = 0.1` as the first practical AdamW guess

### First LR sweep candidates
- `1e-4`
- `3e-4`
- `6e-4`
- `1e-3`
- `2e-3`

Interpretation:
- `1e-3` comes from the Adam paper default
- `3e-4` and `6e-4` are practical smaller nearby anchors
- `2e-3` is included to intentionally probe the edge of instability
- if all of these are stable, add a larger run to force one divergent curve for the handout

### First warmup candidates
- `0`
- `200`
- `500`

Interpretation:
- the paper does not prescribe warmup
- for LM training, warmup is worth testing early because it can separate “bad LR” from “bad cold-start behavior”

### First weight decay candidates
- `0.0`
- `0.01`
- `0.1`

Interpretation:
- the Adam paper discusses L2 regularization in experiments, but not AdamW-style decoupled weight decay defaults
- `0.1` is a reasonable first AdamW anchor for modern Transformer training
- `0.0` is needed as a control

### Only if needed: second-round Adam hyperparameter sweep
- `beta2`: `0.98`, `0.99`, `0.999`
- `beta1`: `0.9`, `0.95`
- `epsilon`: keep `1e-8` unless instability suggests testing `1e-6`

## Recommended first experiment table

| Run family | LR | Warmup | beta1 | beta2 | eps | weight_decay | Purpose |
|---|---:|---:|---:|---:|---:|---:|---|
| lr_sweep_a | 1e-4 | 200 | 0.9 | 0.999 | 1e-8 | 0.1 | low-LR anchor |
| lr_sweep_b | 3e-4 | 200 | 0.9 | 0.999 | 1e-8 | 0.1 | conservative practical baseline |
| lr_sweep_c | 6e-4 | 200 | 0.9 | 0.999 | 1e-8 | 0.1 | mid-range candidate |
| lr_sweep_d | 1e-3 | 200 | 0.9 | 0.999 | 1e-8 | 0.1 | Adam-paper default LR |
| lr_sweep_e | 2e-3 | 200 | 0.9 | 0.999 | 1e-8 | 0.1 | instability probe |

## If one LR looks best, next table

Hold the best LR fixed, then test:

| Run family | LR | Warmup | beta1 | beta2 | eps | weight_decay | Purpose |
|---|---:|---:|---:|---:|---:|---:|---|
| warmup_a | best | 0 | 0.9 | 0.999 | 1e-8 | 0.1 | no warmup control |
| warmup_b | best | 200 | 0.9 | 0.999 | 1e-8 | 0.1 | short warmup |
| warmup_c | best | 500 | 0.9 | 0.999 | 1e-8 | 0.1 | longer warmup |
| wd_a | best | best | 0.9 | 0.999 | 1e-8 | 0.0 | no weight decay |
| wd_b | best | best | 0.9 | 0.999 | 1e-8 | 0.01 | weak decay |
| wd_c | best | best | 0.9 | 0.999 | 1e-8 | 0.1 | stronger decay |

## Key caution
- The Adam paper is about Adam, not AdamW.
- So:
  - `beta1 = 0.9`
  - `beta2 = 0.999`
  - `eps = 1e-8`
  are directly supported by the paper,
  but `weight_decay` and `warmup` are still assignment-specific empirical choices.

## Next action
- Start with the LR sweep table above.
- Record each run in:
  - `artifacts/experiments/chapter7/experiment_log.md`

## LR sweep decision rules

After the first LR sweep, choose the next step using these rules:

1. If one run clearly has the best validation curve and stays stable:
- keep `beta1=0.9`, `beta2=0.999`, `eps=1e-8`
- fix that LR
- move to warmup sweep

2. If the best two LRs are adjacent and both stable:
- choose the better one by:
  - lower best val loss
  - faster wallclock-to-loss improvement
- then optionally add one interpolation LR between them

3. If all tested LRs are stable but none is clearly near instability:
- add one larger LR to force an edge-of-stability comparison
- this helps satisfy the Chapter 7 requirement for at least one divergent run

4. If all tested LRs are unstable or clearly poor:
- lower the sweep range by about `2x` to `3x`
- check whether warmup `200` is too short and consider trying `500`

5. If instability happens only at the first few steps:
- treat this as a possible warmup problem, not automatically an LR problem
- next step should be warmup sweep before changing `beta` values

6. Only touch `beta2`, `beta1`, or `epsilon` after LR and warmup:
- unless there is obvious numerical instability
- or training curves are noisy/erratic across otherwise reasonable LR choices
