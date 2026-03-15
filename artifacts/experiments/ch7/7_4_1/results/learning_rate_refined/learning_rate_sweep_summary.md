# Learning-rate sweep summary

| LR | Best val loss | Best step | Final step | Wallclock (s) |
|---:|---:|---:|---:|---:|
| 0.0015 | 5.260350 | 400 | 400 | 146.73 |
| 0.002 | 5.240200 | 400 | 400 | 146.89 |
| 0.0025 | 5.228240 | 400 | 400 | 146.61 |
| 0.003 | 5.244215 | 400 | 400 | 146.56 |
| 0.0035 | 5.250064 | 400 | 400 | 146.66 |

## Best LR

- Best LR: `0.0025`
- Best validation loss: `5.228240`
- Recommended OWT main-run config: `bs=128`, `lr=0.0025`, `min_lr=0.00025`, `max_steps=10000`
- Runtime estimate from the pilot: about `146.6s / 400 steps`, or roughly `61 minutes / 10000 steps` before extra TensorBoard and checkpoint overhead
