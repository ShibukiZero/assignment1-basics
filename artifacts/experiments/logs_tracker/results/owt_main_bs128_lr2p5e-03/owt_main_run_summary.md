# OWT main run summary

- Run: `owt_main_bs128_lr2p5e-03`
- Dataset: `OpenWebText`
- Model: `vocab_size=32000`, `context_length=256`, `d_model=512`, `d_ff=1344`, `num_layers=4`, `num_heads=16`, `rope_theta=10000`
- Optimization: `batch_size=128`, `learning_rate=2.5e-3`, `min_learning_rate=2.5e-4`, `warmup_iters=200`, `cosine_cycle_iters=10000`, `betas=(0.9, 0.999)`, `weight_decay=0.1`
- Best validation loss: `3.888615`
- Best step: `10000`
- Final step: `10000`
- Processed tokens: `327,680,000`
- Total wallclock: `3470.46s` (`57.84 min`)
- Best checkpoint: `/root/autodl-tmp/training_runs/owt_main_bs128_lr2p5e-03/best_checkpoint.pt`
- Latest checkpoint: `/root/autodl-tmp/training_runs/owt_main_bs128_lr2p5e-03/latest_checkpoint.pt`
- TensorBoard: `/root/tf-logs/owt_main_bs128_lr2p5e-03`

Interpretation:

- This run completed the matched `10000`-step Chapter 7.4 budget cleanly.
- The best validation loss occurred at the final evaluation, so this curve does not show an obvious overtraining signal within the matched budget.
