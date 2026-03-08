from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor

# Support both:
# 1) `python -m cs336_basics.experiments.train_transformer_lm`
# 2) `python cs336_basics/experiments/train_transformer_lm.py`
if __package__:
    from ..optim import AdamW, lr_cosine_schedule
    from ..training import (
        cross_entropy,
        get_batch,
        gradient_clipping,
        load_checkpoint,
        perplexity,
        save_checkpoint,
    )
    from ..transformer import TransformerLM
else:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from cs336_basics.optim import AdamW, lr_cosine_schedule
    from cs336_basics.training import (
        cross_entropy,
        get_batch,
        gradient_clipping,
        load_checkpoint,
        perplexity,
        save_checkpoint,
    )
    from cs336_basics.transformer import TransformerLM


@dataclass
class EvalMetrics:
    step: int
    split: str
    loss: float
    perplexity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scaffold training script for Assignment 1 Chapter 5.",
    )
    parser.add_argument("--train-npy", type=Path, required=True)
    parser.add_argument("--val-npy", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=500)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--cosine-cycle-iters", type=int, default=1_000)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--resume-from", type=Path, default=None)
    return parser.parse_args()


def resolve_dtype(raw: str) -> torch.dtype:
    if raw == "float32":
        return torch.float32
    if raw == "bfloat16":
        return torch.bfloat16
    if raw == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {raw}")


def load_token_array(path: Path) -> np.ndarray:
    """Load token IDs lazily from a `.npy` file using numpy memmap mode."""
    return np.load(path, mmap_mode="r")


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def flatten_lm_batch(
    logits: Float[Tensor, "batch_size sequence_length vocab_size"],
    targets: Int[Tensor, "batch_size sequence_length"],
) -> tuple[Float[Tensor, "batch_size_times_sequence_length vocab_size"], Int[Tensor, "batch_size_times_sequence_length"]]:
    """Flatten `(B, T, V)` logits and `(B, T)` targets into cross-entropy inputs."""
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    return flat_logits, flat_targets


def evaluate_loss(
    *,
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
    step: int,
    split: str,
) -> EvalMetrics:
    """Evaluate average loss / perplexity on a split."""
    was_training = model.training
    model.eval()
    loss_values: list[float] = []
    with torch.inference_mode():
        for _ in range(num_batches):
            x, y = get_batch(
                dataset=dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )
            logits = model(x)
            flat_logits, flat_targets = flatten_lm_batch(logits, y)
            loss = cross_entropy(flat_logits, flat_targets)
            loss_values.append(float(loss.detach().cpu().item()))

    losses = torch.tensor(loss_values, dtype=torch.float32)
    avg_loss = float(losses.mean().item())
    ppl = float(perplexity(losses).item())

    if was_training:
        model.train()

    return EvalMetrics(
        step=step,
        split=split,
        loss=avg_loss,
        perplexity=ppl,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_tokens = load_token_array(args.train_npy)
    val_tokens = load_token_array(args.val_npy)
    dtype = resolve_dtype(args.dtype)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=torch.device(args.device),
        dtype=dtype,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume_from is not None:
        start_step = load_checkpoint(args.resume_from, model=model, optimizer=optimizer)

    metrics_path = args.output_dir / "metrics.jsonl"
    latest_checkpoint = args.output_dir / "latest_checkpoint.pt"

    for step in range(start_step, args.max_steps):
        learning_rate = lr_cosine_schedule(
            it=step,
            max_learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )
        set_learning_rate(optimizer, learning_rate)

        x, y = get_batch(
            dataset=train_tokens,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)

        flat_logits, flat_targets = flatten_lm_batch(logits, y)

        loss = cross_entropy(flat_logits, flat_targets)

        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.eval_interval == 0:
            train_metrics = evaluate_loss(
                model=model,
                dataset=train_tokens,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
                num_batches=args.eval_batches,
                step=step,
                split="train",
            )

            val_metrics = evaluate_loss(
                model=model,
                dataset=val_tokens,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
                num_batches=args.eval_batches,
                step=step,
                split="val",
            )

            for metrics in (train_metrics, val_metrics):
                append_jsonl(metrics_path, asdict(metrics))
                print(
                    f"step={metrics.step} split={metrics.split} "
                    f"loss={metrics.loss:.6f} perplexity={metrics.perplexity:.6f}"
                )

        if step > start_step and step % args.checkpoint_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=step,
                out=latest_checkpoint,
            )


if __name__ == "__main__":
    main()
