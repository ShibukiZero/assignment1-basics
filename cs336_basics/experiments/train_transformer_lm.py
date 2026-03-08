from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

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
    run_name: str
    timestamp_utc: str
    wallclock_seconds: float
    step: int
    tokens_seen: int
    split: str
    loss: float
    perplexity: float
    learning_rate: float
    batch_size: int
    context_length: int


@dataclass
class DiagnosticsMetrics:
    run_name: str
    timestamp_utc: str
    wallclock_seconds: float
    step: int
    tokens_seen: int
    learning_rate: float
    grad_norm_pre_clip: float
    grad_norm_post_clip: float
    param_norm: float


@dataclass
class RunSummary:
    run_name: str
    final_step: int
    total_wallclock_seconds: float
    best_val_loss: float | None
    best_val_step: int | None
    latest_checkpoint: str
    best_checkpoint: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chapter 7 training script with experiment logging.",
    )
    parser.add_argument("--train-npy", type=Path, required=True)
    parser.add_argument("--val-npy", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--tensorboard-root", type=Path, default=Path("/root/tf-logs"))
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

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
    raise ValueError(
        "Only float32 training is supported in this assignment script; "
        f"got dtype={raw}."
    )


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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def maybe_make_summary_writer(
    *,
    tensorboard_dir: Path,
    disable_tensorboard: bool,
) -> SummaryWriter | None:
    if disable_tensorboard:
        print("tensorboard=disabled_by_flag")
        return None
    if SummaryWriter is None:
        print(
            "tensorboard=unavailable "
            "(torch.utils.tensorboard could not be imported; "
            "structured logs will still be written)"
        )
        return None
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    print(f"tensorboard=enabled log_dir={tensorboard_dir}")
    return SummaryWriter(log_dir=str(tensorboard_dir))


def global_grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    grad_norm_sq = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        grad_norm_sq += float(torch.sum(grad * grad).item())
    return grad_norm_sq**0.5


def global_param_norm(parameters: list[torch.nn.Parameter]) -> float:
    param_norm_sq = 0.0
    for param in parameters:
        value = param.detach()
        param_norm_sq += float(torch.sum(value * value).item())
    return param_norm_sq**0.5


def resolve_log_dir(*, requested_log_dir: Path | None, run_name: str) -> Path:
    if requested_log_dir is not None:
        return requested_log_dir
    return Path(".agents/logs") / run_name


def make_run_config(
    *,
    args: argparse.Namespace,
    run_name: str,
    log_dir: Path,
    tensorboard_dir: Path,
    tensorboard_enabled: bool,
) -> dict:
    return {
        "run_name": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_npy": str(args.train_npy),
        "val_npy": str(args.val_npy),
        "output_dir": str(args.output_dir),
        "log_dir": str(log_dir),
        "tensorboard_dir": str(tensorboard_dir),
        "tensorboard_enabled": tensorboard_enabled,
        "device": args.device,
        "dtype": args.dtype,
        "seed": args.seed,
        "model": {
            "vocab_size": args.vocab_size,
            "context_length": args.context_length,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "d_ff": args.d_ff,
            "rope_theta": args.rope_theta,
        },
        "optimization": {
            "batch_size": args.batch_size,
            "max_steps": args.max_steps,
            "eval_interval": args.eval_interval,
            "eval_batches": args.eval_batches,
            "checkpoint_interval": args.checkpoint_interval,
            "learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "warmup_iters": args.warmup_iters,
            "cosine_cycle_iters": args.cosine_cycle_iters,
            "betas": list(args.betas),
            "eps": args.eps,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
        },
        "resume_from": str(args.resume_from) if args.resume_from is not None else None,
    }


def flatten_lm_batch(
    logits: Float[Tensor, "batch_size sequence_length vocab_size"],
    targets: Int[Tensor, "batch_size sequence_length"],
) -> tuple[Float[Tensor, "batch_size_times_sequence_length vocab_size"], Int[Tensor, "batch_size_times_sequence_length"]]:
    """Flatten `(B, T, V)` logits and `(B, T)` targets into cross-entropy inputs."""
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    return flat_logits, flat_targets


def save_training_checkpoint(
    *,
    model: TransformerLM,
    optimizer: AdamW,
    iteration: int,
    output_path: Path,
) -> None:
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=iteration,
        out=output_path,
    )


def evaluate_loss(
    *,
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
    step: int,
    run_name: str,
    split: str,
    wallclock_seconds: float,
    tokens_seen: int,
    learning_rate: float,
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
        run_name=run_name,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        wallclock_seconds=wallclock_seconds,
        step=step,
        tokens_seen=tokens_seen,
        split=split,
        loss=avg_loss,
        perplexity=ppl,
        learning_rate=learning_rate,
        batch_size=batch_size,
        context_length=context_length,
    )


def main() -> None:
    args = parse_args()
    if args.eval_interval <= 0:
        raise ValueError(f"Expected eval_interval > 0, got {args.eval_interval}.")
    if args.checkpoint_interval <= 0:
        raise ValueError(
            f"Expected checkpoint_interval > 0, got {args.checkpoint_interval}."
        )
    if args.eval_batches <= 0:
        raise ValueError(f"Expected eval_batches > 0, got {args.eval_batches}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or args.output_dir.name
    log_dir = resolve_log_dir(requested_log_dir=args.log_dir, run_name=run_name)
    tensorboard_dir = args.tensorboard_root / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    train_tokens = load_token_array(args.train_npy)
    val_tokens = load_token_array(args.val_npy)
    dtype = resolve_dtype(args.dtype)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
        start_step = load_checkpoint(
            args.resume_from,
            model=model,
            optimizer=optimizer,
            map_location=torch.device(args.device),
        )

    metrics_path = log_dir / "metrics.jsonl"
    diagnostics_path = log_dir / "diagnostics.jsonl"
    config_path = log_dir / "config.json"
    summary_path = log_dir / "summary.json"
    latest_checkpoint = args.output_dir / "latest_checkpoint.pt"
    best_checkpoint = args.output_dir / "best_checkpoint.pt"
    writer = maybe_make_summary_writer(
        tensorboard_dir=tensorboard_dir,
        disable_tensorboard=args.disable_tensorboard,
    )
    write_json(
        config_path,
        make_run_config(
            args=args,
            run_name=run_name,
            log_dir=log_dir,
            tensorboard_dir=tensorboard_dir,
            tensorboard_enabled=writer is not None,
        ),
    )

    start_time = time.perf_counter()
    best_val_loss: float | None = None
    best_val_step: int | None = None
    model_parameters = [param for param in model.parameters()]

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
        grad_norm_pre_clip = global_grad_norm(model_parameters)
        gradient_clipping(model.parameters(), args.grad_clip)
        grad_norm_post_clip = global_grad_norm(model_parameters)
        optimizer.step()
        param_norm = global_param_norm(model_parameters)
        completed_steps = step + 1

        if completed_steps % args.eval_interval == 0:
            wallclock_seconds = time.perf_counter() - start_time
            tokens_seen = completed_steps * args.batch_size * args.context_length
            diagnostics = DiagnosticsMetrics(
                run_name=run_name,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                wallclock_seconds=wallclock_seconds,
                step=completed_steps,
                tokens_seen=tokens_seen,
                learning_rate=learning_rate,
                grad_norm_pre_clip=grad_norm_pre_clip,
                grad_norm_post_clip=grad_norm_post_clip,
                param_norm=param_norm,
            )
            train_metrics = evaluate_loss(
                model=model,
                dataset=train_tokens,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
                num_batches=args.eval_batches,
                step=completed_steps,
                run_name=run_name,
                split="train",
                wallclock_seconds=wallclock_seconds,
                tokens_seen=tokens_seen,
                learning_rate=learning_rate,
            )

            val_metrics = evaluate_loss(
                model=model,
                dataset=val_tokens,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
                num_batches=args.eval_batches,
                step=completed_steps,
                run_name=run_name,
                split="val",
                wallclock_seconds=wallclock_seconds,
                tokens_seen=tokens_seen,
                learning_rate=learning_rate,
            )

            append_jsonl(diagnostics_path, asdict(diagnostics))
            print(
                f"step={diagnostics.step} diagnostics "
                f"grad_pre={diagnostics.grad_norm_pre_clip:.6f} "
                f"grad_post={diagnostics.grad_norm_post_clip:.6f} "
                f"param_norm={diagnostics.param_norm:.6f}"
            )

            for metrics in (train_metrics, val_metrics):
                append_jsonl(metrics_path, asdict(metrics))
                print(
                    f"step={metrics.step} split={metrics.split} "
                    f"loss={metrics.loss:.6f} perplexity={metrics.perplexity:.6f} "
                    f"lr={metrics.learning_rate:.6g} "
                    f"wallclock={metrics.wallclock_seconds:.2f}s "
                    f"tokens={metrics.tokens_seen}"
                )
                if writer is not None:
                    writer.add_scalar(f"loss/{metrics.split}", metrics.loss, metrics.step)
                    writer.add_scalar(
                        f"perplexity/{metrics.split}",
                        metrics.perplexity,
                        metrics.step,
                    )

            if writer is not None:
                writer.add_scalar("lr", learning_rate, completed_steps)
                writer.add_scalar(
                    "time/wallclock_seconds", wallclock_seconds, completed_steps
                )
                writer.add_scalar("tokens/seen", tokens_seen, completed_steps)
                writer.add_scalar(
                    "grad/global_norm_pre_clip",
                    diagnostics.grad_norm_pre_clip,
                    completed_steps,
                )
                writer.add_scalar(
                    "grad/global_norm_post_clip",
                    diagnostics.grad_norm_post_clip,
                    completed_steps,
                )
                writer.add_scalar(
                    "param/global_norm",
                    diagnostics.param_norm,
                    completed_steps,
                )
                writer.flush()

            if best_val_loss is None or val_metrics.loss < best_val_loss:
                best_val_loss = val_metrics.loss
                best_val_step = val_metrics.step
                save_training_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=completed_steps,
                    output_path=best_checkpoint,
                )

        if completed_steps % args.checkpoint_interval == 0:
            save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=completed_steps,
                output_path=latest_checkpoint,
            )

    final_iteration = args.max_steps
    if final_iteration > start_step:
        save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            iteration=final_iteration,
            output_path=latest_checkpoint,
        )

    total_wallclock_seconds = time.perf_counter() - start_time
    write_json(
        summary_path,
        asdict(
            RunSummary(
                run_name=run_name,
                final_step=args.max_steps,
                total_wallclock_seconds=total_wallclock_seconds,
                best_val_loss=best_val_loss,
                best_val_step=best_val_step,
                latest_checkpoint=str(latest_checkpoint),
                best_checkpoint=str(best_checkpoint) if best_val_loss is not None else None,
            )
        ),
    )
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
