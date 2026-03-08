from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor

if __package__:
    from ..decoding import decode
    from ..optim import AdamW, lr_cosine_schedule
    from ..tokenizer import Tokenizer
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

    from cs336_basics.decoding import decode
    from cs336_basics.optim import AdamW, lr_cosine_schedule
    from cs336_basics.tokenizer import Tokenizer
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
class SanityReport:
    train_losses: list[float]
    val_loss: float
    val_perplexity: float
    checkpoint_iteration: int
    restored_iteration: int
    logits_match_after_restore: bool
    generated_token_ids: list[int]
    generated_text: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a short end-to-end sanity check for the training pipeline.",
    )
    parser.add_argument("--train-npy", type=Path, required=True)
    parser.add_argument("--val-npy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--cosine-cycle-iters", type=int, default=3)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eot-token-id", type=int, default=9999)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
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
    return np.load(path, mmap_mode="r")


def maybe_load_tokenizer(tokenizer_dir: Path | None) -> Tokenizer | None:
    if tokenizer_dir is None:
        return None

    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.json"
    if not vocab_path.exists() or not merges_path.exists():
        raise FileNotFoundError(
            f"Tokenizer directory must contain vocab.json and merges.json: {tokenizer_dir}",
        )

    return Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=["<|endoftext|>"],
    )


def build_model_and_optimizer(args: argparse.Namespace) -> tuple[TransformerLM, AdamW]:
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=torch.device(args.device),
        dtype=resolve_dtype(args.dtype),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    return model, optimizer


def flatten_lm_batch(
    logits: Float[Tensor, "batch_size sequence_length vocab_size"],
    targets: Int[Tensor, "batch_size sequence_length"],
) -> tuple[
    Float[Tensor, "batch_size_times_sequence_length vocab_size"],
    Int[Tensor, "batch_size_times_sequence_length"],
]:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    return flat_logits, flat_targets


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def train_step(
    *,
    model: TransformerLM,
    optimizer: AdamW,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    step: int,
    args: argparse.Namespace,
) -> float:
    learning_rate = lr_cosine_schedule(
        it=step,
        max_learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.cosine_cycle_iters,
    )
    set_learning_rate(optimizer, learning_rate)

    x, y = get_batch(
        dataset=dataset,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    flat_logits, flat_targets = flatten_lm_batch(logits, y)
    loss = cross_entropy(flat_logits, flat_targets)
    loss.backward()
    gradient_clipping(model.parameters(), args.grad_clip)
    optimizer.step()
    return float(loss.detach().cpu().item())


def evaluate_once(
    *,
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    losses: list[float] = []
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
            losses.append(float(loss.detach().cpu().item()))

    if was_training:
        model.train()

    losses_tensor = torch.tensor(losses, dtype=torch.float32)
    return float(losses_tensor.mean().item()), float(perplexity(losses_tensor).item())


def compare_restore_logits(
    original_model: TransformerLM,
    restored_model: TransformerLM,
    prompt: Int[Tensor, "prompt_length"],
) -> bool:
    with torch.inference_mode():
        original_logits = original_model(prompt.unsqueeze(0))
        restored_logits = restored_model(prompt.unsqueeze(0))
    return bool(torch.allclose(original_logits, restored_logits, atol=1e-6, rtol=1e-5))


def main() -> None:
    args = parse_args()
    if args.train_steps <= 0:
        raise ValueError(f"Expected train_steps > 0, got {args.train_steps}.")
    if args.eval_batches <= 0:
        raise ValueError(f"Expected eval_batches > 0, got {args.eval_batches}.")
    if args.prompt_length <= 0:
        raise ValueError(f"Expected prompt_length > 0, got {args.prompt_length}.")
    if args.prompt_length > args.context_length:
        raise ValueError(
            "Expected prompt_length <= context_length, "
            f"got prompt_length={args.prompt_length}, context_length={args.context_length}.",
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_tokens = load_token_array(args.train_npy)
    val_tokens = load_token_array(args.val_npy)
    if len(val_tokens) < args.prompt_length:
        raise ValueError(
            "Validation token array is shorter than prompt_length: "
            f"len(val_tokens)={len(val_tokens)}, prompt_length={args.prompt_length}.",
        )
    tokenizer = maybe_load_tokenizer(args.tokenizer_dir)

    model, optimizer = build_model_and_optimizer(args)
    train_losses: list[float] = []
    for step in range(args.train_steps):
        loss = train_step(
            model=model,
            optimizer=optimizer,
            dataset=train_tokens,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
            step=step,
            args=args,
        )
        train_losses.append(loss)
        print(f"train_step={step} loss={loss:.6f}")

    val_loss, val_ppl = evaluate_once(
        model=model,
        dataset=val_tokens,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=args.device,
        num_batches=args.eval_batches,
    )
    print(f"val_loss={val_loss:.6f} val_perplexity={val_ppl:.6f}")

    checkpoint_path = args.output_dir / "sanity_checkpoint.pt"
    checkpoint_iteration = args.train_steps
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=checkpoint_iteration,
        out=checkpoint_path,
    )
    print(f"checkpoint_saved={checkpoint_path}")

    restored_model, restored_optimizer = build_model_and_optimizer(args)
    restored_iteration = load_checkpoint(
        checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        map_location=torch.device(args.device),
    )

    prompt_tokens = torch.tensor(
        np.asarray(val_tokens[: args.prompt_length]),
        device=args.device,
        dtype=torch.long,
    )
    restore_ok = compare_restore_logits(model, restored_model, prompt_tokens)
    print(f"restored_iteration={restored_iteration}")
    print(f"logits_match_after_restore={restore_ok}")

    generated_ids = decode(
        restored_model,
        prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        end_of_text_token_id=args.eot_token_id,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    generated_ids_list = generated_ids.detach().cpu().tolist()
    print(f"generated_token_ids={generated_ids_list}")

    generated_text: str | None = None
    if tokenizer is not None:
        generated_text = tokenizer.decode(generated_ids_list)
        print("generated_text:")
        print(generated_text)

    report = SanityReport(
        train_losses=train_losses,
        val_loss=val_loss,
        val_perplexity=val_ppl,
        checkpoint_iteration=checkpoint_iteration,
        restored_iteration=restored_iteration,
        logits_match_after_restore=restore_ok,
        generated_token_ids=generated_ids_list,
        generated_text=generated_text,
    )
    report_path = args.output_dir / "sanity_report.json"
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(f"report_written={report_path}")


if __name__ == "__main__":
    main()
