from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import Tensor

from .transformer import softmax


def temperature_scaled_probs(
    logits: Float[Tensor, "vocab_size"],
    temperature: float,
) -> Float[Tensor, "vocab_size"]:
    """
    Convert next-token logits into a probability distribution with temperature scaling.

    Behavior:
    - if `temperature == 0`, return a one-hot distribution at the argmax
    - otherwise, divide logits by temperature and apply softmax
    """
    if temperature < 0:
        raise ValueError(f"Expected temperature >= 0, got {temperature}.")
    if logits.ndim != 1:
        raise ValueError(f"Expected 1D logits, got shape {tuple(logits.shape)}.")

    if temperature == 0:
        probs = torch.zeros_like(logits)
        probs[torch.argmax(logits)] = 1.0
        return probs

    return softmax(logits / temperature, dim=-1)


def top_p_filtering(
    probs: Float[Tensor, "vocab_size"],
    top_p: float,
) -> Float[Tensor, "vocab_size"]:
    """
    Apply nucleus / top-p truncation to a probability distribution.

    Contract:
    - sort tokens by probability descending
    - keep the smallest prefix whose cumulative probability is >= `top_p`
    - zero out everything else
    - renormalize the kept mass to sum to 1
    """
    if probs.ndim != 1:
        raise ValueError(f"Expected 1D probs, got shape {tuple(probs.shape)}.")
    if not 0 < top_p <= 1:
        raise ValueError(f"Expected top_p in (0, 1], got {top_p}.")

    if top_p == 1.0:
        return probs

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)

    keep_mask = cumulative <= top_p
    # Always keep at least the most likely token, and keep the first token that
    # crosses the top-p threshold so the retained set reaches probability >= top_p.
    keep_mask[0] = True
    first_crossing = torch.nonzero(cumulative >= top_p, as_tuple=False)
    if first_crossing.numel() > 0:
        keep_mask[first_crossing[0].item()] = True

    filtered_sorted = torch.where(
        keep_mask,
        sorted_probs,
        torch.zeros_like(sorted_probs),
    )

    filtered = torch.zeros_like(probs)
    filtered[sorted_indices] = filtered_sorted
    return filtered / filtered.sum()


def sample_next_token(
    logits: Float[Tensor, "vocab_size"],
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Int[Tensor, ""]:
    """
    Sample one next token from logits using temperature scaling and optional top-p.
    """
    probs = temperature_scaled_probs(logits, temperature=temperature)
    probs = top_p_filtering(probs, top_p=top_p)
    return torch.multinomial(probs, num_samples=1).squeeze(0)


def decode(
    model: torch.nn.Module,
    prompt: Int[Tensor, "prompt_length"],
    *,
    max_new_tokens: int,
    end_of_text_token_id: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Int[Tensor, "total_length"]:
    """
    Autoregressively decode from a decoder-only language model.

    Expected behavior:
    - start from the user-provided prompt
    - repeatedly run the model on the current prefix
    - take the last-position logits
    - sample one next token using temperature / top-p
    - append it to the sequence
    - stop if the sampled token is `end_of_text_token_id`
      or after `max_new_tokens`

    Notes:
    - `prompt` is expected to be a 1D token-ID tensor
    - most decoder-only models expect a batch dimension, so a common pattern is
      to temporarily unsqueeze to shape `(1, seq_len)`
    """
    if prompt.ndim != 1:
        raise ValueError(f"Expected 1D prompt, got shape {tuple(prompt.shape)}.")
    if max_new_tokens < 0:
        raise ValueError(f"Expected max_new_tokens >= 0, got {max_new_tokens}.")
    if prompt.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"Expected integer prompt IDs, got {prompt.dtype}.")

    was_training = model.training
    generated = prompt.clone()

    try:
        model.eval()
        context_length = getattr(model, "context_length", None)
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                # Decoder-only models can only attend to the most recent context window.
                if context_length is None:
                    model_input = generated
                else:
                    model_input = generated[-context_length:]

                logits: Float[Tensor, "1 current_length vocab_size"] = model(
                    model_input.unsqueeze(0)
                )
                next_token_logits: Float[Tensor, "vocab_size"] = logits[0, -1, :]
                next_token = sample_next_token(
                    next_token_logits,
                    temperature=temperature,
                    top_p=top_p,
                )
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=0)
                if next_token.item() == end_of_text_token_id:
                    break
    finally:
        model.train(was_training)

    return generated
