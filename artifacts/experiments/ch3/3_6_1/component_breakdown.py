"""Compute FLOP breakdowns for Transformer accounting questions."""

from __future__ import annotations


def print_breakdown(
    *,
    name: str,
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
) -> None:
    d_k = d_model // num_heads

    q_proj = 2 * context_length * d_model * d_model
    k_proj = q_proj
    v_proj = q_proj
    output_proj = q_proj
    qk = 2 * num_heads * context_length * d_k * context_length
    av = qk
    w1 = 2 * context_length * d_model * d_ff
    w3 = w1
    w2 = 2 * context_length * d_ff * d_model

    per_layer_components = {
        "attention_projections": q_proj + k_proj + v_proj + output_proj,
        "attention_scores": qk,
        "attention_values": av,
        "ffn": w1 + w3 + w2,
    }
    model_components = {
        key: value * num_layers for key, value in per_layer_components.items()
    }
    model_components["lm_head"] = 2 * context_length * d_model * vocab_size

    total_flops = sum(model_components.values())

    print(f"== {name} ==")
    print(f"context_length={context_length}, num_layers={num_layers}, d_model={d_model}, num_heads={num_heads}, d_ff={d_ff}")
    print(f"total_forward_flops={total_flops}")
    for key, value in model_components.items():
        print(f"{key}={value} ({value / total_flops:.6%})")
    print()


def main() -> None:
    vocab_size = 50_257
    print_breakdown(
        name="gpt2_xl",
        vocab_size=vocab_size,
        context_length=1_024,
        num_layers=48,
        d_model=1_600,
        num_heads=25,
        d_ff=6_400,
    )
    print_breakdown(
        name="gpt2_small",
        vocab_size=vocab_size,
        context_length=1_024,
        num_layers=12,
        d_model=768,
        num_heads=12,
        d_ff=3_072,
    )
    print_breakdown(
        name="gpt2_medium",
        vocab_size=vocab_size,
        context_length=1_024,
        num_layers=24,
        d_model=1_024,
        num_heads=16,
        d_ff=4_096,
    )
    print_breakdown(
        name="gpt2_large",
        vocab_size=vocab_size,
        context_length=1_024,
        num_layers=36,
        d_model=1_280,
        num_heads=20,
        d_ff=5_120,
    )
    print_breakdown(
        name="gpt2_xl_ctx16384",
        vocab_size=vocab_size,
        context_length=16_384,
        num_layers=48,
        d_model=1_600,
        num_heads=25,
        d_ff=6_400,
    )


if __name__ == "__main__":
    main()
