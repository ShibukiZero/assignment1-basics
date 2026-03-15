"""Verify GPT-2 XL accounting values used in writeup.md."""

from __future__ import annotations


def main() -> None:
    vocab_size = 50_257
    context_length = 1_024
    num_layers = 48
    d_model = 1_600
    num_heads = 25
    d_ff = 6_400
    d_k = d_model // num_heads

    token_embeddings = vocab_size * d_model
    attention_params_per_block = 4 * d_model * d_model
    ffn_params_per_block = 3 * d_model * d_ff
    norm_params_per_block = 2 * d_model
    params_per_block = (
        attention_params_per_block + ffn_params_per_block + norm_params_per_block
    )
    final_norm = d_model
    lm_head = vocab_size * d_model

    total_parameters = (
        token_embeddings + num_layers * params_per_block + final_norm + lm_head
    )
    fp32_parameter_bytes = total_parameters * 4

    q_proj_flops = 2 * context_length * d_model * d_model
    k_proj_flops = q_proj_flops
    v_proj_flops = q_proj_flops
    output_proj_flops = q_proj_flops
    qk_flops = 2 * num_heads * context_length * d_k * context_length
    av_flops = qk_flops
    w1_flops = 2 * context_length * d_model * d_ff
    w3_flops = w1_flops
    w2_flops = 2 * context_length * d_ff * d_model
    flops_per_block = (
        q_proj_flops
        + k_proj_flops
        + v_proj_flops
        + output_proj_flops
        + qk_flops
        + av_flops
        + w1_flops
        + w3_flops
        + w2_flops
    )
    all_blocks_flops = num_layers * flops_per_block
    lm_head_flops = 2 * context_length * d_model * vocab_size
    total_forward_flops = all_blocks_flops + lm_head_flops

    print("GPT-2 XL accounting verification")
    print(f"d_k: {d_k}")
    print(f"params_per_block: {params_per_block}")
    print(f"total_parameters: {total_parameters}")
    print(f"fp32_parameter_bytes: {fp32_parameter_bytes}")
    print(f"fp32_parameter_gb: {fp32_parameter_bytes / 1e9:.6f}")
    print(f"fp32_parameter_gib: {fp32_parameter_bytes / (1024 ** 3):.6f}")
    print(f"q_proj_flops: {q_proj_flops}")
    print(f"qk_flops: {qk_flops}")
    print(f"w1_flops: {w1_flops}")
    print(f"flops_per_block: {flops_per_block}")
    print(f"all_blocks_flops: {all_blocks_flops}")
    print(f"lm_head_flops: {lm_head_flops}")
    print(f"total_forward_flops: {total_forward_flops}")


if __name__ == "__main__":
    main()
