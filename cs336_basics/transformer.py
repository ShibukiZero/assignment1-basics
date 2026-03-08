from __future__ import annotations

import math

import torch
from einops import einsum, reduce, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn


class Linear(nn.Module):
    """Bias-free linear layer used in Assignment 1, section 3.4.2."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Store W with shape (d_out, d_in), matching the handout contract.
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        # Init: N(0, 2/(d_in + d_out)), truncated to [-3*std, 3*std].
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (..., in_features).

        Returns:
            Tensor of shape (..., out_features).
        """
        # Shape checkpoint:
        # - input last dim: in_features
        # - output last dim: out_features
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected x.shape[-1] == {self.in_features}, got {x.shape[-1]}."
            )
        return einsum(
            x,
            self.weight,
            "... in_features, out_features in_features -> ... out_features",
        )


class Embedding(nn.Module):
    """Token embedding layer used in Assignment 1, section 3.4.3."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Store embedding table with shape (vocab_size, d_model).
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        # Init: N(0, 1), truncated to [-3, 3].
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Args:
            token_ids: Tensor of shape (...) containing token IDs.

        Returns:
            Tensor of shape (..., embedding_dim).
        """
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                f"Expected token_ids dtype int32 or int64, got {token_ids.dtype}."
            )

        # Shape checkpoint:
        # - input: (...)
        # - output: (..., embedding_dim)
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """RMSNorm layer used in Assignment 1, section 3.5.1."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (..., d_model).

        Returns:
            Tensor of shape (..., d_model).
        """
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected x.shape[-1] == {self.d_model}, got {x.shape[-1]}."
            )

        in_dtype = x.dtype
        x_float = x.to(torch.float32)
        mean_sq: Float[Tensor, "... 1"] = reduce(
            x_float.square(),
            "... d_model -> ... 1",
            "mean",
        )
        rms: Float[Tensor, "... 1"] = torch.sqrt(mean_sq + self.eps)
        normalized: Float[Tensor, "... d_model"] = (x_float / rms) * self.weight
        return normalized.to(dtype=in_dtype)


def silu(in_features: Tensor) -> Tensor:
    """
    Elementwise SiLU activation.

    Args:
        in_features: Tensor of arbitrary shape.

    Returns:
        Tensor of the same shape.
    """
    return in_features * torch.sigmoid(in_features)


class SwiGLU(nn.Module):
    """Position-wise feed-forward layer from section 3.5.2."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # FFN(x) = W2( SiLU(W1 x) * (W3 x) )
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, in_features: Tensor) -> Tensor:
        """
        Args:
            in_features: Tensor of shape (..., d_model).

        Returns:
            Tensor of shape (..., d_model).
        """
        if in_features.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected in_features.shape[-1] == {self.d_model}, got {in_features.shape[-1]}."
            )

        up: Float[Tensor, "... d_ff"] = self.w1(in_features)
        gate: Float[Tensor, "... d_ff"] = self.w3(in_features)
        activated: Float[Tensor, "... d_ff"] = silu(up)
        gated: Float[Tensor, "... d_ff"] = activated * gate
        out_features: Float[Tensor, "... d_model"] = self.w2(gated)
        return out_features


class RotaryPositionalEmbedding(nn.Module):
    """RoPE module used in Assignment 1, section 3.5.3."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"Expected even d_k, got {d_k}.")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        half_d_k = d_k // 2

        i_seq: Int[Tensor, "max_seq_len"] = torch.arange(max_seq_len, device=device)
        k_seq: Int[Tensor, "half_d_k"] = torch.arange(half_d_k, device=device)
        theta_seq: Float[Tensor, "max_seq_len half_d_k"] = i_seq[:, None] / (
            theta ** ((2 * k_seq[None, :]) / d_k)
        )
        self.register_buffer("cos_cached", torch.cos(theta_seq), persistent=False)
        self.register_buffer("sin_cached", torch.sin(theta_seq), persistent=False)

    def forward(
        self,
        in_query_or_key: Float[Tensor, "... sequence_length d_k"],
        token_positions: Int[Tensor, "... sequence_length"],
    ) -> Float[Tensor, "... sequence_length d_k"]:
        """
        Apply RoPE to query/key features.

        Args:
            in_query_or_key: Tensor of shape (..., sequence_length, d_k).
            token_positions: Tensor of shape (..., sequence_length).

        Returns:
            Tensor of shape (..., sequence_length, d_k).
        """
        if in_query_or_key.shape[-1] != self.d_k:
            raise ValueError(
                f"Expected in_query_or_key.shape[-1] == {self.d_k}, got {in_query_or_key.shape[-1]}."
            )
        if token_positions.max() >= self.max_seq_len:
            raise ValueError(
                f"Expected token positions < {self.max_seq_len}, got max={token_positions.max().item()}."
            )

        in_dtype = in_query_or_key.dtype

        cos_selected: Float[Tensor, "... sequence_length half_d_k"] = self.cos_cached[token_positions]
        sin_selected: Float[Tensor, "... sequence_length half_d_k"] = self.sin_cached[token_positions]
        x_pair = rearrange(in_query_or_key, "... seq_len (half two) -> ... seq_len half two", two=2)
        x_even = x_pair[..., 0]
        x_odd = x_pair[..., 1]

        # Align cached RoPE factors with any extra dimensions in query/key
        # (e.g., num_heads) by inserting singleton axes before sequence_length.
        while cos_selected.ndim < x_even.ndim:
            cos_selected = cos_selected.unsqueeze(-3)
            sin_selected = sin_selected.unsqueeze(-3)

        rot_even = x_even * cos_selected - x_odd * sin_selected
        rot_odd = x_even * sin_selected + x_odd * cos_selected
        rot_pair = torch.stack([rot_even, rot_odd], dim=-1)
        out_features: Float[Tensor, "... sequence_length d_k"] = rearrange(
            rot_pair,
            "... seq_len half two -> ... seq_len (half two)",
        )
        return out_features.to(dtype=in_dtype)


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Numerically stable softmax over a chosen dimension.

    Args:
        in_features: Tensor of arbitrary shape.
        dim: Dimension to normalize over.

    Returns:
        Tensor of the same shape with probabilities along `dim`.
    """
    max_vals: Float[Tensor, " ..."] = in_features.max(dim=dim, keepdim=True).values
    shifted: Float[Tensor, " ..."] = in_features - max_vals
    exp_vals: Float[Tensor, " ..."] = torch.exp(shifted)
    out_features: Float[Tensor, " ..."] = exp_vals / exp_vals.sum(dim=dim, keepdim=True)

    return out_features


def scaled_dot_product_attention(
    Q: Float[Tensor, "batch_size ... queries d_k"],
    K: Float[Tensor, "batch_size ... keys d_k"],
    V: Float[Tensor, "batch_size ... keys d_v"],
    mask: Bool[Tensor, "batch_size ... queries keys"] | None = None,
) -> Float[Tensor, "batch_size ... queries d_v"]:
    """
    Scaled dot-product attention from section 3.5.4.

    Args:
        Q: Query tensor of shape (batch_size, ..., queries, d_k).
        K: Key tensor of shape (batch_size, ..., keys, d_k).
        V: Value tensor of shape (batch_size, ..., keys, d_v).
        mask: Optional bool mask of shape (batch_size, ..., queries, keys),
            where True means "visible" and False means "masked out".

    Returns:
        Tensor of shape (batch_size, ..., queries, d_v).
    """
    if Q.shape[-1] != K.shape[-1]:
        raise ValueError(
            f"Expected Q and K to share d_k, got {Q.shape[-1]} and {K.shape[-1]}."
        )
    if K.shape[-2] != V.shape[-2]:
        raise ValueError(
            f"Expected K and V to share key length, got {K.shape[-2]} and {V.shape[-2]}."
        )
    if mask is not None and mask.shape[-2:] != (Q.shape[-2], K.shape[-2]):
        raise ValueError(
            "Expected mask.shape[-2:] == (queries, keys), "
            f"got {mask.shape[-2:]} vs {(Q.shape[-2], K.shape[-2])}."
        )

    scale: float = math.sqrt(Q.shape[-1])
    attention_logits: Float[Tensor, "batch_size ... queries keys"] = einsum(
        Q,
        K,
        "batch_size ... queries d_k, batch_size ... keys d_k -> batch_size ... queries keys",
    )
    attention_logits: Float[Tensor, "batch_size ... queries keys"] = attention_logits / scale
    if mask is not None:
        attention_logits: Float[Tensor, "batch_size ... queries keys"] = (
            attention_logits.masked_fill(~mask, -float("inf"))
        )
    attention_weights: Float[Tensor, "batch_size ... queries keys"] = softmax(
        attention_logits,
        dim=-1,
    )
    out_features: Float[Tensor, "batch_size ... queries d_v"] = einsum(
        attention_weights,
        V,
        "batch_size ... queries keys, batch_size ... keys d_v -> batch_size ... queries d_v",
    )
    return out_features.to(dtype=Q.dtype)


class CausalMultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention from section 3.5.5."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"Expected d_model divisible by num_heads, got {d_model} and {num_heads}."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.use_rope = max_seq_len is not None and theta is not None

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope: RotaryPositionalEmbedding | None = None
        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device,
            )

    def forward(
        self,
        in_features: Float[Tensor, "batch_size ... sequence_length d_model"],
        token_positions: Int[Tensor, "batch_size ... sequence_length"] | None = None,
    ) -> Float[Tensor, "batch_size ... sequence_length d_model"]:
        """
        Args:
            in_features: Tensor of shape (batch_size, ..., sequence_length, d_model).
            token_positions: Optional tensor of shape (batch_size, ..., sequence_length).
                Required when RoPE is enabled.

        Returns:
            Tensor of shape (batch_size, ..., sequence_length, d_model).
        """
        if in_features.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected in_features.shape[-1] == {self.d_model}, got {in_features.shape[-1]}."
            )
        if self.use_rope:
            if token_positions is None:
                raise ValueError("Expected token_positions when RoPE is enabled.")
            if token_positions.shape[-1] != in_features.shape[-2]:
                raise ValueError(
                    "Expected token_positions.shape[-1] == in_features.shape[-2], "
                    f"got {token_positions.shape[-1]} and {in_features.shape[-2]}."
                )

        sequence_length = in_features.shape[-2]
        q_proj: Float[Tensor, "batch_size ... sequence_length d_model"] = self.q_proj(
            in_features
        )
        k_proj: Float[Tensor, "batch_size ... sequence_length d_model"] = self.k_proj(
            in_features
        )
        v_proj: Float[Tensor, "batch_size ... sequence_length d_model"] = self.v_proj(
            in_features
        )
        q_heads: Float[Tensor, "batch_size ... num_heads sequence_length d_k"] = rearrange(
            q_proj,
            "batch_size ... sequence_length (num_heads d_k) -> batch_size ... num_heads sequence_length d_k",
            num_heads=self.num_heads,
            d_k=self.d_k,
        )
        k_heads: Float[Tensor, "batch_size ... num_heads sequence_length d_k"] = rearrange(
            k_proj,
            "batch_size ... sequence_length (num_heads d_k) -> batch_size ... num_heads sequence_length d_k",
            num_heads=self.num_heads,
            d_k=self.d_k,
        )
        v_heads: Float[Tensor, "batch_size ... num_heads sequence_length d_k"] = rearrange(
            v_proj,
            "batch_size ... sequence_length (num_heads d_k) -> batch_size ... num_heads sequence_length d_k",
            num_heads=self.num_heads,
            d_k=self.d_k,
        )
        if self.use_rope:
            q_heads = self.rope(q_heads, token_positions)
            k_heads = self.rope(k_heads, token_positions)
        causal_mask: Bool[Tensor, "sequence_length sequence_length"] = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=in_features.device,
            )
        )
        attn_out_heads: Float[Tensor, "batch_size ... num_heads sequence_length d_k"] = scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            causal_mask,
        )
        merged_heads: Float[Tensor, "batch_size ... sequence_length d_model"] = rearrange(
            attn_out_heads,
            "batch_size ... num_heads sequence_length d_k -> batch_size ... sequence_length (num_heads d_k)",
        )
        out_features: Float[Tensor, "batch_size ... sequence_length d_model"] = self.output_proj(
            merged_heads
        )
        return out_features.to(dtype=in_features.dtype)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block from sections 3.5 and 3.6."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        in_features: Float[Tensor, "batch_size sequence_length d_model"],
        token_positions: Int[Tensor, "batch_size sequence_length"] | None = None,
    ) -> Float[Tensor, "batch_size sequence_length d_model"]:
        """
        Args:
            in_features: Tensor of shape (batch_size, sequence_length, d_model).
            token_positions: Optional tensor of shape (batch_size, sequence_length).

        Returns:
            Tensor of shape (batch_size, sequence_length, d_model).
        """
        if in_features.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected in_features.shape[-1] == {self.d_model}, got {in_features.shape[-1]}."
            )
        if token_positions is None:
            token_positions = torch.arange(
                in_features.shape[-2],
                device=in_features.device,
                dtype=torch.int64,
            )
            token_positions = rearrange(token_positions, "sequence_length -> 1 sequence_length")
            token_positions = token_positions.expand(in_features.shape[0], -1)

        normed_attn_in: Float[Tensor, "batch_size sequence_length d_model"] = self.ln1(
            in_features
        )
        attn_out: Float[Tensor, "batch_size sequence_length d_model"] = self.attn(
            normed_attn_in,
            token_positions=token_positions,
        )
        residual_attn: Float[Tensor, "batch_size sequence_length d_model"] = (
            in_features + attn_out
        )
        normed_ffn_in: Float[Tensor, "batch_size sequence_length d_model"] = self.ln2(
            residual_attn
        )
        ffn_out: Float[Tensor, "batch_size sequence_length d_model"] = self.ffn(
            normed_ffn_in
        )
        out_features: Float[Tensor, "batch_size sequence_length d_model"] = (
            residual_attn + ffn_out
        )
        return out_features.to(dtype=in_features.dtype)


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model from section 3.6."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        in_indices: Int[Tensor, "batch_size sequence_length"],
    ) -> Float[Tensor, "batch_size sequence_length vocab_size"]:
        """
        Args:
            in_indices: Tensor of shape (batch_size, sequence_length).

        Returns:
            Tensor of shape (batch_size, sequence_length, vocab_size).
        """
        if in_indices.shape[-1] > self.context_length:
            raise ValueError(
                f"Expected sequence length <= {self.context_length}, got {in_indices.shape[-1]}."
            )

        token_positions: Int[Tensor, "batch_size sequence_length"] = torch.arange(
            in_indices.shape[-1],
            device=in_indices.device,
            dtype=torch.int64,
        )
        token_positions = rearrange(token_positions, "sequence_length -> 1 sequence_length")
        token_positions = token_positions.expand(in_indices.shape[0], -1)

        hidden_states: Float[Tensor, "batch_size sequence_length d_model"] = (
            self.token_embeddings(in_indices)
        )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                token_positions=token_positions,
            )
        final_hidden: Float[Tensor, "batch_size sequence_length d_model"] = self.ln_final(
            hidden_states
        )
        logits: Float[Tensor, "batch_size sequence_length vocab_size"] = self.lm_head(
            final_hidden
        )

        return logits
