from __future__ import annotations

import math

import torch
from einops import einsum, reduce
from jaxtyping import Float
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
        return normalized.to(in_dtype)


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
