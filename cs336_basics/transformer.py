from __future__ import annotations

import math

import torch
from einops import einsum
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
