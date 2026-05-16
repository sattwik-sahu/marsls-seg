from typing import Any

import torch
from tensordict import tensorclass
import math
from einops import rearrange


@tensorclass
class AttentionOutput:
    x_out: torch.Tensor
    attention_score: torch.Tensor


class MultiheadAttention(torch.nn.Module):
    """The multi-head attention module."""

    def __init__(self, n_heads: int, dim: int, dim_embed: int | None = None) -> None:
        """
        Create a multi-head attention block.

        Args:
            n_heads (int): The number of heads.
            dim (int): The input dimension.
            dim_embed (int): The embedding dimension used for attention computation.
                If not specified, defaults to the input dimension.
        """
        super().__init__()

        self._n_heads: int = n_heads
        self._dim: int = dim
        self._dim_embed: int = dim_embed or dim

        assert self._dim_embed % self._n_heads == 0, (
            f"dim_embed (got {dim_embed}) should be divisible by n_heads (got {n_heads})"
        )

        self._head_dim: int = self._dim_embed // self._n_heads

        # Initialize the weights
        self._W_query = torch.nn.Linear(
            in_features=self._dim, out_features=self._dim_embed
        )
        self._W_key = torch.nn.Linear(
            in_features=self._dim, out_features=self._dim_embed
        )
        self._W_value = torch.nn.Linear(
            in_features=self._dim, out_features=self._dim_embed
        )
        self._W_out = torch.nn.Linear(
            in_features=self._dim_embed, out_features=self._dim
        )

    def _split_into_heads(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b n (h d) -> b h n d", h=self._n_heads)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> AttentionOutput:
        # Obtain query, key, value
        x_query: torch.Tensor = self._W_query(query)
        x_key: torch.Tensor = self._W_key(key)
        x_value: torch.Tensor = self._W_value(value)

        # Split into heads
        x_query = self._split_into_heads(x=x_query)
        x_key = self._split_into_heads(x=x_key)
        x_value = self._split_into_heads(x=x_value)

        # Calculate attention score
        score = x_query @ x_key.transpose(-2, -1) / math.sqrt(self._dim_embed)
        attention_score = torch.softmax(score, dim=-1)

        # Calculate output
        x_out = attention_score @ x_value

        # Merge attention output heads back into one
        x_out = rearrange(x_out, "b h n d -> b n (h d)")
        x_out = self._W_out(x_out)

        return AttentionOutput(attention_score=attention_score, x_out=x_out)
