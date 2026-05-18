import math

import torch
from einops import rearrange, repeat


class MultiheadAttention(torch.nn.Module):
    """The multi-head attention module."""

    def __init__(
        self,
        n_heads: int,
        dim: int,
        dim_embed: int | None = None,
        n_groups: int | None = None,
        p_dropout: float = 0.0,
    ) -> None:
        """
        Create a multi-head attention block.

        Args:
            n_heads (int): The number of heads in the query.
            dim (int): The input dimension.
            dim_embed (int): The embedding dimension used for attention computation.
                If not specified, defaults to the input dimension.
            n_groups (int): The number of groups for grouped-query attention.
                It is equal to the number of KV heads.
            p_dropout (float): The attention dropout rate.
        """
        super().__init__()

        self._n_heads: int = n_heads
        self._dim: int = dim
        self._dim_embed: int = dim_embed or dim
        self._p_dropout: float = p_dropout

        assert self._dim_embed % self._n_heads == 0, (
            f"dim_embed (got {dim_embed}) should be divisible by n_heads (got {n_heads})"
        )

        self._head_dim: int = self._dim_embed // self._n_heads
        self._n_groups: int = n_groups or self._n_heads

        # Initialize the weights
        self._W_query = torch.nn.Linear(
            in_features=self._dim, out_features=self._head_dim * self._n_heads
        )
        self._W_key = torch.nn.Linear(
            in_features=self._dim, out_features=self._head_dim * self._n_groups
        )
        self._W_value = torch.nn.Linear(
            in_features=self._dim, out_features=self._head_dim * self._n_groups
        )
        self._W_out = torch.nn.Linear(
            in_features=self._dim_embed, out_features=self._dim
        )

    def _split_into_heads(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b s (nh dh) -> b nh s dh", nh=self._n_heads)

    def _split_into_grouped_heads(self, x: torch.Tensor) -> torch.Tensor:
        # Split raw channels into grouped head chunks
        x = rearrange(x, "b s (ng dh) -> b ng s dh", ng=self._n_groups)
        return x

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        # Obtain query, key, value
        x_query: torch.Tensor = self._W_query(query)
        x_key: torch.Tensor = self._W_key(key)
        x_value: torch.Tensor = self._W_value(value)

        # Split into heads and groups for GQA
        x_query = self._split_into_heads(x=x_query)
        x_key = self._split_into_grouped_heads(x=x_key)
        x_value = self._split_into_grouped_heads(x=x_value)

        x_out: torch.Tensor = torch.nn.functional.scaled_dot_product_attention(
            query=x_query,
            key=x_key,
            value=x_value,
            dropout_p=self._p_dropout,
            enable_gqa=True,
        )

        # Merge attention output heads back into one
        x_out = rearrange(x_out, "b h n d -> b n (h d)")
        x_out = self._W_out(x_out)

        return x_out
