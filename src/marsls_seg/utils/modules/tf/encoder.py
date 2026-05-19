import torch

from marsls_seg.utils.modules.tf.activation import SwiGLU
from marsls_seg.utils.modules.tf.attention import MultiheadAttention


class TransformerEncoderBlock(torch.nn.Module):
    """
    A transformer encoder block.
    """

    def __init__(self, n_heads: int, dim: int, n_groups: int | None = None) -> None:
        super().__init__()

        self._dim: int = dim
        self._n_heads: int = n_heads

        self._norm1 = torch.nn.LayerNorm(normalized_shape=self._dim)
        self._mha = MultiheadAttention(
            n_heads=self._n_heads, dim=self._dim, n_groups=n_groups
        )
        self._norm2 = torch.nn.LayerNorm(normalized_shape=self._dim)
        self._ff_act = SwiGLU(dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-normalization
        x_norm1 = self._norm1(x)

        # Multi-head attention and residual
        mha_output: torch.Tensor = self._mha(query=x_norm1, key=x_norm1, value=x_norm1)
        x = x + mha_output

        # Normalization 2
        x_norm2 = self._norm2(x)

        # Feedforward and residual
        x = x + self._ff_act(x_norm2)

        return x


class TransformerEncoder(torch.nn.Module):
    """
    A transformer encoder with multiple blocks
    arranged sequentially.
    """

    def __init__(
        self, n_layers: int, n_heads: int, dim: int, n_groups: int | None = None
    ) -> None:
        super().__init__()

        self._n_layers: int = n_layers
        self._n_heads: int = n_heads
        self._dim: int = dim

        self._layers = torch.nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim=self._dim, n_heads=self._n_heads, n_groups=n_groups
                )
                for _ in range(self._n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self._layers:
            x = block(x=x)
        return x
