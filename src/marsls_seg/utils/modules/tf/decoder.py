import torch

from marsls_seg.utils.modules.tf.activation import SwiGLU
from marsls_seg.utils.modules.tf.attention import MultiheadAttention


class TransformerDecoderBlock(torch.nn.Module):
    """
    A transformer decoder block.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_groups: int | None = None,
        p_dropoout: float = 0.1,
    ) -> None:
        super().__init__()

        self._dim: int = dim
        self._n_heads: int = n_heads
        self._p_dropout: float = p_dropoout

        # Initialize the modules
        self._norm1 = torch.nn.LayerNorm(normalized_shape=self._dim)
        self._mha1 = MultiheadAttention(
            n_heads=self._n_heads, dim=dim, n_groups=n_groups, p_dropout=p_dropoout
        )
        self._norm2 = torch.nn.LayerNorm(normalized_shape=self._dim)
        self._mha2 = MultiheadAttention(
            n_heads=self._n_heads, dim=dim, n_groups=n_groups, p_dropout=p_dropoout
        )
        self._norm3 = torch.nn.LayerNorm(normalized_shape=self._dim)
        self._ff_act = SwiGLU(dim=self._dim)

    def forward(self, tgt: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        x = tgt
        x_norm1: torch.Tensor = self._norm1(x)
        mha_output1: torch.Tensor = self._mha1(
            query=x_norm1, key=x_norm1, value=x_norm1
        )
        x = x + mha_output1
        x_norm2 = self._norm2(x)
        mha_output2: torch.Tensor = self._mha2(query=x_norm2, key=mem, value=mem)
        x = x + mha_output2
        x_norm3 = self._norm3(x)
        x = x + self._ff_act(x_norm3)
        return x


class TransformerDecoder(torch.nn.Module):
    """
    A transformer decoder.
    """

    def __init__(
        self, dim: int, n_heads: int, n_layers: int, n_groups: int | None = None
    ) -> None:
        super().__init__()

        self._dim: int = dim
        self._n_heads: int = n_heads
        self._n_layers: int = n_layers

        self._layers = torch.nn.ModuleList(
            [
                TransformerDecoderBlock(
                    dim=self._dim, n_heads=self._n_heads, n_groups=n_groups
                )
                for _ in range(self._n_layers)
            ]
        )

    def forward(self, tgt: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        x = tgt
        for block in self._layers:
            x = block(tgt=x, mem=mem)
        return x
