from typing_extensions import override
import torch
from marsls_seg.utils.modules.activation import SwiGLU


class DecoderLayer(torch.nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()

        self._dim: int = dim
        self._n_heads: int = n_heads

        self._norm_1: torch.nn.LayerNorm = torch.nn.LayerNorm(
            normalized_shape=self._dim
        )
        self._self_attn: torch.nn.MultiheadAttention = torch.nn.MultiheadAttention(
            embed_dim=self._dim, num_heads=self._n_heads, batch_first=True
        )
        self._norm_2: torch.nn.LayerNorm = torch.nn.LayerNorm(
            normalized_shape=self._dim
        )
        self._cross_attn: torch.nn.MultiheadAttention = torch.nn.MultiheadAttention(
            embed_dim=self._dim, num_heads=self._n_heads, batch_first=True
        )
        self._norm_3: torch.nn.LayerNorm = torch.nn.LayerNorm(
            normalized_shape=self._dim
        )
        self._ff: SwiGLU = SwiGLU(dim=self._dim)

    @override
    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = self._norm_1(x)
        x = x + self._self_attn(x, x, x)[0]
        x = self._norm_2(x)
        x = x + self._cross_attn(x, memory, memory)[0]
        x = self._norm_3(x)
        x = x + self._ff(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, dim_embed: int, n_layers: int, n_heads: int):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [DecoderLayer(dim=dim_embed, n_heads=n_heads) for _ in range(n_layers)]
        )

    def forward(self, context: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            query = layer(query, context)
        return query
