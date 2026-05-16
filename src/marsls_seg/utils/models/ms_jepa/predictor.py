from typing_extensions import override

import torch
from marsls_seg.utils.modules.decoder import Decoder


class MultispectralJEPAPredictor(torch.nn.Module):
    def __init__(
        self,
        dim_embed: int,
        image_size: int,
        patch_size: int,
        n_layers: int,
        n_heads: int,
    ) -> None:
        super().__init__()

        self._dim_embed: int = dim_embed
        self._image_size: int = image_size
        self._patch_size: int = patch_size
        self._n_layers: int = n_layers
        self._n_heads: int = n_heads

        self._transformer_decoder: Decoder = Decoder(
            dim_embed=self._dim_embed, n_heads=self._n_heads, n_layers=self._n_layers
        )

    @override
    def forward(self, sx: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self._transformer_decoder(query=z, context=sx)
