from typing import override

import torch
from einops import rearrange
from tensordict import TensorClass

from marsls_seg.utils.modules.encoder.base import BaseImagePatchEncoder
from marsls_seg.utils.modules.tf.encoder import TransformerEncoder


class SSViTInput(TensorClass):
    image: torch.Tensor  # (B, 7, 128, 128)
    mask: torch.Tensor = torch.empty(0)
    """(M) Indices of Visible tokens in the range of (0,num_patches*7)"""


class SpatioSpectralVisionTransformer(BaseImagePatchEncoder[SSViTInput]):
    """
    Vision transformer that tokenizes channels seperately and
    uses the seperate spatial and channel encodings.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_layers: int,
        patch_size: int,
        img_size: int,
        n_channels: int,
        n_groups: int | None = None,
    ) -> None:
        super().__init__(
            dim=dim, n_channels=n_channels, img_size=img_size, patch_size=patch_size
        )

        self._tokenizer: torch.nn.Conv2d = torch.nn.Conv2d(
            in_channels=self._n_channels,
            out_channels=self._dim * self._n_channels,
            kernel_size=self._patch_size,
            stride=self._patch_size,
            groups=self._n_channels,
        )

        self._spatial_embeddings: torch.nn.Parameter = torch.nn.Parameter(
            torch.randn(self._n_patches, self._dim) * 0.02
        )
        self._spectral_embeddings: torch.nn.Parameter = torch.nn.Parameter(
            torch.randn(self._n_channels, self._dim) * 0.02
        )

        self.encoder = TransformerEncoder(
            n_layers=n_layers, n_heads=n_heads, dim=dim, n_groups=n_groups
        )

    @property
    def total_tokens(self) -> int:
        return self.n_patches * self.n_channels

    @property
    def get_full_pos_embed(self) -> torch.Tensor:
        """
        Creates the full positional embedding for all tokens
        by combining the spatial and channel embeddings.
        Shape: (n_channels * n_patches, dim)
        """
        combined = self._spectral_embeddings.unsqueeze(
            1
        ) + self._spatial_embeddings.unsqueeze(0)

        flattened_tokens: torch.Tensor = rearrange(combined, "c p d -> (c p) d")

        return flattened_tokens

    @override
    def forward(self, x: SSViTInput | torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            image = x.image
        else:
            image = x

        b, c, _, _ = image.shape

        tokens: torch.Tensor = rearrange(
            self._tokenizer(image),
            "b (nc d) h w -> b (nc h w) d",
            nc=self._n_channels,
            d=self._dim,
        )

        """
        Adding the identity of the channel to the positional embedding
        allows the model to learn seperate spatial and channel encodings.
        """
        pos_embed = self.get_full_pos_embed  # (B, total_tokens, dim)
        tokens = tokens + pos_embed.unsqueeze(0)

        if isinstance(x, SSViTInput) and x.mask.numel() > 0:
            tokens = tokens[:, x.mask]

        return self.encoder(tokens)
