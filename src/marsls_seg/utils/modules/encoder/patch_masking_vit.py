from typing import Optional

import torch
from einops import rearrange, repeat
from tensordict import TensorClass

from marsls_seg.utils.modules.tf.encoder import TransformerEncoder
from tensordict import TensorClass
from marsls_seg.utils.modules.encoder.base import BaseImageEncoder


class PatchMaskingViTInput(TensorClass):
    image: torch.Tensor
    """All channels merged into a 'fat-image'"""

    mask: torch.Tensor = torch.empty(0)
    """The indexes of the patches left visible after masking."""


class PatchMaskingVisionTransformer(BaseImageEncoder):
    """Vision Transformer (ViT) implementation with patch masking."""

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
        super().__init__()

        self._dim: int = dim
        self._n_heads: int = n_heads
        self._n_layers: int = n_layers
        self._patch_size: int = patch_size
        self._img_size: int = img_size
        self._n_channels: int = n_channels
        self._n_patches: int = int((self._img_size // self._patch_size) ** 2)

        self._patchify = torch.nn.Conv2d(
            in_channels=self._n_channels,
            out_channels=self._dim,
            kernel_size=self._patch_size,
            stride=self._patch_size,
        )
        self._pos_emb = torch.nn.Parameter(
            torch.randn(self._n_patches, self._dim) * 0.02
        )
        self._encoder = TransformerEncoder(
            n_layers=self._n_layers,
            n_heads=self._n_heads,
            dim=self._dim,
            n_groups=n_groups,
        )

    @property
    def pos_emb(self) -> torch.Tensor:
        return self._pos_emb

    @property
    def n_patches(self) -> int:
        return self._n_patches

<<<<<<< HEAD:src/marsls_seg/utils/modules/vit.py
    def forward(self, x: ViTInput) -> torch.Tensor:
=======
    def forward(self, x: PatchMaskingViTInput | torch.Tensor) -> torch.Tensor:
>>>>>>> main:src/marsls_seg/utils/modules/encoder/patch_masking_vit.py
        if not isinstance(x, torch.Tensor):
            image = x.image
        else:
            image = x

        if image.ndim == 3:
            image = image.unsqueeze(0)

        b, _, _, _ = image.shape

        # Patchify the image
        patches: torch.Tensor = self._patchify(image)

        # Flatten to form tokens
        patch_tokens = rearrange(patches, "b d h w -> b (h w) d")

        # Apply positional encoding
        pos_emb = repeat(self._pos_emb, "n d -> b n d", b=b)
        patch_tokens = patch_tokens + pos_emb

        # Apply masking if required
        if isinstance(x, PatchMaskingViTInput) and x.mask:
            mask: torch.Tensor = torch.zeros_like(x.image, dtype=torch.bool)
            mask[:, x.mask] = True
            patch_tokens = patch_tokens.where(mask, 0)

        # Pass through transformer encoder
        return self._encoder(patch_tokens)
