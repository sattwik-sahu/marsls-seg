from typing import override

import torch

from marsls_seg.utils.modules.encoder.patch_masking_vit import (
    PatchMaskingVisionTransformer,
    PatchMaskingViTInput,
)
from marsls_seg.utils.modules.ijepa._typing import IJEPAEncoding, IJEPAInput
from marsls_seg.utils.modules.jepa.base import BaseJEPAEncoder


class IJEPAEncoder(BaseJEPAEncoder[IJEPAInput, IJEPAEncoding]):
    """The IJEPA encoder. Uses `PatchMaskingVisionTransformer` as the model."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_layers: int,
        patch_size: int,
        img_size: int,
        n_channels: int,
        n_groups: int | None = None,
    ):
        super().__init__()

        # Initialize the patch-masking ViT
        self._encoder: PatchMaskingVisionTransformer = PatchMaskingVisionTransformer(
            dim=dim,
            n_heads=n_heads,
            n_layers=n_layers,
            patch_size=patch_size,
            img_size=img_size,
            n_channels=n_channels,
            n_groups=n_groups,
        )

    @property
    def encoder(self) -> PatchMaskingVisionTransformer:
        return self._encoder

    @property
    def n_patches(self) -> int:
        return self._encoder.n_patches

    @override
    def forward(self, x: PatchMaskingViTInput) -> torch.Tensor:
        return self._encoder(x=x)
