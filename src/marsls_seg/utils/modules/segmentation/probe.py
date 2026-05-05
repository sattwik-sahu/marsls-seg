import torch
from marsls_seg.utils.modules.segmentation.base import BaseSegmentationHead


class LinearProbeSegmentationHead(BaseSegmentationHead):
    def __init__(
        self, dim_embed: int, image_size: int, patch_size: int, n_classes: int
    ) -> None:
        super().__init__()

        self._dim_embed: int = dim_embed
        self._image_size: int = image_size
        self._patch_size: int = patch_size
        self._n_classes: int = n_classes

        self._probe_head: torch.nn.Conv2d = torch.nn.Conv2d(
            in_channels=dim_embed * 2, out_channels=n_classes, kernel_size=1
        )

    def forward(
        self,
        vision_encodings: torch.Tensor,
        physics_encodings: torch.Tensor,
        image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x: torch.Tensor = torch.cat([vision_encodings, physics_encodings], dim=-1)
        batch_size, n_patches, dim = x.shape
        h = w = int(n_patches**0.5)
        x = x.transpose(1, 2).view(batch_size, dim, h, w)
        x = torch.nn.functional.interpolate(
            x,
            size=(self._image_size, self._image_size),
            mode="bilinear",
            align_corners=True,
            antialias=True,
        )
        segmentation: torch.Tensor = self._probe_head(x)
        return segmentation
