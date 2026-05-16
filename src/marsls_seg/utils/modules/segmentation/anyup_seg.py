import torch
from einops import rearrange

from marsls_seg.utils.modules.segmentation.base import BaseSegmentationHead


class AnyUpSegmentationHead(BaseSegmentationHead):
    def __init__(self, dim_embed: int, image_size: int, patch_size: int) -> None:
        super().__init__()

        self._image_size: int = image_size
        self._n_patches_per_axis: int = image_size // patch_size

        # Initialize the AnyUp pipeline
        self._anyup: torch.nn.Module = torch.hub.load(
            "wimmerth/anyup", "anyup_multi_backbone", use_natten=True
        )  # type: ignore

        # Create the feature fusion MLP
        self._feature_fusion: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Linear(in_features=dim_embed * 2, out_features=dim_embed * 2),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=dim_embed * 2, out_features=dim_embed),
        )
        self._probe: torch.nn.Linear = torch.nn.Linear(
            in_features=dim_embed, out_features=1
        )

    def _upsample_features(self, x: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return self._anyup(image, x, output_size=(self._image_size, self._image_size))

    def _probe_segmentation(self, dense_features: torch.Tensor) -> torch.Tensor:
        _, _, h, w = dense_features.shape
        x = rearrange(dense_features, "b d h w -> b (h w) d")
        x = self._probe(x).squeeze(-1)
        x = rearrange(x, "b (h w) -> b h w", h=h, w=w)
        return x

    def forward(  # type: ignore
        self,
        vision_encodings: torch.Tensor,
        physics_encodings: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        encodings: torch.Tensor = self._feature_fusion(
            torch.cat((vision_encodings, physics_encodings), dim=-1)
        )  # Shape: [B, N_p, dim_embed]
        encodings = rearrange(
            encodings,
            "b (h w) d -> b d h w",
            h=self._n_patches_per_axis,
            w=self._n_patches_per_axis,
        )  # type: ignore
        with torch.no_grad():
            dense_encodings: torch.Tensor = self._upsample_features(
                x=encodings, image=image
            )  # Shape: (B, D, H, W)
        segmentation: torch.Tensor = self._probe_segmentation(
            dense_features=dense_encodings
        )
        return segmentation.unsqueeze(1)
