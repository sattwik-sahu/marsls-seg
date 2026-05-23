from pathlib import Path
from typing import override

import torch
from einops import rearrange

from marsls_seg.utils.modules._typing import FeatureMap, MultiSpectralImage
from marsls_seg.utils.modules.encoder.spatio_spectral_vit import (
    SpatioSpectralVisionTransformer,
    SSViTInput,
)
from marsls_seg.utils.modules.ijepa import IJEPAEncoder, IJEPAEncoding, IJEPAInput
from marsls_seg.utils.modules.segmentation.base import (
    BaseMultiSpectralImage_SMP_Backbone,
)


class IJEPA_SMP_Backbone(
    BaseMultiSpectralImage_SMP_Backbone[IJEPAEncoder, IJEPAInput, IJEPAEncoding]
):
    def __init__(
        self, jepa_ckpt_path: str | Path, jepa_config_path: str | Path
    ) -> None:
        super().__init__(jepa_ckpt_path, jepa_config_path)

    @override
    def _convert_to_encoder_input(self, x: MultiSpectralImage) -> IJEPAInput:
        return IJEPAInput(image=x)

    @override
    def _convert_to_feature_map(self, encoding: IJEPAEncoding) -> FeatureMap:
        hp = wp = int(self.encoder.n_patches**0.5)
        return rearrange(encoding, "b (hp wp) d -> b d hp wp", hp=hp, wp=wp)


class SSJEPA_SMP_Backbone(
    BaseMultiSpectralImage_SMP_Backbone[
        SpatioSpectralVisionTransformer, SSViTInput, torch.Tensor
    ]
):
    def __init__(
        self,
        jepa_ckpt_path: str | Path,
        jepa_config_path: str | Path,
        fusion_mlp_hidden_size: int,
        **kwargs,
    ) -> None:
        super().__init__(jepa_ckpt_path, jepa_config_path, **kwargs)

        # Create fusion module
        self._fusion: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self._encoder._dim * self._encoder._n_channels,
                out_features=fusion_mlp_hidden_size,
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(
                in_features=fusion_mlp_hidden_size, out_features=self._encoder._dim
            ),
        )

    @override
    def _convert_to_encoder_input(self, x: MultiSpectralImage) -> SSViTInput:
        return SSViTInput(image=x)

    @override
    def _convert_to_feature_map(self, encoding: torch.Tensor) -> FeatureMap:
        # Concatenate channel features along last dimension
        concatenated_features: torch.Tensor = rearrange(
            encoding, "b (nc np) d -> b np (nc d)"
        )

        # Fuse features using fusion layer
        fused_features: torch.Tensor = self._fusion(concatenated_features)

        # Rearrange to form feature maps from fused features
        hp = wp = int(self._encoder._n_patches**0.5)
        feature_map: FeatureMap = rearrange(
            fused_features, "b (hp wp) d -> b d hp wp", hp=hp, wp=wp
        )

        return feature_map
