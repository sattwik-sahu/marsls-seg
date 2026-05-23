from typing import override

import segmentation_models_pytorch as smp
import torch

from marsls_seg.utils.modules.segmentation.base import Base_SMP_EncoderWrapper


class SegmentationModelWithPretainedEncoder(torch.nn.Module):
    """Segmentation model wrapper over SMP models, with custom pretrained encoder."""

    _CUSTOM_ENCODER_KEY: str = "marsls_jepa_encoder"

    def __init__(self, arch: str, backbone: Base_SMP_EncoderWrapper) -> None:
        super().__init__()

        # The architecture of the segmentation head
        self._seg_arch: str = arch
        self._backbone = backbone

        # Freeze backbone encoder
        self._backbone.freeze_encoder()

        # Register the encoder
        self._register_encoder()

        # Create the segmentation model
        self._model: torch.nn.Module = smp.create_model(
            arch=self._seg_arch,
            encoder_name=self._CUSTOM_ENCODER_KEY,
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )

    @property
    def encoder_dim(self) -> int:
        return self._backbone.dim_embed

    def _register_encoder(self) -> None:
        smp.encoders.encoders[self._CUSTOM_ENCODER_KEY] = dict(
            encoder=lambda **kwargs: self._backbone,
            pretrained_settings={
                "custom": {
                    "mean": [0],
                    "std": [1],
                    "url": None,
                    "repo_id": None,
                    "input_space": "raw",
                    "input_range": [0, 1],
                }
            },
            params={},
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self._model(x)
        return logits
