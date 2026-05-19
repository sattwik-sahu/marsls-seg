import segmentation_models_pytorch as smp
import torch
from marsls_seg.utils.modules.segmentation.wrapper import SMPWrapper
from pathlib import Path
from typing import override
from marsls_seg.utils.data.mmls import MultimodalMartianLandslideSample


class SegmentationModelWithPretainedEncoder(torch.nn.Module):
    """Segmentation model wrapper over SMP models, with custom pretrained encoder."""

    _CUSTOM_ENCODER_KEY: str = "marsls_jepa_encoder"

    def __init__(
        self, arch: str, encoder_ckpt_path: Path, encoder_config_path: Path
    ) -> None:
        super().__init__()

        # The architecture of the segmentation head
        self._seg_arch: str = arch
        self._encoder_ckpt_path: Path = encoder_ckpt_path
        self._encoder_config_path: Path = encoder_config_path

        # Load the wrapper on the encoder
        self._encoder: SMPWrapper = SMPWrapper(
            jepa_ckpt_path=encoder_ckpt_path, jepa_config_path=encoder_config_path
        )

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

    def _register_encoder(self) -> None:
        smp.encoders.encoders[self._CUSTOM_ENCODER_KEY] = dict(
            encoder=SMPWrapper,
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
            params={
                # "jepa_ckpt_path": self._encoder_ckpt_path,
                # "jepa_config_path": self._encoder_config_path,
            },
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image: torch.Tensor = x
        logits: torch.Tensor = self._model(image)
        return logits
