from abc import ABC, abstractmethod
from pathlib import Path
from typing import override

import torch
from segmentation_models_pytorch.encoders._base import EncoderMixin

from marsls_seg.utils.data._typing import MultimodalMartianLandslideSample
from marsls_seg.utils.modules._typing import FeatureMap, TensorData
from marsls_seg.utils.modules.encoder.base import BaseImageEncoder
from marsls_seg.utils.modules.jepa.extras import BaseImageJEPA
from marsls_seg.utils.modules.load import load_model_from_ckpt


class Base_SMP_EncoderWrapper[
    TInput: TensorData,
    TEncoderInput: TensorData,
    TEncoder: BaseImageEncoder,
    TEncoding: TensorData,
](torch.nn.Module, EncoderMixin, ABC):
    """
    The base wrapper for a segmentation model using one of the image
    encoder backbones from MarsJEPA and a segmentation head from the
    `segmentation_models_python` package.
    """

    def __init__(
        self, jepa_ckpt_path: str | Path, jepa_config_path: str | Path, **kwargs
    ) -> None:
        super().__init__()

        if not isinstance(jepa_ckpt_path, Path):
            jepa_ckpt_path = Path(jepa_ckpt_path)
        if not isinstance(jepa_config_path, Path):
            jepa_config_path = Path(jepa_config_path)

        self._jepa_model: BaseImageJEPA = load_model_from_ckpt(
            ckpt_path=jepa_ckpt_path,
            config_path=jepa_config_path,
            model_class=BaseImageJEPA,
        )
        self._encoder: TEncoder = self._jepa_model.context_encoder
        self._dim_embed: int = self._encoder.dim
        self._in_channels: int = self._encoder.n_channels

        self._depth: int = 5
        self._output_stride: int = 16

        self.is_dilated: bool = False

        self._out_channels: list[int] = [
            self._in_channels,
            self._in_channels,
            self._in_channels,
            self._dim_embed,
            self._dim_embed,
            self._dim_embed,
        ]

        self._skip_dropout = torch.nn.Dropout2d(p=0.1)

    @property
    def dim_embed(self) -> int:
        return self._dim_embed

    @property
    def encoder(self) -> TEncoder:
        return self._encoder

    @property
    def in_channels(self) -> int:
        return self._in_channels

    def make_dilated(self, output_stride) -> None:
        self.is_dilated = True

    @abstractmethod
    def _convert_to_encoder_input(self, x: TInput) -> TEncoderInput:
        pass

    @abstractmethod
    def _convert_to_feature_map(self, encoding: TEncoding) -> FeatureMap:
        pass

    def freeze_encoder(self) -> None:
        self._encoder.eval()
        for parameter in self._encoder.parameters():
            parameter.requires_grad = False

    @override
    def forward(self, x: TInput) -> list[torch.Tensor]:
        encoder_input: TEncoderInput = self._convert_to_encoder_input(x=x)
        encoding: TEncoding = self._encoder(encoder_input)
        feature_map: FeatureMap = self._convert_to_feature_map(encoding=encoding)

        f0 = self._skip_dropout(x)
        f1 = self._skip_dropout(
            torch.nn.functional.max_pool2d(f0, kernel_size=2, stride=2)
        )
        f2 = self._skip_dropout(
            torch.nn.functional.max_pool2d(f1, kernel_size=2, stride=2)
        )
        f3 = feature_map
        f4 = torch.nn.functional.max_pool2d(f3, kernel_size=2, stride=2)

        if self.is_dilated:
            f5 = f4
        else:
            f5 = torch.nn.functional.max_pool2d(f4, kernel_size=2, stride=2)

        return [f0, f1, f2, f3, f4, f5]


class BaseMMLS_SMP_Wrapper[
    TEncoder: BaseImageEncoder,
    TEncoderInput: TensorData,
    TEncoding: TensorData,
](
    Base_SMP_EncoderWrapper[
        MultimodalMartianLandslideSample, TEncoderInput, TEncoder, TEncoding
    ]
):
    pass
