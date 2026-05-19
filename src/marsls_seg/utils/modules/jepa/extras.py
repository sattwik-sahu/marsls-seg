from typing import override

from marsls_seg.utils.modules._typing import TensorData
from marsls_seg.utils.modules.jepa.base import BaseJEPA, BaseJEPALoss, BaseJEPAPredictor
from marsls_seg.utils.modules.encoder.base import (
    BaseImageEncoder,
    BaseImagePatchEncoder,
)
from marsls_seg.utils.modules.tf.decoder import TransformerDecoder
import torch


class BaseImageJEPA[
    TInput: TensorData,
    TEncoder: BaseImageEncoder,
    TEncoding: TensorData,
    TLatent: TensorData,
    TPredictor: BaseJEPAPredictor,
    TLoss: BaseJEPALoss,
](BaseJEPA[TInput, TEncoder, TEncoding, TLatent, TPredictor, TLoss]):
    pass


class BaseImagePatchJEPA[
    TInput: TensorData,
    TEncoder: BaseImagePatchEncoder,
    TEncoding: TensorData,
    TLatent: TensorData,
    TPredictor: BaseJEPAPredictor,
    TLoss: BaseJEPALoss,
](BaseImageJEPA[TInput, TEncoder, TEncoding, TLatent, TPredictor, TLoss]):
    pass


class SimpleTransformerDecoderPredictor(BaseJEPAPredictor[torch.Tensor, torch.Tensor]):
    """A simple transformer decoder predictor for JEPA."""

    def __init__(
        self, dim: int, n_heads: int, n_layers: int, n_groups: int | None = None
    ) -> None:
        super().__init__()

        # Initialize the decoder
        self._decoder: TransformerDecoder = TransformerDecoder(
            dim=dim, n_heads=n_heads, n_layers=n_layers, n_groups=n_groups
        )

    @override
    def forward(self, s_x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self._decoder(tgt=z, mem=s_x)
