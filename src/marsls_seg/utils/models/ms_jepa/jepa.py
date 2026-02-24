from typing import TypedDict

import torch

from marsls_seg.utils.models.ms_jepa.predictor import Predictor
from marsls_seg.utils.models.ms_jepa.vit import VisionTransformer
from marsls_seg.utils.modules.masking import Mask


class JEPAOutput(TypedDict):
    sx: torch.Tensor
    sy_hat: torch.Tensor
    sy: torch.Tensor


class JEPA(torch.nn.Module):
    def __init__(
        self,
        context_encoder: VisionTransformer,
        predictor: Predictor,
        target_encoder: VisionTransformer | None = None,
    ) -> None:
        super().__init__()

        self._context_encoder: VisionTransformer = context_encoder
        self._predictor: Predictor = predictor
        self._target_encoder: VisionTransformer = (
            context_encoder if target_encoder is None else self._context_encoder
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, mask: Mask
    ) -> JEPAOutput:
        sx: torch.Tensor = self._context_encoder(x, mask=mask)
        sy: torch.Tensor = self._target_encoder(y)
        sy_hat: torch.Tensor = self._predictor(sx, z)
        return JEPAOutput(sx=sx, sy=sy, sy_hat=sy_hat)
