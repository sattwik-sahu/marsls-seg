import torch
from marsls_seg.utils.modules.jepa.base import BaseJEPA, BaseJEPALoss
from marsls_seg.utils.modules.sigreg import SIGReg
from marsls_seg.utils.modules.vit import VisionTransformer, ViTInput
from marsls_seg.utils.modules.tf.decoder import TransformerDecoder
from typing_extensions import override


class TransformerDecoderPredictor(torch.nn.Module):
    def __init__(self, decoder: TransformerDecoder) -> None:
        super().__init__()
        self._decoder = decoder

    @override
    def forward(self, s_x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self._decoder(tgt=z, mem=s_x)


class IJEPALoss(BaseJEPALoss):
    sigreg: torch.Tensor
    pred: torch.Tensor


class IJEPA(
    BaseJEPA[
        ViTInput,
        VisionTransformer,
        torch.Tensor,
        torch.Tensor,
        TransformerDecoder,
        IJEPALoss,
    ]
):
    def __init__(
        self,
        encoder: VisionTransformer,
        predictor: TransformerDecoder,
        sigreg: SIGReg,
        sigreg_lambda: float,
    ) -> None:
        super().__init__(context_encoder=encoder, predictor=predictor)

        self._sigreg = sigreg
        self._sigreg_lambda: float = sigreg_lambda

    @override
    def _calculate_loss(
        self, s_x: torch.Tensor, s_y: torch.Tensor, s_y_hat: torch.Tensor
    ) -> IJEPALoss:
        # Calculate JEPA prediction loss
        loss_pred = torch.nn.functional.mse_loss(s_y_hat, s_y)

        # Calculate SIGReg loss
        # Transpose batch and token dims
        loss_sigreg = self._sigreg(s_x.transpose(0, 1))

        # Calculate total loss
        loss_total = (
            1 - self._sigreg_lambda
        ) * loss_pred + self._sigreg_lambda * loss_sigreg

        return IJEPALoss(sigreg=loss_sigreg, pred=loss_pred, total=loss_total)
