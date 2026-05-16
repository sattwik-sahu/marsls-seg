import torch
from marsls_seg.utils.modules.jepa.base import BaseJEPA, JEPALoss
from marsls_seg.utils.modules.sigreg import SIGReg
from marsls_seg.utils.modules.vit import VisionTransformer, ViTInput
from marsls_seg.utils.modules.tf.decoder import TransformerDecoder
from typing_extensions import override


class IJEPALoss(JEPALoss):
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
        loss_sigreg = self._sigreg(s_x)

        # Calculate total loss
        loss_total = (
            1 - self._sigreg_lambda
        ) * loss_pred + self._sigreg_lambda * loss_sigreg

        return IJEPALoss(sigreg=loss_sigreg, pred=loss_pred, total=loss_total)
