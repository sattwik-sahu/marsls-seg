from typing import override

import torch

from marsls_seg.helpers.device import DEVICE
from marsls_seg.utils.modules.encoder.spatio_spectral_vit import (
    SpatioSpectralVisionTransformer,
)
from marsls_seg.utils.modules.jepa.extras import (
    BaseImagePatchJEPA,
    SimpleTransformerDecoderPredictor,
)
from marsls_seg.utils.modules.sigreg import SIGReg

from ._typing import SSJEPAEncoding, SSJEPAInput, SSJEPALatent, SSJEPALoss, SSJEPAOutput

DEVICE = DEVICE


class SpatioSpectralJEPA(
    BaseImagePatchJEPA[
        SSJEPAInput,
        SpatioSpectralVisionTransformer,
        SSJEPAEncoding,
        SSJEPALatent,
        SimpleTransformerDecoderPredictor,
        SSJEPALoss,
    ]
):
    """
    Joint Embedding Predictive Architecture for Spatio-Spectral Fetaures.
    Learns relationship between Different Martian Modalites
    by predicting masked channel patches.
    """

    def __init__(
        self,
        encoder: SpatioSpectralVisionTransformer,
        predictor: SimpleTransformerDecoderPredictor,
        sigreg: SIGReg,
        sigreg_lambda: float,
    ) -> None:
        super().__init__(context_encoder=encoder, predictor=predictor)
        self._sigreg = sigreg
        self._sigreg_lambda: float = sigreg_lambda

        # variable for storing the vissible mask from the last forward pass
        self._last_visible_mask: torch.Tensor = torch.empty(0)

    def _get_masked_token_indexes(self, visible_mask: torch.Tensor) -> torch.Tensor:
        """
        Finds the indices that were hidden (dropped) using set operations.
        Returns a tensor of integers (indices).
        """
        # 1. Get the full range of tokens (1792)
        # Using self.context_encoder.total_tokens ensures we cover all 7 channels
        all_indexes: set = set(range(self.context_encoder.total_tokens))

        # 2. Convert the visible mask (ids_keep) to a Python set of integers
        visible_indexes: set = set(visible_mask.flatten().tolist())

        # 3. Find the difference (the indices that are NOT visible)
        masked_indexes: set = all_indexes.difference(visible_indexes)

        return torch.as_tensor(sorted(list(masked_indexes)), device=visible_mask.device)

    @override
    def _calculate_loss(
        self, s_x: torch.Tensor, s_y: torch.Tensor, s_y_hat: torch.Tensor
    ) -> SSJEPALoss:
        """
        Calculate the loss for a single Spatio-Spectral JEPA step.

        Args:
            s_x (torch.Tensor): The context encoder output (visible tokens).
            s_y (torch.Tensor): The target encoder output (all 1792 tokens).
            s_y_hat (torch.Tensor): The predictor output (predictions for masked tokens).

        Returns:
            SSJEPALoss: TensorClass containing pred, sigreg, and total loss.
        """

        ids_drop = self._get_masked_token_indexes(self._last_visible_mask)

        # s_y: [B, 1792, dim] -> target_masked: [B, num_masked, dim]
        target_masked: torch.Tensor = s_y[:, ids_drop]

        loss_pred = torch.nn.functional.mse_loss(s_y_hat, target_masked)

        loss_sigreg_context: torch.Tensor = self._sigreg(s_x.transpose(0, 1))
        loss_sigreg_target: torch.Tensor = self._sigreg(s_y.transpose(0, 1))
        loss_sigreg = loss_sigreg_context + loss_sigreg_target

        loss_total = (
            1 - self._sigreg_lambda
        ) * loss_pred + self._sigreg_lambda * loss_sigreg

        return SSJEPALoss(sigreg=loss_sigreg, pred=loss_pred, total=loss_total)

    @override
    def forward(self, x: SSJEPAInput, y: SSJEPAInput, z: torch.Tensor) -> SSJEPAOutput:
        """
        Perform a forward pass through the Spatio-Spectral JEPA.

        Args:
            x: Input for context encoder (image + mask of visible indices)
            y: Input for target encoder (full image)
            z: The query for predictor (positional embeddings of masked indices)
        """

        self._last_visible_mask = x.mask

        return super().forward(x=x, y=y, z=z)
