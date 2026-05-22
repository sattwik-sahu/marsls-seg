import torch
from marsls_seg.utils.modules.jepa.extras import BaseImagePatchJEPA
from marsls_seg.utils.modules.sigreg import SIGReg
from typing import override

from marsls_seg.utils.modules.ijepa._typing import (
    IJEPAInput,
    IJEPAEncoding,
    IJEPALatent,
    IJEPALoss,
    IJEPAOutput,
)
from marsls_seg.utils.modules.encoder.patch_masking_vit import (
    PatchMaskingVisionTransformer as IJEPAEncoder,
)

from marsls_seg.utils.modules.jepa.extras import (
    SimpleTransformerDecoderPredictor as IJEPAPredictor,

)



class IJEPA(
    BaseImagePatchJEPA[
        IJEPAInput, IJEPAEncoder, IJEPAEncoding, IJEPALatent, IJEPAPredictor, IJEPALoss
    ]
):
    def __init__(
        self,
        encoder: IJEPAEncoder,
        predictor: IJEPAPredictor,
        sigreg: SIGReg,
        sigreg_lambda: float,
    ) -> None:
        super().__init__(context_encoder=encoder, predictor=predictor)

        self._sigreg = sigreg
        self._sigreg_lambda: float = sigreg_lambda
        self._last_context_mask: torch.Tensor = torch.empty(0)

    def _get_masked_patch_indexes(self, visible_mask: torch.Tensor) -> torch.Tensor:
        all_indexes: set = set(range(self.context_encoder.n_patches))
        visible_indexes: set = set(visible_mask.flatten().tolist())
        masked_indexes: set = all_indexes.difference(visible_indexes)
        return torch.as_tensor(sorted(list(masked_indexes)))

    @override
    def _calculate_loss(
        self, s_x: torch.Tensor, s_y: torch.Tensor, s_y_hat: torch.Tensor
    ) -> IJEPALoss:
        # Since ids_keep and ids_drop are both sorted, we can compute the
        # dropped patch indices natively on the GPU using boolean masking
        device = s_y.device
        visible_mask = self._last_context_mask  # This is ids_keep

        # Create a boolean mask of all patches
        is_dropped = torch.ones(
            self.context_encoder.n_patches, dtype=torch.bool, device=device
        )
        is_dropped[visible_mask] = False  # Set kept patches to False

        # Non-zero indices gives us exactly ids_drop in sorted order!
        masked_patch_indexes = torch.nonzero(is_dropped).squeeze(1)

        # Slice out the ground-truth target embeddings
        masked_s_y: torch.Tensor = s_y[:, masked_patch_indexes]

        # --- VALID LOGICAL ALIGNMENT ---
        # s_y_hat:      [Prediction for Drop Patch A, Prediction for Drop Patch B, ...]
        # masked_s_y:   [True Embedding for Drop Patch A, True Embedding for Drop Patch B, ...]
        loss_pred = torch.nn.functional.mse_loss(s_y_hat, masked_s_y)

        # Calculate SIGReg loss on both s_x, s_y
        loss_sigreg_context: torch.Tensor = self._sigreg(s_x.transpose(0, 1))
        loss_sigreg_target: torch.Tensor = self._sigreg(s_y.transpose(0, 1))
        loss_sigreg = loss_sigreg_context + loss_sigreg_target

        # Calculate total loss
        loss_total = (
            1 - self._sigreg_lambda
        ) * loss_pred + self._sigreg_lambda * loss_sigreg

        return IJEPALoss(sigreg=loss_sigreg, pred=loss_pred, total=loss_total)

    @override
    def forward(self, x: IJEPAInput, y: IJEPAInput, z: IJEPAEncoding) -> IJEPAOutput:
        self._last_context_mask = x.mask
        return super().forward(x=x, y=y, z=z)
