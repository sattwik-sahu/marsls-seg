import torch

from marsls_seg.utils.modules.encoder.patch_masking_vit import PatchMaskingViTInput
from marsls_seg.utils.modules.jepa.base import BaseJEPALoss, JEPAOutput

IJEPAInput = PatchMaskingViTInput
IJEPAEncoding = torch.Tensor
IJEPALatent = torch.Tensor


class IJEPALoss(BaseJEPALoss):
    pred: torch.Tensor
    sigreg: torch.Tensor


type IJEPAOutput = JEPAOutput[IJEPAEncoding, IJEPALoss]
