import torch

from marsls_seg.utils.modules.encoder.spatio_spectral_vit import SSViTInput
from marsls_seg.utils.modules.jepa.base import BaseJEPALoss, JEPAOutput

SSJEPAInput = SSViTInput
SSJEPAEncoding = torch.Tensor
SSJEPALatent = torch.Tensor


class SSJEPALoss(BaseJEPALoss):
    """
    Class defining the loss components for spatio_spectral JEPA.
    """

    pred: torch.Tensor
    """Prediction loss for masked patches"""

    sigreg: torch.Tensor
    """SigReg loss to prevent the model from collapsing"""


type SSJEPAOutput = JEPAOutput[torch.Tensor, SSJEPALoss]
