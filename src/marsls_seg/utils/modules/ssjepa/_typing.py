import torch
from tensordict import TensorClass

from marsls_seg.utils.modules.encoder.spatio_spectral_vit import SSVitInput
from marsls_seg.utils.modules.jepa.base import BaseJEPALoss, JEPAOutput

SSJEPAInput = SSVitInput

class SSJEPALoss(BaseJEPALoss):
    """
        class defines the loss components for spatio_spectral jepa
    """

    pred : torch.Tensor # Prediction loss for masked patches
    sigreg : torch.Tensor # SigReg loss to prevent the model from collapsing
    total : torch.Tensor # total loss to be optimized 


type SSJEPAOutput = JEPAOutput[torch.Tensor , SSJEPALoss]


