import torch
from typing import override
from marsls_seg.utils.modules.jepa.extras import BaseImagePatchJEPA, SimpleTransformerDecoderPredictor
from marsls_seg.utils.modules.sigreg import SIGReg
from ._typing import SSJEPAInput, SSJEPAOutput, SSJEPALoss , SSJEPALatent , SSJEPAEncoding
from marsls_seg.utils.modules.encoder.spatio_spectral_vit import SpatioSpectralVisionTransformer
from marsls_seg.helpers.device import DEVICE

DEVICE=DEVICE

class SpatioSpectralJepa(
    BaseImagePatchJEPA[
        SSJEPAInput , 
        SpatioSpectralVisionTransformer , 
        SSJEPAEncoding , 
        SSJEPALatent ,
        SimpleTransformerDecoderPredictor ,
        SSJEPALoss
    ]):
    """
        Joint Embedding Predictive Architecture for Spatio-Spectral Fetaures.
        Learns relationship between Different Martian Modalites 
        by predicting masked channel patches.
    """
    def __init__(self,
                 encoder: SpatioSpectralVisionTransformer,
                 predictor: SimpleTransformerDecoderPredictor,
                 sigreg : SIGReg,
                 sigred_lambda : float
                 ) -> None:
        super().__init__(context_encoder=encoder , predictor=predictor)
        self._sigreg = sigreg
        self._sigreg_lambda : float = sigred_lambda
        
        #variable for storing the vissible mask from the last forward pass
        self._last_visible_mask : torch.Tensor = torch.empty(0)

    def _get_masked_patch_indexes(self,
                                  visible_mask : torch.Tensor,
                                  device : torch.device = DEVICE
                                  ) -> torch.Tensor:
        """
            Finds the indices that were hidden given the visible mask.

            Args:
                visible_mask (torch.Tensor) : 1D tensor of indices that are currently visible

            Returns :
                torch.Tensor (torch.Tensor) : 1D tensor of indices that were masked


        """ 
        device = visible_mask.device
        total_tokens = self.context_encoder.total_tokens

        is_dropped = torch.ones(total_tokens , dtype=torch.bool , device=device)

        is_dropped[visible_mask] = False

        return torch.nonzero(is_dropped).squeeze(1)
        #TODO: 
    
    @override
    def _calculate_loss(
        self,
        s_x : torch.Tensor,
        s_y : torch.Tensor,
        s_y_hat : torch.Tensor,
        device : torch.device = DEVICE,
    ) -> SSJEPALoss:
        """
           Caculates the loss between the Target Encodings and Predictions
           Args:
                s_x (torch.Tensor) : The input spatio-spectral sequence
                s_y (torch.Tensor) : The target encodings for the masked patches
                s_y_hat (torch.Tensor) : The predicted encodings for the masked tokens

        """

        device = s_y.device

        total_tokens = self.context_encoder.total_tokens
        is_dropped = torch.ones(total_tokens,dtype=torch.bool,device=device)

        is_dropped[self._last_visible_mask] = False
        ids_drop = torch.nonzero(is_dropped).squeeze(1)

        target_masked = s_y[:,ids_drop] 
        # transforming s_y's shape from B, Total_tokens,dim to B,M,dim

        loss_pred = torch.nn.functional.mse_loss(s_y_hat,target_masked)

        loss_sigreg_context = self._sigreg(s_x.transpose(0, 1))
        loss_sigreg_target = self._sigreg(s_y.transpose(0, 1))
        loss_sigreg = loss_sigreg_context + loss_sigreg_target

        loss_total = (1 - self._sigreg_lambda) * loss_pred + self._sigreg_lambda * loss_sigreg

        return SSJEPALoss(total=loss_total, pred=loss_pred, sigreg=loss_sigreg)
    

    @override
    def forward(self, x: torch.Tensor)




