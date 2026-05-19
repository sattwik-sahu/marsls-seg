import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from segmentation_models_pytorch.encoders._base import EncoderMixin

from marsls_seg.utils.modules.ijepa.jepa import IJEPA
from marsls_seg.utils.modules.encoder.patch_masking_vit import PatchMaskingViTInput


class SMPWrapper(nn.Module,EncoderMixin):
    """
    Wraps the IJEPA ContextEncoder to be universally compatible with segmentation_models_python

    """

    def __init__(self,jepa_model:IJEPA):
        super().__init__()
        self.encoder=jepa_model.context_encoder
        self.patch_size=self.encoder.encoder.patch_size
        self.dim_embed=self.encoder.encoder.dim
        self._in_channels=self.encoder.encoder.n_channels

        self._depth = 5 
        self._output_stride = 16 
        self.is_dilated = False 
        
        self._out_channels =[
            self._in_channels,   
            self._in_channels,   
            self._in_channels,   
            self.dim_embed,  
            self.dim_embed,  
            self.dim_embed   
        ]

        self.skip_dropout=nn.Dropout2d(p=0.1)

    def make_dilated(self,output_stride):
        self.is_dilated = True

    def forward(self,x:torch.Tensor)->list[torch.Tensor]:
        vit_input=PatchMaskingViTInput(image=x)
        encodings=self.encoder(vit_input)

        B,N,D=encodings.shape
        H_p=self.encoder.encoder.n_patches
        feat=encodings.permute(0,2,1).reshape(B,D,H_p,H_p)

        # we will define the feature pyramid
        f0=self.skip_dropout(x)
        f1=self.skip_dropout(F.max_pool2d(f0,kernel_size=2,stride=2))
        f2=self.skip_dropout(F.max_pool2d(f1,kernel_size=2,stride=2))
        f3=feat
        f4=F.max_pool2d(f3,kernel_size=2,stride=2)

        if self.is_dilated:
            f5=f4
        else:
            f5=F.max_pool2d(f4,kernel_size=2,stride=2)



        return [f0,f1,f2,f3,f4,f5]





    

