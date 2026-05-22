import torch
import torch.nn as nn

from typing import override
from einops import repeat, rearrange
from tensordict import TensorClass

from marsls_seg.utils.modules.tf.encoder import TransformerEncoder
from marsls_seg.utils.modules.encoder.base import BaseImagePatchEncoder


class SSVitInput(TensorClass):
    image: torch.Tensor # (B,7,128,128)
    mask:  torch.Tensor # (M) Indices of Visible tokens in the range of (0,num_patches*7)


class SpatioSpectralVisionTransformer(BaseImagePatchEncoder[SSVitInput]):
    """
        Vision transformer that tokenizes channels seperately and 
        uses the seperate spatial and channel encodings
    """
    def __init__(
            self,
            dim : int,
            n_heads : int,
            n_layers : int,
            patch_size : int,
            img_size : int,
            n_channels : int,
            n_groups : int | None = None,
            
    )->None:
        super().__init__(dim=dim,n_channels=n_channels,img_size=img_size,patch_size=patch_size)

        self.tokenizer = nn.ModuleList[(
            nn.Conv2d(in_channels=1,out_channels=dim,kernel_size=patch_size,stride=patch_size)
            for _ in range(n_channels)
            )]

        # learned positional encodings
        self.spatial_embeddings = nn.Embeddings(self.n_patches,dim)
        self.channel_embeddings = nn.Embeddings(n_channels,dim)

        self.encoder = TransformerEncoder(n_layers=n_layers, n_heads=n_heads, dim=dim,)

    @property
    def total_tokens(self) -> int:
        return self.n_patches * self.n_channels
    
    @property
    def get_full_pos_embed(self, device) -> torch.Tensor :
        """
        creates the full positional embedding for all tokens
        by combining the spatial and channel embeddings
        """
        s_idx = torch.arange(self.n_patches,device=device)
        c_idx = torch.arange(self.n_channels,device=device)

        s_emb=self.spatial_embeddings(s_idx)
        c_emb=self.channel_embeddings(c_idx)

        combined = c_emb.unsqueeze(1) + s_emb.unsqueeze(0)

        return combined.rearrange(combined, "c p d -> (c p) d")
        

    @override
    def forward (self, x: SSVitInput | torch.Tensor) -> torch.Tensor :
        if not isinstance(x, torch.Tensor):
            image = x.image
        else:
            image = x

        b,c,h,w = image.shape

        channel_tokens = []
        for i in range(c):
            t=self.tokenizer[i](image[:,i:i+1])
            channel_tokens.append(rearrange(t,"b,d,h,w -> b (h w) d")) 
            #TODO: check to parallelize

        tokens = torch.cat(channel_tokens,dim=1)

        pos_embed=self.get_full_pos_embed(image.device) # (B, total_tokens,dim)
        tokens = tokens + repeat(pos_embed, "n d -> b n d" , b=b)
        # adding the identity of the channel to the positional embedding allows the model to learn seperate spatial and channel encodings

        if isinstance(x, SSVitInput) and x.mask.numel() >0:
            tokens = tokens[: , x.mask]

        return self.encoder(tokens)
                
            

        



