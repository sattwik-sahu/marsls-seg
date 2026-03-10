import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    TODO: Define the attributes of the class
    """
    def __init__(self,in_ch:int,out_ch:int)->None:
        super().__init__()
        self._in_ch:int=in_ch
        self._out_ch:int=out_ch

        self._block=nn.Sequential(
            nn.Conv2d(self._in_ch,self._out_ch,3,padding=1),
            nn.BatchNorm2d(self._out_ch),
            nn.GELU(),

            nn.Conv2d(self._out_ch,self._out_ch,3,padding=1),
            nn.BatchNorm2d(self._out_ch),
            nn.GELU(),
        )

    def forward(self,x):
        return self._block(x)
    


class UpBlock(nn.Module):
    """
    TODO: Define the attributes of the function
    """

    def __init__(self,in_ch:int,out_ch:int)->None:
        super().__init__()
        self._in_ch:int=in_ch
        self._out_ch:int=out_ch

        self._up=nn.ConvTranspose2d(self._in_ch,self._out_ch,kernel_size=2,stride=2)
        self._conv=ConvBlock(self._out_ch,self._in_ch)

    def forward(self,x):
        x=self._up(x)
        x=self._conv(x)
        return x
    



class VIT_Unet_SegmentationHead(torch.nn.Module):
    """TODO: Describe the attributes of the class that we have"""
    def __init__(self,emb_dim:int, img_size:int,patch_size:int,num_classes:int=2):
        super().__init__()
        self._img_size=img_size
        self._patch_size=patch_size
        self._emb_dim=emb_dim
        self._num_classes=num_classes

        self._bottlneck=ConvBlock(self._emb_dim,512)
        self._up1=UpBlock(in_ch=512,out_ch=256)
        self._up2=UpBlock(in_ch=256,out_ch=128)
        self._up3=UpBlock(in_ch=128,out_ch=64)
        self._up4=UpBlock(in_ch=64,out_ch=32)

        self._final_conv=nn.Conv2d(32,num_classes,kernel_size=1)
     
    def forward(self,tokens:torch.Tensor):
        tokens=tokens[:,1:,:]
        B,N,D=tokens.shape
        w=int(N**0.5)
        h=int(N**0.5)
        x=tokens.permute(0,2,1)
        x=x.view(B,D,h,w)

        x=self._bottlneck(x)

        x=self._up1(x)
        x=self._up2(x)
        x=self._up3(x)
        x=self._up4(x)

        x=self._final_conv()

        x=F.interpolate(x,size=self._img_size,mode="Bilinear",align_corners=False)
        
