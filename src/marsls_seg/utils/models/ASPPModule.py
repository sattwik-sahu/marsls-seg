import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPModule(nn.Module):
    """
    ASPP Module isnpired from Atrous Spatial Pyramid Pooling
    DESCIPTION: Defines the dilated convoulutions with different different dilation r=[6,12,13]
    Atrous is a french word that meaning holes. The dilation rate r defines we are picking neighbours that are r apart for convolutions.
    The effective kernel size comes from the formula k+[k-1][r-1].
    So if my kernel size is 3 then with dilation 6 our effective kernel becomes 3+10 =13


    Source:https://ieeexplore.ieee.org/document/10116882 Read more here 
    """
    def __init__(self,in_ch:int,out_ch:int,dilations:list)->None:
        super().__init__()
        self._in_ch:int=in_ch
        self._out_ch:int=out_ch
        self._dilations:list=dilations

        self._branches=nn.ModuleList()
        self._branches.append(nn.Sequential(
            nn.Conv2d(in_channels=self._in_ch,out_channels=self._out_ch,kernel_size=1,bias=False),
            nn.LayerNorm([self._out_ch,14,14]),
            nn.GELU()
        ))

        for d in self._dilations:
            self._branches.append(nn.Sequential(
               nn.Conv2d(in_channels=self._in_ch,out_channels=self._out_ch,kernel_size=d,dilations=d,bias=False),
               nn.LayerNorm([self._out_ch,14,14]),
               nn.GELU() 
            ))

        self._global_average_pooling=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self._in_ch,out_channels=self._out_channels,kernel_size=1,bias=False),
            nn.GELU(),
        )

        self._bottleneck=nn.Sequential(
            nn.Conv2d(in_channels=(self._out_ch*(len(self._dilations))+2),out_channels=self._out_ch,kernel_size=1,bias=False),
            nn.GELU(),
            nn.Dropout(0.1)
        )



    def forward(self,x)->torch.Tensor:
        res=[]
        for branch in self._branches:
            res.append(branch(x))


        global_feat=self._global_average_pooling(x)
        global_feat=F.interpolate(global_feat,size=x.shape[2:],mode="bilinear",align_corners=True)
        res.append(global_feat)
        x=torch.cat(res,dim=1)

        return self._bottleneck(x)
           

        



class MarsLS(nn.Module):
    """
    Prepares the tokens coming from the pretrained vit for Atrous Spatial Pyramid Pooling.
    TODO: Define the attributes of the function

    """
    def __init__(self,embed_dim:int,num_clases:int=2)->None:
        super().__init__()
        self._embed_dim:int=embed_dim
        self._num_classes=num_clases
        self._dilations:list=[6,12,18]

        self._aspp_module=ASPPModule(in_ch=embed_dim,out_ch=256,dilations=self._dilations)
        self._classifier=nn.Conv2d(256,self._num_classes,kernel_size=1)



    def forward(self,x:torch.Tensor):
        B,N,D=x.shape # Assumig that there are no cls token
        h=int(N**2)
        w=h
        x=x.permute(0,2,1)
        x=x.view(B,D,h,w)

        logits=self._classifier(x)
        return F.interpolate(logits,scale_factor=16,mode="Bilinear",align_corners=True)
































