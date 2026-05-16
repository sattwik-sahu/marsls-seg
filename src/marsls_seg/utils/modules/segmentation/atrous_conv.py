import torch
import torch.nn as nn
import torch.nn.functional as F
from marsls_seg.utils.modules.segmentation.base import BaseSegmentationHead


class ASPPModule(nn.Module):
    """
    ASPP Module inspired from Atrous Spatial Pyramid Pooling
    DESCRIPTION: Defines the dilated convolutions with different dilation r=[6,12,18]
    Atrous is a french word meaning holes. The dilation rate r defines we are picking neighbours that are r apart for convolutions.
    The effective kernel size comes from the formula k+[k-1][r-1].
    So if my kernel size is 3 then with dilation 6 our effective kernel becomes 3+10=13
    Source: https://ieeexplore.ieee.org/document/10116882
    """

    def __init__(self, in_ch: int, out_ch: int, dilations: list) -> None:
        super().__init__()
        self._in_ch: int = in_ch
        self._out_ch: int = out_ch
        self._dilations: list = dilations
        self._branches = nn.ModuleList()

        self._branches.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self._in_ch,
                    out_channels=self._out_ch,
                    kernel_size=1,
                    bias=False,
                ),
                nn.LayerNorm([self._out_ch, 16, 16]),
                nn.GELU(),
            )
        )

        for d in self._dilations:
            self._branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self._in_ch,
                        out_channels=self._out_ch,
                        kernel_size=3,
                        dilation=d,
                        padding=d,
                        bias=False,
                    ),
                    nn.LayerNorm([self._out_ch, 16, 16]),
                    nn.GELU(),
                )
            )

        self._global_average_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=self._in_ch,
                out_channels=self._out_ch,
                kernel_size=1,
                bias=False,
            ),
            nn.GELU(),
        )

        self._bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=(self._out_ch * (len(self._dilations) + 2)),
                out_channels=self._out_ch,
                kernel_size=1,
                bias=False,
            ),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x) -> torch.Tensor:
        res = []
        for branch in self._branches:
            res.append(branch(x))
        global_feat = self._global_average_pooling(x)
        global_feat = F.interpolate(
            global_feat, size=x.shape[2:], mode="bilinear", align_corners=True
        )
        res.append(global_feat)
        x = torch.cat(res, dim=1)
        return self._bottleneck(x)


class AtrousConvolutionSegHead(BaseSegmentationHead):
    def __init__(self, dim_embed: int, n_classes: int = 2) -> None:
        super().__init__()
        self._embed_dim: int = 2 * dim_embed

        self._num_classes = n_classes
        self._dilations: list = [6, 12, 18]
        self._aspp_module = ASPPModule(
            in_ch=self._embed_dim, out_ch=256, dilations=self._dilations
        )
        self._classifier = nn.Conv2d(256, self._num_classes, kernel_size=1)

    def forward(
        self, vision_encodings: torch.Tensor, physics_encodings: torch.Tensor
    ) -> torch.Tensor:
        x: torch.Tensor = torch.cat([vision_encodings, physics_encodings], dim=-1)

        B, N, D = x.shape

        h = int(N**0.5)
        w = h
        x = x.permute(0, 2, 1)
        x = x.view(B, D, h, w)
        x = self._aspp_module(x)
        logits = self._classifier(x)

        return F.interpolate(
            logits, scale_factor=8, mode="bilinear", align_corners=True
        )
