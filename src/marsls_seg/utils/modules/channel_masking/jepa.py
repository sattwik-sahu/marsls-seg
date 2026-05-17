import torch
from torch.nn.modules.module import Module
from marsls_seg.utils.modules.jepa.base import BaseJEPA, BaseJEPALoss
from marsls_seg.utils.modules.channel_masking.encoder import (
    ChannelMaskingViT,
    ChannelMaskingViTInput,
)
from tensordict import TensorClass
from typing_extensions import override


class ChannelMaskingJEPALoss(BaseJEPALoss):
    sigreg: torch.Tensor
    pred: torch.Tensor
    # NOTE Add other losses or regularization terms if you implement them


class ChannelMaskingJEPA(
    BaseJEPA[
        ChannelMaskingViTInput,
        ChannelMaskingViT,
        torch.Tensor,
        torch.Tensor,  # TODO Decide on how one can identify a particular channel
        torch.nn.Module,  # TODO Decide how you want to implement predictor
        ChannelMaskingJEPALoss,
    ]
):
    def __init__(
        self,
        context_encoder: ChannelMaskingViT,
        predictor: Module,
        target_encoder: ChannelMaskingViT | None = None,
    ) -> None:
        super().__init__(context_encoder, predictor, target_encoder)

    @override
    def _calculate_loss(
        self, s_x: torch.Tensor, s_y: torch.Tensor, s_y_hat: torch.Tensor
    ) -> ChannelMaskingJEPALoss:
        # TODO Calculate all losses here
        return ChannelMaskingJEPALoss(
            sigreg=0.0,
            pred=0.0,
            total=0.0,  # IMPORTANT
        )
