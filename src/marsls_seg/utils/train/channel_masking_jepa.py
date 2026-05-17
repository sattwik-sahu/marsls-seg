from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from typing_extensions import override
import torch
from marsls_seg.utils.modules.channel_masking.jepa import (
    ChannelMaskingJEPA,
    ChannelMaskingJEPALoss,
)
from marsls_seg.utils.modules.channel_masking.encoder import (
    ChannelMaskingViT,
    ChannelMaskingViTInput,
)
from marsls_seg.utils.train.base import BaseTrainer
from marsls_seg.utils.data.mmls import MultimodalMartianLandslideSample


class ChannelMaskingJEPATrainer(
    BaseTrainer[ChannelMaskingJEPA, MultimodalMartianLandslideSample]
):
    """
    Implementation of the channel masking JEPA trainer.
    """

    def __init__(self, p1: float, p2: int) -> None:
        super().__init__()

        self._p1 = p1
        self._p2 = p2 - 5

    @override
    def train_epoch(
        self,
        model: ChannelMaskingJEPA,
        dataloader: DataLoader[MultimodalMartianLandslideSample],
        optimizer: Optimizer,
        device: torch.device,
        epoch: int,
    ) -> None:
        """TODO Implement on epoch of training of the channel masking JEPA here."""
        return super().train_epoch(model, dataloader, optimizer, device, epoch)

    @override
    def evaluate(
        self,
        model: ChannelMaskingJEPA,
        dataloader: DataLoader[MultimodalMartianLandslideSample],
        device: torch.device,
    ):
        """TODO Implement one run of evaluation of the channel masking JEPA here."""
        return super().evaluate(model, dataloader, device)
