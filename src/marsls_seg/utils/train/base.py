from abc import ABC, abstractmethod

import torch
from marsls_seg.utils.modules._typing import TensorData


class BaseTrainer[
    TModel: torch.nn.Module,
    TData: TensorData,
    TLog: dict[str, float | int | str | list],
](ABC):
    """
    The base training pipeline.
    """

    @abstractmethod
    def train_epoch(
        self,
        model: TModel,
        dataloader: torch.utils.data.DataLoader[TData],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
    ) -> TLog:
        pass

    @abstractmethod
    def evaluate(
        self,
        model: TModel,
        dataloader: torch.utils.data.DataLoader[TData],
        device: torch.device,
    ):
        pass
