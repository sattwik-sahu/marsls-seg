from abc import ABC, abstractmethod

import torch
from tensordict import TensorClass, TensorDict


class BaseTrainer[
    TModel: torch.nn.Module,
    TData: torch.Tensor | TensorClass | TensorDict,
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
    ) -> None:
        pass

    @abstractmethod
    def evaluate(
        self,
        model: TModel,
        dataloader: torch.utils.data.DataLoader[TData],
        device: torch.device,
    ):
        pass
