from abc import ABC, abstractmethod

import torch
from marsls_seg.utils.modules._typing import TensorData
from wandb import Run as WandbRun


class BaseTrainer[
    TModel: torch.nn.Module,
    TData: TensorData,
    TLog: dict[str, float | int | str | list],
](ABC):
    """
    The base training pipeline.
    """

    def __init__(
        self,
        model: TModel,
        n_epochs: int,
        lr: float,
        device: torch.device,
        wandb: WandbRun,
    ) -> None:
        super().__init__()

        self._n_epochs: int = n_epochs
        self._lr: float = lr
        self._wandb: WandbRun = wandb
        self._model: TModel = model
        self._device: torch.device = device
        self._optimizer: torch.optim.Optimizer = self._create_optimizer(
            model=self._model
        )

    @abstractmethod
    def _create_optimizer(self, model: TModel) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def train_epoch(
        self, dataloader: torch.utils.data.DataLoader[TData], epoch: int
    ) -> None:
        pass

    @abstractmethod
    def evaluate(self, dataloader: torch.utils.data.DataLoader[TData]) -> TLog:
        pass
