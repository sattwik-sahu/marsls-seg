import torch
from abc import ABC, abstractmethod
from marsls_seg.utils.modules._typing import TensorData


class BaseImageEncoder[TInput: TensorData](torch.nn.Module, ABC):
    """Base class for an image encoder."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: TInput | torch.Tensor) -> torch.Tensor:
        pass
