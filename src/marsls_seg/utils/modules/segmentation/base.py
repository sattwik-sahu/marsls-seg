from typing import Any

import torch

from abc import abstractmethod, ABC
from typing_extensions import override


class BaseSegmentationHead(torch.nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    @override
    @abstractmethod
    def forward(
        self,
        vision_encodings: torch.Tensor,
        physics_encodings: torch.Tensor,
        image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass
