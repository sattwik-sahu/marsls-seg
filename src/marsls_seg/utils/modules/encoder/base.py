import torch
from abc import ABC, abstractmethod
from marsls_seg.utils.modules._typing import TensorData


class BaseEncoder[TInput: TensorData, TEncoding: TensorData](torch.nn.Module, ABC):
    """Base class for an encoder."""

    def __init__(self, dim: int) -> None:
        super().__init__()

        self._dim: int = dim

    @property
    def dim(self) -> int:
        return self._dim

    @abstractmethod
    def forward(self, x: TInput) -> TEncoding:
        pass


class BaseImageEncoder[TInput: TensorData, TEncoding: TensorData](
    BaseEncoder[TInput | torch.Tensor, TEncoding], ABC
):
    """Base class for an image encoder."""

    def __init__(self, dim: int, n_channels: int) -> None:
        super().__init__(dim=dim)

        self._n_channels: int = n_channels

    @property
    def n_channels(self) -> int:
        return self._n_channels


class BaseImagePatchEncoder[TInput: TensorData](
    BaseImageEncoder[TInput, torch.Tensor], ABC
):
    """Base class for a patch-based image encoder."""

    def __init__(
        self, dim: int, n_channels: int, img_size: int, patch_size: int
    ) -> None:
        super().__init__(dim=dim, n_channels=n_channels)

        self._img_size: int = img_size
        self._patch_size: int = patch_size
        self._n_patches: int = int((self._img_size // self._patch_size) ** 2)

    @property
    def img_size(self) -> int:
        return self._img_size

    @property
    def patch_size(self) -> int:
        return self.patch_size

    @property
    def n_patches(self) -> int:
        return self._n_patches
