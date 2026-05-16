import torch
from tensordict import TypedTensorDict
from typing import Literal


class MarsLS_Sample(TypedTensorDict):
    """
    A sample of the MarsLS segmentation dataset.
    """

    rgb: torch.Tensor
    """The RGB data. Shape: `(3, 128, 128)`"""

    dem: torch.Tensor
    """The digital elevation map. Shape: `(128, 128)`"""

    slope: torch.Tensor
    """The slope map. Shape: `(128, 128)`"""

    thermal: torch.Tensor
    """The thermal map. Shape: `(128, 128)`"""

    grayscale: torch.Tensor
    """The grayscale image. Shape: `(128, 128)`"""

    label: torch.Tensor
    """
    The segmentation mask. Set to `torch.empty()` if
    no label is available for the sample.
    Shape: `(128, 128)`
    """


type SplitName = Literal["train", "val", "test"]
"""A split from the dataset. Can be one of `"train"`, `"val"` or `"test"`."""
