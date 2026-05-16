from typing import Literal

import torch
from tensordict import TensorClass


class MultimodalMartianLandslideSample(TensorClass):
    """
    A sample of the Multimodal Martian Landslide detection dataset.
    """

    rgb: torch.Tensor
    """The RGB data. Shape: `(3, 128, 128)`"""

    dem: torch.Tensor
    """The digital elevation map. Shape: `(1, 128, 128)`"""

    slope: torch.Tensor
    """The slope map. Shape: `(1, 128, 128)`"""

    thermal_inertial: torch.Tensor
    """The thermal inertial map. Shape: `(1, 128, 128)`"""

    grayscale: torch.Tensor
    """The grayscale image. Shape: `(1, 128, 128)`"""

    label: torch.Tensor
    """
    The segmentation mask. Set to `torch.empty()` if
    no label is available for the sample.
    Shape: `(1, 128, 128)`
    """

    def merge_channels(self) -> torch.Tensor:
        """
        Returns the "fat-image" of a sample, i.e., all channels
        concatenated into a single image with multiple channels.

        Returns:
            torch.Tensor: The merged multi-channel image.
        """
        return torch.cat([self.get(key) for key in self.keys()], dim=-1)


type SplitName = Literal["train", "val", "test"]
"""A split from the dataset. Can be one of `"train"`, `"val"` or `"test"`."""
