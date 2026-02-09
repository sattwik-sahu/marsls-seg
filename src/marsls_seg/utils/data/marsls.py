import torch
from typing import TypedDict, Literal, Sized
from pathlib import Path
from glob import glob
from tifffile import imread as tiff_imread
import os


class MarsLS_Sample(TypedDict):
    thermal_inertial: torch.Tensor
    """Thermal inertia data derived from THEMIS observations (band 1)"""

    dem: torch.Tensor
    """Slope maps computed from the DEM (band 2)"""

    slope: torch.Tensor
    """Digital Elevation Model (DEM) (band 3)"""

    rgb: torch.Tensor
    """RGB imagery derived from Viking mission data (bands 5, 6, 7)"""

    gray: torch.Tensor
    """Grayscale imagery (band 4)"""

    label: torch.Tensor | None
    """Segmentation mask"""


type _SplitName = Literal["train", "test", "val"]


class MarsLS_Dataset(Sized, torch.utils.data.Dataset[MarsLS_Sample]):
    _THERMAL_INERTIAL_INDEX: int = 0
    _SLOPE_INDEX: int = 1
    _DEM_INDEX: int = 2
    _GRAY_INDEX: int = 3
    _RGB_INDEX: slice = slice(4, 7)

    _IMAGES_DIR: str = "images"
    _LABELS_DIR: str = "masks"

    def __init__(self, data_root: Path, split: _SplitName) -> None:
        super().__init__()

        self._data_dir: Path = data_root / split

        if not (self._data_dir.is_dir() and self._data_dir.exists()):
            raise FileNotFoundError(
                f"No directory exists at `{self._data_dir.as_posix()}`"
            )

        # Extract image and label paths
        image_path_pattern: str = str(
            self._data_dir / f"{MarsLS_Dataset._IMAGES_DIR}/*.tif"
        )
        label_path_pattern: str = str(
            self._data_dir / f"{MarsLS_Dataset._LABELS_DIR}/*.tif"
        )
        self._image_paths: list[str] = glob(image_path_pattern)
        self._label_paths: list[str] = glob(label_path_pattern)

        self._labels_exist: bool = len(self._label_paths) > 0

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, index) -> MarsLS_Sample:
        image_path: str = self._image_paths[index]
        image: torch.Tensor = torch.from_numpy(tiff_imread(image_path)).permute(2, 0, 1)

        label: torch.Tensor | None = None
        if self._labels_exist:
            label_path: str = self._label_paths[index]
            label = torch.from_numpy(tiff_imread(label_path))

        return MarsLS_Sample(
            thermal_inertial=image[MarsLS_Dataset._THERMAL_INERTIAL_INDEX],
            dem=image[MarsLS_Dataset._DEM_INDEX],
            slope=image[MarsLS_Dataset._SLOPE_INDEX],
            gray=image[MarsLS_Dataset._GRAY_INDEX],
            rgb=image[MarsLS_Dataset._RGB_INDEX, ...],
            label=label,
        )
