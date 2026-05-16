from glob import glob
from pathlib import Path
from typing import Final, Sized

import torch
from tifffile import imread as tiff_imread

from marsls_seg.utils.data._typing import MarsLS_Sample, SplitName


class MultimodalMartianLandslideDataset(Sized, torch.utils.data.Dataset[MarsLS_Sample]):
    """
    The Multimodal Martian Landslide Detection (MMLSv2) dataset [1]. It contains data about the following channels:
    1. RGB
    2. Digital Elevation Map (DEM)
    3. Slope
    4. Thermal Inertial
    5. Grayscale

    There are seven channels in total, but the samples in this class the RGB channels together for
    convenient image processing.

    ### References
    - [1]: Paheding, Sidike, et al. "MMLSv2: A Multimodal Dataset for Martian Landslide Detection in Remote Sensing Imagery." arXiv preprint arXiv:2602.08112 (2026).
    """

    _THERMAL_INERTIAL_INDEX: Final[slice] = slice(0, 1)
    _SLOPE_INDEX: Final[slice] = slice(1, 2)
    _DEM_INDEX: Final[slice] = slice(2, 3)
    _GRAY_INDEX: Final[slice] = slice(3, 4)
    _RGB_INDEX: Final[slice] = slice(4, 7)

    _IMAGES_DIR: Final[str] = "images"
    _LABELS_DIR: Final[str] = "masks"

    def __init__(
        self, data_root: Path, split: SplitName, phase: int | None = None
    ) -> None:
        """
        Creates a `torch.utils.Dataset` object for the MMLSv2 dataset.

        Args:
            data_root (Path): The path object to the root dir to the dataset.
            split (SplitName): The split name to extract. Should be one of `["train", "test", "split"]`.
            phase: (int | None): *Optional, Deprecated*. The phase of the CVPR2026 competition.
                Do not use when not writing code for the competition. Default: `None`.
        """
        super().__init__()

        if phase is not None:
            self._data_dir: Path = data_root / f"phase-{str(phase).zfill(2)}" / split
        else:
            self._data_dir: Path = data_root / split

        # Check if the root dir exists
        self._check_root_dir_exists()

        # Extract image and label paths
        image_path_pattern: str = str(
            self._data_dir / f"{MultimodalMartianLandslideDataset._IMAGES_DIR}/*.tif"
        )
        label_path_pattern: str = str(
            self._data_dir / f"{MultimodalMartianLandslideDataset._LABELS_DIR}/*.tif"
        )
        self._image_paths: list[str] = glob(image_path_pattern)
        self._label_paths: list[str] = glob(label_path_pattern)

        # Labels exist? (Might not exist for certain versions in which test split has no labels)
        self._labels_exist: bool = len(self._label_paths) > 0

    @property
    def root_dir(self) -> Path:
        return self._data_dir

    @property
    def image_paths(self) -> list[Path]:
        return [Path(p) for p in self._image_paths]

    @property
    def label_paths(self) -> list[Path]:
        return [Path(p) for p in self._label_paths]

    @property
    def labels_exist(self) -> bool:
        return self._labels_exist

    def _check_root_dir_exists(self) -> None:
        """
        Check if the root directory exists.

        Raises:
            FileNotFoundError: If the root directory passed does not exist.
        """
        if not (self._data_dir.is_dir() and self._data_dir.exists()):
            raise FileNotFoundError(
                f"No directory exists at `{self._data_dir.as_posix()}`"
            )

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, index) -> MarsLS_Sample:
        # Retrieve path from path list
        image_path: str = self._image_paths[index]

        # Read image from path
        image: torch.Tensor = torch.from_numpy(tiff_imread(image_path)).permute(2, 0, 1)

        # Retrieve the label (if exists)
        label: torch.Tensor = torch.empty(0)
        if self._labels_exist:
            label_path: str = self._label_paths[index]
            label = torch.from_numpy(tiff_imread(label_path))

        # Construct the sample
        return MarsLS_Sample(
            thermal_inertial=image[
                MultimodalMartianLandslideDataset._THERMAL_INERTIAL_INDEX
            ],
            dem=image[MultimodalMartianLandslideDataset._DEM_INDEX],
            slope=image[MultimodalMartianLandslideDataset._SLOPE_INDEX],
            gray=image[MultimodalMartianLandslideDataset._GRAY_INDEX],
            rgb=image[MultimodalMartianLandslideDataset._RGB_INDEX],
            label=label,
        )
