from pathlib import Path
from typing import Final

from tensordict import TensorDict
from yaml import safe_load as load_yaml

from marsls_seg.utils.data._typing import MultimodalMartianLandslideSample


class MultimodalMarsLandslideDataProcessor:
    """
    Data processor for the MMLSv2 dataset. Takes in a file containing the
    normalization parameters for each of the channels and returns the normalized
    channel data.
    """

    _KEYS: Final[list[str]] = ["thermal_inertial", "dem", "slope", "grayscale", "rgb"]

    def __init__(self, params_file: Path) -> None:
        """
        Creates a data processor for the MMLSv2 dataset.

        Args:
            params_file (Path): Path to the yaml file containing the
                normalization parameters for each channel.
        """
        self._params_file: Path = params_file
        with open(self._params_file, "r") as f:
            p = load_yaml(f)
            self._params: TensorDict = TensorDict(p)

    def __call__(
        self, x: MultimodalMartianLandslideSample
    ) -> MultimodalMartianLandslideSample:
        features = x.exclude("label")  # type: ignore
        params = self._params.to(device=x.device)  # type: ignore
        normalized_features = (features - params["mean"]) / params["std"]  # type: ignore
        x.update_(normalized_features)  # type: ignore
        return x
