from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from datasets import Array2D, Array3D, Dataset, DatasetDict, Features

from marsls_seg.utils.data._typing import SplitName
from marsls_seg.utils.data.mmls import (
    MultimodalMartianLandslideDataset,
    MultimodalMartianLandslideSample,
)


def main():
    # Dataset root folder
    DATA_ROOT: Path = Path("data/marsls-seg/mmlsv2")

    # Load the splits
    train_data = MultimodalMartianLandslideDataset(data_root=DATA_ROOT, split="train")
    val_data = MultimodalMartianLandslideDataset(data_root=DATA_ROOT, split="val")
    test_data = MultimodalMartianLandslideDataset(data_root=DATA_ROOT, split="test")

    # Generator for the data
    def _get_generator(
        dataset_split: MultimodalMartianLandslideDataset,
    ) -> Callable[[], Iterable[dict[str, np.ndarray]]]:
        def generator():
            for sample in dataset_split:
                yield {
                    "rgb": sample.rgb.numpy().astype(np.float32),
                    "dem": sample.dem.numpy().astype(np.float32),
                    "thermal_inertial": sample.thermal_inertial.numpy().astype(
                        np.float32
                    ),
                    "grayscale": sample.grayscale.numpy().astype(np.float32),
                    "label": sample.label.numpy().astype(np.float32),
                }

        return generator

    # Define features
    features = Features(
        {
            "rgb": Array3D(shape=(3, 128, 128), dtype="float32"),
            "dem": Array3D(shape=(1, 128, 128), dtype="float32"),
            "thermal_inertial": Array3D(shape=(1, 128, 128), dtype="float32"),
            "grayscale": Array3D(shape=(1, 128, 128), dtype="float32"),
            "label": Array2D(shape=(128, 128), dtype="float32"),
        }
    )

    # Create HF dataset for each split
    hf_train = Dataset.from_generator(
        generator=_get_generator(dataset_split=train_data), features=features
    )
    hf_val = Dataset.from_generator(
        generator=_get_generator(dataset_split=val_data), features=features
    )
    hf_test = Dataset.from_generator(
        generator=_get_generator(dataset_split=test_data), features=features
    )

    # Combine datasets into one dataset
    hf_dataset_dict = DatasetDict(
        {
            "train": hf_train,
            "val": hf_val,
            "test": hf_test,
        }
    )

    # Push to hub
    hf_dataset_dict.push_to_hub(repo_id="sattwik21/mmls-v2")


if __name__ == "__main__":
    # Push to hub
    main()
