import torch
from datasets import load_dataset

from marsls_seg.utils.data._typing import MultimodalMartianLandslideSample, SplitName


class MultimodalMartianLandslideHFDataset(torch.utils.data.Dataset):
    """The MMLSv2 dataset loaded from Huggingface."""

    _REPO_ID: str = "sattwik21/mmls-v2"

    def __init__(self, split: SplitName, repo_name: str = _REPO_ID) -> None:
        self._repo_name: str = repo_name
        self._split: SplitName = split
        self._dataset = load_dataset(self._repo_name, split=self._split)
        self._dataset = self._dataset.with_format(type="torch")

    @property
    def repo_name(self) -> str:
        return self._repo_name

    @property
    def split(self) -> SplitName:
        return self._split

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index) -> MultimodalMartianLandslideSample:
        return MultimodalMartianLandslideSample(**self._dataset[index])
