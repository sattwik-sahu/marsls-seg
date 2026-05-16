import torch

from marsls_seg.utils.data.mmls import MultimodalMartianLandslideSample


def group_channels(
    sample: MultimodalMartianLandslideSample, channel_names: list[str]
) -> torch.Tensor:
    return torch.cat([sample[key] for key in channel_names], dim=1)
