import torch

from marsls_seg.utils.data.marsls import MarsLS_Sample


def group_channels(sample: MarsLS_Sample, channel_names: list[str]) -> torch.Tensor:
    return torch.cat([sample[key] for key in channel_names], dim=1)
