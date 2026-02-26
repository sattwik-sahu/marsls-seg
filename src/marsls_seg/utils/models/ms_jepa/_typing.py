import torch
from typing import TypedDict


class MultispectralJEPAEncoderOutput(TypedDict):
    vision_encoding: torch.Tensor
    physics_encoding: torch.Tensor
