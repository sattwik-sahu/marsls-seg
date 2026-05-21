import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import override
from einops import repeat, rearrange
from tensordict import TensorClass

from marsls_seg.utils.modules.tf.encoder import TransformerEncoder
from marsls.seg.utils.modules.encoder.base import BaseImagePatchEncoder


class SSVitInput(TensorClass):
    image: torch.Tensor # (B,7,128,128)
    mask: torch.Tensor # (M) Indices of Visible tokens in the range of 
