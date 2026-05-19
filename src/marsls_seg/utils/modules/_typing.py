import torch
from tensordict import TensorClass, TensorDict


type TensorData = torch.Tensor | TensorClass | TensorDict
