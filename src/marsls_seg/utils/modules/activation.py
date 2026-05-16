import torch.nn as nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 8 // 3)  # Standard SwiGLU expansion
        self.w2 = nn.Linear(dim, dim * 8 // 3)
        self.w3 = nn.Linear(dim * 8 // 3, dim)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
