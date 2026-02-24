import torch
from typing_extensions import override
from einops import rearrange


class PatchTokenizer(torch.nn.Module):
    def __init__(self, dim_embed: int, n_channels: int, patch_size: int) -> None:
        super().__init__()

        self._dim_embed: int = dim_embed
        self._n_channels: int = n_channels
        self._patch_size: int = patch_size

        self._patch_proj: torch.nn.Conv2d = torch.nn.Conv2d(
            in_channels=self._n_channels,
            out_channels=self._dim_embed,
            kernel_size=self._patch_size,
            stride=self._patch_size,
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_tokens: torch.Tensor = self._patch_proj(x)
        x_tokens = rearrange(x_tokens, "b d h w -> b (h w) d")
        return x_tokens
