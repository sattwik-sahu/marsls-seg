# from typing import Any

# import torch
# from typing_extensions import override


# class CrossAttentionFeatureFusion(torch.nn.Module):
#     def __init__(
#         self, image_size: int, patch_size: int, dim_embed: int, n_heads: int
#     ) -> None:
#         super().__init__()

#         self._image_size: int = image_size
#         self._patch_size: int = patch_size
#         self._dim_embed: int = dim_embed
#         self._n_heads: int = n_heads

#         self._n_patches: int = int((self._image_size // self._patch_size) ** 2)

#         self._vision_emb: torch.nn.Parameter = torch.nn.Parameter(
#             torch.randn(1, 1, self._dim_embed)
#         )
#         self._physics_emb: torch.nn.Parameter = torch.nn.Parameter(
#             torch.randn(1, 1, self._dim_embed)
#         )
#         self._query: torch.nn.Parameter = torch.nn.Parameter(
#             torch.randn(1, self._n_patches, self._dim_embed)
#         )
#         self._query_pos_emb: torch.nn.Parameter = torch.nn.Parameter(
#             torch.randn(1, self._n_patches, self._dim_embed)
#         )

#         self._mha: torch.nn.MultiheadAttention = torch.nn.MultiheadAttention(
#             embed_dim=self._dim_embed,
#             num_heads=self._n_heads,
#             batch_first=True,
#         )
#         self._dropout: torch.nn.Dropout = torch.nn.Dropout(p=0.1)
#         self._layer_norm: torch.nn.LayerNorm = torch.nn.LayerNorm(self._dim_embed)

#     @override
#     def forward(
#         self, vision_tokens: torch.Tensor, physics_tokens: torch.Tensor
#     ) -> torch.Tensor:
#         vision_tokens = vision_tokens + self._vision_emb
#         physics_tokens = physics_tokens + self._physics_emb
#         key = value = torch.cat([vision_tokens, physics_tokens], dim=1)
#         query = self._query.expand_as(vision_tokens) + self._query_pos_emb
#         attn_output, _ = self._mha(query, key, value)
#         norm_output = self._layer_norm(self._dropout(attn_output) + query)
#         return norm_output


# class UpsampleBlock(torch.nn.Module):
#     def __init__(
#         self, in_channels: int, out_channels: int, dropout_p: float = 0.2
#     ) -> None:
#         super().__init__()
#         # Use Bilinear + Conv instead of Transposed Conv to reduce artifacts
#         self._upsample = torch.nn.Upsample(
#             scale_factor=2, mode="bilinear", align_corners=False
#         )

#         self._refine = torch.nn.Sequential(
#             torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             torch.nn.GroupNorm(
#                 num_groups=min(16, out_channels), num_channels=out_channels
#             ),
#             torch.nn.GELU(),
#             torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             torch.nn.GroupNorm(
#                 num_groups=min(16, out_channels), num_channels=out_channels
#             ),
#             torch.nn.GELU(),
#             torch.nn.Dropout2d(p=dropout_p),  # Regularize spatial features
#         )

#         # Residual connection to preserve latent info
#         self._shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     @override
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x_up = self._upsample(x)
#         return self._shortcut(x_up) + self._refine(x_up)


# class SegmentationHead(torch.nn.Module):
#     def __init__(self, dim_embed: int) -> None:
#         super().__init__()

#         # Rapidly reduce channel depth: 768 -> 128 -> 64 -> 32
#         # This prevents the model from having enough "memory" to overfit noise
#         self._layers = torch.nn.Sequential(
#             UpsampleBlock(in_channels=dim_embed, out_channels=128),  # 16x16 -> 32x32
#             UpsampleBlock(in_channels=128, out_channels=64),  # 32x32 -> 64x64
#             UpsampleBlock(in_channels=64, out_channels=32),  # 64x64 -> 128x128
#             torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
#         )

#     @override
#     def forward(self, features: torch.Tensor) -> torch.Tensor:
#         batch_size, n_tokens, c = features.shape
#         h = w = int(n_tokens**0.5)
#         features = features.transpose(1, 2).reshape(batch_size, c, h, w)
#         return self._layers(features)


import torch
import torch.nn as nn
from typing_extensions import override


class MLPFeatureFusion(nn.Module):
    def __init__(self, dim_embed: int, dropout_p: float = 0.1) -> None:
        super().__init__()
        self._dim_embed = dim_embed

        # Fuse via concatenation and project back to original dim_embed
        self._fusion_mlp = nn.Sequential(
            nn.Linear(dim_embed * 2, dim_embed),
            nn.LayerNorm(dim_embed),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

    @override
    def forward(
        self, vision_tokens: torch.Tensor, physics_tokens: torch.Tensor
    ) -> torch.Tensor:
        # tokens: [B, N, C]
        fused = torch.cat([vision_tokens, physics_tokens], dim=-1)  # [B, N, 2*C]
        return self._fusion_mlp(fused)


class SegmentationHead(nn.Module):
    def __init__(self, dim_embed: int) -> None:
        super().__init__()

        # Simple 8x Bilinear Upsampling + 1x1 Conv (Linear Probe)
        self._upsample = nn.Upsample(
            scale_factor=8, mode="bilinear", align_corners=False
        )
        self._linear_probe = nn.Conv2d(dim_embed, 1, kernel_size=1)

    @override
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, n_tokens, c = features.shape
        h = w = int(n_tokens**0.5)

        # Reshape to 2D: [B, C, 16, 16]
        x = features.transpose(1, 2).reshape(batch_size, c, h, w)

        # Upsample to [B, C, 128, 128]
        x = self._upsample(x)

        # Apply Linear Probe
        return self._linear_probe(x)
