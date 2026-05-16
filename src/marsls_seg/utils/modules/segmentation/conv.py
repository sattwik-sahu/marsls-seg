import torch
import torch.nn as nn
from typing_extensions import override


class ConvSegmentationDecoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        dim_embed: int,
        n_classes: int = 1,
        base_channels: int = 128,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size  # e.g., 128 // 8 = 16

        # 1. Initial projection to reduce JEPA latent dim to something lighter
        # If you are fusing Vision + Physics, dim_embed might be 2 * original_dim
        self.proj = nn.Conv2d(
            in_channels=2 * dim_embed, out_channels=base_channels, kernel_size=1
        )

        # 2. Calculate number of upsampling steps (e.g., for patch_size 8, we need 3 doublings)
        # 8 -> 4 -> 2 -> 1 (log2 of 8 is 3)
        num_upsamples = int(torch.log2(torch.tensor(patch_size)).item())

        layers = []
        curr_channels = base_channels
        for i in range(num_upsamples):
            out_channels = curr_channels // 2
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=curr_channels,
                        out_channels=out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                )
            )
            curr_channels = out_channels

        self.upsample_blocks = nn.Sequential(*layers)

        # 3. Final segmentation head
        self.final_head = nn.Conv2d(curr_channels, n_classes, kernel_size=1)

    @override
    def forward(
        self, vision_encodings: torch.Tensor, physics_encodings: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_vision: [B, N, D] latents
            z_physics: [B, N, D] latents
        """
        # Fuse the multimodal latents (Concatenation is standard for JEPA)
        z = torch.cat([vision_encodings, physics_encodings], dim=-1)  # [B, 256, 2*D]

        # Reshape to 2D grid: [B, 2*D, 16, 16]
        B, N, D = z.shape
        z = z.transpose(1, 2).reshape(B, D, self.grid_size, self.grid_size)

        # Project and Upsample
        x = self.proj(z)
        x = self.upsample_blocks(x)

        # Output logits: [B, 1, 128, 128]
        return self.final_head(x)
