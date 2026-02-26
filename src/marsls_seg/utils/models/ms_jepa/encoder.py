from typing import override

import torch
from x_transformers import ContinuousTransformerWrapper, Encoder

from marsls_seg.utils.models.ms_jepa.tokenizer import PatchTokenizer
from marsls_seg.utils.modules.masking import Mask, apply_mask
from marsls_seg.utils.models.ms_jepa._typing import MultispectralJEPAEncoderOutput


class MultispectralJEPAEncoder(torch.nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        dim_embed: int,
        n_layers: int,
        n_heads: int,
        n_vision_channels: int,
        n_physics_channels: int,
    ) -> None:
        super().__init__()

        self._image_size: int = image_size
        self._dim_embed: int = dim_embed
        self._patch_size: int = patch_size
        self._n_vision_channels: int = n_vision_channels
        self._n_physics_channels: int = n_physics_channels
        self._n_layers: int = n_layers
        self._n_heads: int = n_heads

        self._vision_tokenizer: PatchTokenizer = PatchTokenizer(
            dim_embed=self._dim_embed,
            n_channels=self._n_vision_channels,
            patch_size=self._patch_size,
        )
        self._physics_tokenizer: PatchTokenizer = PatchTokenizer(
            dim_embed=self._dim_embed,
            n_channels=self._n_physics_channels,
            patch_size=self._patch_size,
        )

        n_patches: int = int((self._image_size // self._patch_size) ** 2)

        self._vision_pos_emb: torch.nn.Parameter = torch.nn.Parameter(
            torch.randn(1, n_patches, self._dim_embed) * 0.02
        )
        self._physics_pos_emb: torch.nn.Parameter = torch.nn.Parameter(
            torch.randn(1, n_patches, self._dim_embed) * 0.02
        )

        self._transformer_encoder: ContinuousTransformerWrapper = (
            ContinuousTransformerWrapper(
                dim_in=self._dim_embed,
                dim_out=self._dim_embed,
                max_seq_len=n_patches,
                use_abs_pos_emb=False,
                attn_layers=Encoder(
                    dim=self._dim_embed,
                    depth=self._n_layers,
                    heads=self._n_heads,
                    ff_swish=True,
                    ff_glu=True,
                    rotary_pos_emb=False,
                ),
            )
        )

    def _tokenize(
        self, vision_image: torch.Tensor, physics_image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vision_patch_tokens: torch.Tensor = (
            self._vision_tokenizer(vision_image) + self._vision_pos_emb
        )
        physics_patch_tokens: torch.Tensor = (
            self._physics_tokenizer(physics_image) + self._physics_pos_emb
        )
        return vision_patch_tokens, physics_patch_tokens

    @override
    def forward(
        self,
        vision_image: torch.Tensor,
        physics_image: torch.Tensor,
        mask: Mask | None = None,
    ) -> MultispectralJEPAEncoderOutput:
        # Tokenize the patches
        vision_tokens, physics_tokens = self._tokenize(
            vision_image=vision_image, physics_image=physics_image
        )

        # Mask patch tokens if required
        if mask is not None:
            vision_tokens, _ = apply_mask(x=vision_tokens, mask=mask)
            physics_tokens, _ = apply_mask(x=physics_tokens, mask=mask)

        # Pass through the transformer
        tokens: torch.Tensor = torch.cat([vision_tokens, physics_tokens], dim=0)
        encoding: torch.Tensor = self._transformer_encoder(tokens)
        vision_encoding, physics_encoding = torch.chunk(encoding, chunks=2, dim=0)

        return MultispectralJEPAEncoderOutput(
            vision_encoding=vision_encoding, physics_encoding=physics_encoding
        )
