from typing import override

import torch
from x_transformers import ContinuousTransformerWrapper, Encoder

from marsls_seg.utils.models.ms_jepa.tokenizer import PatchTokenizer
from marsls_seg.utils.modules.masking import Mask, apply_mask


class VisionTransformer(torch.nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        n_channels: int,
        dim_embed: int,
        n_layers: int,
        n_heads: int,
    ) -> None:
        super().__init__()

        self._image_size: int = image_size
        self._dim_embed: int = dim_embed
        self._patch_size: int = patch_size
        self._n_channels: int = n_channels
        self._n_layers: int = n_layers
        self._n_heads: int = n_heads

        self._tokenizer: PatchTokenizer = PatchTokenizer(
            dim_embed=self._dim_embed,
            n_channels=self._n_channels,
            patch_size=self._patch_size,
        )

        n_patches: int = int((self._image_size // self._patch_size) ** 2)

        self._pos_emb: torch.nn.Parameter = torch.nn.Parameter(
            torch.randn(1, n_patches, self._dim_embed)
        )

        self._transformer_encoder: ContinuousTransformerWrapper = (
            ContinuousTransformerWrapper(
                dim_in=dim_embed,
                dim_out=dim_embed,
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

    @override
    def forward(self, image: torch.Tensor, mask: Mask | None = None) -> torch.Tensor:
        # Tokenize the patches
        patch_tokens: torch.Tensor = self._tokenizer(image)

        # Add learned positional embeddings
        patch_tokens = patch_tokens + self._pos_emb

        # Mask patch tokens if required
        if mask is not None:
            patch_tokens, _ = apply_mask(x=patch_tokens, mask=mask)

        # Pass through the transformer
        patch_encodings: torch.Tensor = self._transformer_encoder(patch_tokens)

        return patch_encodings
