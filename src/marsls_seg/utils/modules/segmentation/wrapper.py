import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders._base import EncoderMixin

from marsls_seg.utils.modules.encoder.base import BaseImagePatchEncoder
from marsls_seg.utils.modules.encoder.patch_masking_vit import PatchMaskingViTInput
from marsls_seg.utils.modules.jepa.extras import BaseImagePatchJEPA

from einops import rearrange

from pathlib import Path
from marsls_seg.utils.modules.load import load_model_from_ckpt


class SMPWrapper(nn.Module, EncoderMixin):
    """
    Wraps the IJEPA ContextEncoder to be universally compatible with segmentation_models_python
    """

    def __init__(self, jepa_ckpt_path: Path, jepa_config_path: Path, **kwargs) -> None:
        super().__init__()

        self._jepa_model: BaseImagePatchJEPA = load_model_from_ckpt(
            ckpt_path=jepa_ckpt_path,
            config_path=jepa_config_path,
            model_class=BaseImagePatchJEPA,
        )
        self._encoder: BaseImagePatchEncoder = self._jepa_model.context_encoder
        self._patch_size: int = self._encoder.patch_size
        self._dim_embed: int = self._encoder.dim
        self._in_channels: int = self._encoder.n_channels

        self._depth: int = 5
        self._output_stride: int = 16

        # Required by SMP for Deeplabv3 and PAN
        self.is_dilated: bool = False

        self._out_channels: list[int] = [
            self._in_channels,
            self._in_channels,
            self._in_channels,
            self._dim_embed,
            self._dim_embed,
            self._dim_embed,
        ]

        self._skip_dropout = nn.Dropout2d(p=0.1)

    def make_dilated(self, output_stride) -> None:
        self.is_dilated = True

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        vit_input = PatchMaskingViTInput(image=x)
        encodings: torch.Tensor = self._encoder(vit_input)

        # Define number of patches in height and width
        hp = wp = int((self._encoder.n_patches) ** 0.5)

        # Rearrange encodings to form feature maps
        feat: torch.Tensor = rearrange(
            encodings, "b (hp wp) d -> b d hp wp", hp=hp, wp=wp
        )

        # we will define the feature pyramid
        f0 = self._skip_dropout(x)
        f1 = self._skip_dropout(F.max_pool2d(f0, kernel_size=2, stride=2))
        f2 = self._skip_dropout(F.max_pool2d(f1, kernel_size=2, stride=2))
        f3 = feat
        f4 = F.max_pool2d(f3, kernel_size=2, stride=2)

        if self.is_dilated:
            f5 = f4
        else:
            f5 = F.max_pool2d(f4, kernel_size=2, stride=2)

        return [f0, f1, f2, f3, f4, f5]
