from typing import override

from marsls_seg.utils.modules.ijepa._typing import IJEPAEncoding, IJEPALatent
from marsls_seg.utils.modules.jepa.base import BaseJEPAPredictor
from marsls_seg.utils.modules.tf.decoder import TransformerDecoder


class IJEPAPredictor(BaseJEPAPredictor[IJEPAEncoding, IJEPALatent]):
    """The IJEPA predictor. Uses the transformer decoder under the hood."""

    def __init__(
        self, dim: int, n_heads: int, n_layers: int, n_groups: int | None = None
    ) -> None:
        super().__init__()

        # Initialize the decoder
        self._decoder: TransformerDecoder = TransformerDecoder(
            dim=dim, n_heads=n_heads, n_layers=n_layers, n_groups=n_groups
        )

    @override
    def forward(self, s_x: IJEPAEncoding, z: IJEPALatent) -> IJEPAEncoding:
        return self._decoder(tgt=z, mem=s_x)
