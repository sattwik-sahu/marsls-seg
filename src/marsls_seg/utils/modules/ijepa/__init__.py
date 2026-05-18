"""Implementation of IJEPA for Martian landslide segmentation"""

from marsls_seg.utils.modules.ijepa.encoder import IJEPAEncoder
from marsls_seg.utils.modules.ijepa.predictor import IJEPAPredictor
from marsls_seg.utils.modules.ijepa._typing import (
    IJEPAInput,
    IJEPALatent,
    IJEPAEncoding,
    IJEPALoss,
)
from marsls_seg.utils.modules.ijepa.jepa import IJEPA

__all__ = [
    "IJEPAEncoder",
    "IJEPAPredictor",
    "IJEPAInput",
    "IJEPALatent",
    "IJEPAEncoding",
    "IJEPALoss",
    "IJEPA",
]
