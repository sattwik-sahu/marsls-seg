from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from tensordict import TensorClass, TensorDict
from typing_extensions import override


class BaseJEPALoss(TensorClass):
    total: torch.Tensor


class JEPAOutput[TEncoding, TLoss: BaseJEPALoss](TensorClass):
    context_encoding: TEncoding
    target_encoding: TEncoding
    prediction: TEncoding
    loss: Optional[TLoss]


class BaseJEPA[
    TInput: torch.Tensor | TensorClass | TensorDict,
    TEncoder: torch.nn.Module,
    TEncoding: torch.Tensor | TensorClass | TensorDict,
    TLatent: torch.Tensor | TensorClass | TensorDict,
    TPredictor: torch.nn.Module,
    TLoss: BaseJEPALoss,
](torch.nn.Module, ABC):
    """
    Base class for the Joint Embedding Predictive Architecture (JEPA).
    """

    def __init__(
        self,
        context_encoder: TEncoder,
        predictor: TPredictor,
        target_encoder: TEncoder | None = None,
    ) -> None:
        super().__init__()

        # Initialize the module
        self._context_encoder: TEncoder = context_encoder
        self._target_encoder: TEncoder = target_encoder or self._context_encoder
        self._shared_weight: bool = target_encoder is None
        self._predictor: TPredictor = predictor

    @property
    def context_encoder(self) -> TEncoder:
        return self._context_encoder

    @property
    def target_encoder(self) -> TEncoder:
        return self._target_encoder

    @property
    def predictor(self) -> TPredictor:
        return self._predictor

    @abstractmethod
    def _calculate_loss(
        self, s_x: TEncoding, s_y: TEncoding, s_y_hat: TEncoding
    ) -> TLoss:
        pass

    @override
    def forward(self, x: TInput, y: TInput, z: TLatent) -> JEPAOutput[TEncoding, TLoss]:
        s_x: TEncoding = self._context_encoder(x)
        s_y: TEncoding = self._target_encoder(y)

        s_y_hat: TEncoding = self._predictor(s_x=s_x, z=z)

        jepa_output = JEPAOutput(
            context_encoding=s_x, target_encoding=s_y, prediction=s_y_hat, loss=None
        )

        # if self.training:
        jepa_output.loss = self._calculate_loss(s_x=s_x, s_y=s_y, s_y_hat=s_y_hat)

        return jepa_output
