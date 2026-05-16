import torch
from tensordict import TensorDict, TensorClass
from typing_extensions import override
from typing import Optional
from abc import ABC, abstractmethod


class JEPALoss(TensorClass):
    total: torch.Tensor


class JEPAOutput[TEncoding, TLoss: JEPALoss](TensorClass):
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
    TLoss: JEPALoss,
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
        s_x: TEncoding
        s_y: TEncoding
        if self._shared_weight:
            s_x, s_y = torch.chunk(
                self._context_encoder(torch.cat([x, y], dim=0)),  # type: ignore
                chunks=2,
            )  # type: ignore
        else:
            s_x = self._context_encoder(x)
            s_y = self._target_encoder(y)

        s_y_hat: TEncoding = self._predictor(s_x, z)

        jepa_output = JEPAOutput(
            context_encoding=s_x, target_encoding=s_y, prediction=s_y_hat, loss=None
        )

        if self.training:
            jepa_output.loss = self._calculate_loss(s_x=s_x, s_y=s_y, s_y_hat=s_y_hat)

        return jepa_output
