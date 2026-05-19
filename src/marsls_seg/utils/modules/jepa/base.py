from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self, override
import hydra
from omegaconf import OmegaConf
import torch
from tensordict import TensorClass

from marsls_seg.utils.modules._typing import TensorData
from marsls_seg.utils.modules.encoder.base import BaseEncoder


class BaseJEPALoss(TensorClass):
    """
    Base class for JEPA loss.

    Must be extended by other methods to create their own loss class.
    Implementations of JEPA must put the loss to be backpropagated
    in the `total` field.
    """

    total: torch.Tensor
    """The total loss to be backpropagated."""


BaseJEPAEncoder = BaseEncoder
"""The base JEPA encoder. It is an alias for `BaseEncoder`."""


class BaseJEPAPredictor[TEncoding: TensorData, TLatent: TensorData](
    torch.nn.Module, ABC
):
    """Base class for a JEPA predictor."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, s_x: TEncoding, z: TLatent) -> TEncoding:
        """
        Predicts the unperturbed input encoding from the perturbed
        input encoding `s_x`, conditioned on the latent `z`.

        Args:
            s_x (TEncoding): The perturbed input encoding.
            z (TLatent): The latent variable.

        Returns:
            TEncoding: The prediction for the encoding of the
                unperturbed input.
        """
        pass


class JEPAOutput[TEncoding, TLoss: BaseJEPALoss](TensorClass):
    """The output of a JEPA model."""

    context_encoding: TEncoding
    """Encoding of the context (perturbed input)."""

    target_encoding: TEncoding
    """Encoding of the target (unperturbed input)."""

    prediction: TEncoding
    """Prediction of the encoding of target."""

    loss: TLoss
    """The loss for the JEPA."""


class BaseJEPA[
    TInput: TensorData,
    TEncoder: BaseJEPAEncoder,
    TEncoding: TensorData,
    TLatent: TensorData,
    TPredictor: BaseJEPAPredictor,
    TLoss: BaseJEPALoss,
](torch.nn.Module, ABC):
    """Base class for the Joint Embedding Predictive Architecture (JEPA)."""

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
        self._predictor: TPredictor = predictor

    @property
    def context_encoder(self) -> TEncoder:
        """The context encoder."""
        return self._context_encoder

    @property
    def target_encoder(self) -> TEncoder:
        """The target encoder."""
        return self._target_encoder

    @property
    def predictor(self) -> TPredictor:
        """The predictor."""
        return self._predictor

    @abstractmethod
    def _calculate_loss(
        self, s_x: TEncoding, s_y: TEncoding, s_y_hat: TEncoding
    ) -> TLoss:
        """
        Calculate the loss for a single step.

        Args:
            s_x (TEncoding): The context encoder output.
            s_y (TEncoding): The target encoder output.
            s_y_hat (TEncoding): The predictor output.

        Returns:
            TLoss: The loss for the single step.
        """
        pass

    @override
    def forward(self, x: TInput, y: TInput, z: TLatent) -> JEPAOutput:
        """
        Perform a forward pass through the JEPA.

        Args:
            x (TInput): The context input.
            y (TInput): The target input.
            z (TLatent): The latent variable.

        Returns:
            JEPAOutput[TEncoding, TLoss]:
                The output of the JEPA, containing:
                - Context encoding
                - Target encoding
                - Target prediction
                - Loss
        """
        # Context encoding
        s_x: TEncoding = self._context_encoder(x)

        # Target encoding
        s_y: TEncoding = self._target_encoder(y)

        # Predict target from context encoding and latent variable
        s_y_hat: TEncoding = self._predictor(s_x=s_x, z=z)

        # Calculate the loss
        loss: TLoss = self._calculate_loss(s_x=s_x, s_y=s_y, s_y_hat=s_y_hat)

        # Construct the output of the JEPA
        jepa_output = JEPAOutput(
            context_encoding=s_x, target_encoding=s_y, prediction=s_y_hat, loss=loss
        )

        return jepa_output

    @classmethod
    def load_from_ckpt(cls, ckpt_path: Path, config_path: Path) -> Self:
        # Check if checkpoint file exists
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No ckpt found at {ckpt_path.as_posix()}")
        if not config_path.exists():
            raise FileNotFoundError(f"No config found at {ckpt_path.as_posix()}")

        # Create the omegaconf container
        model_cfg = OmegaConf.load(config_path)

        # Instantiate the model with hydra
        model: Self = hydra.utils.instantiate(config=model_cfg)

        # Load the weights and states into the model
        ckpt_state_dict = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(ckpt_state_dict)

        return model
