import torch


class Swish(torch.nn.Module):
    """The Swish activation function."""

    def __init__(self, beta: float) -> None:
        super().__init__()

        # Define the parameter
        self._beta: float = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.nn.functional.sigmoid(self._beta * x)


class FFNGatedLinearUnit(torch.nn.Module):
    """
    Feedforward network Gated Linear Unit (GLU) variant for a particular activation function [1].

    ### References
    - [1] Shazeer, Noam. "Glu variants improve transformer." arXiv preprint arXiv:2002.05202 (2020).
    """

    def __init__(
        self, sigma: torch.nn.Module, dim: int, dim_feedforward: int = 3072
    ) -> None:
        super().__init__()

        self._sigma: torch.nn.Module = sigma
        self._dim: int = dim
        # This is d_ff, chosen as specified in the paper
        self._dim_feedforward: int = (dim_feedforward * 2) // 3

        # Linear layer without bias is same as weight matrix multiplcation
        # NOTE This choice was made to handle batched inputs gracefully
        self._WV = torch.nn.Linear(
            in_features=self._dim, out_features=2 * self._dim_feedforward, bias=False
        )
        self._W2 = torch.nn.Linear(
            in_features=self._dim_feedforward, out_features=self._dim, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xW, xV = torch.chunk(self._WV(x), chunks=2, dim=-1)
        return self._W2(self._sigma(xW) * xV)


class SwiGLU(FFNGatedLinearUnit):
    """
    The SwiGLU activation function.
    """

    def __init__(self, dim: int, dim_feedforward: int = 3072) -> None:
        # SiLU is the same as Swish_1
        super().__init__(
            sigma=torch.nn.SiLU(), dim=dim, dim_feedforward=dim_feedforward
        )
