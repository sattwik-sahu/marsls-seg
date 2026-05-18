import torch
from marsls_seg.helpers.device import DEVICE


class MaskingRatioScheduler:
    """Masking ratio scheduler."""

    def __init__(self, start: float, end: float, T: int) -> None:
        """
        Create a masking ratio scheduler.

        Args:
            start (float): The start value of the masking ratio.
            end (float): The final value of the masking ratio.
            T (int): The number of steps in which masking ratio reaches final value.
        """
        self._end: float = end
        self._value: float = start
        self._update: float = (end - start) / T
        self._schedule_increasing: bool = start < end

    @property
    def value(self) -> float:
        """The current value of the masking ratio"""
        return self._value

    def step(self) -> float:
        if (self._schedule_increasing and self._value < self._end) or (
            not self._schedule_increasing and self._value > self._end
        ):
            self._value += self._update
        return self._value


def generate_uniform_mask(
    mask_ratio: float, n_patches: int, device: torch.device = DEVICE
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a 1D tensor of patch indices to keep and drop.

    Args:
        n_patches (int): Total number of patches in the image.
        mask_ratio (float): Fraction of patches to drop (e.g., 0.6 to drop 60%).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The tuple `ids_keep, ids_drop`
    """
    num_keep = int(n_patches * (1 - mask_ratio))

    # Randomly shuffle all patch coordinates
    shuffled_indices = torch.randperm(n_patches, device=device)

    # Slice out the patches we want to keep and drop
    ids_keep = shuffled_indices[:num_keep]
    ids_drop = shuffled_indices[num_keep:]

    # Sort the ids so that token ordering errors don't occur later
    ids_keep, _ = torch.sort(ids_keep)
    ids_drop, _ = torch.sort(ids_drop)

    return ids_keep, ids_drop


def construct_latent(
    z_full: torch.Tensor, ids: torch.Tensor, batch_size: int | None = None
) -> torch.Tensor:
    z = z_full
    z = z_full[ids]
    if batch_size is not None:
        z = z.unsqueeze(0).expand(batch_size, *z.shape)
    return z
