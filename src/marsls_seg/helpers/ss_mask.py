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

def generate_spatio_spectral_mask(
        mask_ratio: float,
        n_patches: int,
        n_channels: int,
        device : torch.device=DEVICE
) -> tuple[torch.Tensor, torch.Tensor]:
    
    """
        Generate a mask across the entire spatio-spectral Sequence.
        Args:
            mask_ratio (float): Fraction of tokens to hide
            n_patches (int) : total number of patches in the image
            n_channels (int) : total number of channels in the image

        Returns:
            ids_keep (torch.tensor): 1D tensor of indices to keep
            ids_drop (torch.tensor): 1D tensor of indices to drop



    """ 

    total_tokens = n_patches * n_channels
    num_keep = int(total_tokens * (1-mask_ratio))

    perm = torch.randperm(total_tokens,device=device)

    ids_keep = torch.sort(perm[:num_keep])
    ids_drop = torch.sort(perm[num_keep:])

    ids_keep, _ = torch.sort(ids_keep)
    ids_drop, _ = torch.sort(ids_drop)

    return ids_keep, ids_drop



def construct_latent(
        z_full : torch.Tensor,
        ids_drop : torch.Tensor,
        batch_size : int | None = None
) -> torch.Tensor :
    """
        Construct the latent (z) the (query) for the predictor.

        Args:
            z_full (torcrh.Tensor): The full spatial spectral learned embeddings (total_tokns,dim)
            ids_drop (torch.Tensor): The indices of the tokens to drop (mask)
            batch_size(int) : The current batch

        Returns:
            z (torch.Tensor) : The query tensor of shape (Batch, M, dim)
    """ 

    z=z_full[ids_drop]

    if batch_size is not None:
        z=z.unsqueeze(0).expand(batch_size,*z.shape)

    return z
