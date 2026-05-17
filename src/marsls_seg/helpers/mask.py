import torch
from marsls_seg.helpers.device import DEVICE


def generate_uniform_mask(
    n_patches: int, mask_ratio: float, device: torch.device = DEVICE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a 1D tensor of patch indices to keep and drop.

    Args:
        n_patches (int): Total number of patches in the image.
        mask_ratio (float): Fraction of patches to drop (e.g., 0.6 to drop 60%).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The tuple `ids_keep, ids_drop`
    """
    num_keep = int(n_patches * (1 - mask_ratio))

    # Randomly shuffle all patch coordinates
    shuffled_indices = torch.randperm(n_patches, device=device)

    # Slice out only the number of patches we want to keep
    ids_keep = shuffled_indices[:num_keep]
    ids_drop = shuffled_indices[num_keep:]
    return ids_keep, ids_drop


def construct_latent(
    z_full: torch.Tensor, ids: torch.Tensor, batch_size: int | None = None
) -> torch.Tensor:
    z = z_full
    # z = z_full[ids]
    if batch_size is not None:
        z = z.unsqueeze(0).expand(batch_size, *z.shape)
    return z
