import torch
from typing import TypedDict


class Mask(TypedDict):
    id_mask: torch.Tensor
    id_visible: torch.Tensor


def generate_mask(n_tokens: int, mask_ratio: float) -> Mask:
    """
    Decides exactly which indices are context and which are targets.
    """
    n_mask = int(n_tokens * mask_ratio)
    n_keep = n_tokens - n_mask

    # Shuffle all indices
    ids_shuffle = torch.argsort(torch.rand(n_tokens))

    # Split them based on the ratio once
    id_visible = ids_shuffle[:n_keep]
    id_mask = ids_shuffle[n_keep:]

    return Mask(id_mask=id_mask, id_visible=id_visible)


def invert_mask(mask: Mask) -> Mask:
    return Mask(id_mask=mask["id_visible"], id_visible=mask["id_mask"])


def apply_mask(x: torch.Tensor, mask: Mask) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pure gathering function. No logic, just mapping.
    """
    B, _, D = x.shape

    # Helper for batch-wise gathering
    def gather_at(indices) -> torch.Tensor:
        # indices shape: [n_idx] -> Expand to [B, n_idx, D]
        idx = indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, D).to(device=x.device)
        return torch.gather(x, dim=1, index=idx)

    visible_tokens = gather_at(mask["id_visible"])
    masked_tokens = gather_at(mask["id_mask"])

    return visible_tokens, masked_tokens
