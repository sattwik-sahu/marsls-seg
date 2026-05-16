import torch
import pytest
from marsls_seg.utils.modules.masking import generate_mask, apply_mask, Mask


def test_masking_logic():
    # Setup params for Martian patches (16x16 grid)
    B, N, D = 8, 256, 384
    ratio = 0.6
    n_visible = N - int(N * ratio)
    n_masked = int(N * ratio)

    # Generate dummy tokens
    x = torch.randn(B, N, D)

    # 1. Test generate_mask structure and counts
    mask: Mask = generate_mask(N, ratio)

    assert isinstance(mask, dict)
    assert "id_mask" in mask and "id_visible" in mask
    assert len(mask["id_visible"]) == n_visible
    assert len(mask["id_mask"]) == n_masked

    # Ensure all tokens are accounted for and unique
    combined = torch.cat([mask["id_visible"], mask["id_mask"]])
    assert torch.unique(combined).size(0) == N, (
        "Mask indices must be unique and cover all tokens."
    )

    # 2. Test apply_mask execution
    # Note: Based on your code, masked_tokens uses id_visible, visible_tokens uses id_mask
    visible_for_enc, masked_for_pred = apply_mask(x, mask)

    assert visible_for_enc.shape == (B, n_visible, D)
    assert masked_for_pred.shape == (B, n_masked, D)


def test_masking_content_integrity():
    """Verify that the tokens returned are actually the ones from the original tensor."""
    B, N, D = 1, 4, 2
    # [Patch 0, Patch 1, Patch 2, Patch 3]
    x = torch.tensor([[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]]])

    # Manual mask: Visible=[0, 2], Masked=[1, 3]
    mask = Mask(id_visible=torch.tensor([0, 2]), id_mask=torch.tensor([1, 3]))

    visible_for_enc, masked_for_pred = apply_mask(x, mask)

    # Check visible tokens (id_visible)
    assert torch.allclose(visible_for_enc[0, 0], torch.tensor([0.1, 0.1]))
    assert torch.allclose(visible_for_enc[0, 1], torch.tensor([0.3, 0.3]))

    # Check masked tokens (id_mask)
    assert torch.allclose(masked_for_pred[0, 0], torch.tensor([0.2, 0.2]))
    assert torch.allclose(masked_for_pred[0, 1], torch.tensor([0.4, 0.4]))


def test_device_compatibility():
    """Ensure gather works across devices (CPU indices to GPU tensor)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, N, D = 2, 100, 64
    x = torch.randn(B, N, D).cuda()
    mask = generate_mask(N, 0.5)  # Indices start on CPU

    try:
        visible, masked = apply_mask(x, mask)
        assert visible.is_cuda
        assert masked.is_cuda
    except TypeError as e:
        pytest.fail(f"Type error during device transfer: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")


@pytest.mark.parametrize("ratio", [0.25, 0.5, 0.75])
def test_mask_ratios(ratio):
    N = 100
    mask = generate_mask(N, ratio)
    # n_mask = int(N * ratio)
    assert len(mask["id_mask"]) == int(N * ratio)
