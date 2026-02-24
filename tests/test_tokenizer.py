import torch
import pytest
from marsls_seg.utils.models.ms_jepa.tokenizer import PatchTokenizer


def test_patch_tokenizer_output_shape():
    B, C, H, W = 4, 3, 128, 128
    D, P = 384, 8
    n_tokens = (H // P) * (W // P)  # 256

    tokenizer = PatchTokenizer(dim_embed=D, n_channels=C, patch_size=P)
    x = torch.randn(B, C, H, W)

    tokens = tokenizer(x)

    assert tokens.shape == (B, n_tokens, D), (
        f"Expected shape {(B, n_tokens, D)}, got {tokens.shape}"
    )


def test_patch_tokenizer_multimodal_channels():
    """Verify it handles different channel counts (e.g., 4 for Visual, 3 for Physics)."""
    B, H, W = 2, 64, 64
    D, P = 256, 4

    # Test for 4 channels (RGB + Gray)
    vis_tokenizer = PatchTokenizer(dim_embed=D, n_channels=4, patch_size=P)
    tokens_v = vis_tokenizer(torch.randn(B, 4, H, W))
    assert tokens_v.shape == (B, (H // P) * (W // P), D)

    # Test for 3 channels (DEM + Slope + Thermal)
    phy_tokenizer = PatchTokenizer(dim_embed=D, n_channels=3, patch_size=P)
    tokens_p = phy_tokenizer(torch.randn(B, 3, H, W))
    assert tokens_p.shape == (B, (H // P) * (W // P), D)


def test_patch_tokenizer_gradient_flow():
    """Ensure the tokenizer is a trainable module."""
    tokenizer = PatchTokenizer(dim_embed=128, n_channels=3, patch_size=16)
    x = torch.randn(1, 3, 32, 32)
    tokens = tokenizer(x)

    loss = tokens.sum()
    loss.backward()

    # Check if weights have gradients
    assert tokenizer._patch_proj.weight.grad is not None
    assert not torch.allclose(
        tokenizer._patch_proj.weight.grad,
        torch.zeros_like(tokenizer._patch_proj.weight.grad),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_patch_tokenizer_device_move():
    tokenizer = PatchTokenizer(dim_embed=384, n_channels=3, patch_size=8).cuda()
    x = torch.randn(1, 3, 64, 64).cuda()

    tokens = tokenizer(x)
    assert tokens.is_cuda
