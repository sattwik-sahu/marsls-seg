import torch
import pytest
from marsls_seg.utils.modules.masking import generate_mask
from marsls_seg.utils.models.ms_jepa.encoder import MultispectralJEPAEncoder


@pytest.fixture
def vit_config():
    return {
        "image_size": 128,
        "patch_size": 8,
        "n_channels": 4,  # RGB + Gray
        "dim_embed": 384,
        "n_layers": 4,
        "n_heads": 6,
    }


def test_vit_forward_full(vit_config):
    """Test encoding a full image (no masking)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultispectralJEPAEncoder(**vit_config).to(device)

    # Batch of 2 images
    x = torch.randn(2, 4, 128, 128).to(device)

    # Expected number of tokens: (128/8)^2 = 256
    out = model(x)

    assert out.shape == (2, 256, vit_config["dim_embed"])
    assert not torch.isnan(out).any(), "Model produced NaNs"


def test_vit_forward_masked(vit_config):
    """Test encoding only the visible tokens using a mask."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultispectralJEPAEncoder(**vit_config).to(device)

    n_tokens = (vit_config["image_size"] // vit_config["patch_size"]) ** 2
    ratio = 0.6
    mask = generate_mask(n_tokens, ratio)

    x = torch.randn(2, 4, 128, 128).to(device)

    # apply_mask returns (masked_tokens, visible_tokens)
    # Based on your logic: model sees masked_tokens (which are the 'id_visible' indices)
    out = model(x, mask=mask)

    expected_len = n_tokens - int(n_tokens * ratio)
    assert out.shape == (2, expected_len, vit_config["dim_embed"])


def test_vit_gradient_flow(vit_config):
    """Ensure gradients flow back to the patch tokenizer."""
    model = MultispectralJEPAEncoder(**vit_config)
    x = torch.randn(1, 4, 128, 128)

    out = model(x)
    loss = out.mean()
    loss.backward()

    # Check weight of the first conv layer in tokenizer
    grad = model._tokenizer._patch_proj.weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_vit_device_consistency(vit_config):
    """Verify the model works on GPU with masking."""
    model = MultispectralJEPAEncoder(**vit_config).cuda()
    x = torch.randn(1, 4, 128, 128).cuda()

    n_tokens = (vit_config["image_size"] // vit_config["patch_size"]) ** 2
    mask = generate_mask(n_tokens, 0.5)

    # This checks if apply_mask handles the device transfer for ids
    out = model(x, mask=mask)
    assert out.is_cuda
