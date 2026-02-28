import torch
import pytest
from marsls_seg.utils.models.ms_jepa.predictor import MultispectralJEPAPredictor
from marsls_seg.utils.modules.decoder import Decoder, DecoderLayer

@pytest.fixture
def model_params():
    return {
        "dim_embed": 128,
        "image_size": 128,
        "patch_size": 8,
        "n_layers": 2,
        "n_heads": 4
    }

def test_decoder_layer_shapes(model_params):
    """Verify DecoderLayer handles self and cross attention shapes."""
    dim = model_params["dim_embed"]
    layer = DecoderLayer(dim=dim, n_heads=model_params["n_heads"])
    
    query = torch.randn(2, 10, dim)    # e.g., 10 masked tokens
    context = torch.randn(2, 20, dim)  # e.g., 20 context tokens
    
    out = layer(query, context)
    
    assert out.shape == (2, 10, dim), "Output shape must match query shape."
    assert not torch.isnan(out).any(), "DecoderLayer produced NaNs."

def test_predictor_forward_logic(model_params):
    """Test full Predictor integration with realistic JEPA sizes."""
    predictor = MultispectralJEPAPredictor(**model_params)
    
    B = 4
    n_visible = 100
    n_masked = 156 # 256 total tokens
    D = model_params["dim_embed"]
    
    sx = torch.randn(B, n_visible, D) # Context from ViT
    z = torch.randn(B, n_masked, D)   # Mask tokens + PosEmb
    
    out = predictor(sx, z)
    
    assert out.shape == (B, n_masked, D)

def test_predictor_non_causality(model_params):
    """
    CRITICAL TEST: Ensure the first predicted token 
    can see the last context token (Non-causal).
    """
    predictor = MultispectralJEPAPredictor(**model_params)
    D = model_params["dim_embed"]
    
    z = torch.randn(1, 5, D)
    sx = torch.randn(1, 5, D)
    
    # Baseline
    out_1 = predictor(sx, z)
    
    # Modify the very last token of the context
    sx_mod = sx.clone()
    sx_mod[0, -1] += 10.0
    
    out_2 = predictor(sx_mod, z)
    
    # If bidirectional, the first token of output MUST change
    diff = (out_1[0, 0] - out_2[0, 0]).abs().sum()
    assert diff > 1e-5, "Predictor is causal! First token