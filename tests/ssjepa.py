import torch
import torch.nn as nn
from marsls_seg.utils.data._typing import MultimodalMartianLandslideSample
from marsls_seg.utils.modules.encoder.spatio_spectral_vit import SpatioSpectralVisionTransformer, SSViTInput
from marsls_seg.utils.modules.ssjepa.jepa import SpatioSpectralJEPA
from marsls_seg.utils.modules.jepa.extras import SimpleTransformerDecoderPredictor
from marsls_seg.utils.modules.sigreg import SIGReg
from marsls_seg.helpers.mask import generate_spatio_spectral_mask, construct_latent

def test_pipeline():
    
    batch_size = 2
    dim = 64
    patch_size = 8
    img_size = 128
    n_channels = 7
    n_patches = (img_size // patch_size) ** 2  # 256
    total_tokens = n_patches * n_channels      # 1792
    mask_ratio = 0.7
    device = torch.device("cpu") 

    print(f"--- Initializing Spatio-Spectral Pipeline ---")
    print(f"Total Spatio-Spectral Tokens: {total_tokens}")


    dummy_image = torch.randn(batch_size, n_channels, img_size, img_size)
    sample = MultimodalMartianLandslideSample(
        rgb=dummy_image[:, 0:3],
        dem=dummy_image[:, 3:4],
        slope=dummy_image[:, 4:5],
        thermal_inertial=dummy_image[:, 5:6],
        grayscale=dummy_image[:, 6:7],
        label=torch.empty(batch_size, 1, img_size, img_size)
    )

   
    encoder = SpatioSpectralVisionTransformer(
        dim=dim, n_heads=4, n_layers=2, 
        patch_size=patch_size, img_size=img_size, n_channels=n_channels
    )

    predictor = SimpleTransformerDecoderPredictor(
        dim=dim, n_heads=4, n_layers=2
    )

    sigreg = SIGReg(n_directions=128) 

    model = SpatioSpectralJEPA(
        encoder=encoder,
        predictor=predictor,
        sigreg=sigreg,
        sigreg_lambda=0.1
    )

    
    ids_keep, ids_drop = generate_spatio_spectral_mask(
        mask_ratio=mask_ratio,
        n_patches=n_patches,
        n_channels=n_channels,
        device=device
    )
    
    num_visible = ids_keep.numel()
    num_masked = ids_drop.numel()
    print(f"Masking: {num_visible} visible, {num_masked} dropped.")

    
    image_fat = sample.merge_channels() 
    
    x = SSViTInput(image=image_fat, mask=ids_keep)
    y = SSViTInput(image=image_fat)
    
    z_full = model.context_encoder.get_full_pos_embed(device)
    z = construct_latent(z_full=z_full, ids=ids_drop, batch_size=batch_size)

    
    print("\n--- Running Forward Pass ---")
    output = model(x=x, y=y, z=z)

 
    assert output.context_encoding.shape == (batch_size, num_visible, dim), \
        f"Context shape mismatch: {output.context_encoding.shape}"
    

    assert output.target_encoding.shape == (batch_size, total_tokens, dim), \
        f"Target shape mismatch: {output.target_encoding.shape}"

   
    assert output.prediction.shape == (batch_size, num_masked, dim), \
        f"Prediction shape mismatch: {output.prediction.shape}"

   
    print(f"Loss Total: {output.loss.total.item():.4f}")
    print(f"Loss Pred: {output.loss.pred.item():.4f}")
    print(f"Loss SIGReg: {output.loss.sigreg.item():.4f}")

    assert not torch.isnan(output.loss.total), "Loss is NaN!"
    
    print("\n[SUCCESS] Pipeline forward pass is verified.")

if __name__ == "__main__":
    test_pipeline()