import torch

from marsls_seg.utils.models.ms_jepa.encoder import MultispectralJEPAEncoder
from marsls_seg.utils.models.ms_jepa.predictor import MultispectralJEPAPredictor


def build_encoder_predictor(
    config: dict, device: torch.device
) -> tuple[MultispectralJEPAEncoder, MultispectralJEPAPredictor]:
    encoder: MultispectralJEPAEncoder = MultispectralJEPAEncoder(
        image_size=config["data"]["image_size"],
        patch_size=config["model"]["patch_size"],
        dim_embed=config["model"]["dim"],
        n_vision_channels=sum(
            layer["n_channels"] for layer in config["model"]["feature_layers"]["vision"]
        ),
        n_physics_channels=sum(
            layer["n_channels"]
            for layer in config["model"]["feature_layers"]["physics"]
        ),
        n_heads=config["model"]["encoder"]["n_heads"],
        n_layers=config["model"]["encoder"]["n_layers"],
    ).to(device=device)
    predictor: MultispectralJEPAPredictor = MultispectralJEPAPredictor(
        dim_embed=config["model"]["dim"],
        image_size=config["data"]["image_size"],
        patch_size=config["model"]["patch_size"],
        n_heads=config["model"]["predictor"]["n_heads"],
        n_layers=config["model"]["predictor"]["n_layers"],
    ).to(device=device)
    return encoder, predictor
