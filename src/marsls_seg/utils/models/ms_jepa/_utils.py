from pathlib import Path
from typing import TypedDict

import torch
import yaml

from marsls_seg.utils.data.marsls import MarsLS_Sample
from marsls_seg.utils.models.ms_jepa.encoder import MultispectralJEPAEncoder
from marsls_seg.utils.models.ms_jepa.predictor import MultispectralJEPAPredictor
from marsls_seg.helpers.processing import group_channels


class MultispectralJEPALoaderOutput(TypedDict):
    encoder: MultispectralJEPAEncoder
    predictor: MultispectralJEPAPredictor
    config: dict


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


def load_models(
    dir_path: Path, device: torch.device = torch.device("cpu")
) -> MultispectralJEPALoaderOutput:
    """
    Loads the Multispectral JEPA encoder and predictor models from the `dir_path`
    directory, created after training the model.

    Args:
        dir_path (Path): The `Path` to the directory where the model weights
            and config are saved after training.
        device (torch.device): The device to load the models on.
            Default: `torch.device("cpu")`
    """

    # Parse config
    with open(dir_path / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Build encoder and predictor models
    encoder, predictor = build_encoder_predictor(config=config, device=device)

    # Construct paths to weight files
    encoder_path: Path = dir_path / "encoder.pt"
    predictor_path: Path = dir_path / "predictor.pt"

    # Load the weights from the files into the models
    encoder.load_state_dict(torch.load(encoder_path, weights_only=True))
    predictor.load_state_dict(torch.load(predictor_path, weights_only=True))

    # Load models onto target device and return
    return MultispectralJEPALoaderOutput(
        encoder=encoder.to(device=device),
        predictor=predictor.to(device=device),
        config=config,
    )


def preprocess_batch(batch: MarsLS_Sample, config: dict) -> MarsLS_Sample:
    batch["slope"] = batch["slope"].clamp_min(min=-50)
    for layer in config["data"]["feature_info"]:
        info = layer["layer"]
        name = info["name"]
        mean = torch.as_tensor(info["mean"]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        std = torch.as_tensor(info["std"]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        batch[name] = (batch[name] - mean) / std
    return batch


def get_vision_physics_images(batch: MarsLS_Sample, config: dict, device: torch.device):
    vision_image: torch.Tensor = group_channels(
        sample=batch,
        channel_names=[
            layer["name"] for layer in config["model"]["feature_layers"]["vision"]
        ],
    ).to(device=device)
    physics_image: torch.Tensor = group_channels(
        sample=batch,
        channel_names=[
            layer["name"] for layer in config["model"]["feature_layers"]["physics"]
        ],
    ).to(device=device)
    return vision_image, physics_image
