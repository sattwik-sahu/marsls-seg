import torch
from marsls_seg.utils.models.ms_jepa.encoder import MultispectralJEPAEncoder
from marsls_seg.utils.models.ms_jepa.predictor import (
    MultispectralJEPAPredictor as MultispectralJEPAPredictor,
)
import yaml
from pathlib import Path
from typing import TypedDict
from marsls_seg.utils.models.ms_jepa._utils import build_encoder_predictor


class MultispectralJEPAModels(TypedDict):
    encoder: MultispectralJEPAEncoder
    predictor: MultispectralJEPAPredictor


def load_ms_jepa_models(path: Path, device: torch.device) -> MultispectralJEPAModels:
    # Parse config
    with open(path / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Build encoder and predictor models
    encoder, predictor = build_encoder_predictor(config=config, device=device)

    # Construct paths to weight files
    encoder_path: Path = path / "encoder.pt"
    predictor_path: Path = path / "predictor.pt"

    # Load the weights from the files into the models
    encoder.load_state_dict(torch.load(encoder_path, weights_only=True))
    predictor.load_state_dict(torch.load(predictor_path, weights_only=True))

    return MultispectralJEPAModels(encoder=encoder, predictor=predictor)
