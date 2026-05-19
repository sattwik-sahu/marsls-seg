from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf


def load_model_from_ckpt(ckpt_path: Path, config_path: Path) -> torch.nn.Module:
    """
    Load a model using the checkpoint path and the config path.

    Args:
        ckpt_path (Path): The path to the checkpoints weights file.
        config_path (Path): The path to the config hydra file.

    Returns:
        torch.nn.Module:
            The model loaded using the config from the config file
            and the checkpoint weights.
    """
    # Check if checkpoint file exists
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No ckpt found at {ckpt_path.as_posix()}")
    if not config_path.exists():
        raise FileNotFoundError(f"No config found at {ckpt_path.as_posix()}")

    # Create the omegaconf container
    model_cfg = OmegaConf.load(config_path)

    # Instantiate the model with hydra
    model: torch.nn.Module = hydra.utils.instantiate(config=model_cfg)

    # Load the weights and states into the model
    ckpt_state_dict = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(ckpt_state_dict)

    return model
