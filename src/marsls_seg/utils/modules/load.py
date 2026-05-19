from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf


def load_model_from_ckpt[TModel: torch.nn.Module](
    ckpt_path: Path, config_path: Path, model_class: type[TModel]
) -> TModel:
    """
    Load a model using the checkpoint path and the config path.

    Args:
        ckpt_path (Path): The path to the checkpoints weights file.
        config_path (Path): The path to the config hydra file.
        model_class (type[TModel]): The class of the model that will
            be loaded with this function.

    Returns:
        torch.nn.Module:
            The model loaded using the config from the config file
            and the checkpoint weights.

    Example:
        >>> from marsls_seg.utils.modules.ijepa import IJEPA
        >>> ckpt_path: Path = Path("path/to/ckpt/weights.pt")
        >>> config_path: Path = Path("path/to/ckpt/config.yaml")
        >>> model: IJEPA = load_model_from_ckpt(
        ...    ckpt_path=ckpt_path,
        ...    config_path=config_path,
        ...    model_class=IJEPA
        ... )
    """
    # Check if checkpoint file exists
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No ckpt found at {ckpt_path.as_posix()}")
    if not config_path.exists():
        raise FileNotFoundError(f"No config found at {ckpt_path.as_posix()}")

    # Create the omegaconf container
    model_cfg = OmegaConf.load(config_path)

    # Instantiate the model with hydra
    model: TModel = hydra.utils.instantiate(config=model_cfg)

    # Load the weights and states into the model
    ckpt_state_dict = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(ckpt_state_dict)

    return model
