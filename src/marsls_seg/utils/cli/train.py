import typer
from typing_extensions import Annotated
from marsls_seg.utils.train.ms_jepa import train as train_ms_jepa
from marsls_seg.utils.train.ms_jepa_sw import train as train_ms_jepa_sw
from pathlib import Path

app = typer.Typer(name="train", help="Train models for Mars Landslide Segmentation")


@app.command(name="ms_jepa", help="Train a Multispectral JEPA (only SSL encoders)")
def ms_jepa(
    config_path: Annotated[
        Path, typer.Argument(help="The path to the YAML config file")
    ],
) -> None:
    train_ms_jepa(config_path=config_path)


@app.command(name="ms_jepa_sw", help="Train a Multispectral JEPA (only SSL encoders)")
def ms_jepa_sw(
    config_path: Annotated[
        Path, typer.Argument(help="The path to the YAML config file")
    ],
) -> None:
    train_ms_jepa_sw(config_path=config_path)
