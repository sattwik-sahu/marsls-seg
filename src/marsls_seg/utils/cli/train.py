from pathlib import Path

import typer
from typing_extensions import Annotated, Optional

from marsls_seg.utils.train.ms_jepa import train as train_ms_jepa
from marsls_seg.utils.train.ms_jepa_seg import train as train_ms_jepa_seg
from marsls_seg.utils.train.ms_jepa_sw import train as train_ms_jepa_sw

app = typer.Typer(name="train", help="Train models for Mars Landslide Segmentation")


@app.command(name="ms_jepa", help="Train a Multispectral JEPA (only SSL encoders)")
def ms_jepa(
    config_path: Annotated[
        Path, typer.Argument(help="The path to the YAML config file")
    ],
) -> None:
    train_ms_jepa(config_path=config_path)


@app.command(
    name="ms_jepa_sw",
    help="Train a Multispectral JEPA with shared weights (only SSL encoders) [Recommended]",
)
def ms_jepa_sw(
    config_path: Annotated[Path, typer.Option(help="The path to the YAML config file")],
    pretrained_path: Annotated[
        Optional[Path], typer.Option(help="Path to pretrained weights")
    ] = None,
) -> None:
    train_ms_jepa_sw(config_path=config_path, pretrained_path=pretrained_path)


@app.command(
    name="ms_jepa_seg",
    help="Train a segmentation head on top of a pretrained Multispectral JEPA",
)
def ms_jepa_seg(
    weights_dir: Annotated[
        Path,
        typer.Option(
            help="The directory containing the pretrained weights and config.yaml"
        ),
    ],
) -> None:
    train_ms_jepa_seg(weights_dir=weights_dir)
