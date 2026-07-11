from math import sqrt
from typing import override

import hydra
import lightning as L
import segmentation_models_pytorch as smp
import torch
from datasets.load import load_dataset
from einops import rearrange
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from segmentation_models_pytorch.encoders._base import EncoderMixin
from torchvision.transforms.v2 import Normalize
from vit_pytorch import ViT as VisionTransformer

from marsls_seg.utils.data._typing import SplitName


class MMLSv2HFDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    _MODALITIES: tuple[str, ...] = (
        "rgb",
        "grayscale",
        "dem",
        "slope",
        "thermal_inertial",
    )

    def __init__(
        self, repo_id: str = "sattwik21/mmls-v2", split: SplitName = "train"
    ) -> None:
        self._data = load_dataset(repo_id, split=split).with_format("torch")
        self._normalize: dict[str, Normalize] = {
            "dem": Normalize(
                mean=[0.5251417756080627],
                std=[0.24374613165855408],
            ),
            "grayscale": Normalize(
                mean=[0.4956520199775696],
                std=[0.0815693736076355],
            ),
            "rgb": Normalize(
                mean=[
                    0.5196613073348999,
                    0.33910998702049255,
                    0.29424601793289185,
                ],
                std=[
                    0.23257732391357422,
                    0.2374347299337387,
                    0.23585809767246246,
                ],
            ),
            "slope": Normalize(
                mean=[0.19128184020519257],
                std=[0.1708942949771881],
            ),
            "thermal_inertial": Normalize(
                mean=[0.4918023943901062],
                std=[0.1527264565229416],
            ),
        }

    def _transform(
        self, x: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        label = x["label"]
        image = torch.cat(
            list(
                {key: self._normalize[key](x[key]) for key in self._MODALITIES}.values()
            ),
            dim=0,
        )

        return image, label

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item: dict[str, torch.Tensor] = self._data[index]
        return self._transform(item)


class ViTEncoder(torch.nn.Module, EncoderMixin):
    """A ViT encoder for Segmentation Models Pytorch."""

    def __init__(self, vit: VisionTransformer, dim: int) -> None:
        super().__init__()

        self._out_channels: list[int] = [7, 7, 7, dim, dim, dim]
        self._in_channels: int = 7
        self._depth: int = 5
        self._output_stride: int = 16
        self._is_dilated: bool = False

        self._vit: VisionTransformer = vit
        self._vit_output_patch_tokens: torch.Tensor = torch.empty(0)
        self._skip_dropout = torch.nn.Dropout2d(p=0.1)

    def make_dilated(self, output_stride) -> None:
        self._is_dilated = True

    def _register_vit_hook(self) -> None:
        @self._vit.transformer.register_forward_hook  # type: ignore
        def _(module: torch.nn.Module, arg: torch.Tensor, output: torch.Tensor) -> None:
            self._vit_output_patch_tokens = output[:, 1:]

    def _convert_patch_tokens_to_feature_map(
        self, patch_tokens: torch.Tensor
    ) -> torch.Tensor:
        b, n, d = patch_tokens.shape
        h = w = int(sqrt(n))
        return rearrange(tensor=patch_tokens, pattern="b (h w) d -> b d h w", h=h, w=w)

    @override
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Forward pass through ViT
        self._vit(x)

        # Get the patch tokens and convert them to feature maps
        patch_tokens: torch.Tensor = self._vit_output_patch_tokens
        feature_map: torch.Tensor = self._convert_patch_tokens_to_feature_map(
            patch_tokens=patch_tokens
        )

        # Create the output feature maps
        f0 = self._skip_dropout(x)
        f1 = self._skip_dropout(
            torch.nn.functional.max_pool2d(f0, kernel_size=2, stride=2)
        )
        f2 = self._skip_dropout(
            torch.nn.functional.max_pool2d(f1, kernel_size=2, stride=2)
        )
        f3 = feature_map
        f4 = torch.nn.functional.max_pool2d(f3, kernel_size=2, stride=2)

        if self.is_dilated:
            f5 = f4
        else:
            f5 = torch.nn.functional.max_pool2d(f4, kernel_size=2, stride=2)

        return [f0, f1, f2, f3, f4, f5]


def _register_vit_encoder(vit: VisionTransformer, dim: int) -> None:
    smp.encoders.encoders["vanilla_vit"] = {
        "encoder": ViTEncoder,
        "pretrained_settings": {
            "custom": {
                "mean": [0],
                "std": [1],
                "url": None,
                "repo_id": None,
                "input_space": "raw",
                "input_range": [0, 1],
            }
        },
        "params": {"vit": vit, "dim": dim},
    }  # type: ignore


class ViTSegmentationModel(L.LightningModule):
    """The lightning module for the ViT segmentation model."""

    def __init__(
        self,
        vit: VisionTransformer,
        dim: int,
        decoder_arch: str,
        lr: float,
        weight_decay: float,
    ) -> None:
        super().__init__()

        self._encoder: ViTEncoder = ViTEncoder(vit=vit, dim=dim)
        self._model: torch.nn.Module = smp.create_model(
            arch=decoder_arch, encoder_name="vanilla_vit", in_channels=7, classes=1
        )
        self._lr: float = lr
        self._weight_decay: float = weight_decay

    @override
    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx
    ) -> torch.Tensor:
        image, label = batch
        logits: torch.Tensor = self._model(image)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label)
        return loss

    @override
    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx
    ) -> None:
        image, label = batch
        logits: torch.Tensor = self._model(image)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label)
        self.log("val_loss", loss)

    @override
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(), lr=self._lr, weight_decay=self._weight_decay
        )
        return optimizer


@hydra.main(version_base=None, config_path="../", config_name="config")
def main(cfg: DictConfig) -> None:
    train_dataset: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]] = (
        MMLSv2HFDataset(split="train")
    )
    val_dataset: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]] = (
        MMLSv2HFDataset(split="val")
    )
    # test_dataset: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]] = (
    #     MMLSv2HFDataset(split="test")
    # )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    # test_loader = torch.utils.data.DataLoader(
    #     dataset=test_dataset,
    #     batch_size=cfg.batch_size,
    #     shuffle=True,
    #     num_workers=cfg.num_workers,
    # )

    model = hydra.utils.instantiate(cfg.model)
    checkpoint_callback: ModelCheckpoint = ModelCheckpoint(
        dirpath=cfg.ckpt_dir, every_n_epochs=cfg.ckpt_interval
    )
    logger: WandbLogger = WandbLogger(
        offline=cfg.wandb.mode == "offline", project=cfg.wandb.project
    )
    trainer: Trainer = Trainer(
        accelerator=cfg.accelerator,
        precision=cfg.precision,
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=cfg.n_epochs,
        accumulate_grad_batches=cfg.n_grad_accum_batches,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
