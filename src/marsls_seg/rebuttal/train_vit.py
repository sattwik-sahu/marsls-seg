# src/mars_jepa/train_vit.py
import os
from math import sqrt
from typing import Literal, override

import hydra
import lightning as L
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms.functional as F
from datasets.load import load_dataset
from einops import rearrange
from hydra.core.hydra_config import HydraConfig
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, LitLogger
from omegaconf import DictConfig, OmegaConf
from segmentation_models_pytorch.encoders._base import EncoderMixin
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    F1Score,
    JaccardIndex,
    Precision,
    Recall,
)
from torchvision.transforms.v2 import Normalize
from vit_pytorch import ViT as VisionTransformer

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Normalization stats live at module level (not inside the Dataset class) so
# that anything that needs them (e.g. the LightningModule, for de-normalizing
# images before logging) can grab them without instantiating - and therefore
# re-downloading/re-loading - the HF dataset.
MODALITIES: tuple[str, ...] = (
    "rgb",
    "grayscale",
    "dem",
    "slope",
    "thermal_inertial",
)

NORM_STATS: dict[str, dict[str, list[float]]] = {
    "dem": {"mean": [0.5251417756080627], "std": [0.24374613165855408]},
    "grayscale": {"mean": [0.4956520199775696], "std": [0.0815693736076355]},
    "rgb": {
        "mean": [0.5196613073348999, 0.33910998702049255, 0.29424601793289185],
        "std": [0.23257732391357422, 0.2374347299337387, 0.23585809767246246],
    },
    "slope": {"mean": [0.19128184020519257], "std": [0.1708942949771881]},
    "thermal_inertial": {"mean": [0.4918023943901062], "std": [0.1527264565229416]},
}


class MMLSv2HFDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        repo_id: str = "sattwik21/mmls-v2",
        split: Literal["train", "val", "test"] = "train",
        cache_dir: str | None = None,
    ) -> None:
        # On Lightning AI Studios, pass a cache_dir that lives on persistent
        # storage (e.g. /teamspace/studios/this_studio/.hf_cache) so the
        # dataset isn't re-downloaded every time the Studio restarts.
        self._data = load_dataset(
            repo_id, split=split, cache_dir=cache_dir
        ).with_format("torch")
        self._normalize: dict[str, Normalize] = {
            key: Normalize(mean=stats["mean"], std=stats["std"])
            for key, stats in NORM_STATS.items()
        }

    def _transform(
        self, x: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.cat(
            [self._normalize[key](x[key].float()) for key in MODALITIES],
            dim=0,
        )
        label = x["label"].float()
        return image, label

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item: dict[str, torch.Tensor] = self._data[index]
        return self._transform(item)


# ---------------------------------------------------------------------------
# ViT -> SMP encoder adapter
# ---------------------------------------------------------------------------


class ViTEncoder(torch.nn.Module, EncoderMixin):
    """Wraps a vit_pytorch ViT so it can act as an encoder for segmentation_models_pytorch."""

    def __init__(self, vit: VisionTransformer, dim: int, **kwargs) -> None:
        super().__init__()

        self._out_channels: list[int] = [7, 7, 7, dim, dim, dim]
        self._in_channels: int = 7
        self._depth: int = 5
        self._output_stride: int = 16
        self._is_dilated: bool = False

        self._vit: VisionTransformer = vit
        self._vit_output_patch_tokens: torch.Tensor = torch.empty(0)
        self._register_vit_hook()
        self._skip_dropout = torch.nn.Dropout2d(p=0.1)

    def make_dilated(self, output_stride) -> None:
        self._is_dilated = True

    def _register_vit_hook(self) -> None:
        def _hook(
            module: torch.nn.Module, arg: torch.Tensor, output: torch.Tensor
        ) -> None:
            self._vit_output_patch_tokens = output[:, 1:]

        self._vit.transformer.register_forward_hook(_hook)  # type: ignore

    def _convert_patch_tokens_to_feature_map(
        self, patch_tokens: torch.Tensor
    ) -> torch.Tensor:
        b, n, d = patch_tokens.shape
        h = w = int(sqrt(n))
        return rearrange(tensor=patch_tokens, pattern="b (h w) d -> b d h w", h=h, w=w)

    @override
    def set_in_channels(self, in_channels: int, pretrained: bool = True) -> None:
        self._in_channels = in_channels

    @override
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        self._vit(x)

        patch_tokens: torch.Tensor = self._vit_output_patch_tokens
        feature_map: torch.Tensor = self._convert_patch_tokens_to_feature_map(
            patch_tokens=patch_tokens
        )

        f0 = self._skip_dropout(x)
        f1 = self._skip_dropout(
            torch.nn.functional.max_pool2d(f0, kernel_size=2, stride=2)
        )
        f2 = self._skip_dropout(
            torch.nn.functional.max_pool2d(f1, kernel_size=2, stride=2)
        )
        f3 = feature_map
        f4 = torch.nn.functional.max_pool2d(f3, kernel_size=2, stride=2)
        f5 = (
            f4
            if self._is_dilated
            else torch.nn.functional.max_pool2d(f4, kernel_size=2, stride=2)
        )

        return [f0, f1, f2, f3, f4, f5]


def register_vit_encoder(vit: VisionTransformer, dim: int) -> None:
    smp.encoders.encoders["vanilla_vit"] = {
        "encoder": ViTEncoder,
        "params": dict(vit=vit, dim=dim),
        "pretrained_settings": {},
    }  # type: ignore


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------


class ViTSegmentationModel(L.LightningModule):
    def __init__(
        self,
        vit: VisionTransformer,
        dim: int,
        decoder_arch: str,
        lr: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["vit"])

        register_vit_encoder(vit=vit, dim=dim)
        self._model: torch.nn.Module = smp.create_model(
            arch=decoder_arch,
            encoder_name="vanilla_vit",
            in_channels=7,
            classes=1,
            encoder_weights=None,
        )
        self._lr = lr
        self._weight_decay = weight_decay

        self._bce_loss = torch.nn.BCEWithLogitsLoss()
        self._dice_loss = smp.losses.DiceLoss(
            mode=smp.losses.BINARY_MODE, from_logits=True
        )

        # Metrics
        self._train_metrics = MetricCollection(
            metrics={
                "miou": JaccardIndex(task="binary"),
                "precision": Precision(task="binary"),
                "recall": Recall(task="binary"),
                "f1_score": F1Score(task="binary"),
            },
            prefix="train/",
        )
        self._val_metrics = self._train_metrics.clone(prefix="val/")

    @override
    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        image, label = batch
        logits = self._model(image).squeeze(1)
        bce_loss = self._bce_loss(logits, label)
        dice_loss = self._dice_loss(logits, label)
        loss = bce_loss + dice_loss

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/bce_loss", bce_loss)
        self.log("train/dice_loss", dice_loss)

        batch_metrics = self._train_metrics(logits, label)
        self.log_dict(batch_metrics)
        return loss

    @override
    def on_train_epoch_end(self) -> None:
        self._train_metrics.reset()

    @override
    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        image, label = batch
        logits = self._model(image).squeeze(1)

        bce_loss = self._bce_loss(logits, label)
        dice_loss = self._dice_loss(logits, label)
        loss = bce_loss + dice_loss

        pred = (torch.sigmoid(logits) > 0.5).float()
        intersection = (pred * label).sum(dim=(1, 2))
        union = pred.sum(dim=(1, 2)) + label.sum(dim=(1, 2)) - intersection
        miou = ((intersection + 1e-6) / (union + 1e-6)).mean()

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/bce_loss", bce_loss, sync_dist=True)
        self.log("val/dice_loss", dice_loss, sync_dist=True)
        self.log("val/mIoU", miou, prog_bar=True, sync_dist=True)

        self.log_dict(self._val_metrics(logits, label))

        if batch_idx == 0:
            self._log_sample_images(image, label, pred)

        return loss

    @override
    def on_validation_epoch_end(self) -> None:
        self._val_metrics.reset()

    def _log_sample_images(
        self, image: torch.Tensor, label: torch.Tensor, pred: torch.Tensor
    ) -> None:
        """Log one RGB/GT/pred/overlay sample. Only works with TensorBoardLogger
        (which exposes .experiment.add_image); silently skipped otherwise."""
        experiment = getattr(self.logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "add_image"):
            return

        rgb_std = torch.tensor(NORM_STATS["rgb"]["std"]).view(3, 1, 1)
        rgb_mean = torch.tensor(NORM_STATS["rgb"]["mean"]).view(3, 1, 1)

        rgb_norm = image[0, :3].detach().cpu()
        rgb_denorm = torch.clamp(rgb_norm * rgb_std + rgb_mean, 0, 1)

        label_single = label[0].detach().cpu()
        pred_single = pred[0].detach().cpu()

        step = self.global_step
        experiment.add_image("val/sample_rgb", rgb_denorm, step)
        experiment.add_image("val/sample_ground_truth", label_single.unsqueeze(0), step)
        experiment.add_image("val/sample_prediction", pred_single.unsqueeze(0), step)

        rgb_np = (rgb_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        label_np = label_single.numpy()
        pred_np = pred_single.numpy()

        overlay = rgb_np.copy()
        overlay[pred_np > 0.5, 0] = np.clip(
            overlay[pred_np > 0.5, 0].astype(int) + 50, 0, 255
        )
        overlay[pred_np > 0.5, 1] = np.clip(
            overlay[pred_np > 0.5, 1].astype(int) - 50, 0, 255
        )
        overlay[label_np > 0.5, 1] = np.clip(
            overlay[label_np > 0.5, 1].astype(int) + 50, 0, 255
        )
        overlay[label_np > 0.5, 0] = np.clip(
            overlay[label_np > 0.5, 0].astype(int) - 50, 0, 255
        )

        overlay_tensor = torch.from_numpy(overlay).permute(2, 0, 1)
        experiment.add_image("val/sample_overlay", overlay_tensor, step)

    @override
    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.parameters(), lr=self._lr, weight_decay=self._weight_decay
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="config", config_name="config_vit")
def main(cfg: DictConfig) -> None:
    # L40S / Ampere+ benefit from TF32 matmuls for the non-autocast ops.
    torch.set_float32_matmul_precision("high")

    L.seed_everything(cfg.seed, workers=True)

    train_dataset = MMLSv2HFDataset(split="train", cache_dir=cfg.hf_cache_dir)
    val_dataset = MMLSv2HFDataset(split="test", cache_dir=cfg.hf_cache_dir)

    # When running under the joblib launcher, each hydra job already lives
    # inside a loky subprocess. Torch's DataLoader workers must use the
    # "fork" start method explicitly here, or spawning them tries to pickle
    # worker args across the nested loky/multiprocessing boundary and fails
    # with a confusing "Process object has no attribute 'env'" error.
    mp_context = "fork" if cfg.num_workers > 0 else None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        multiprocessing_context=mp_context,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        multiprocessing_context=mp_context,
        drop_last=False,
    )

    model = hydra.utils.instantiate(cfg.model)

    # Each hydra (multi)run gets its own hydra-generated output dir; reuse it
    # so every decoder_arch sweep leg writes to its own place automatically.
    run_dir = HydraConfig.get().runtime.output_dir
    run_name = f"vit_{cfg.model.decoder_arch}"

    loggers = [
        # Hosted dashboard at lightning.ai: metric charts, hyperparameter
        # comparison across the decoder_arch sweep, optional checkpoint
        # upload. Auto-detects credentials when run inside a Studio.
        LitLogger(
            root_dir=run_dir,
            name=run_name,
            teamspace=cfg.get("teamspace", None),
            metadata=OmegaConf.to_container(cfg),  # type: ignore
            log_model=True,
            save_logs=True,
        ),
        # Kept for the sample RGB/GT/pred/overlay images logged in
        # validation_step, which use TensorBoard's add_image API directly.
        # TensorBoardLogger(save_dir=run_dir, name="tb", version=run_name),
        CSVLogger(save_dir=run_dir, name="csv", version=run_name),
    ]

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.ckpt_dir, run_name),
        every_n_epochs=cfg.ckpt_interval,
        save_top_k=1,
        monitor="val/mIoU",
        mode="max",
    )

    trainer = Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        logger=loggers,
        callbacks=[checkpoint_callback],
        max_epochs=cfg.n_epochs,
        accumulate_grad_batches=cfg.n_grad_accum_batches,
        log_every_n_steps=1,
        default_root_dir=run_dir,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
