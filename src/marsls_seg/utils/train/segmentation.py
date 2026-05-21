from typing import override

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import InterpolationMode, v2
from transformers import get_cosine_schedule_with_warmup

from marsls_seg.utils.data._typing import MultimodalMartianLandslideSample
from marsls_seg.utils.modules.segmentation.model import (
    SegmentationModelWithPretainedEncoder,
)
from marsls_seg.utils.train.base import BaseTrainer
from wandb import Run as WandbRun


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> tuple:
    if preds.shape[2:] != labels.shape[2:]:
        preds = F.interpolate(
            preds, size=labels.shape[2:], mode="bilinear", align_corners=False
        )

    preds = (torch.sigmoid(preds) > 0.5).float()
    labels = labels.float()

    TP = (preds * labels).sum().item()
    FP = (preds * (1 - labels)).sum().item()
    FN = ((1 - preds) * labels).sum().item()
    TN = ((1 - preds) * (1 - labels)).sum().item()

    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    iou_fg = TP / (TP + FP + FN + 1e-7)
    iou_bg = TN / (TN + FP + FN + 1e-7)
    miou = (iou_fg + iou_bg) / 2

    return precision, recall, f1, iou_bg, iou_fg, miou


class SegmentationTrainer(
    BaseTrainer[
        SegmentationModelWithPretainedEncoder,
        MultimodalMartianLandslideSample,
        dict[str, float | int | str | list],
    ]
):
    """
    Implementation of Base Trainer for Segmentation Head training.
    """

    def __init__(
        self,
        model: SegmentationModelWithPretainedEncoder,
        n_epochs: int,
        n_warmup_epochs: int,
        lr: float,
        apply_aug: bool,
        device: torch.device,
        wandb: WandbRun,
    ) -> None:
        super().__init__(
            model=model, n_epochs=n_epochs, lr=lr, device=device, wandb=wandb
        )

        # Save the encoder dim of the model
        self._wandb.config.update(dict(encoder_dim=self._model.encoder_dim))

        self._apply_aug: bool = apply_aug

        self._diceloss = smp.losses.DiceLoss(mode="binary")
        self._bceloss = torch.nn.BCEWithLogitsLoss(reduction="none")

        self._max_miou: float = 0.0

        # Create LR scheduler
        self._lr_scheduler: torch.optim.lr_scheduler.LRScheduler = (
            get_cosine_schedule_with_warmup(
                optimizer=self._optimizer,
                num_warmup_steps=n_warmup_epochs,
                num_training_steps=self._n_epochs,
            )
        )

        # Create augmentations
        self._augmentation: v2.Compose = self._create_augmentation()

    def _create_augmentation(self) -> v2.Compose:
        return v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                # Use 'Nearest' for everything to ensure labels never get corrupted
                v2.RandomRotation(
                    degrees=(90, 90), interpolation=InterpolationMode.NEAREST
                ),
                v2.Identity(),  # Does nothing, used for debugging
            ]
        )

    def _apply_augmentation(
        self, image: torch.Tensor, label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        aug_image, aug_label = self._augmentation(image, label)
        return aug_image, aug_label

    @override
    def _create_optimizer(
        self, model: SegmentationModelWithPretainedEncoder
    ) -> optim.Optimizer:
        return torch.optim.AdamW(params=model.parameters(), lr=self._lr)

    @override
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader[MultimodalMartianLandslideSample],
        epoch: int,
    ) -> None:

        self._model.train()

        train_loss = 0.0
        train_dice_loss = 0.0
        train_bce_loss = 0.0

        batch: MultimodalMartianLandslideSample
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(self._device)

            image_size = batch.label.shape[-2:]
            image: torch.Tensor = batch.merge_channels()
            label: torch.Tensor = batch.label.unsqueeze(1)

            if self._apply_aug:
                image, label = self._apply_augmentation(image=image, label=label)

            self._optimizer.zero_grad()

            with torch.amp.autocast(  # type: ignore
                device_type=self._device.type, dtype=torch.bfloat16
            ):
                logits = self._model(image)

                if logits.shape[-2:] != label.shape[-2:]:
                    logits = F.interpolate(
                        logits,
                        size=image_size,
                        mode="bilinear",
                        align_corners=False,
                    )

                l_dice = self._diceloss(logits, label)
                l_bce = self._bceloss(logits, label).mean()

                loss = l_dice + l_bce

            loss.backward()
            self._optimizer.step()

            train_loss += loss.item()
            train_dice_loss += l_dice.item()
            train_bce_loss += l_bce.item()

            self._wandb.log(
                {
                    "train/batch_total_loss": loss.item(),
                    "train/batch_dice_loss": l_dice.item(),
                    "train/batch_bce_loss": l_bce.item(),
                }
            )

        # Update the learning rate after the epoch
        self._lr_scheduler.step()

        epoch_total_loss = train_loss / len(dataloader)
        epoch_dice_loss = train_dice_loss / len(dataloader)
        epoch_bce_loss = train_bce_loss / len(dataloader)

        self._wandb.log(
            {
                "train/epoch_total_loss": epoch_total_loss,
                "train/epoch_dice_loss": epoch_dice_loss,
                "train/epoch_bce_loss": epoch_bce_loss,
                "epoch": epoch,
            }
        )

        # return {
        #     "epoch_loss": float(epoch_total_loss),
        #     "epoch_dice_loss": float(epoch_dice_loss),
        #     "epoch_bce_loss": float(epoch_bce_loss),
        # }

    @override
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader[MultimodalMartianLandslideSample],
    ) -> dict[str, float | int | str | list]:

        self._model.eval()
        val_loss = 0.0

        all_metrics = []

        with torch.no_grad():
            batch: MultimodalMartianLandslideSample
            for batch in dataloader:
                batch = batch.to(self._device)

                image: torch.Tensor = batch.merge_channels()
                label: torch.Tensor = batch.label.unsqueeze(1).to(torch.float32)

                with torch.amp.autocast(  # type: ignore
                    device_type=self._device.type, dtype=torch.bfloat16
                ):
                    logits = self._model(image)

                    if logits.shape[2:] != label.shape[2:]:
                        logits = F.interpolate(
                            logits,
                            size=label.shape[2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                    l_dice = self._diceloss(logits, label)
                    l_bce = self._bceloss(logits, label).mean()
                    loss = l_dice + l_bce

                val_loss += loss.item()

                metrics = compute_metrics(logits, label)
                all_metrics.append(metrics)
        avg_metrics = tuple(
            sum(m[i] for m in all_metrics) / len(all_metrics)
            for i in range(len(all_metrics[0]))
        )

        miou: float = float(avg_metrics[5])
        self._max_miou = max(miou, self._max_miou)

        log = {
            "epoch_loss": float(val_loss / len(dataloader)),
            "precision": float(avg_metrics[0]),
            "recall": float(avg_metrics[1]),
            "f1_score": float(avg_metrics[2]),
            "iou_bg": float(avg_metrics[3]),
            "iou_fg": float(avg_metrics[4]),
            "miou": miou,
            "miou_max": self._max_miou,
        }

        self._wandb.log(log)

        return log  # type: ignore
