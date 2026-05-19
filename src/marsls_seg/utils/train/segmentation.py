import pandas as pd
from pathlib import Path
from typing import Any, override

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from wandb import wandb as WandbRun
from tqdm import tqdm
import segmentation_models_pytorch as smp

from marsls_seg.helpers.device import DEVICE
from marsls_seg.utils.data.mmls import MultimodalMartianLandslideDataset
from marsls_seg.utils.data._typing import MultimodalMartianLandslideSample
from marsls_seg.utils.data.processing import ProcessedMMLSv2Dataset

from marsls_seg.utils.train.base import BaseTrainer
from marsls_seg.utils.modules.segmentation.model import (
    SegmentationModelWithPretainedEncoder,
)


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

    def __init__(self, wandb: WandbRun) -> None:
        super().__init__(wandb=wandb)

        self._diceloss = smp.losses.DiceLoss(mode="binary")
        self._bceloss = torch.nn.BCEWithLogitsLoss(reduction="none")

    @override
    def train_epoch(
        self,
        model: SegmentationModelWithPretainedEncoder,
        dataloader: torch.utils.data.DataLoader[MultimodalMartianLandslideSample],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
    ) -> dict[str, float | int | str | list]:

        model.train()

        train_loss = 0.0
        train_dice_loss = 0.0
        train_bce_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)

            image_size = batch.label.shape[-2:]
            label = batch.label.unsqueeze(1).to(torch.float32)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(batch)

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
            optimizer.step()

            train_loss += loss.item()
            train_dice_loss += l_dice.item()
            train_bce_loss += l_bce.item()

            
            self._wandb.log({
                "train/batch_total_loss": loss.item(),
                "train/batch_dice_loss": l_dice.item(),
                "train/batch_bce_loss": l_bce.item(),
            })

            

        epoch_total_loss = train_loss / len(dataloader)
        epoch_dice_loss = train_dice_loss / len(dataloader)
        epoch_bce_loss = train_bce_loss / len(dataloader)

        
        self._wandb.log({
            "train/epoch_total_loss": epoch_total_loss,
            "train/epoch_dice_loss": epoch_dice_loss,
            "train/epoch_bce_loss": epoch_bce_loss,
            "epoch": epoch,
        })

        

        return {
            "epoch_loss": float(epoch_total_loss),
            "epoch_dice_loss": float(epoch_dice_loss),
            "epoch_bce_loss": float(epoch_bce_loss),
        }
    @override
    def evaluate(
        self,
        model: SegmentationModelWithPretainedEncoder,
        dataloader: torch.utils.data.DataLoader[MultimodalMartianLandslideSample],
        device: torch.device,
    ) -> dict[str, float | int | str | list]:

        model.eval()
        val_loss = 0.0

        all_metrics = []

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                label = batch.label.unsqueeze(1).to(torch.float32)

                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits = model(batch)

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

        return {
            "epoch_loss": float(val_loss / len(dataloader)),
            "precision": float(avg_metrics[0]),
            "recall": float(avg_metrics[1]),
            "f1_score": float(avg_metrics[2]),
            "iou_bg": float(avg_metrics[3]),
            "iou_fg": float(avg_metrics[4]),
            "miou": float(avg_metrics[5]),
        }
