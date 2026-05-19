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
from marsls_seg.utils.modules.segmentation.model import SegmentationModelWithPretainedEncoder

def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> tuple:
    if preds.shape[2:] != labels.shape[2:]:
        preds = F.interpolate(preds, size=labels.shape[2:], mode="bilinear", align_corners=False)

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
    BaseTrainer[SegmentationModelWithPretainedEncoder,
                MultimodalMartianLandslideSample,
                dict[str, float | int | str | list]
                ]
):
    """
    Implementation of Base Trainer for Segmentation Head training.
    """
    def __init__(self,wandb: WandbRun)->None:
        super().__init__(wandb=wandb)

        self._diceloss = smp.losses.DiceLoss(mode="binary")
        self._bceloss = smp.losses.BCEloss








