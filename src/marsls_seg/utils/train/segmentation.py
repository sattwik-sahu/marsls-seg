from typing import override

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision.transforms import InterpolationMode, v2
from transformers import get_cosine_schedule_with_warmup

import wandb
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
                v2.RandomResizedCrop(
                    size=(128, 128),
                    scale=(0.333, 1.00),
                    interpolation=InterpolationMode.NEAREST,
                ),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                # Use 'Nearest' for everything to ensure labels never get corrupted
                v2.RandomRotation(
                    degrees=(-90, 90), interpolation=InterpolationMode.NEAREST
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

    def _visualize_and_log(
        self,
        batch: MultimodalMartianLandslideSample,
        logits: torch.Tensor,
        epoch: int,
        num_samples: int = 5,
    ) -> None:
        """
        Creates a high-res grid of multispectral inputs, GT, and Preds for WandB/Paper.
        """
        # Ensure we don't try to plot more than what's in the batch
        num_samples = min(num_samples, batch.batch_size[0])
        indices = torch.arange(num_samples)

        # Move to CPU and float32 for plotting
        # images = image[indices].detach().cpu().float().numpy()
        labels = batch.label[indices].detach().cpu().float().numpy()
        preds = (torch.sigmoid(logits[indices]) > 0.5).detach().cpu().float().numpy()

        cols = 7  # RGB, Gray, DEM, Slope, Thermal, GT, Prediction
        rows = num_samples
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), dpi=100)

        # Titles for the columns (Placeholders for your specific names)
        titles = [
            "RGB",
            "Grayscale",
            "DEM",
            "Slope",
            "Thermal-Inertial",
            "Ground Truth",
            "Prediction",
        ]

        for r in range(rows):
            # Extract channels based on your provided slices
            # img_rgb = images[r, 0:3].transpose(1, 2, 0)
            # img_dem = images[r, 3:4].squeeze()
            # img_slope = images[r, 4:5].squeeze()
            # img_thermal = images[r, 5:6].squeeze()
            # img_gray = images[r, 6:7].squeeze()
            img_rgb = batch.rgb[r].detach().cpu().permute(1, 2, 0).numpy()
            img_dem = batch.dem[r].detach().cpu().squeeze()
            img_slope = batch.slope[r].detach().cpu().squeeze()
            img_thermal = batch.thermal_inertial[r].detach().cpu().squeeze()
            img_gray = batch.grayscale[r].detach().cpu().squeeze()
            gt = labels[r].squeeze()
            pred = preds[r].squeeze()

            imgs_to_plot = [
                img_rgb,
                img_gray,
                img_dem,
                img_slope,
                img_thermal,
                gt,
                pred,
            ]

            for c in range(cols):
                ax = axes[r, c]
                curr_img = imgs_to_plot[c]

                # Normalize 1-channel images for better visibility in paper
                if c > 0:  # Gray, DEM, Slope, Thermal, GT, Pred
                    if c < 5:  # Scientific channels: use 'viridis' or 'magma'
                        cmap = "magma"
                        # Min-Max scaling for visualization contrast
                        curr_img = (curr_img - curr_img.min()) / (
                            curr_img.max() - curr_img.min() + 1e-7
                        )
                    else:  # Binary masks
                        cmap = "gray"
                    ax.imshow(curr_img, cmap=cmap)
                else:
                    # RGB normalization
                    curr_img = (curr_img - curr_img.min()) / (
                        curr_img.max() - curr_img.min() + 1e-7
                    )
                    ax.imshow(curr_img)

                if r == 0:
                    ax.set_title(titles[c], fontsize=20, fontweight="bold")

                ax.axis("off")

        plt.tight_layout()

        # Log to WandB
        self._wandb.log(
            {"visuals/segmentation": wandb.Image(fig), "train/epoch": epoch}
        )

        # Optional: Save locally for paper use
        # plt.savefig(f"mars_jepa_eval_epoch_{epoch}.png", bbox_inches='tight', dpi=300)

        plt.close(fig)

    @override
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader[MultimodalMartianLandslideSample],
    ) -> dict[str, float | int | str | list]:

        self._model.eval()
        val_loss = 0.0
        all_metrics = []

        # Track one batch for visualization
        viz_data = None

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
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

                # Store the first batch of the validation set for visualization
                # if i == 0:
                #     viz_data = (batch, logits)

        # Trigger Visualization
        # if viz_data is not None:
        #     # We use a placeholder for epoch, you might want to pass it into evaluate()
        #     current_epoch = self._lr_scheduler.last_epoch
        #     self._visualize_and_log(*viz_data, epoch=current_epoch)

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
