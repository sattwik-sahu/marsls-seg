from typing import Any, override

import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import v2
from transformers import get_cosine_schedule_with_warmup

from marsls_seg.helpers.mask import MaskingRatioScheduler, generate_uniform_mask
from marsls_seg.utils.data.mmls import MultimodalMartianLandslideSample
from marsls_seg.utils.modules.encoder.patch_masking_vit import (
    PatchMaskingViTInput as ViTInput,
)
from marsls_seg.utils.modules.ijepa._typing import IJEPAOutput
from marsls_seg.utils.modules.ijepa.jepa import IJEPA, IJEPALoss
from marsls_seg.utils.modules.jepa.base import JEPAOutput
from marsls_seg.utils.train.base import BaseTrainer
from wandb import Run as WandbRun


class IJEPATrainer(BaseTrainer[IJEPA, MultimodalMartianLandslideSample, dict]):
    """
    IJEPA style trainer for ViT on fat images from MMLSv2 dataset
    with advanced interactive latent space diagnostics.
    """

    def __init__(
        self,
        model: IJEPA,
        mask_ratio: tuple[float, float],
        lr: float,
        n_epochs: int,
        n_warmup_epochs: int,
        wandb: WandbRun,
        device: torch.device,
    ) -> None:
        super().__init__(
            model=model,
            n_epochs=n_epochs,
            lr=lr,
            wandb=wandb,
            device=device,
        )

        self._mask_ratio_scheduler: MaskingRatioScheduler = MaskingRatioScheduler(
            start=mask_ratio[0], end=mask_ratio[1], T=n_epochs
        )

        # Create a cosine annealing with linear warmup lr scheduler
        self._lr_scheduler: torch.optim.lr_scheduler.LRScheduler = (
            get_cosine_schedule_with_warmup(
                optimizer=self._optimizer,
                num_warmup_steps=n_warmup_epochs,
                num_training_steps=self._n_epochs,
            )
        )

        # Create augmentation
        self._augmentation: v2.Compose = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=[-90, 90]),
                v2.RandomResizedCrop(
                    size=(128, 128), antialias=True, scale=(0.333, 1.00)
                ),
                v2.GaussianNoise(),
            ]
        )

    def _create_x_y_z(
        self,
        batch: MultimodalMartianLandslideSample,
        ids_keep: torch.Tensor,
        ids_drop: torch.Tensor,
        device: torch.device,
        pos_emb: torch.Tensor,
    ) -> tuple[ViTInput, ViTInput, torch.Tensor]:
        image = batch.merge_channels()

        # Apply augmentation while training
        if self._model.training:
            image = self._augmentation(image)

        x = ViTInput(image=image, mask=ids_keep).to(device=device)
        y = ViTInput(image=image).to(device=device)
        z = pos_emb[ids_drop].unsqueeze(0).repeat(batch.batch_size[0], 1, 1)
        return x, y, z

    def _generate_svd_chart(self, target_encodings: torch.Tensor) -> go.Figure | None:
        """Computes singular values of target encodings and returns an interactive Plotly figure."""
        # Flatten batch and patch sequences to matrix rows: (B * N_patches, Feature_Dim)
        flat_features = (
            target_encodings.reshape(-1, target_encodings.shape[-1]).detach().float()
        )

        # Center the feature vectors
        flat_features = flat_features - flat_features.mean(dim=0, keepdim=True)

        try:
            # Perform singular value decomposition on the GPU
            singular_values = torch.linalg.svdvals(flat_features)
            # Normalize against peak singular value for absolute comparison scaling
            normalized_sv = (
                (singular_values / (singular_values[0] + 1e-8)).cpu().numpy()
            )
        except RuntimeError:
            # Failsafe hook if matrix decomposition conditions break on unstable gradient step
            return None

        # Build native Plotly trace
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(normalized_sv))),
                y=normalized_sv,
                mode="lines+markers",
                name="Singular Value Decay",
                line=dict(color="indigo", width=2.5),
                marker=dict(size=4),
            )
        )

        fig.update_layout(
            title="Latent Space Singular Value Distribution (SIGReg Health Check)",
            xaxis_title="Dimension Feature Index",
            yaxis_title="Normalized Singular Value Magnitude",
            template="plotly_white",
            hovermode="x unified",
        )
        return fig

    @override
    def _create_optimizer(self, model: IJEPA) -> Optimizer:
        return torch.optim.AdamW(params=model.parameters(), lr=self._lr)

    @override
    def train_epoch(
        self,
        dataloader: DataLoader[MultimodalMartianLandslideSample],
        epoch: int,
    ) -> None:
        batch: MultimodalMartianLandslideSample
        self._model.train()

        for batch in dataloader:
            ids_keep, ids_drop = generate_uniform_mask(
                mask_ratio=self._mask_ratio_scheduler.value,
                n_patches=self._model.context_encoder.n_patches,
            )
            x, y, z = self._create_x_y_z(
                batch=batch,
                ids_keep=ids_keep,
                ids_drop=ids_drop,
                device=self._device,
                pos_emb=self._model.context_encoder.pos_emb,
            )

            self._optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):  # type: ignore
                jepa_output: IJEPAOutput = self._model(x=x, y=y, z=z)
                jepa_output.loss.total.backward()

            self._optimizer.step()

            # Standard scalar telemetry logging
            self._wandb.log(
                {
                    "train/loss/total": jepa_output.loss.total.item(),
                    "train/loss/sigreg": jepa_output.loss.sigreg.item(),
                    "train/loss/pred": jepa_output.loss.pred.item(),
                    "train/mask_ratio": self._mask_ratio_scheduler.value,
                    "train/epoch": epoch,
                    "train/lr": self._lr_scheduler.get_last_lr()[0],
                }
            )

        # Update the masking ratio after every epoch
        self._mask_ratio_scheduler.step()

        # Update the learning rate
        self._lr_scheduler.step()

        # Return some stuff to log
        return dict(
            loss=jepa_output.loss.total.item(),  # type: ignore
            loss_sigreg=jepa_output.loss.sigreg.item(),  # type: ignore
            loss_pred=jepa_output.loss.pred.item(),  # type: ignore
            lr=self._lr_scheduler.get_last_lr()[0],
        )

    @override
    def evaluate(
        self,
        dataloader: DataLoader[MultimodalMartianLandslideSample],
    ) -> dict[str, int | float | str | list]:
        self._model.eval()

        losses: list[IJEPALoss] = []
        cos_sims: list[float] = []
        latent_stds: list[float] = []
        latent_pred_stds: list[float] = []
        target_encodings: list[torch.Tensor] = []

        # batch: MultimodalMartianLandslideSample = next(iter(dataloader))
        for batch in dataloader:
            ids_keep, ids_drop = generate_uniform_mask(
                mask_ratio=self._mask_ratio_scheduler.value,
                n_patches=self._model.context_encoder.n_patches,
            )
            x, y, z = self._create_x_y_z(
                batch=batch,
                ids_keep=ids_keep,
                ids_drop=ids_drop,
                device=self._device,
                pos_emb=self._model.context_encoder.pos_emb,
            )

            with torch.no_grad():
                jepa_output: JEPAOutput[torch.Tensor, IJEPALoss] = self._model(
                    x=x, y=y, z=z
                )

                losses.append(jepa_output.loss.unsqueeze(0))  # type: ignore
                target_encodings.append(jepa_output.target_encoding.detach().cpu())

                # --- 1. Compute Prediction Alignment Metric (Cosine Similarity) ---
                pred = jepa_output.prediction  # Shape: (B, num_drop, dim)

                # Use the sorted ids_drop to select the exact ground-truth patches
                target = jepa_output.target_encoding[
                    :, ids_drop
                ]  # Shape: (B, num_drop, dim)

                # Both match in sequence order and point to identical patch indices!
                cos_sim = F.cosine_similarity(pred, target, dim=-1).mean().item()
                cos_sims.append(cos_sim)

                # --- 2. Compute Latent Vector Column-Wise Variance ---
                flat_target = jepa_output.target_encoding.reshape(
                    -1, jepa_output.target_encoding.shape[-1]
                )
                latent_std = flat_target.std(dim=0).mean().item()
                latent_stds.append(latent_std)

                flat_pred = jepa_output.prediction.reshape(
                    -1, jepa_output.prediction.shape[-1]
                )
                latent_pred_std = flat_pred.std(dim=0).mean().item()
                latent_pred_stds.append(latent_pred_std)

        # Build the interactive validation chart
        svd_plotly_fig = self._generate_svd_chart(torch.cat(target_encodings, dim=0))

        # Build log payload dictionary
        mean_loss: IJEPALoss = torch.cat(losses).mean()  # type: ignore

        metrics_payload: dict[str, Any] = {
            "val/loss/total": mean_loss.total.item(),  # type: ignore
            "val/loss/sigreg": mean_loss.sigreg.item(),  # type: ignore
            "val/loss/pred": mean_loss.pred.item(),  # type: ignore
            "val/diagnostics/prediction_cosine_similarity": sum(cos_sims)
            / len(cos_sims),
            "val/diagnostics/latent_dimension_std": sum(latent_stds) / len(latent_stds),
            "val/diagnostics/latent_pred_std": sum(latent_pred_stds)
            / len(latent_pred_stds),
        }

        # Inject interactive figure if matrix conversion succeeded
        if svd_plotly_fig is not None:
            metrics_payload["val/plots/singular_value_distribution"] = svd_plotly_fig

        self._wandb.log(metrics_payload)

        return {
            "loss/pred": round(mean_loss.pred.item(), 4),
            "loss/sigreg": round(mean_loss.sigreg.item(), 4),
            "loss/total": round(mean_loss.total.item(), 4),
        }
