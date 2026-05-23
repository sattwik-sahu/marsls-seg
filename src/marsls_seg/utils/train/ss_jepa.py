from typing import Any, override

import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from transformers import get_cosine_schedule_with_warmup

from marsls_seg.helpers.mask import (
    MaskingRatioScheduler,
    construct_latent,
    generate_uniform_mask,
)
from marsls_seg.utils.data._typing import MultimodalMartianLandslideSample
from marsls_seg.utils.modules.encoder.spatio_spectral_vit import SSVitInput
from marsls_seg.utils.modules.ssjepa._typing import SSJEPAOutput
from marsls_seg.utils.modules.ssjepa.jepa import SpatioSpectralJEPA, SSJEPALoss
from marsls_seg.utils.train.base import BaseTrainer
from wandb import Run as WandbRun


class SSJEPATrainer(
    BaseTrainer[SpatioSpectralJEPA, MultimodalMartianLandslideSample, dict]
):
    """
    Trainer for spatio-spectral jepa, handles the Training of 7 channels.
    """

    def __init__(
        self,
        model: SpatioSpectralJEPA,
        mask_ratio: tuple[float, float],
        lr: float,
        n_epochs: int,
        n_warmup_epochs: int,
        wandb: WandbRun,
        device: torch.device,
    ) -> None:
        super().__init__(
            model=model, n_epochs=n_epochs, lr=lr, wandb=wandb, device=device
        )

        self._mask_ratio_scheduler = MaskingRatioScheduler(
            start=mask_ratio[0], end=mask_ratio[1], T=n_epochs
        )

        self._lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self._optimizer,
            num_warmup_steps=n_warmup_epochs,
            num_training_steps=self._n_epochs,
        )

        self._augmentation = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=[-90, 90]),
                v2.RandomResizedCrop(size=(128, 128), antialias=True),
            ]
        )

    def _create_x_y_z(
        self,
        batch: MultimodalMartianLandslideSample,
        ids_keep: torch.Tensor,
        ids_drop: torch.Tensor,
        device: torch.device,
    ) -> tuple[SSVitInput, SSVitInput, torch.Tensor]:
        """Creates the context(x), target(y) and Query (z) tensors"""
        image = batch.merge_channels()  # (B, 7, 128, 128)
        if self._model.training:
            image = self._augmentation(image)

        # Input Image + indices of set of visible patches
        x = SSVitInput(image=image, mask=ids_keep).to(device=device)

        # Full image for the target encoder
        y = SSVitInput(image=image).to(device=device)

        # This is our query
        z_full = self._model.context_encoder.get_full_pos_embed

        z = construct_latent(
            z_full=z_full, ids=ids_drop, batch_size=batch.batch_size[0]
        )

        return x, y, z

    @override
    def _create_optimizer(self, model: SpatioSpectralJEPA) -> torch.optim.Optimizer:
        return torch.optim.AdamW(params=model.parameters(), lr=self._lr)

    @override
    def train_epoch(
        self,
        dataloader: DataLoader[MultimodalMartianLandslideSample],
        epoch: int,
    ) -> None:
        self._model.train()

        losses: list[SSJEPALoss] = []

        for batch in dataloader:
            ids_keep, ids_drop = generate_uniform_mask(
                mask_ratio=self._mask_ratio_scheduler.value,
                n_patches=self._model.context_encoder.total_tokens,
                device=self._device,
            )

            x, y, z = self._create_x_y_z(batch, ids_keep, ids_drop, self._device)

            self._optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):  # type: ignore
                output: SSJEPAOutput = self._model(x=x, y=y, z=z)
                losses.append(output.loss)
                output.loss.total.backward()

            self._optimizer.step()
            self._wandb.log(
                {
                    "train/loss/total": output.loss.total.item(),
                    "train/loss/pred": output.loss.pred.item(),
                    "train/loss/sigreg": output.loss.sigreg.item(),
                    "train/mask_ratio": self._mask_ratio_scheduler.value,
                    "train/lr": self._lr_scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                }
            )

        self._mask_ratio_scheduler.step()
        self._lr_scheduler.step()

    def _generate_svd_chart(self, target_encodings: torch.Tensor) -> go.Figure | None:
        """SVD Health Check for the 1792 tokens."""
        flat_features = (
            target_encodings.reshape(-1, target_encodings.shape[-1]).detach().float()
        )
        flat_features = flat_features - flat_features.mean(dim=0, keepdim=True)
        try:
            singular_values = torch.linalg.svdvals(flat_features)
            normalized_sv = (
                (singular_values / (singular_values[0] + 1e-8)).cpu().numpy()
            )
        except RuntimeError:
            return None

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(y=normalized_sv, mode="lines+markers", name="SVD Decay")
        )
        fig.update_layout(
            title="Spatio-Spectral Latent Health", template="plotly_white"
        )
        return fig

    @override
    def evaluate(
        self,
        dataloader: DataLoader[MultimodalMartianLandslideSample],
    ) -> dict[str, int | float | str | list]:
        self._model.eval()

        losses: list[SSJEPALoss] = []
        cos_sims: list[float] = []
        latent_stds: list[float] = []
        latent_pred_stds: list[float] = []
        target_encodings: list[torch.Tensor] = []

        for batch in dataloader:
            # 1. GENERATE SS-MASK (Indices 0-1791)

            ids_keep, ids_drop = generate_uniform_mask(
                mask_ratio=self._mask_ratio_scheduler.value,
                n_patches=self._model.context_encoder.total_tokens,
                device=self._device,
            )

            x, y, z = self._create_x_y_z(
                batch=batch, ids_keep=ids_keep, ids_drop=ids_drop, device=self._device
            )

            with torch.no_grad():
                # Forward returns SSJEPAOutput
                jepa_output: SSJEPAOutput = self._model(x=x, y=y, z=z)

                # Collect for mean loss calculation
                losses.append(jepa_output.loss.unsqueeze(0))
                target_encodings.append(jepa_output.target_encoding.detach().cpu())

                #  Pred is (B, num_masked, dim). Target is (B, 1792, dim).

                pred = jepa_output.prediction
                target_masked = jepa_output.target_encoding[:, ids_drop]

                cos_sim = F.cosine_similarity(pred, target_masked, dim=-1).mean().item()
                cos_sims.append(cos_sim)

                # We flatten (B, 1792, dim) -> (B*1792, dim).

                flat_target = jepa_output.target_encoding.reshape(
                    -1, jepa_output.target_encoding.shape[-1]
                )

                # std(dim=0) calculates the variation per feature-dimension.

                latent_std = flat_target.std(dim=0).mean().item()
                latent_stds.append(latent_std)

                flat_pred = jepa_output.prediction.reshape(
                    -1, jepa_output.prediction.shape[-1]
                )
                latent_pred_std = flat_pred.std(dim=0).mean().item()
                latent_pred_stds.append(latent_pred_std)

        svd_plotly_fig = self._generate_svd_chart(torch.cat(target_encodings, dim=0))

        mean_loss: SSJEPALoss = torch.cat(losses).mean()  # type: ignore

        metrics_payload: dict[str, Any] = {
            "val/loss/total": mean_loss.total.item(),
            "val/loss/sigreg": mean_loss.sigreg.item(),
            "val/loss/pred": mean_loss.pred.item(),
            "val/diagnostics/prediction_cosine_similarity": sum(cos_sims)
            / len(cos_sims),
            "val/diagnostics/latent_dimension_std": sum(latent_stds) / len(latent_stds),
            "val/diagnostics/latent_pred_std": sum(latent_pred_stds)
            / len(latent_pred_stds),
        }

        if svd_plotly_fig is not None:
            metrics_payload["val/plots/singular_value_distribution"] = svd_plotly_fig

        self._wandb.log(metrics_payload)

        return {
            "loss/pred": round(mean_loss.pred.item(), 4),
            "loss/sigreg": round(mean_loss.sigreg.item(), 4),
            "loss/total": round(mean_loss.total.item(), 4),
        }
