from typing import override

import torch
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader

from marsls_seg.utils.data.mmls import (
    MultimodalMartianLandslideDataset,
    MultimodalMartianLandslideSample,
)
from marsls_seg.utils.data.processing import MultimodalMarsLandslideDataProcessor
from marsls_seg.utils.modules.jepa.base import JEPAOutput
from marsls_seg.utils.modules.jepa.ijepa import IJEPA, IJEPALoss
from marsls_seg.utils.modules.tf.decoder import TransformerDecoder
from marsls_seg.utils.modules.vit import VisionTransformer, ViTInput
from marsls_seg.utils.train.base import BaseTrainer
from wandb import Run
from marsls_seg.helpers.mask import generate_uniform_mask, construct_latent


class IJEPATrainer(BaseTrainer[IJEPA, MultimodalMartianLandslideSample]):
    """
    IJEPA style trainer for ViT on fat images from MMLSv2 dataset.
    """

    def __init__(
        self, mask_ratio: tuple[float, float], n_epochs: int, wandb: Run
    ) -> None:
        super().__init__()

        self._mask_ratio: float = mask_ratio[0]

        self._delta_mask_ratio: float = (mask_ratio[1] - mask_ratio[0]) / n_epochs
        self._wandb = wandb

    def _create_x_y_z(
        self,
        batch: MultimodalMartianLandslideSample,
        ids_keep: torch.Tensor,
        ids_drop: torch.Tensor,
        device: torch.device,
        pos_emb: torch.Tensor,
    ) -> tuple[ViTInput, ViTInput, torch.Tensor]:
        image = batch.merge_channels()
        x = ViTInput(image=image, mask=ids_keep).to(device=device)
        y = ViTInput(image=image).to(device=device)
        z = construct_latent(
            z_full=pos_emb, ids=ids_drop, batch_size=batch.batch_size[0]
        )
        return x, y, z

    @override
    def train_epoch(
        self,
        model: IJEPA,
        dataloader: DataLoader[MultimodalMartianLandslideSample],
        optimizer: Optimizer,
        device: torch.device,
        epoch: int,
    ) -> None:
        batch: MultimodalMartianLandslideSample
        for batch in dataloader:
            ids_keep, ids_drop = generate_uniform_mask(
                mask_ratio=self._mask_ratio, n_patches=model.context_encoder.n_patches
            )
            x, y, z = self._create_x_y_z(
                batch=batch,
                ids_keep=ids_keep,
                ids_drop=ids_drop,
                device=device,
                pos_emb=model.context_encoder.pos_emb,
            )
            jepa_output: JEPAOutput[torch.Tensor, IJEPALoss] = model(x=x, y=y, z=z)
            optimizer.zero_grad()
            if jepa_output.loss is not None:
                jepa_output.loss.total.backward()
                print(f"Loss:\n{jepa_output.loss.to_dict()}")
                optimizer.step()
                self._wandb.log(
                    {
                        "loss/total": jepa_output.loss.total,
                        "loss/sigreg": jepa_output.loss.sigreg,
                        "loss/pred": jepa_output.loss.pred,
                    }
                )
        print(f"Trained epoch {epoch}")

    @override
    def evaluate(
        self,
        model: IJEPA,
        dataloader: DataLoader[MultimodalMartianLandslideSample],
        device: torch.device,
    ):
        batch: MultimodalMartianLandslideSample = next(iter(dataloader))
        ids_keep, ids_drop = generate_uniform_mask(
            mask_ratio=self._mask_ratio, n_patches=model.context_encoder.n_patches
        )
        x, y, z = self._create_x_y_z(
            batch=batch,
            ids_keep=ids_keep,
            ids_drop=ids_drop,
            device=device,
            pos_emb=model.context_encoder.pos_emb,
        )
        with torch.inference_mode():
            jepa_output: JEPAOutput[torch.Tensor, IJEPALoss] = model(x=x, y=y, z=z)
        self._wandb.log({"val/loss/total": jepa_output.loss.total})  # type: ignore
        print("Evaluation complete")
