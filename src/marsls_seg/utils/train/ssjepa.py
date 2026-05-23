import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.transforms import v2
from transformers import get_cosine_schedule_with_warmup

from typing import override, Any
import plotly.graph_objects as go

from marsls_seg.utils.data._typing import MultimodalMartianLandslideSample
from marsls_seg.utils.modules.encoder.spatio_spectral_vit import SSVitInput
from marsls_seg.utils.modules.ssjepa.jepa import SpatioSpectralJepa, SSJEPALoss

from marsls_seg.utils.modules.ssjepa._typing import SSJEPAOutput
from marsls_seg.utils.train.base import BaseTrainer
from marsls_seg.helpers.mask import MaskingRatioScheduler, generate_uniform_mask,construct_latent
from wandb import Run as WandbRun



class SSJEPAtrainer(BaseTrainer[SpatioSpectralJepa,MultimodalMartianLandslideSample , dict]):
    """
        Trainer for spatio-spectral jepa, handles the Training of 7 channels.

    """

    def __init__(
            self,
            model: SpatioSpectralJepa,
            mask_ratio: tuple[float,float],
            lr: float,
            n_epochs: int,
            n_warmup_epochs: int,
            wandb: WandbRun,
            device: torch.device,
    ) -> None :
        super().__init__(
            model=model,
            n_epochs=n_epochs,
            lr=lr,
            wandb=wandb,
            device=device

        )

        self._mask_ratio_scheduler = MaskingRatioScheduler(start=mask_ratio[0],end=mask_ratio[1],T=n_epochs)

        self._lr_scheduler = get_cosine_schedule_with_warmup(optimzer=self._optimzer,
                                                             num_warmup_steps=n_warmup_epochs,
                                                             num_training_stps=self._n_epochs,
                                                             )
        
        self._augmentation =v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RanodmRotation(degrees=[-90,90]),
            v2.RandomResizedCrop(size=(128,128),antialias=True),
        ])

    def _create_x_y_z(
            self,
            batch: MultimodalMartianLandslideSample,
            ids_keep: torch.Tensor,
            ids_drop: torch.Tensor,
            device: torch.device
    ) -> tuple[SSVitInput,SSVitInput,torch.Tensor]:
        """Creates the context(x), target(y) and Query (z) tensors"""
        image=batch.merge_channels() #(B,7,128,128)
        if self._model_training:
            image= self._augmentation(image)

        #Input Image + indices of set of visible patches
        x= SSVitInput(image=image,mask=ids_keep).to(device)

        # Full image for the target encoder
        y=SSVitInput(image=image).to(device)

        # This is our query
        z_full=self._model.context_encoder.get_full_pos_embed(device)

        z=construct_latent(z_full=z_full,ids=ids_drop,batch_size=batch.batch_size[0])

        return x, y, z
    
    @override
    def _create_optimizer(self, model: SpatioSpectralJepa) -> torch.optim.Optimizer:
        return torch.optim.AdamW(params=model.parameters(),lr=self._lr)
    
    @override
    def train_epoch(
        self,
        dataloader:
    )
