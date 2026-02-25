import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

import wandb
from marsls_seg.helpers.device import DEVICE
from marsls_seg.helpers.timestamp import get_timestamp_now
from marsls_seg.utils.data.marsls import MarsLS_Dataset, MarsLS_Sample
from marsls_seg.utils.models.ms_jepa.predictor import Predictor
from marsls_seg.utils.models.ms_jepa.segmentation import (
    MLPFeatureFusion,
    SegmentationHead,
)
from marsls_seg.utils.models.ms_jepa.sigreg import SIGReg
from marsls_seg.utils.models.ms_jepa.vit import VisionTransformer
from marsls_seg.utils.modules.masking import Mask, apply_mask, generate_mask
from marsls_seg.utils.train.ms_jepa import (
    build_encoder_predictor,
    group_channels,
    parse_config,
    create_dataloaders,
    preprocess_batch,
)
from marsls_seg.utils.modules.loss import LogCoshDiceLoss


def load_config(weights_dir: Path) -> dict:
    config_file: Path = weights_dir / "config.yaml"
    config: dict = parse_config(config_file=config_file)
    return config


def load_encoders(
    weights_dir: Path, config: dict
) -> tuple[VisionTransformer, VisionTransformer]:
    vision_encoder, _ = build_encoder_predictor(
        config=config, device=DEVICE, model_name="vision"
    )
    physics_encoder, _ = build_encoder_predictor(
        config=config, device=DEVICE, model_name="physics"
    )

    # Load pre-trained weights
    vision_encoder.load_state_dict(
        torch.load(weights_dir / "vision_encoder.pt", weights_only=True)
    )
    physics_encoder.load_state_dict(
        torch.load(weights_dir / "physics_encoder.pt", weights_only=True)
    )

    return vision_encoder.eval(), physics_encoder.eval()


def get_inputs(sample: MarsLS_Sample) -> tuple[torch.Tensor, torch.Tensor]:
    vision_input = group_channels(sample=sample, channel_names=["rgb", "gray"])
    physics_input = group_channels(
        sample=sample, channel_names=["dem", "slope", "thermal_inertial"]
    )
    return vision_input, physics_input


def build_segmentation_modules(
    config: dict,
) -> tuple[MLPFeatureFusion, SegmentationHead]:
    fusion = MLPFeatureFusion(
        dim_embed=config["model"]["dim"],
        # image_size=config["data"]["image_size"],
        # n_heads=8,
        # patch_size=config["model"]["patch_size"],
    ).to(device=DEVICE)

    segmentation = SegmentationHead(dim_embed=config["model"]["dim"]).to(device=DEVICE)
    return fusion, segmentation


def train(weights_dir: Path) -> None:
    config: dict = load_config(weights_dir=weights_dir)

    # Initialize WandB
    wandb.init(project="MarsLS-JEPA", config=config)

    vision_encoder, physics_encoder = load_encoders(
        config=config, weights_dir=weights_dir
    )
    fusion, segmentation = build_segmentation_modules(config=config)

    train_loader, val_loader, _ = create_dataloaders(
        data_root=Path(config["data"]["root_dir"]), batch_size=196
    )

    # 1. Lower the Learning Rates
    optimizer = torch.optim.AdamW(
        [
            {"params": fusion.parameters(), "lr": 3e-5, "weight_decay": 0.5},
            {"params": segmentation.parameters(), "lr": 1e-4, "weight_decay": 0.5},
        ]
    )

    # 2. Add an explicit scheduler to catch the Epoch 14 divergence
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.5, patience=3
    # )

    # Hybrid Loss Components
    dice_loss_fn = LogCoshDiceLoss()
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(15_000):
        fusion.train()
        segmentation.train()
        epoch_train_losses = []

        for batch in train_loader:
            optimizer.zero_grad()
            batch = preprocess_batch(batch=batch, config=config)
            vision_input, physics_input = get_inputs(sample=batch)
            label = batch["label"].unsqueeze(1).to(device=DEVICE, dtype=torch.float32)

            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                with torch.no_grad():
                    v_tokens = vision_encoder(vision_input.to(DEVICE))
                    p_tokens = physics_encoder(physics_input.to(DEVICE))
                logits = segmentation(
                    fusion(vision_tokens=v_tokens, physics_tokens=p_tokens)
                )

                # Hybrid Loss Calculation
                loss_dice = dice_loss_fn(logits, label)
                loss_bce = bce_loss_fn(logits, label)
                total_loss = loss_dice + loss_bce

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(segmentation.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_losses.append(total_loss.item())

        # Validation Step
        avg_val_loss, avg_val_iou, val_sample = validate(
            fusion,
            segmentation,
            vision_encoder,
            physics_encoder,
            val_loader,
            dice_loss_fn,
            bce_loss_fn,
            config,
        )
        # scheduler.step(avg_val_loss)

        # Logging
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": np.mean(epoch_train_losses),
                "val_loss": avg_val_loss,
                "val_mIoU": avg_val_iou,
                "visuals": [wandb.Image(val_sample, caption=f"Epoch {epoch} Eval")],
            }
        )

        print(
            f"Epoch {epoch} | Train: {np.mean(epoch_train_losses):.4f} | Val: {avg_val_loss:.4f}"
        )


def calculate_iou(preds: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
    # Convert logits to binary mask
    preds = (torch.sigmoid(preds) > threshold).float()

    intersection = (preds * labels).sum(dim=(1, 2, 3))
    union = (preds + labels).clamp(0, 1).sum(dim=(1, 2, 3))

    # Add small epsilon to avoid division by zero
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou.mean()


def validate(fusion, segmentation, v_enc, p_enc, loader, d_fn, b_fn, config):
    fusion.eval()
    segmentation.eval()
    val_losses = []
    val_ious = []
    plot_img = None

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = preprocess_batch(batch=batch, config=config)
            v_in, p_in = get_inputs(sample=batch)
            label = batch["label"].unsqueeze(1).to(DEVICE, dtype=torch.float32)

            # Forward pass
            v_tokens = v_enc(v_in.to(DEVICE))
            p_tokens = p_enc(p_in.to(DEVICE))
            logits = segmentation(fusion(v_tokens, p_tokens))

            # Metrics
            loss = d_fn(logits, label) + b_fn(logits, label)
            iou = calculate_iou(logits, label)

            val_losses.append(loss.item())
            val_ious.append(iou.item())

            # Grab one random sample for plotting from the first batch
            if i == 0:
                idx = np.random.randint(0, v_in.size(0))
                # Assuming first 3 channels of vision_input are RGB
                rgb = v_in[idx, :3].cpu().permute(1, 2, 0).numpy()
                gt = label[idx, 0].cpu().numpy()
                pred = torch.sigmoid(logits[idx, 0]).cpu().numpy()

                # Normalize RGB for plotting if it isn't [0,1]
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                ax[0].imshow(rgb)
                ax[0].set_title("RGB Image")
                ax[1].imshow(gt, cmap="gray")
                ax[1].set_title("Ground Truth")
                ax[2].imshow(pred, cmap="inferno")
                ax[2].set_title("Prediction")
                for a in ax:
                    a.axis("off")

                plt.tight_layout()
                # Replace the conversion block with this:
                fig.canvas.draw()
                # Use buffer_hlinear_rgb or a more modern approach
                rgba_buffer = fig.canvas.buffer_rgba()
                plot_img = np.asarray(rgba_buffer)[
                    :, :, :3
                ]  # Strip Alpha channel to get RGB
                plt.close(fig)

        return np.mean(val_losses), np.mean(val_ious), plot_img


def main():
    train(weights_dir=Path("data/runs/ms-jepa/20260224-135544"))


if __name__ == "__main__":
    main()
