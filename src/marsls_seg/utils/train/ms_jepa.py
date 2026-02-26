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
from marsls_seg.utils.models.ms_jepa.sigreg import SIGReg
from marsls_seg.utils.models.ms_jepa.vit import VisionTransformer
from marsls_seg.utils.modules.masking import Mask, apply_mask, generate_mask


def parse_config(config_file: Path) -> dict:
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(
    data_root: Path, batch_size: int
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    def _create_dataloader(split: Literal["train", "test", "val"]):
        dataset: MarsLS_Dataset = MarsLS_Dataset(data_root=data_root, split=split)
        loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        return loader

    return (
        _create_dataloader(split="train"),
        _create_dataloader(split="val"),
        _create_dataloader(split="test"),
    )


def preprocess_batch(batch: MarsLS_Sample, config: dict) -> MarsLS_Sample:
    batch["slope"] = batch["slope"].clamp_min(min=-50)
    for layer in config["data"]["feature_info"]:
        info = layer["layer"]
        name = info["name"]
        mean = torch.as_tensor(info["mean"]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        std = torch.as_tensor(info["std"]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        batch[name] = (batch[name] - mean) / std
    return batch


def build_encoder_predictor(
    model_name: Literal["vision", "physics"], config: dict, device: torch.device
) -> tuple[VisionTransformer, Predictor]:
    encoder: VisionTransformer = VisionTransformer(
        image_size=config["data"]["image_size"],
        patch_size=config["model"]["patch_size"],
        dim_embed=config["model"]["dim"],
        n_channels=sum(
            layer["n_channels"]
            for layer in config["model"][model_name]["encoder"]["feature_layers"]
        ),
        n_heads=config["model"][model_name]["encoder"]["n_heads"],
        n_layers=config["model"][model_name]["encoder"]["n_layers"],
    ).to(device=device)
    predictor: Predictor = Predictor(
        dim_embed=config["model"]["dim"],
        image_size=config["data"]["image_size"],
        patch_size=config["model"]["patch_size"],
        n_heads=config["model"][model_name]["predictor"]["n_heads"],
        n_layers=config["model"][model_name]["predictor"]["n_layers"],
    ).to(device=device)
    return encoder, predictor


def group_channels(sample: MarsLS_Sample, channel_names: list[str]) -> torch.Tensor:
    return torch.cat([sample[key] for key in channel_names], dim=1)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer, config: dict
) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer=optimizer,
                start_factor=config["training"]["lr_scheduler"]["start_factor"],
                end_factor=config["training"]["lr_scheduler"]["end_factor"],
                total_iters=config["training"]["lr_scheduler"]["n_warmup_epochs"],
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                # eta_min=config["training"]["lr"]
                # * config["training"]["lr_scheduler"]["start_factor"],
                eta_min=config["training"]["lr_scheduler"]["eta_min"],
                T_max=config["training"]["n_epochs"]
                - config["training"]["lr_scheduler"]["n_warmup_epochs"],
            ),
        ],
        milestones=[config["training"]["lr_scheduler"]["n_warmup_epochs"]],
    )


def log_latent_pca(
    sample_vision_image: torch.Tensor,
    sample_physics_image: torch.Tensor,
    vision_tokens: torch.Tensor,
    physics_tokens: torch.Tensor,
    vision_pred_tokens: torch.Tensor,
    physics_pred_tokens: torch.Tensor,
    config: dict,
):
    """
    vision_tokens/physics_tokens: [n_patches, dim_embed]
    sample_img: [3, H, W] tensor
    """

    # 1. Compute PCA for both
    def get_pca_map(tokens):
        # Tokens: [N, D] -> [N, 3]
        pca = PCA(n_components=3)
        # Normalize to [0, 1] for visualization
        tokens_pca = pca.fit_transform(tokens.cpu().numpy())
        tokens_pca = (tokens_pca - tokens_pca.min()) / (
            tokens_pca.max() - tokens_pca.min()
        )

        # Reshape to grid
        grid_size = int(np.sqrt(tokens.shape[0]))
        return tokens_pca.reshape(grid_size, grid_size, 3)

    vis_pca = get_pca_map(vision_tokens)
    phy_pca = get_pca_map(physics_tokens)
    vis_pred_pca = get_pca_map(vision_pred_tokens)
    phy_pred_pca = get_pca_map(physics_pred_tokens)

    # 2. Convert original image for plotting
    img_np = sample_vision_image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    # 3. Create Figure
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 8))
    axes[0][0].imshow(vis_pca)
    axes[0][0].set_title("Vision Latent PCA")
    axes[0][1].imshow(phy_pca)
    axes[0][1].set_title("Physics Latent PCA")
    axes[0][2].imshow(vis_pred_pca)
    axes[0][2].set_title("Vision Latent Prediction PCA")
    axes[0][3].imshow(phy_pred_pca)
    axes[0][3].set_title("Physics Latent Prediction PCA")

    axes[1][0].imshow(img_np)
    axes[1][0].set_title("RGB Image")
    for i in range(sample_physics_image.shape[0]):
        axes[1][i + 1].imshow(sample_physics_image[i].cpu().numpy(), cmap="inferno")
        axes[1][i + 1].set_title(
            config["model"]["physics"]["encoder"]["feature_layers"][i]["name"]
            .replace("_", " ")
            .upper()
        )

    for ax in axes:
        for ax_ in ax:
            ax_.axis("off")
    plt.tight_layout()

    # 4. Log to WandB
    wandb.log({"val/latents": wandb.Image(fig)})
    plt.close(fig)


def train(config_path: Path):
    config: dict = parse_config(config_file=config_path)
    training_config: dict = config["training"]
    data_config: dict = config["data"]

    # Load data
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=Path(data_config["root_dir"]),
        batch_size=training_config["batch_size"],
    )

    # Create models
    vision_encoder, vision_predictor = build_encoder_predictor(
        model_name="vision", config=config, device=DEVICE
    )
    physics_encoder, physics_predictor = build_encoder_predictor(
        model_name="physics", config=config, device=DEVICE
    )

    # SIGReg regularization
    sigreg: SIGReg = SIGReg().to(device=DEVICE)

    # Optimizer
    optimizer: torch.optim.AdamW = torch.optim.AdamW(
        params=[
            {
                "params": vision_encoder.parameters(),
                "lr": training_config["vision"]["encoder"]["lr"],
                "weight_decay": training_config["vision"]["encoder"]["weight_decay"],
            },
            {
                "params": physics_encoder.parameters(),
                "lr": training_config["physics"]["encoder"]["lr"],
                "weight_decay": training_config["physics"]["encoder"]["weight_decay"],
            },
            {
                "params": vision_predictor.parameters(),
                "lr": training_config["vision"]["predictor"]["lr"],
                "weight_decay": training_config["vision"]["predictor"]["weight_decay"],
            },
            {
                "params": physics_predictor.parameters(),
                "lr": training_config["physics"]["predictor"]["lr"],
                "weight_decay": training_config["physics"]["predictor"]["weight_decay"],
            },
        ]
    )
    # scheduler = torch.optim.lr_scheduler.LinearLR(
    #     optimizer=optim,
    # )
    scheduler: torch.optim.lr_scheduler.LRScheduler = build_lr_scheduler(
        optimizer=optimizer, config=config
    )

    wandb.init(project="MarsLS-JEPA", config=config)

    # Start training
    batch: MarsLS_Sample
    for epoch in range(training_config["n_epochs"]):
        mask_ratio: float = (
            config["training"]["mask_ratio"]["start"]
            + epoch
            * (
                config["training"]["mask_ratio"]["end"]
                - config["training"]["mask_ratio"]["start"]
            )
            / training_config["n_epochs"]
        )

        for batch in train_loader:
            batch = preprocess_batch(batch=batch, config=config)
            vision_image: torch.Tensor = group_channels(
                sample=batch,
                channel_names=[
                    layer["name"]
                    for layer in config["model"]["vision"]["encoder"]["feature_layers"]
                ],
            ).to(device=DEVICE)
            physics_image: torch.Tensor = group_channels(
                sample=batch,
                channel_names=[
                    layer["name"]
                    for layer in config["model"]["physics"]["encoder"]["feature_layers"]
                ],
            ).to(device=DEVICE)

            # Input preparation
            n_tokens: int = int(
                (data_config["image_size"] // config["model"]["patch_size"]) ** 2
            )
            mask: Mask = generate_mask(n_tokens=n_tokens, mask_ratio=mask_ratio)

            optimizer.zero_grad()

            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                # Model inference
                sx_vision: torch.Tensor = physics_encoder(physics_image, mask=mask)
                sx_physics: torch.Tensor = vision_encoder(vision_image, mask=mask)

                # Do not detach z
                z_vision: torch.Tensor = vision_encoder._pos_emb.repeat(
                    training_config["batch_size"], 1, 1
                )
                z_physics: torch.Tensor = physics_encoder._pos_emb.repeat(
                    training_config["batch_size"], 1, 1
                )

                _, sy_hat_vision = apply_mask(
                    vision_predictor(sx=sx_vision, z=z_vision), mask=mask
                )
                _, sy_hat_physics = apply_mask(
                    physics_predictor(sx=sx_physics, z=z_physics), mask=mask
                )

                _, sy_physics = apply_mask(
                    vision_encoder(vision_image).detach(), mask=mask
                )
                _, sy_vision = apply_mask(
                    physics_encoder(physics_image).detach(), mask=mask
                )

                # Calculate losses
                loss_jepa_vision = torch.nn.functional.mse_loss(
                    sy_hat_vision, sy_vision
                )
                loss_jepa_physics = torch.nn.functional.mse_loss(
                    sy_hat_physics, sy_physics
                )
                loss_jepa = 0.5 * (loss_jepa_vision + loss_jepa_physics)

                loss_sigreg_vision = sigreg(sx_vision.transpose(0, 1))
                loss_sigreg_physics = sigreg(sx_physics.transpose(0, 1))
                loss_sigreg = 0.5 * (loss_sigreg_vision + loss_jepa_physics)

                lambd = training_config["lambda"]
                loss = (1 - lambd) * loss_jepa + lambd * loss_sigreg

            loss.backward()

            grad_norm: float = 0
            grad_norm += torch.nn.utils.clip_grad_norm_(
                physics_encoder.parameters(), training_config["grad_clip_norm"]
            ).item()
            grad_norm += torch.nn.utils.clip_grad_norm_(
                physics_predictor.parameters(), training_config["grad_clip_norm"]
            ).item()
            grad_norm += torch.nn.utils.clip_grad_norm_(
                vision_encoder.parameters(), training_config["grad_clip_norm"]
            ).item()
            grad_norm += torch.nn.utils.clip_grad_norm_(
                vision_encoder.parameters(), training_config["grad_clip_norm"]
            ).item()
            grad_norm /= 4
            optimizer.step()

            # 4. W&B Logging
            wandb.log(
                {
                    "train/total_loss": loss.item(),
                    "train/jepa_vision": loss_jepa_vision.item(),
                    "train/jepa_physics": loss_jepa_physics.item(),
                    "train/sigreg": (loss_sigreg_vision + loss_sigreg_physics).item(),
                    "train/grad_norm": grad_norm,
                    "meta/lr": optimizer.param_groups[0]["lr"],
                    "meta/mask_ratio": mask_ratio,
                    "meta/epoch": epoch,
                }
            )

        vision_encoder.eval()
        physics_encoder.eval()
        with torch.no_grad():
            # 1. Grab a single validation sample
            val_batch = next(iter(val_loader))

            # Use your group_channels helper   gggg
            v_img: torch.Tensor = group_channels(
                sample=val_batch,
                channel_names=[
                    layer["name"]
                    for layer in config["model"]["vision"]["encoder"]["feature_layers"]
                ],
            ).to(device=DEVICE)
            p_img: torch.Tensor = group_channels(
                sample=val_batch,
                channel_names=[
                    layer["name"]
                    for layer in config["model"]["physics"]["encoder"]["feature_layers"]
                ],
            ).to(device=DEVICE)

            # 2. Get full latent maps (no masking)
            # We use batch[0] to only look at the first image in the batch
            sx_vis = vision_encoder(v_img[0:1])  # [1, 256, D]
            sx_phy = physics_encoder(p_img[0:1])  # [1, 256, D]
            z_vision: torch.Tensor = vision_encoder._pos_emb.repeat(
                sx_vis.shape[0], 1, 1
            ).detach()
            z_physics: torch.Tensor = physics_encoder._pos_emb.repeat(
                sx_phy.shape[0], 1, 1
            ).detach()
            sy_hat_vis = vision_predictor(sx=sx_vis, z=z_vision)
            sy_hat_phy = vision_predictor(sx=sx_phy, z=z_physics)

            # 3. Log the PCA
            # We pass v_img[0][:3] assuming the first 3 channels are RGB
            log_latent_pca(
                sample_vision_image=v_img[0][:3],
                sample_physics_image=p_img[0],
                vision_tokens=sx_vis[0],
                physics_tokens=sx_phy[0],
                vision_pred_tokens=sy_hat_vis[0],
                physics_pred_tokens=sy_hat_phy[0],
                config=config,
            )

        vision_encoder.train()
        physics_encoder.train()

        scheduler.step()

    # Save the models
    weights_save_dir: Path = Path(
        f"{config['data']['weights_dir']}/{get_timestamp_now()}/"
    )
    os.mkdir(weights_save_dir)
    print("Saving model...")
    torch.save(vision_encoder.state_dict(), weights_save_dir / "vision_encoder.pt")
    torch.save(vision_predictor.state_dict(), weights_save_dir / "vision_predictor.pt")
    torch.save(physics_encoder.state_dict(), weights_save_dir / "physics_encoder.pt")
    torch.save(
        physics_predictor.state_dict(), weights_save_dir / "physics_predictor.pt"
    )
    with open(weights_save_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
