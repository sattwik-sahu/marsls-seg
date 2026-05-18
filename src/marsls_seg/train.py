from pathlib import Path
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from marsls_seg.helpers.device import DEVICE
from marsls_seg.helpers.timestamp import get_timestamp_now
from marsls_seg.utils.data.mmls import (
    MultimodalMartianLandslideDataset,
    MultimodalMartianLandslideSample,
)
from marsls_seg.utils.data.processing import MultimodalMarsLandslideDataProcessor
from rich.console import Console

# Import rich progress layout components
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from marsls_seg.utils.train.base import BaseTrainer


class ProcessedMMLSv2Dataset(
    torch.utils.data.Dataset[MultimodalMartianLandslideSample]
):
    def __init__(
        self,
        dataset: MultimodalMartianLandslideDataset,
        processor: MultimodalMarsLandslideDataProcessor,
    ) -> None:
        super().__init__()

        self._dataset = dataset
        self._processor = processor

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index) -> MultimodalMartianLandslideSample:
        return self._processor(self._dataset[index])


@hydra.main(version_base=None, config_path="../configs", config_name="pretrain")
def main(cfg: DictConfig) -> None:
    timestamp = get_timestamp_now()
    console = Console()
    console.log("Initializing training...", style="yellow")

    wandb_run = wandb.init(
        project=cfg.wandb.project,
        name=f"run-{timestamp}",
        config=OmegaConf.to_container(cfg=cfg, resolve=True),  # type: ignore
        mode=cfg.wandb.mode,
    )

    # Create processor
    console.log(f"Initializing processor: [magenta]{cfg.data.processor._target_}[/]")
    processor: MultimodalMarsLandslideDataProcessor = hydra.utils.instantiate(
        cfg.data.processor
    )

    # Create datasets
    train_data = MultimodalMartianLandslideDataset(
        data_root=Path(cfg.data.root_dir), split="train"
    )
    val_data = MultimodalMartianLandslideDataset(
        data_root=Path(cfg.data.root_dir), split="val"
    )

    # Wrap the raw datasets with your processor to apply channel normalization
    processed_train_dataset = ProcessedMMLSv2Dataset(train_data, processor)
    processed_val_dataset = ProcessedMMLSv2Dataset(val_data, processor)

    # Create dataloaders using the normalized datasets
    train_loader = torch.utils.data.DataLoader[MultimodalMartianLandslideSample](
        dataset=processed_train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        collate_fn=torch.stack,
    )
    val_loader = torch.utils.data.DataLoader[MultimodalMartianLandslideSample](
        dataset=processed_val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        collate_fn=torch.stack,
    )

    # Create model
    model: torch.nn.Module = hydra.utils.instantiate(cfg.model).to(device=DEVICE)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)

    # Create trainer
    trainer: BaseTrainer = hydra.utils.instantiate(cfg.trainer, wandb=wandb_run)

    # Setup Checkpoint Directory
    checkpoint_root = Path(cfg.training.ckpt_dir) / timestamp
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    console.log(f"Starting model training: [green]{cfg.model._target_}[/]")

    # --- SIMPLIFIED PROGRESS DISPLAY ---
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),  # Track step count (e.g., Epoch 34/50)
        TextColumn("[yellow]ETA:[/]"),
        TimeRemainingColumn(),  # Tracks estimated remaining computation time
        console=console,
    ) as progress:
        # Initialize tracking task
        training_task = progress.add_task(
            "[bold cyan]Training Progress[/]", total=cfg.trainer.n_epochs
        )

        for epoch in range(1, cfg.trainer.n_epochs + 1):
            trainer.train_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=DEVICE,
                epoch=epoch,
            )

            if epoch % cfg.training.eval_int == 0:
                trainer.evaluate(model=model, dataloader=val_loader, device=DEVICE)

            # Advance progress counter by 1 step
            progress.update(training_task, advance=1)

            # --- Periodic Local Checkpointing ---
            if epoch % cfg.training.save_int == 0 or epoch == cfg.trainer.n_epochs:
                checkpoint_path = checkpoint_root / f"ckpt_epoch_{epoch}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                # Logging through progress.console guarantees smooth bar redraws
                progress.console.log(
                    f"💾 Saved local checkpoint to: [cyan]{checkpoint_path}[/]"
                )

    console.log("Run complete, preparing final artifact upload...", style="bold green")

    # --- Upload the Final Model Checkpoint to WandB ---
    final_checkpoint_path = checkpoint_root / f"ckpt_epoch_{cfg.trainer.n_epochs}.pt"

    if final_checkpoint_path.exists():
        model_artifact = wandb.Artifact(
            name="marsls-ijepa-vit",
            type="model",
            description="Self-supervised IJEPA Vision Transformer backbone trained on MMLSv2 imagery.",
        )
        model_artifact.add_file(str(final_checkpoint_path))
        wandb_run.log_artifact(model_artifact)
        console.log(
            "🚀 Final model checkpoint successfully uploaded to WandB Artifacts!"
        )
    else:
        console.log(
            "[red]⚠️ Warning: Final checkpoint file not found. Upload failed.[/]"
        )

    wandb.finish()


if __name__ == "__main__":
    main()
