import re
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from torchinfo import summary

import wandb
from marsls_seg.helpers.device import DEVICE
from marsls_seg.helpers.timestamp import get_timestamp_now
from marsls_seg.utils.data.mmls import (
    MultimodalMartianLandslideDataset,
    MultimodalMartianLandslideSample,
)
from marsls_seg.utils.data.processing import ProcessedMMLSv2Dataset
from marsls_seg.utils.train.base import BaseTrainer


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Get current timestamp
    timestamp: str = get_timestamp_now()

    # Create the rich console
    console: Console = Console()

    # Initialize wandb
    wandb_run: wandb.Run = wandb.init(
        project=cfg.wandb.project,
        name=f"run-{timestamp}",
        config=OmegaConf.to_container(cfg=cfg, resolve=True),  # type: ignore
        group=cfg.wandb.group,
        mode=cfg.wandb.mode,
    )

    # Initialize datasets
    data_root_dir: Path = Path(cfg.data.root_dir)
    data_params_file: Path = Path(cfg.data.params_file)
    train_data: ProcessedMMLSv2Dataset = ProcessedMMLSv2Dataset(
        dataset=MultimodalMartianLandslideDataset(
            data_root=data_root_dir, split="train"
        ),
        params_file=data_params_file,
    )
    val_data: ProcessedMMLSv2Dataset = ProcessedMMLSv2Dataset(
        dataset=MultimodalMartianLandslideDataset(
            data_root=data_root_dir, split=cfg.data.eval_split
        ),
        params_file=data_params_file,
    )

    # Create dataloader
    train_loader: torch.utils.data.DataLoader[MultimodalMartianLandslideSample] = (
        torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=cfg.training.num_workers,
            pin_memory=cfg.training.pin_memory,
            collate_fn=torch.stack,
        )
    )
    val_loader: torch.utils.data.DataLoader[MultimodalMartianLandslideSample] = (
        torch.utils.data.DataLoader(
            dataset=val_data,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=cfg.training.num_workers,
            pin_memory=cfg.training.pin_memory,
            collate_fn=torch.stack,
        )
    )

    # Create the model and load to device
    model: torch.nn.Module = hydra.utils.instantiate(cfg.model)
    model = model.to(device=DEVICE)

    # Create trainer
    trainer: BaseTrainer = hydra.utils.instantiate(
        cfg.trainer,
        wandb=wandb_run,
        model=model,
        n_epochs=cfg.training.n_epochs,
        lr=cfg.training.lr,
        device=DEVICE,
    )

    # Setup checkpoint directory
    ckpt_root_dir: Path = Path(cfg.training.ckpt_dir)
    ckpt_dir = ckpt_root_dir / timestamp
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    # Save the model config with the checkpoints
    config_path: Path = ckpt_dir / "config.yaml"
    OmegaConf.save(config=cfg.model, f=config_path, resolve=True)

    # Start training

    # Get model name
    # Match everything after the final literal dot
    match = re.search(r"[^.]+$", cfg.model._target_)
    model_name: str = match.group() if match else ""

    # Display model summary
    console.log("=========== Model summary ===========")
    summary(model=model)

    # Create progress bar
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("[yellow]ETA:[/]"),
        TimeRemainingColumn(),
    ) as progress:
        # Initialize training task
        training_task = progress.add_task(
            f"Train {model_name}", total=cfg.training.n_epochs
        )

        # Start training by epochs
        for epoch in range(1, cfg.training.n_epochs + 1):
            trainer.train_epoch(dataloader=train_loader, epoch=epoch)

            # Evaluate model every `eval_int` epochs
            if epoch % cfg.training.eval_int == 0:
                log = trainer.evaluate(dataloader=val_loader)
                progress.log(log)

            # Update progress bar by one step
            progress.update(task_id=training_task, advance=1)

            # Save model every `save_int` epochs
            if epoch % cfg.training.save_int == 0 or epoch == cfg.training.n_epochs:
                ckpt_path: Path = ckpt_dir / f"ckpt_ep_{epoch}.pt"
                torch.save(model.state_dict(), ckpt_path)
                progress.log(f"Saved local checkpoint --> [cyan]{ckpt_path}[/]")

    console.log("Run complete. Uploading artifacts...", style="bold green")

    # Upload the final artifact to wandb
    final_ckpt_path: Path = ckpt_dir / f"ckpt_ep_{cfg.training.n_epochs}.pt"

    if final_ckpt_path.exists():
        model_artifact = wandb.Artifact(
            name=cfg.wandb.artifact_name, type="model", description=cfg.wandb.model_desc
        )
        model_artifact.add_file(final_ckpt_path.as_posix())  # Save the model weights
        model_artifact.add_file(config_path.as_posix())  # Save the model config
        wandb_run.log_artifact(model_artifact)
        console.log(":tada: Uploaded artifact successfully!")
    else:
        console.log(
            ":warning: Warning: Final checkpoint file does not exist.", style="yellow"
        )

    # Finish the run
    wandb.finish()


if __name__ == "__main__":
    main()
