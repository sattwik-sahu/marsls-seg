import time
import math
import pandas as pd
from pathlib import Path

from marsls_seg.utils import train
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders._base import EncoderMixin

from marsls_seg.helpers.device import DEVICE
from marsls_seg.utils.models.ms_jepa import load_ms_jepa
from marsls_seg.utils.modules.loss import LogCoshDiceLoss
from marsls_seg.utils.train.ms_jepa_sw import create_dataloaders, group_channels, preprocess_batch

# ==========================================
# 1. UNIVERSAL SMP WRAPPER
# ==========================================
class UniversalSMPJEPAWrapper(nn.Module, EncoderMixin):
    def __init__(self, jepa_encoder, config):
        super().__init__()
        self.jepa = jepa_encoder
        self.patch_size = config["model"]["patch_size"]
        self.dim_embed = config["model"]["dim"]
        
        self._n_vision = sum(l["n_channels"] for l in config["model"]["feature_layers"]["vision"])
        self._n_physics = sum(l["n_channels"] for l in config["model"]["feature_layers"]["physics"])
        self._in_channels = self._n_vision + self._n_physics
        
        self._depth = 5 
        self._output_stride = 16 
        
        self._out_channels =[
            self._in_channels,   
            self._in_channels,   
            self._in_channels,   
            self.dim_embed * 2,  
            self.dim_embed * 2,  
            self.dim_embed * 2   
        ]

    def make_dilated(self, output_stride):
        pass

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        v_img = x[:, :self._n_vision, :, :]
        p_img = x[:, self._n_vision:, :, :]
        
        enc = self.jepa(vision_image=v_img, physics_image=p_img)
        jepa_feat = torch.cat([enc["vision_encoding"], enc["physics_encoding"]], dim=-1)
        
        B, N, D2 = jepa_feat.shape
        H_p = int(math.sqrt(N))
        jepa_feat = jepa_feat.transpose(1, 2).reshape(B, D2, H_p, H_p)

        f0 = x
        f1 = F.max_pool2d(f0, kernel_size=2, stride=2)
        f2 = F.max_pool2d(f1, kernel_size=2, stride=2)
        f3 = jepa_feat                                  
        f4 = F.max_pool2d(f3, kernel_size=2, stride=2)  
        f5 = F.max_pool2d(f4, kernel_size=2, stride=2)  
            
        return [f0, f1, f2, f3, f4, f5]

# ==========================================
# 2. METRICS & TIMING ENGINE
# ==========================================
def compute_metrics(preds, labels):
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

def measure_inference_time(model, dummy_input):
    model.eval()
    with torch.no_grad():
        for _ in range(10): _ = model(dummy_input)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(50): _ = model(dummy_input)
        end_event.record()
        torch.cuda.synchronize()
        return (start_event.elapsed_time(end_event) / 50.0) / 1000.0

# ==========================================
# 3. BENCHMARKING LOOP
# ==========================================
def run_benchmark(weights_dir: Path):
    ms_jepa = load_ms_jepa(dir_path=weights_dir, device=DEVICE)
    config, base_jepa = ms_jepa["config"], ms_jepa["encoder"]
    config["data"]["root_dir"] = "/home/moonlab/Downloads/Mars_data/Mars_LSc_2025_dataset_1st_phase"
    
    smp.encoders.encoders["multispectral_jepa"] = {
        "encoder": lambda **kwargs: UniversalSMPJEPAWrapper(base_jepa, config),
        "pretrained_settings": {"custom": {"mean": [0], "std": [1], "url": None, "repo_id": None, "input_space": "raw", "input_range": [0,1]}},
        "params": {},
    }
    
    architectures = {
        "U-Net": smp.Unet,
        "U-Net++": smp.UnetPlusPlus,
        "FPN": smp.FPN,
        "MAnet": smp.MAnet,
        "Linknet": smp.Linknet,
    }
    
    train_loader, val_loader, _ = create_dataloaders(Path(config["data"]["root_dir"]), batch_size=8, phase=config["data"]["phase"])
    dice_loss_fn = LogCoshDiceLoss()
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    
    results =[]

    for model_name, ModelClass in architectures.items():
        print(f"\n{'='*40}\nTraining {model_name} Decoder (Fine-Tuning)\n{'='*40}")
        
        wandb.init(
            project="MarsLS-Benchmark",
            name=f"{model_name}-Finetune",
            config=config,
            reinit=True 
        )
        
        model = ModelClass(
            encoder_name="multispectral_jepa",
            encoder_weights=None,
            in_channels=3,  
            classes=1,
        ).to(DEVICE)
        
        decoder_params = list(model.decoder.parameters()) + list(model.segmentation_head.parameters())
        optimizer = torch.optim.AdamW([
            {"params": decoder_params, "lr": 1e-3},
            {"params": model.encoder.parameters(), "lr": 5e-5}, 
        ])
        
        # 🚀 ADDED SCHEDULER: 10 Epochs Warmup -> 190 Epochs Cosine Decay
        epochs = 75
        warmup_epochs = 10
        
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(epochs - warmup_epochs), eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )
        
        best_miou = 0.0
        best_metrics = []
        
        for epoch in range(epochs):
            # --- TRAINING ---
            model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
                batch = preprocess_batch(batch, config)
                v_in = group_channels(batch, [l["name"] for l in config["model"]["feature_layers"]["vision"]])
                p_in = group_channels(batch, [l["name"] for l in config["model"]["feature_layers"]["physics"]])
                
                x = torch.cat([v_in, p_in], dim=1).to(DEVICE)
                label = batch["label"].unsqueeze(1).to(DEVICE, dtype=torch.float32)
                
                optimizer.zero_grad()
                with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                    logits = model(x)
                    if logits.shape[2:] != label.shape[2:]:
                        logits = F.interpolate(logits, size=label.shape[2:], mode="bilinear", align_corners=False)
                    loss = dice_loss_fn(logits, label) + bce_loss_fn(logits, label)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            # --- VALIDATION ---
            model.eval()
            all_metrics =[]
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                    batch = preprocess_batch(batch, config)
                    v_in = group_channels(batch, [l["name"] for l in config["model"]["feature_layers"]["vision"]])
                    p_in = group_channels(batch, [l["name"] for l in config["model"]["feature_layers"]["physics"]])
                    
                    x = torch.cat([v_in, p_in], dim=1).to(DEVICE)
                    label = batch["label"].unsqueeze(1).to(DEVICE, dtype=torch.float32)
                    
                    with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                        logits = model(x)
                        if logits.shape[2:] != label.shape[2:]:
                            logits = F.interpolate(logits, size=label.shape[2:], mode="bilinear", align_corners=False)
                        loss = dice_loss_fn(logits, label) + bce_loss_fn(logits, label)
                        
                    val_loss += loss.item()
                    all_metrics.append(compute_metrics(logits, label))
            
            avg_metrics = [sum(m)/len(m) for m in zip(*all_metrics)]
            current_miou = avg_metrics[5]
            
            # 🚀 SAVE THE BEST EPOCH
            if current_miou > best_miou:
                best_miou = current_miou
                best_metrics = avg_metrics
            
            # 🚀 STEP SCHEDULER
            scheduler.step()
            
            # 🚀 LOG METRICS & LRs TO WANDB
            wandb.log({
                "epoch": epoch + 1,
                "train/lr_decoder": optimizer.param_groups[0]["lr"],
                "train/lr_encoder": optimizer.param_groups[1]["lr"],
                "train/epoch_loss": train_loss / len(train_loader),
                "val/epoch_loss": val_loss / len(val_loader),
                "val/precision": avg_metrics[0],
                "val/recall": avg_metrics[1],
                "val/f1_score": avg_metrics[2],
                "val/iou_bg": avg_metrics[3],
                "val/iou_fg": avg_metrics[4],
                "val/mIoU": current_miou,
                "val/best_mIoU": best_miou
            })
        
        # We append the BEST metrics achieved during the 200 epochs to the table!
        dummy_input = torch.randn(1, 7, config["data"]["image_size"], config["data"]["image_size"]).to(DEVICE)
        inf_time = measure_inference_time(model, dummy_input)
        
        results.append([model_name, *best_metrics, inf_time])
        wandb.finish()

    # ==========================================
    # 4. GENERATE THE PAPER TABLE
    # ==========================================
    columns = ["Model", "Precision", "Recall", "F1-score", "IoU_BG", "IoU_FG", "mIoU", "Inference Time (s)"]
    df = pd.DataFrame(results, columns=columns)
    for col in ["Precision", "Recall", "F1-score", "IoU_BG", "IoU_FG", "mIoU", "Inference Time (s)"]:
        df[col] = df[col].apply(lambda x: f"{x:.3f}")
        
    print("\n\n" + "="*80)
    print("FINAL BENCHMARK RESULTS FOR BMVC PAPER")
    print("="*80)
    print(df.to_markdown(index=False))
    
    Path("data/runs/segmentation").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/runs/segmentation/benchmark_results.csv", index=False)

if __name__ == "__main__":
    from pathlib import Path
    print("🚀 Script execution started!")
    weights = Path("/home/moonlab/marsls-seg/data/runs/ms-jepa-sw/20260511-172652_rare-snowball-146")
    try:
        run_benchmark(weights_dir=weights)
    except NameError:
        train(weights_dir=weights)