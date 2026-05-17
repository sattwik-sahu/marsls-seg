import time
import math
import pandas as pd
from pathlib import Path

from marsls_seg.utils import train
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders._base import EncoderMixin

from marsls_seg.helpers.device import DEVICE
from marsls_seg.utils.data.marsls import MarsLS_Sample
from marsls_seg.utils.models.ms_jepa import load_ms_jepa
from marsls_seg.utils.modules.loss import LogCoshDiceLoss
from marsls_seg.utils.train.ms_jepa_sw import create_dataloaders, group_channels, preprocess_batch

VISION_CHANNELS  = ["rgb", "gray"]                      # 4 channels
PHYSICS_CHANNELS = ["dem", "slope", "thermal_inertial"] # 3 channels
N_VISION  = 4
N_PHYSICS = 3
N_TOTAL   = N_VISION + N_PHYSICS  # 7



# ==========================================
# 1. UNIVERSAL SMP WRAPPER
# ==========================================
class UniversalSMPJEPAWrapper(nn.Module, EncoderMixin):
    def __init__(self, jepa_encoder, config):
        super().__init__()
        self.jepa = jepa_encoder
        self.patch_size = config["model"]["patch_size"]
        self.dim_embed = config["model"]["dim"]
        
        self._n_vision =N_VISION
        self._n_physics =N_PHYSICS
        self._in_channels = self._n_vision + self._n_physics
        
        # We enforce a standard SMP Depth of 5 for universal compatibility
        self._depth = 5 
        self._output_stride = 32  # Final stride of the encoder (after 5 levels of downsampling)
        
        # The channels at each stride: [s1, s2, s4, s8(JEPA), s16, s32]
        self._out_channels =[
            self._in_channels,   # Stride 1 (128x128)
            self._in_channels,   # Stride 2 (64x64)
            self._in_channels,   # Stride 4 (32x32)
            self.dim_embed * 2,  # Stride 8 (16x16) - JEPA Features
            self.dim_embed * 2,  # Stride 16 (8x8)
            self.dim_embed * 2   # Stride 32 (4x4)
        ]
    

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        v_img = x[:, :self._n_vision, :, :]
        p_img = x[:, self._n_vision:, :, :]


       
        
        with torch.no_grad(): # Keep Encoder frozen during segmentation benchmarking
            enc = self.jepa(vision_image=v_img, physics_image=p_img)
            jepa_feat = torch.cat([enc["vision_encoding"], enc["physics_encoding"]], dim=-1)
            
            B, N, D2 = jepa_feat.shape
            H_p = int(math.sqrt(N))
            jepa_feat = jepa_feat.transpose(1, 2).reshape(B, D2, H_p, H_p)

        # Build standard 6-level pyramid for all SMP decoders
        f0 = x
        f1 = F.max_pool2d(f0, kernel_size=2, stride=2)
        f2 = F.max_pool2d(f1, kernel_size=2, stride=2)
        f3 = jepa_feat                                  # Stride 8 (16x16)
        f4 = F.max_pool2d(f3, kernel_size=2, stride=2)  # Stride 16 (8x8)
        f5 = F.max_pool2d(f4, kernel_size=2, stride=2)  # Stride 32 (4x4)
        
        return [f0, f1, f2, f3, f4, f5]


# ==========================================
# 2. METRICS & TIMING ENGINE
# ==========================================
def compute_metrics(preds, labels):
    """Calculates all metrics exactly as required for the BMVC table."""
    preds = (torch.sigmoid(preds) > 0.5).float()
    labels = labels.float()
    
    # True Positives, False Positives, False Negatives, True Negatives
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
    """Measures precise GPU inference time."""
    model.eval()
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
        
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(50): # Average over 50 runs
        _ = model(dummy_input)
    end_event.record()
    torch.cuda.synchronize()
    
    return (start_event.elapsed_time(end_event) / 50.0) / 1000.0 # Return in seconds


# ==========================================
# 3. BENCHMARKING LOOP
# ==========================================
def run_benchmark(weights_dir: Path):
    # 1. Load Pre-trained JEPA
    ms_jepa = load_ms_jepa(dir_path=weights_dir, device=DEVICE)
    config, base_jepa = ms_jepa["config"], ms_jepa["encoder"]
    config["data"]["root_dir"] = "/home/moonlab/Downloads/Mars_data/Mars_LSc_2025_dataset_1st_phase"
    
    # 2. Register Universal Wrapper
   # 2. Register Universal Wrapper
    smp.encoders.encoders["multispectral_jepa"] = {
        "encoder": lambda **kwargs: UniversalSMPJEPAWrapper(base_jepa, config),
        "pretrained_settings": {
            "custom": {
                "mean": [0],
                "std": [1],
                "url": None,

        # REQUIRED BY SMP 0.5.x
                "repo_id": None,
                "revision": None,

                "input_space": "raw",
                "input_range": [0, 1],
    }
},
        "params": {},
    }
    
    total_in_channels = sum(l["n_channels"] for l in config["model"]["feature_layers"]["vision"] + config["model"]["feature_layers"]["physics"])

    # 3. Define the 8 Architectures to benchmark
    architectures = {
        "Segformer": smp.Segformer,
        "U-Net": smp.Unet,
        "U-Net++": smp.UnetPlusPlus,
        "FPN": smp.FPN,
        # "DeepLabV3": smp.DeepLabV3,
        # "DeepLabV3+": smp.DeepLabV3Plus,
        "MAnet": smp.MAnet,
        "Linknet": smp.Linknet,
        # "PAN": smp.PAN,
    }
    
    # 4. Dataloaders & Loss
    train_loader, val_loader, _ = create_dataloaders(Path(config["data"]["root_dir"]), batch_size=16, phase=config["data"]["phase"])
    dice_loss_fn = LogCoshDiceLoss()
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    
    results =[]

    # 5. Train and Evaluate each architecture
    for model_name, ModelClass in architectures.items():
        print(f"\n{'='*40}\nTraining {model_name} Decoder\n{'='*40}")
        
        # Instantiate SMP model
        model = ModelClass(
            encoder_name="multispectral_jepa",
            encoder_weights=None,
            in_channels=3,
            classes=1,
).to(DEVICE)
        
        # Only optimize the segmentation head (decoder)
        optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=1e-3)
        
        # Train for a limited number of epochs for benchmarking (e.g., 25)
        epochs = 200
        for epoch in range(epochs):
            model.train()
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                batch = preprocess_batch(batch, config)
                v_in = group_channels(batch, VISION_CHANNELS)
                p_in = group_channels(batch, PHYSICS_CHANNELS)

                
                x = torch.cat([v_in, p_in], dim=1).to(DEVICE)
                label = batch["label"].unsqueeze(1).to(DEVICE, dtype=torch.float32)
                
                optimizer.zero_grad()
                with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                    logits = model(x)
                    loss = dice_loss_fn(logits, label) + bce_loss_fn(logits, label)
                loss.backward()
                optimizer.step()

        # Evaluation Phase
        model.eval()
        all_metrics =[]
        with torch.no_grad():
            for batch in val_loader:
                batch = preprocess_batch(batch, config)
                v_in = group_channels(batch, VISION_CHANNELS)
                p_in = group_channels(batch, PHYSICS_CHANNELS)

                
                x = torch.cat([v_in, p_in], dim=1).to(DEVICE)
                label = batch["label"].unsqueeze(1).to(DEVICE, dtype=torch.float32)
                
                logits = model(x)
                all_metrics.append(compute_metrics(logits, label))
                
        # Average the metrics
        avg_metrics = [sum(x)/len(x) for x in zip(*all_metrics)]
        
        # Measure Inference time (Batch size 1 for true latency measurement)
        dummy_input = torch.randn(1, total_in_channels, config["data"]["image_size"], config["data"]["image_size"]).to(DEVICE)
        inf_time = measure_inference_time(model, dummy_input)
        
        # Save results
        results.append([model_name, *avg_metrics, inf_time])

    # ==========================================
    # 4. GENERATE THE PAPER TABLE
    # ==========================================
    columns = ["Model", "Precision", "Recall", "F1-score", "IoU_BG", "IoU_FG", "mIoU", "Inference Time (s)"]
    df = pd.DataFrame(results, columns=columns)
    
    # Format exactly like the image
    for col in ["Precision", "Recall", "F1-score", "IoU_BG", "IoU_FG", "mIoU", "Inference Time (s)"]:
        df[col] = df[col].apply(lambda x: f"{x:.3f}")
        
    print("\n\n" + "="*80)
    print("FINAL BENCHMARK RESULTS FOR BMVC PAPER")
    print("="*80)
    print(df.to_markdown(index=False))
    
    # Save to CSV
    df.to_csv("data/runs/segmentation/benchmark_results.csv", index=False)
    print("Results saved to data/runs/segmentation/benchmark_results.csv")


# --- PUT THIS AT THE ABSOLUTE BOTTOM OF THE FILE ---

if __name__ == "__main__":
    print("🚀 Script execution started!")
    
    from pathlib import Path
    
    # 1. Provide the exact path to your pre-trained JEPA weights
    weights = Path("/home/moonlab/marsls-seg/data/runs/ms-jepa-sw/20260511-172652_rare-snowball-146")
    
    print(f"Loading weights from: {weights}")
    
    # 2. Call the main training/benchmarking function 
    # (If your function is named 'train', change 'run_benchmark' to 'train')
    try:
        run_benchmark(weights_dir=weights) 
    except NameError:
        # Just in case you kept the old function name
        train(weights_dir=weights)
        
    print("✅ Script execution finished!")
   