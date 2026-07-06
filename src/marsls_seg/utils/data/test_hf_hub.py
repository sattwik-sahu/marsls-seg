from datasets import load_dataset

# 1. Load a specific split from the Hub (train, val, or test)
dataset = load_dataset("sattwik21/mmls-v2", split="train")

# 2. Automatically format all arrays as native PyTorch Tensors
dataset = dataset.with_format("torch")

# 3. Pull a sample to verify
sample = dataset[0]

# 4. Verify tensor dimensions
print("RGB shape:", sample["rgb"].shape)  # Expected: torch.Size([3, 128, 128])
print("DEM shape:", sample["dem"].shape)  # Expected: torch.Size([1, 128, 128])
print(
    "Thermal shape:", sample["thermal_inertial"].shape
)  # Expected: torch.Size([1, 128, 128])
print(
    "Grayscale shape:", sample["grayscale"].shape
)  # Expected: torch.Size([1, 128, 128])
print("Label shape:", sample["label"].shape)  # Expected: torch.Size([128, 128])
