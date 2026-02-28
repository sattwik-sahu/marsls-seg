# MARS Landslide Segmentation

Team MOON Lab (IISER Bhopal)

---

## Installation

*Coming soon*

## Usage

### Training

#### Multispectral JEPA

```bash
marsls-seg train ms_jepa_sw path/to/config.yaml
```

### Model Inference

#### Multispectral JEPA (Pre-trained)

You can load the pre-trained Multispectral JEPA encoder and predictor as shown below.

```python
from marsls_seg.utils.models.ms_jepa import load_ms_jepa
from pathlib import Path

# The path to the directory generated after training
dir_path: Path = Path("path/to/save/dir.yaml")
device: torch.device = ... # torch.device -> {cpu, cuda, mps}
ms_jepa = load_ms_jepa(dir_path=dir_path, device=device)

encoder = ms_jepa["encoder"]
predictor = ms_jepa["predictor"]  # If required
```
