# Khai báo utilities
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Khai báo thư viện
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import densenet121, DenseNet121_Weights
from mambapy.mamba import Mamba, MambaConfig

from training.models import run_experiment
from utilities.dataset import NIHChestXrayDataset

# ============================================================
# MODEL SETTINGS
MODEL_NAME = "CNN Mamba_v1.2.1"
MAMBA_DIM = 256
MAMBA_LAYERS = 2
DROPOUT = 0.35
PRETRAINED_CNN = True
dataset_class = NIHChestXrayDataset

# ============================================================
# Mô hình: CNN + MAMBA MODEL
class CNNMamba(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        
        # DenseNet121
        weights = (DenseNet121_Weights.IMAGENET1K_V1 if PRETRAINED_CNN else None)
        backbone = densenet121(weights=weights)

        # Bỏ classifier gốc (Output: [B, 1024, 7, 7] khi input = 224 x 224.)
        self.cnn = (backbone.features)
        
        # Projection
        self.projection = nn.Conv2d(
            in_channels=1024,
            out_channels=MAMBA_DIM,
            kernel_size=1,
            bias=False,
        )
        
        self.projection_norm = (nn.BatchNorm2d(MAMBA_DIM))
        
        # Mamba
        mamba_config = MambaConfig(
            d_model=MAMBA_DIM,
            n_layers=MAMBA_LAYERS,
        )
        
        self.mamba = Mamba(mamba_config)
        self.final_norm = (nn.LayerNorm(MAMBA_DIM))
        
        # Classifier
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(MAMBA_DIM * 2, num_classes)

    def forward(self, x):
        
        # CNN feature extractor (Expected: [B, 1024, 7, 7])
        x = self.cnn(x)
        x = F.relu(x, inplace=False)

        # Projection (Expected: [B, 256, 7, 7])
        x = self.projection(x)
        x = self.projection_norm(x)
        x = F.silu(x)
        
        # 2D feature map -> sequence | [B, D, H, W] -> [B, H*W, D] (Expected: [B, 49, 256])
        x = x.flatten(start_dim=2)
        x = x.transpose(1, 2)
        
        # Mamba
        x = self.mamba(x)
        x = self.final_norm(x)

        # Sequence pooling (Expected: [B, 256])
        avg_pool = x.mean(dim=1)
        max_pool = x.amax(dim=1)

        x = torch.cat([avg_pool, max_pool], dim=1,)
        
        # Classification
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits

LOSS_CONFIG = {
    "name": "asl",
    "gamma_neg": 4.0,
    "gamma_pos": 0.0,
    "clip": 0.05,
}

if __name__ == "__main__":
    run_experiment(
        model_name      = MODEL_NAME,
        dataset_class   = NIHChestXrayDataset, 
        model_factory   = CNNMamba,
        model_kwargs    = {},
        loss_config=LOSS_CONFIG,
    )
