# Khai báo utilities
import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Khai báo thư viện
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

from torchvision.models import densenet121, DenseNet121_Weights
from mambapy.mamba import Mamba, MambaConfig

from training.models import run_experiment
from utilities.dataset import NIHChestXrayDataset

# ============================================================
# MODEL SETTINGS
MODEL_NAME = "CNN BiMamba_v1"
FEAT_HW = config.IMAGE_SIZE // 32
MAMBA_DIM = 256
MAMBA_LAYERS = 2
DROPOUT = 0.2
dataset_class = NIHChestXrayDataset

# ============================================================
# Mô hình: Bi Mamba
class BiMamba(nn.Module):
    def __init__(self, d, n_layers):
        super().__init__()
        cfg = MambaConfig(d_model=d, n_layers=n_layers)
        self.fwd, self.bwd = Mamba(cfg), Mamba(cfg)
        self.fuse = nn.Linear(2 * d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        b = self.bwd(x.flip(1)).flip(1)
        return self.norm(x + self.fuse(torch.cat([self.fwd(x), b], dim=-1)))

# ============================================================
# Mô hình: Class Query Head
class ClassQueryHead(nn.Module):
    def __init__(self, d, num_classes, heads=4, dropout=0.1):
        super().__init__()
        self.q = nn.Parameter(torch.randn(num_classes, d) * 0.02)
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.w = nn.Parameter(torch.randn(num_classes, d) * 0.02)
        self.b = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):                                 # [B, N, D]
        kv = self.norm(x)
        q = self.q.unsqueeze(0).expand(x.size(0), -1, -1)
        out, _ = self.attn(q, kv, kv)                     # [B, C, D]
        return (out * self.w).sum(-1) + self.b            # [B, C]
    
# ============================================================
# Mô hình: CNN + MAMBA MODEL
class CNNMamba(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).features
        self.projection = nn.Conv2d(1024, MAMBA_DIM, 1, bias=False)
        self.projection_norm = nn.BatchNorm2d(MAMBA_DIM)

        self.pos = nn.Parameter(torch.zeros(1, FEAT_HW * FEAT_HW, MAMBA_DIM))
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.mamba = BiMamba(MAMBA_DIM, MAMBA_LAYERS)
        self.head = ClassQueryHead(MAMBA_DIM, num_classes)

        # Nhánh skip từ GAP, giúp ổn định giai đoạn đầu
        self.skip = nn.Linear(MAMBA_DIM, num_classes)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = F.relu(self.cnn(x))
        x = F.silu(self.projection_norm(self.projection(x)))   # [B, D, H, W]
        gap = x.mean(dim=(2, 3))                               # [B, D]
        x = x.flatten(2).transpose(1, 2)                       # [B, N, D]
        if x.size(1) != self.pos.size(1):
            raise ValueError(f'Số token {x.size(1)} khác pos {self.pos.size(1)}; kiểm tra IMG_SIZE.')
        x = self.mamba(x + self.pos)
        
        return self.head(self.dropout(x)) + self.skip(self.dropout(gap))

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
        loss_config     = LOSS_CONFIG,
    )
