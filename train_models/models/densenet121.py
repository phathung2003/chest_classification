# Khai báo utilities
import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Khai báo thư viện
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import densenet121, DenseNet121_Weights

import config
from training.experiment import run_experiment
from utilities.dataset import NIHChestXrayDataset

# ============================================================
# MODEL SETTINGS
MODEL_NAME = "DenseNet121 Baseline_v1"
dataset_class = NIHChestXrayDataset

# ============================================================
# Mô hình: DenseNet-121
class DenseNetBaseline(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = densenet121(weights = DenseNet121_Weights.IMAGENET1K_V1).features
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.cnn(x))                                # [B, 1024, H/32, W/32]
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)             # [B, 1024]
        return self.classifier(x)

if __name__ == "__main__":
    run_experiment(
        model_name     = MODEL_NAME,
        dataset_class  = NIHChestXrayDataset,
        model_factory  = DenseNetBaseline,
        model_kwargs   = {},
        **config.EXPERIMENT_PROTOCOL,
    )
