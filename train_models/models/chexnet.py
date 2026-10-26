
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

from torchvision.models import densenet121, DenseNet121_Weights

from training.models import run_experiment
from utilities.dataset import NIHChestXrayDataset

# ============================================================
# MODEL SETTINGS
MODEL_NAME = "CheXNet_v1"
PRETRAINED_CNN = True

dataset_class = NIHChestXrayDataset

# ============================================================
# Mô hình: CheXNet (DenseNet121)
class CheXNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # DenseNet121
        weights = (
            DenseNet121_Weights.IMAGENET1K_V1
            if PRETRAINED_CNN else None
        )

        self.cnn = densenet121(weights=weights)

        # Thay classifier gốc bằng bộ phân loại đa nhãn
        in_features = self.cnn.classifier.in_features
        self.cnn.classifier = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):
        # Output: [B, num_classes] (logits)
        return self.cnn(x)


# ============================================================
# Loss: Giữ nguyên ASL để so sánh với CNN-BiMamba
LOSS_CONFIG = {
    "name": "asl",
    "gamma_neg": 4.0,
    "gamma_pos": 0.0,
    "clip": 0.05,
}

# ============================================================
# Run Experiment
if __name__ == "__main__":
    run_experiment(
        model_name=MODEL_NAME,
        dataset_class=NIHChestXrayDataset,
        model_factory=CheXNet,
        model_kwargs={},
        loss_config=LOSS_CONFIG,
    )
